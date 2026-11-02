// testworker -- a standalone C++ driver for the `ovgenai worker` subprocess and its line protocol
// (one <command>:<session>:<body> line each way).  It is the test-side counterpart to the worker
// (the "consume" side a client / web server would be): it spawns ONE worker, talks to it over a real
// stdin/stdout, and gathers the worker's stdout lines until a request completes.
//
// WHY a separate binary (and not an extension of worker_cli_test.cpp):
//   * worker_cli_test.cpp feeds the subprocess /dev/null on stdin and reads stdout only AS A
//     WHOLE after the child exits (its runOvgenai).  The worker is the opposite -- a long-lived
//     process that streams JSON in BOTH directions -- so that one-shot shape is the wrong tool.
//   * the goal is to drive MANY sessions in parallel, from different threads, over ONE subprocess
//     (the "multiple sessions for one" requirement).  GoogleTest runs its cases one at a time and
//     is a poor home for that; a plain binary built on std::thread is the natural fit.
//
// WHAT this release actually does (the for-now scope): EXACTLY one session over the single
// subprocess, asserting only the PLUMBING -- the worker starts and emits READY, the one GENERATE
// streams a turn (per-chunk <item> lines then a terminal <done>), and the worker shuts down cleanly
// (exit 0).  Per the user, the worker's
// OUTPUT is not inspected (a random-weight model emits rubbish); only the shapes that prove the
// channel live.  `--sessions N` (default 1) is wired through main so N sessions can be driven over
// the one subprocess from N threads; the ctest entry ships with N==1, and the multi-session run is
// intentionally NOT registered there, because making the PRODUCTION worker fully session-/
// thread-aware is a future session -- only this testInfrastructure is delivered here.
//
// Thread-/session-safety of the Supervisor (the "prepared" part), WITHOUT touching production:
//   * ONE subprocess; many sessions ride its one stdin/stdout, each a distinct (session_id,req_id).
//   * ONE stdout reader thread splits the byte stream into complete lines and pushes them onto a
//     single inbox guarded by a mutex + a condition_variable.
//   * awaitResult(req_id) pops ONLY the first inbox line whose req_id matches -- so N concurrent
//     session-threads, each awaiting its own (unique) req_id, can never consume each other's
//     response: the same mutex the reader takes (to push) and every waiter takes (to pop its own)
//     serialises them, and a waiter never removes a line carrying a different req_id.
//   * writeLine() locks the single stdin and writes the whole line + "\n" in a full-write loop, so
//     two session-threads writing at once can never split a line.
//
// No third-party deps in THIS file (POSIX fork/pipe/waitpid + the C++ standard library only).  The
// wire is the framing <command>:<session>:<body> -- the 1st token is the command / message type, the
// 2nd is the session id (so a client reads it then takes the body into its own response envelope), and
// the body is the JSON payload (split off verbatim after the 2nd colon).  We match the few keys we
// read (req_id, num_generated_tokens) by SUBSTRING against the body, not a full grammar.  No
// new/delete/malloc/free in the sense the project forbids: the Supervisor is owned through a
// std::unique_ptr (built with std::make_unique -- a subprocess is uniquely owned) and closed via an
// idempotent, RAII stop() the dtor calls (it joins the helper threads, reaps the child, closes every
// int fd); no raw new/delete.
#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <iostream>       // std::cerr
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>         // std::ostream (a member argument)
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>         // std::move
#include <vector>

namespace {

// One protocol line the worker emitted, split into its framing slots -- <command>:<sessionId>:
// <body>  (the wire format; here we mirror the worker's own parse on the reader side).  The command
// is the 1st token, the session id the 2nd (the routing key the reader/client reads WITHOUT parsing
// the JSON body), and the body the JSON payload (taken verbatim after the 2nd colon, so its own
// colons survive into it).  We keep req_id pulled from the body's "req_id" member for per-request
// correlation; `raw` is the whole physical line, so the per-request token count (a body member) is
// still pickInt-able for the session-count assertion.
struct Line {
  std::string type;        // the command / message-type token: READY / PONG / CANCEL / ITEM / DONE / ERROR
  std::string sessionId;   // the 2nd header slot ("" where the worker emitted none)
  std::string body;        // the JSON payload after the 2nd colon ("" only on a line with no body slot)
  std::int64_t reqId = 0;  // pulled from the body's "req_id" (0 where the body carries none)
  std::string raw;         // the whole physical line, for diagnostics
};

// The point of the new framing (the refinement): the session id is a header slot (l.sessionId),
// NOT a JSON member of the body -- so a client (a web server) reads the session, then drops l.body
// straight into its own response envelope (an OpenAI chat-completions "choices", which has no
// session id).  This asserts the crux the tests check: the body carries no "session_id" member.
bool bodyHasNoSessionId(const Line& l) {
  return l.body.find("\"session_id\"") == std::string::npos;
}

// One finished turn, gathered off the worker's stdout: every per-chunk <item> delta the turn STREAMED
// (drained as they arrive, so the shared inbox never piles up) PLUS its single terminal <done> -- whose
// body carries the WHOLE turn (content / reasoning_content / tool_calls / finish_reason / metrics), so a
// client keyed only on the <done> key still gets everything.  Both halves are matched by the SESSION id in
// the line header (the user's demux key -- not the body's req_id, which a bare <item> delta lacks), so N
// concurrent session-threads each gather only their own turn and never cross.  A `--stream 0` turn has no
// <item> deltas at all: its <done> is the lone line, and awaitTurn returns on it immediately.
struct Turn {
  std::vector<Line>   items;    // the per-chunk <item> deltas (0 for a --stream 0 turn)
  std::optional<Line> terminal; // the terminal <done> (the whole turn's body is in it)
  bool got = false;             // false only if the worker left before the turn's <done> arrived
};

// Escape a string for embedding in a hand-built JSON line: the only bytes that matter are ", \ and
// control chars; non-ASCII (UTF-8) passes through -- mirrors the project's own jsonEscape.
std::string jsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '"':  out += "\\\""; continue;
      case '\\': out += "\\\\"; continue;
      case '\b': out += "\\b";  continue;
      case '\f': out += "\\f";  continue;
      case '\n': out += "\\n";  continue;
      case '\r': out += "\\r";  continue;
      case '\t': out += "\\t";  continue;
      default:
        if (c < 0x20) {
          static const char* hex = "0123456789abcdef";
          out += "\\u00";
          out += hex[(c >> 4) & 0xF] + hex[c & 0xF];
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

// Pull a string member `key` ("key":"value") out of a line, or nullopt.  Best-effort and
// order-insensitive (we never assume the grammar); now it reads the BODY (the payload after the
// 2nd header colon) -- the session id is a header slot, not a body member, so it is never found here.
std::optional<std::string> pickString(const std::string& line, const std::string& key) {
  const std::string kq = "\"" + key + "\"";
  std::size_t j = line.find(kq);                 // the member, e.g.  "result"
  if (j == std::string::npos) return std::nullopt;
  j += kq.size();                                 // just after the key's closing quote
  while (j < line.size() && line[j] != ':') ++j;  // walk over any spaces to the ':'
  if (j >= line.size() || line[j] != ':') return std::nullopt;
  ++j;                                            // skip the ':'
  while (j < line.size() && line[j] == ' ') ++j;  // tolerate an optional space before the value quote
  if (j >= line.size() || line[j] != '"') return std::nullopt;  // the value must be a string
  const std::size_t v = j + 1;                    // the value's opening quote
  std::size_t end = v;
  while (end < line.size()) {
    if (line[end] == '\\') { end += 2; continue; }  // skip an escaped byte (", \, \n, ...)
    if (line[end] == '"') break;
    ++end;
  }
  if (end >= line.size()) return std::string();
  return line.substr(v, end - v);
}

// Pull a numeric member `key` ("key":12) as a signed long, from a line -- used for the body's
// req_id and (in driveOneSession) the result's num_generated_tokens.  The FIRST occurrence is used;
// the keys we read precede any long `result` text in the body's field order, so a stray key inside
// rubbish output cannot shadow the real one.
std::optional<std::int64_t> pickInt(const std::string& line, const std::string& key) {
  const std::string tok = "\"" + key + "\":";
  std::size_t k = line.find(tok);
  if (k == std::string::npos) return std::nullopt;
  k += tok.size();
  while (k < line.size() && line[k] == ' ') ++k;
  if (k >= line.size()) return std::nullopt;
  bool neg = false;
  if (line[k] == '-') { neg = true; ++k; }
  long long v = 0;
  bool any = false;
  while (k < line.size() && line[k] >= '0' && line[k] <= '9') {
    v = v * 10 + (line[k] - '0');
    ++k;
    any = true;
  }
  if (!any) return std::nullopt;
  return neg ? -v : v;
}

// Split a physical stdout line into its framing slots -- <command>:<sessionId>:<body>  -- mirroring
// the worker's own split on the reader side.  A blank line is nullopt (the worker never emits one),
// and a line with fewer than the two ':' is not a protocol line (nullopt, so a stray diagnostic line
// is dropped rather than mis-classified).  The 2nd colon splits off the session-id slot; the body is
// everything after it (verbatim, so its own colons survive).
std::optional<Line> parseLine(const std::string& raw) {
  if (raw.empty()) return std::nullopt;
  const std::size_t c1 = raw.find(':');
  if (c1 == std::string::npos) return std::nullopt;
  const std::size_t c2 = raw.find(':', c1 + 1);
  if (c2 == std::string::npos) return std::nullopt;
  Line l;
  l.type      = raw.substr(0, c1);
  l.sessionId = raw.substr(c1 + 1, c2 - c1 - 1);
  l.body      = raw.substr(c2 + 1);
  l.raw       = raw;                                  // whole line: the body's num_tokens is pickInt-able
  if (auto r = pickInt(l.body, "req_id")) l.reqId = *r;
  return l;
}

// Assemble one OUTGOING line for the worker's stdin: <command>:<sessionId>:<body> -- mirrors the
// worker's buildFramedLine.  The body is a JSON object the worker reads (a req_id, a gen_config:
// req_id for correlation, the body itself the wrappable payload); the session id is the 2nd slot.
std::string frame(const std::string& command, const std::string& sessionId, const std::string& body) {
  return command + ":" + sessionId + ":" + body;
}

// Format an argv (program + args) as one readable line for the logs; quote a token that carries
// a space / tab / quote / backslash.
std::string fmtArgs(const std::vector<std::string>& a) {
  std::string s;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (i) s += ' ';
    const std::string& t = a[i];
    const bool q = (t.find_first_of(" \t\"\\") != std::string::npos);
    if (q) s += '"';
    s += t;
    if (q) s += '"';
  }
  return s;
}

// A one-per-session request as a gen_config object: a single user message plus the max_tokens cap.
// This is exactly the shape the worker consumes it two ways from (verified end-to-end against the
// built worker): the per-request sampling (modelRequestConfigFromJson ignores the unknown `messages`
// key) and the turn's messages (translateGenConfig reads the plain-string `content`).
std::string buildGenConfig(const std::string& prompt, std::int64_t maxTokens) {
  return std::string("{\"messages\":[{\"role\":\"user\",\"content\":\"") +
         jsonEscape(prompt) + "\"}],\"max_tokens\":" + std::to_string(maxTokens) + "}";
}

// ---------------------------------------------------------------------------
// Supervisor: one `ovgenai worker` subprocess + a background reader + stderr drainer.
// ---------------------------------------------------------------------------
class Supervisor {
 public:
  Supervisor(std::string program, std::vector<std::string> args,
             std::chrono::milliseconds reapTimeout)
      : m_program(std::move(program)),
        m_args(std::move(args)),
        m_reapTimeout(std::move(reapTimeout)) {}

  ~Supervisor() { stop(); }
  Supervisor(Supervisor const&) = delete;             // thread / owned-fd members
  Supervisor& operator=(Supervisor const&) = delete;

  // fork + exec + wire the three pipes (stdin / stdout / stderr), then launch the two drainers.
  // Returns false (with the handles already closed) if fork/exec fails -- the caller skips.
  bool start() {
    if (m_started) return true;
    int inp[2], outp[2], errp[2];
    if (::pipe(inp) != 0 || ::pipe(outp) != 0 || ::pipe(errp) != 0) {
      ::perror("testworker: pipe");
      cleanupFds();
      return false;
    }
    const pid_t pid = ::fork();
    if (pid < 0) {
      ::perror("testworker: fork");
      cleanupFds();
      return false;
    }
    if (pid == 0) {
      // Child (the worker): its stdin must point at the input pipe's READ end -- the parent feeds the
      // protocol by writing the pipe's WRITE end (m_inWrite = inp[1]), so a worker reading its own
      // stdin has to hold the READ end (inp[0]).  stdout/stderr point at the pipes' WRITE ends; the
      // parent's reader/drainer threads read their READ ends.  All six pipe ends are then closed in
      // the child -- the dup2'd fd 0/1/2 keep the worker's live copy of each end.
      ::dup2(inp[0], STDIN_FILENO);                // <was inp[1]: the WRITE end is unreadable -> the fix
      ::dup2(outp[1], STDOUT_FILENO);
      ::dup2(errp[1], STDERR_FILENO);
      for (int fd : {inp[0], inp[1], outp[0], outp[1], errp[0], errp[1]})
        ::close(fd);
      std::vector<char*> av;
      av.push_back(const_cast<char*>(m_program.c_str()));
      for (const auto& a : m_args) av.push_back(const_cast<char*>(a.c_str()));
      av.push_back(nullptr);
      ::execv(m_program.c_str(), av.data());
      const char* msg = "testworker: could not exec the worker (missing shared library? run under "
                        "`eval $(./setupenv.sh)`) -- execv failed";
      ssize_t r = ::write(STDERR_FILENO, msg, std::strlen(msg));
      (void)r;
      ::_exit(127);
    }
    // Parent: keep our write end into the worker's stdin and the two read ends the threads drain;
    // close the child's copies of those read/write ends so neither side can reach the other's buffer.
    ::close(inp[0]);
    ::close(outp[1]);
    ::close(errp[1]);
    m_pid = pid;
    m_inWrite = inp[1];                     // we write the protocol here
    m_outRead = outp[0];                    // the reader thread drains this
    m_errRead = errp[0];                    // the drainer thread drains this
    // CLOEXEC hygiene: if this process ever forks again, the pipes must not leak into the grandchild.
    ::fcntl(m_inWrite, F_SETFD, FD_CLOEXEC);
    ::fcntl(m_outRead, F_SETFD, FD_CLOEXEC);
    ::fcntl(m_errRead, F_SETFD, FD_CLOEXEC);
    m_stopped.store(false);
    m_eof.store(false);
    m_reader = std::thread([this] { readLoop(); });
    m_drainer = std::thread([this] { drainLoop(); });
    m_started = true;
    return true;
  }

  // --- writer side (thread-safe): feed the worker one protocol line on its stdin ---------------

  // Write a complete line + "\n" atomically (locked, full-write loop) so two session-threads writing
  // at once can never split a line -- the whole line is serialised on ONE mutex (the kernel only
  // makes a single write() of < PIPE_BUF indivisible, which a long JSON line can exceed).
  void writeLine(const std::string& json) {
    std::lock_guard<std::mutex> lk(m_writeMtx);
    if (m_inWrite < 0) return;              // already closed (stopped): a no-op
    const std::string data = json + "\n";
    std::size_t off = 0;
    while (off < data.size()) {
      ssize_t n = ::write(m_inWrite, data.data() + off, data.size() - off);
      if (n < 0) {
        if (errno == EINTR) continue;
        break;                             // broken pipe (child already gone): stop writing
      }
      off += static_cast<std::size_t>(n);
    }
  }

  // Send a GENERATE for (session, req_id) carrying a gen_config (the request's messages + sampling):
  //   GENERATE:<session>:{"req_id":N,"gen_config":{...}}
  // The session is the 2nd header slot; the body is the wrappable JSON (req_id for correlation + the
  // gen_config the worker folds onto its CLI base).
  void sendRequest(const std::string& session, std::int64_t reqId, const std::string& genConfig) {
    std::string body = std::string("{\"req_id\":") + std::to_string(reqId) +
                       ",\"gen_config\":" + genConfig + "}";
    writeLine(frame("GENERATE", session, body));
  }

  // A bare control op (PING / CANCEL): <op>:<session>:{"req_id":N} -- a req_id-less control just
  // carries {} so the body slot is never blank (the worker skips a blank line only on a fully empty
  // buffer, but a present, parseable {} keeps its ack uniform).  The session id is the 2nd slot.
  void sendControl(const std::string& op, const std::string& session, std::int64_t reqId) {
    const std::string body =
        reqId ? std::string("{\"req_id\":") + std::to_string(reqId) + "}" : std::string("{}");
    writeLine(frame(op, session, body));
  }

  // The terminal shutdown op: a global, session-less, req-less -- UNLOAD::{}  -- the worker stops its
  // pump, emits a lone DONE, exits.
  void unload() { writeLine(frame("UNLOAD", std::string(), "{}")); }

  // --- reader side (the corollary for each request; safe for many concurrent callers) ----------

  // Wait for, then remove, the NEXT inbox line whose req_id matches -- so concurrent session-threads
  // each only take their own response.  Returns nullopt on timeout or once the worker has exited
  // (m_eof) without one.
  std::optional<Line> awaitResult(std::int64_t wantReqId, std::chrono::milliseconds timeout) {
    Line got;
    bool have = false;
    {
      std::unique_lock<std::mutex> lk(m_inbox);
      const auto deadline = std::chrono::steady_clock::now() + timeout;
      m_cv.wait_until(lk, deadline, [&] {
        for (std::size_t i = 0; i < m_lines.size(); ++i) {
          if (m_lines[i].reqId == wantReqId) {
            got = std::move(m_lines[i]);
            m_lines.erase(m_lines.begin() + i);
            have = true;
            return true;
          }
        }
        return m_eof.load();              // a dead worker: stop waiting even if still no match
      });
    }
    return have ? std::optional<Line>(std::move(got)) : std::nullopt;
  }

  // Like awaitResult but matches on `type` (a broadcast control line -- READY has no req_id, so for
  // the warm-up we want the first line of that type regardless of req_id).  Consumes it so it does
  // not confuse a later per-req waiter.
  std::optional<Line> expectControl(const std::string& type, std::chrono::milliseconds timeout) {
    Line got;
    bool have = false;
    {
      std::unique_lock<std::mutex> lk(m_inbox);
      const auto deadline = std::chrono::steady_clock::now() + timeout;
      m_cv.wait_until(lk, deadline, [&] {
        for (std::size_t i = 0; i < m_lines.size(); ++i) {
          if (m_lines[i].type == type) {
            got = std::move(m_lines[i]);
            m_lines.erase(m_lines.begin() + i);
            have = true;
            return true;
          }
        }
        return m_eof.load();
      });
    }
    return have ? std::optional<Line>(std::move(got)) : std::nullopt;
  }

  // Gather one finished turn for `session`: drain EVERY per-chunk <item> the worker streamed for it (so
  // the shared inbox never piles up), then return its single terminal <done> -- whose body carries the
  // whole turn.  We match by the SESSION id in the header, NOT the body's req_id (a bare <item> delta
  // carries no req_id, so a req_id match would grab a stray delta rather than the terminal) -- which is
  // why the multi-session run demuxes by session.  A --stream 0 turn has no <item>s: awaitTurn returns on
  // the lone <done>.  got is false only if the worker left (m_eof) before the <done> arrived.
  Turn awaitTurn(const std::string& session, std::chrono::milliseconds timeout) {
    Turn t;
    std::unique_lock<std::mutex> lk(m_inbox);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    m_cv.wait_until(lk, deadline, [&] {
      for (std::size_t i = 0; i < m_lines.size(); ++i) {
        const std::string& sid = m_lines[i].sessionId;
        if (sid != session) continue;                  // another session's line: leave it for its own gatherer
        Line moved = std::move(m_lines[i]);
        m_lines.erase(m_lines.begin() + i);
        --i;
        if (moved.type == "DONE") {                   // the terminal: capture it and stop (a turn ends with one <done>)
          t.terminal = std::move(moved);
          t.got = true;
          return true;
        }
        t.items.push_back(std::move(moved));          // a streamed <item> delta: drain + remember it
      }
      return m_eof.load();                            // keep waiting until the <done> or a dead worker
    });
    return t;
  }

  // End the child deterministically (close our stdin write end -> clean EOF), reap it (a wedged
  // worker is SIGTERM'd then SIGKILL'd so it can never hang the suite), join both threads, and close
  // every fd.  Idempotent; the dtor calls it.  No raw new/delete: only int fds are closed.
  void stop() {
    if (!m_started) {                       // start() never ran / already stopped: just release fds
      cleanupFds();
      return;
    }
    m_stopped.store(true);
    if (m_inWrite != -1) { ::close(m_inWrite); m_inWrite = -1; }  // clean EOF to the worker
    reapChild();                                                  // reap (with force fallback)
    if (m_reader.joinable()) m_reader.join();                     // the threads exit once their read
    if (m_drainer.joinable()) m_drainer.join();                  // ends EOF (the kernel closed the child's write ends)
    // Safety closes: the threads own m_outRead/m_errRead and set them to -1 before returning, so these
    // are no-ops on the happy path but guard a thread that bailed unexpectedly.
    if (m_outRead != -1) { ::close(m_outRead); m_outRead = -1; }
    if (m_errRead != -1) { ::close(m_errRead); m_errRead = -1; }
    m_started = false;
  }

  // Wait for the child to exit, reaping it; escalate to SIGTERM then SIGKILL if it does not within
  // the grace period, so a hung worker bounds out instead of wedging ctest.  Records the exit status.
  void reapChild() {
    const auto t0 = std::chrono::steady_clock::now();
    int status = 0;
    for (;;) {
      pid_t r = ::waitpid(m_pid, &status, WNOHANG);
      if (r == m_pid) { m_status = status; return; }
      if (r < 0) { m_status = -1; return; }                    // already reaped / no such child
      const auto elapsed = std::chrono::steady_clock::now() - t0;
      if (m_reapTimeout <= elapsed) {
        if (::kill(m_pid, SIGTERM) == 0) {
          for (int i = 0; i < 40; ++i) {   // ~2s grace for a graceful clean-up
            ::usleep(50 * 1000);
            if (::waitpid(m_pid, &status, WNOHANG) == m_pid) { m_status = status; return; }
          }
        }
        ::kill(m_pid, SIGKILL);
        ::waitpid(m_pid, &status, 0);       // a blocking wait is safe: SIGKILL always reaps
        m_status = status;
        return;
      }
      ::usleep(50 * 1000);
    }
  }

  int exitCode() const {
    if (!WIFEXITED(m_status))
      return WIFSIGNALED(m_status) ? 128 + WTERMSIG(m_status) : -1;
    return WEXITSTATUS(m_status);
  }

  pid_t pid() const { return m_pid; }
  const std::string& program() const { return m_program; }

  // A one-line summary of the inbox line types so far, for readable ctest output.
  void dumpLineTypes(std::ostream& os) const;          // defined after <ostream> is in scope
  std::string stderrLog() const;                      // captured worker diagnostics (shown on failure)

 private:
  // The stdout reader thread: read the pipe to completion, split on '\n', push each line (parsed)
  // onto the shared inbox (lock + CV).  The worker flushes per line, so every pushed line is whole.
  void readLoop() {
    std::string partial;
    char buf[8192];
    for (;;) {
      const ssize_t n = ::read(m_outRead, buf, sizeof buf);
      if (n < 0) {
        if (errno == EINTR) continue;
        eof();
        return;
      }
      if (n == 0) { eof(); return; }        // clean EOF: the child's write end closed (it exited)
      partial.append(buf, static_cast<std::size_t>(n));
      std::size_t pos;
      while ((pos = partial.find('\n')) != std::string::npos) {
        const std::string line = partial.substr(0, pos);
        partial.erase(0, pos + 1);
        if (auto l = parseLine(line)) {
          std::lock_guard<std::mutex> lk(m_inbox);
          m_lines.push_back(std::move(*l));
        }
      }
      m_cv.notify_all();
    }
  }

  // The stderr drainer thread: drain the diagnostics pipe continuously (so the worker can never block
  // on a full pipe even if it logs a lot) and append to m_errLog for post-hoc inspection.
  void drainLoop() {
    char buf[4096];
    for (;;) {
      const ssize_t n = ::read(m_errRead, buf, sizeof buf);
      if (n < 0) {
        if (errno == EINTR) continue;
        closeErr();
        return;
      }
      if (n == 0) { closeErr(); return; }
      std::lock_guard<std::mutex> lk(m_errMtx);
      m_errLog.append(buf, static_cast<std::size_t>(n));
    }
  }

  // Mark the protocol channel dead and wake every waiter.  m_eof is taken with m_inbox so a waiter's
  // re-armed predicate sees it on the same lock the reader used to set it (no lost wakeup).
  void eof() {
    {
      std::lock_guard<std::mutex> lk(m_inbox);
      m_eof.store(true);
    }
    m_cv.notify_all();
  }
  void closeErr() {
    if (m_errRead != -1) { ::close(m_errRead); m_errRead = -1; }
  }
  // Release whatever fds exist (used by start() on a fork/exec failure, before any thread runs).
  void cleanupFds() {
    if (m_inWrite != -1) { ::close(m_inWrite); m_inWrite = -1; }
    if (m_outRead != -1) { ::close(m_outRead); m_outRead = -1; }
    if (m_errRead != -1) { ::close(m_errRead); m_errRead = -1; }
  }

  // Data members in declaration order so the ctor init list avoids -Wreorder.
  std::string m_program;             // the worker binary (an absolute, configure-time-baked path)
  std::vector<std::string> m_args;
  std::chrono::milliseconds m_reapTimeout;
  pid_t m_pid = -1;
  int m_inWrite = -1;                // fd into the worker's stdin   (parent writes)
  int m_outRead = -1;                // fd from the worker's stdout  (reader thread reads)
  int m_errRead = -1;                // fd from the worker's stderr  (drainer thread reads)
  int m_status = 0;
  bool m_started = false;

  std::atomic<bool> m_stopped{false};
  std::atomic<bool> m_eof{false};              // set once the stdout pipe reaches EOF (worker gone)

  // Mutable: the const-observer methods (stderrLog / dumpLineTypes) lock these to read shared state --
  // taking a lock is a benign mutation, which is exactly why a std::mutex guarding shared state is mutable.
  mutable std::mutex m_writeMtx;            // serialises whole-line writes on the single stdin
  mutable std::mutex m_inbox;              // guards m_lines (push from the reader, pop from waiters)
  std::condition_variable m_cv;
  std::deque<Line> m_lines;                // the in-flight / not-yet-claimed protocol lines

  mutable std::mutex m_errMtx;
  std::string m_errLog;                   // captured stderr (worker diagnostics; shown on failure)

  std::thread m_reader;
  std::thread m_drainer;
};

std::string Supervisor::stderrLog() const {
  std::lock_guard<std::mutex> lk(m_errMtx);
  return m_errLog;
}

void Supervisor::dumpLineTypes(std::ostream& os) const {
  std::lock_guard<std::mutex> lk(m_inbox);
  os << "  inbox line types: [";
  for (std::size_t i = 0; i < m_lines.size(); ++i) {
    if (i) os << ", ";
    if (m_lines[i].type.empty())
      os << "?" + (m_lines[i].raw.size() > 24 ? std::string("...") + m_lines[i].raw.substr(0, 21)
                                              : m_lines[i].raw);
    else
      os << m_lines[i].type;
  }
  os << "]";
}

// A cheap "can the worker exec at all" probe (mirrors the gtest suite's ovgenaiRuns gate).  Runs
// `<program> version` -- which loads the shared libraries but no model -- so an unprovisioned box
// (LD_LIBRARY_PATH unset) makes a SKIP (ctest SKIP_RETURN_CODE 77) rather than a hard failure.
// On any failure, dump the EVIDENCE by DEFAULT (not just --verbose), so a reader of a ctest log sees
// it at a glance: the worker's own stdout line-types + its captured stderr (what it actually did).
// This test is a pure send + gather client, then -- the worker is the only long-lived external process
// it drives -- so the captured lines ARE the report.  Read them to tell which side is to blame:
//   * the worker emitted the expected lines (READY / <item> stream / <done> / .) but they did not line up ->
//     a PRODUCTION protocol / behaviour bug to fix in the worker (a later session, not this harness);
//   * the worker emitted nothing, or only its load lines, and exited early -> either a production
//     defect (the worker died) or THIS harness's own pipe wiring (start()'s fork / dup2 / 3 pipes) --
//     the captured stderr tells which.  We do NOT paper over a failure; we surface the ground truth.
void reportFail(Supervisor& sup, const std::string& why) {
  std::fprintf(stderr, "testworker: FAIL -- %s\n", why.c_str());
  std::fprintf(stderr, "  the worker emitted on stdout:");
  sup.dumpLineTypes(std::cerr);
  std::fprintf(stderr, "\n  captured worker stderr:\n%s", sup.stderrLog().c_str());
  if (sup.stderrLog().empty()) std::fprintf(stderr, "\n");
  std::fprintf(stderr, "  => the worker is the only external process this test drives -- the lines above "
                      "are the evidence (expected: READY -> a per-turn <item> stream -> <done> -> ...).  Empty / early output "
                      "points at the worker or at this harness; check the captured stderr.\n");
}

bool workerCanExec(const std::string& program) {
  int outp[2];
  if (::pipe(outp) != 0) return false;
  const pid_t pid = ::fork();
  if (pid < 0) { ::close(outp[0]); ::close(outp[1]); return false; }
  if (pid == 0) {
    int devnull = ::open("/dev/null", O_RDONLY);
    ::dup2(outp[1], STDOUT_FILENO);
    ::dup2(outp[1], STDERR_FILENO);
    ::dup2(devnull, STDIN_FILENO);
    ::close(devnull);
    ::close(outp[0]);
    ::close(outp[1]);
    std::vector<char*> av;
    av.push_back(const_cast<char*>(program.c_str()));
    av.push_back(const_cast<char*>("version"));
    av.push_back(nullptr);
    ::execv(program.c_str(), av.data());
    ::_exit(127);
  }
  ::close(outp[1]);
  char buf[4096];
  ssize_t n;
  while ((n = ::read(outp[0], buf, sizeof buf)) > 0) { /* discard the tiny output */ }
  ::close(outp[0]);
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) return false;
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// ---------------------------------------------------------------------------
// The harness: one Supervisor, one or many sessions, then a clean shutdown.
// ---------------------------------------------------------------------------

struct Opts {
  std::string program;                  // default: OVGENAI_WORKER_BIN (baked absolute)
  std::string model;                    // default: OVGENAI_VLM_FIXTURE (baked absolute)
  std::string device = "CPU";
  std::string pipeline = "vlm";         // the run/worker body (vlm is the one with a real body)
  std::int64_t maxTokens = 8;           // small (the user: 5-10 is enough); we never check the output
  std::string prompt = "Hi, who are you?";
  int sessions = 1;                     // for now 1 (registered in ctest); the >1 path is "prepared"
  std::chrono::milliseconds readyTimeout{120000};
  std::chrono::milliseconds reqTimeout{120000};
  std::chrono::milliseconds reapTimeout{20000};
  bool verbose = false;
  // --protocol-check: instead of the multi-session drive, run the framed-protocol assertions
  // (READY/PONG/ITEM/DONE/CANCEL all carry the session id in the 2nd header slot, with no
  // "session_id" leaking into any JSON body) over the one worker.
  bool protocolCheck = false;
};

// Drive ONE session over the (shared, ONE) supervisor: send a GENERATE, then GATHER that session's whole
// turn off stdout -- every per-chunk <item> the worker streamed FOR IT plus the single terminal <done>
// (awaitTurn, matched by the SESSION id in the line header, NOT the body's req_id a bare <item> delta
// lacks -- so N parallel session-threads never cross) -- and return how many tokens the turn generated (the
// <done> body's "num_generated_tokens") -- or -1 if the turn never completed (the <done> never arrived) or
// the protocol was broken (a line whose body carries a "session_id", or a terminal that is not a <done> for
// THIS session -- printed to stderr so the run's final report still shows the ground truth).  This per-
// session count is what the multi-session run checks (the user's "8 tokens per session in the end"; the
// tiny-random fills the --max-tokens cap before any EOS, so a finished turn reports exactly that many).
// The flow is the natural send-then-gather one; a worker that could not stay up / answer while idle would
// leave its <done> never arriving (the timeout -> -1 -> a loud, evidence-dumped FAIL from the (4) block).
std::int64_t driveOneSession(Supervisor& sup, const std::string& sessionId, std::int64_t reqId,
                             const Opts& o) {
  // The natural consumer flow on purpose: send the request, then gather the session's turn until the
  // terminal <done> (no pre-feeding, no waiting on a req_id the bare <item> deltas do not carry).
  sup.sendRequest(sessionId, reqId, buildGenConfig(o.prompt, o.maxTokens));
  const Turn t = sup.awaitTurn(sessionId, o.reqTimeout);
  if (!t.got) {
    if (o.verbose) std::fprintf(stderr, "  [session %s] no terminal <done> for req_id %lld within %lld ms\n",
                                 sessionId.c_str(), static_cast<long long>(reqId),
                                 static_cast<long long>(o.reqTimeout.count()));
    return -1;                                         // the turn's <done> (its completion) never came
  }
  // The refine, on EVERY line of the turn: the session id is a header slot, none in any body.  Do NOT stop
  // the (shared) child here -- only the main thread may, at the end -- so print the diagnostic and report a
  // short count (-1 / 0) that the run's (4) block turns into a loud, evidence-dumped failure.
  if (t.terminal->type != "DONE") {
    std::fprintf(stderr, "  [session %s] protocol error: the turn's terminal is a '%s' line, not a <done>: %s\n",
                 sessionId.c_str(), t.terminal->type.c_str(), t.terminal->raw.c_str());
    return -1;
  }
  if (t.terminal->sessionId != sessionId) {
    std::fprintf(stderr, "  [session %s] protocol error: the terminal <done> echoed a different session id '%s'\n",
                 sessionId.c_str(), t.terminal->sessionId.c_str());
    return -1;
  }
  for (const Line& it : t.items)
    if (!bodyHasNoSessionId(it)) {
      std::fprintf(stderr, "  [session %s] protocol error: an <item> body carries a \"session_id\": %s\n",
                   sessionId.c_str(), it.raw.c_str());
      return -1;
    }
  if (!bodyHasNoSessionId(t.terminal.value())) {
    std::fprintf(stderr, "  [session %s] protocol error: the <done> body carries a \"session_id\": %s\n",
                 sessionId.c_str(), t.terminal->raw.c_str());
    return -1;
  }
  // The count lives in the <done> body's "metrics"; pick the first "num_generated_tokens" (the field order
  // puts the content -- which a random model emits -- before metrics, but a raw token blob does not form a
  // real JSON key, so the first occurrence is the real one; this first-occurrence read is the proven shape
  // this test already passes, kept for byte-identical counting behaviour).
  const std::int64_t tokens = pickInt(t.terminal->raw, "num_generated_tokens").value_or(0);
  if (o.verbose)
    std::fprintf(stderr, "  [session %s] req_id %lld: %zu <item>(%s) -> <done> %lld generated token(s) (want %lld)\n",
                 sessionId.c_str(), static_cast<long long>(reqId), t.items.size(),
                 t.items.empty() ? "none (a --stream 0 turn)" : "",
                 static_cast<long long>(tokens), static_cast<long long>(o.maxTokens));
  return tokens;
}

// --- the --protocol-check mode (separate from the session-count drive) ----------------------------
// Drive the WHOLE framing end-to-end over the real worker and assert the refinement on EVERY message
// the worker emits: the session id is the 2nd header slot, and the JSON body carries no "session_id" --
// so a client (a web server) can read the session, then drop the body into its own response envelope
// (an OpenAI chat-completions "choices", which has no session id) without extracting one.  A missing /
// header-absent / body-embedded session id fails the step loudly.  Steps (1)-(4) self-stop+report on a
// mismatch; (5) drives the clean UNLOAD -> DONE and a 0 exit is asserted.
int runProtocolCheck(Supervisor& sup, const Opts& o) {
  // A failing step: stop the child deterministically, then dump the evidence (captured lines + the
  // worker's own stderr) a reader can reason with.  Returns 1 (a non-77 non-zero, so ctest FAILs).
  auto fail = [&](const std::string& why) -> int {
    sup.stop();
    reportFail(sup, "protocol-check: " + why);
    return 1;
  };
  if (o.verbose)
    std::fprintf(stderr, "[protocol-check] driving the framed lines over pid %ld\n",
                 static_cast<long>(sup.pid()));

  // (1) Up-front READY: a global, session-less line <READY>:"":<{}> -- no session slot, and (the crux)
  //     no "session_id" inside the body.
  {
    const std::optional<Line> l = sup.expectControl("READY", o.readyTimeout);
    if (!l) return fail("no READY line from the worker");
    if (l->type != "READY")
      return fail("the first line is not the up-front READY: found a " + l->type + " line");
    if (!l->sessionId.empty())
      return fail("the up-front READY should carry no session id; the header slot was '"
                  + l->sessionId + "'");
    if (!bodyHasNoSessionId(*l))
      return fail("the READY body must not carry a \"session_id\" member: " + l->body);
    if (o.verbose) std::fprintf(stderr, "  READY  (global) ok\n");
  }

  // (2) PING -> PONG: <PING>:<sid>:{"req_id":N} is echoed <PONG>:<sid>:{"req_id":N} -- the session
  //     rides the header (not the body) and is echoed back.
  {
    sup.sendControl("PING", "ping1", 7);
    const std::optional<Line> l = sup.expectControl("PONG", o.reqTimeout);
    if (!l) return fail("no PONG for the PING");
    if (l->type != "PONG") return fail("expected a PONG, found a " + l->type + " line");
    if (l->sessionId != "ping1")
      return fail("the PONG echoed the wrong session id: '" + l->sessionId + "' (want 'ping1')");
    if (pickInt(l->body, "req_id").value_or(0) != 7)
      return fail("the PONG did not echo the request id: " + l->body);
    if (!bodyHasNoSessionId(*l))
      return fail("the PONG body must not carry a \"session_id\" member: " + l->body);
    if (o.verbose) std::fprintf(stderr, "  PONG   session=%s ok\n", l->sessionId.c_str());
  }

  // (3) GENERATE -> a STREAMED turn: <GENERATE>:<sid>:{"req_id":N,"gen_config":{...}} streams per-chunk
  //     <ITEM>:<sid>:<delta> lines (the worker's default; `--stream` would change the window) and then a
  //     single terminal <DONE>:<sid>:<body> -- whose body is the WHOLE turn (content / reasoning_content /
  //     tool_calls / finish_reason / metrics), so a client keyed only on DONE gets everything.  We drain
  //     the whole turn (awaitTurn, matched by the header session 'pc1', NOT the body's req_id the bare
  //     <item> deltas lack) and assert: at least one <item> streamed, every line's body is session-free,
  //     the terminal is a <DONE> for 'pc1', and the generation ran (>0 tokens).
  {
    sup.sendRequest("pc1", 1, buildGenConfig(o.prompt, o.maxTokens));
    const Turn t = sup.awaitTurn("pc1", o.reqTimeout);
    if (!t.got) return fail("no terminal <done> for the GENERATE (request id 1)");
    if (t.terminal->type != "DONE")
      return fail("expected a terminal <done>; the turn instead ended with a " + t.terminal->type + " line: " + t.terminal->raw);
    if (t.terminal->sessionId != "pc1")
      return fail("the terminal <done> echoed the wrong session id: '" + t.terminal->sessionId + "' (want 'pc1')");
    if (!bodyHasNoSessionId(t.terminal.value()))
      return fail("the <done> body must not carry a \"session_id\" member: " + t.terminal->raw);
    if (t.items.empty())
      return fail("the GENERATE streamed no <item> deltas -- the default --stream should stream a streamed turn: " + t.terminal->raw);
    for (const Line& it : t.items) {
      if (it.sessionId != "pc1")
        return fail("an <item> delta echoed the wrong session id: '" + it.sessionId + "' (want 'pc1')");
      if (!bodyHasNoSessionId(it))
        return fail("an <item> body must not carry a \"session_id\" member: " + it.raw);
    }
    if (pickInt(t.terminal->raw, "num_generated_tokens").value_or(0) <= 0)
      return fail("the <done> reported 0 generated tokens (the run did not execute): " + t.terminal->raw);
    if (o.verbose)
      std::fprintf(stderr, "  <ITEM> x%zu then <DONE> session=pc1 %lld token(s) ok\n", t.items.size(),
                   static_cast<long long>(pickInt(t.terminal->raw, "num_generated_tokens").value_or(0)));
  }

  // (4) CANCEL -> CANCEL ack: <CANCEL>:<sid>:{"req_id":N} is reflected <CANCEL>:<sid>:{"req_id":N}.
  //     Nothing is in flight on this session (the prior result already finished), so the cancel is a
  //     no-op -- but the worker still emits the framed ack, which is what this checks.
  {
    sup.sendControl("CANCEL", "pc2", 2);
    const std::optional<Line> l = sup.expectControl("CANCEL", o.reqTimeout);
    if (!l) return fail("no CANCEL ack for the CANCEL");
    if (l->type != "CANCEL") return fail("expected a CANCEL ack, found a " + l->type + " line");
    if (l->sessionId != "pc2")
      return fail("the CANCEL ack echoed the wrong session id: '" + l->sessionId + "' (want 'pc2')");
    if (pickInt(l->body, "req_id").value_or(0) != 2)
      return fail("the CANCEL ack did not echo the request id: " + l->body);
    if (!bodyHasNoSessionId(*l))
      return fail("the CANCEL body must not carry a \"session_id\" member: " + l->body);
    if (o.verbose) std::fprintf(stderr, "  CANCEL session=%s ok\n", l->sessionId.c_str());
  }

  // (5) Terminal UNLOAD -> <DONE>:"":<{}>, then a clean exit.
  sup.unload();
  {
    const std::optional<Line> l = sup.expectControl("DONE", o.reapTimeout);
    if (!l) return fail("no DONE for the terminal UNLOAD");
    if (l->type != "DONE") return fail("expected a DONE, found a " + l->type + " line");
    if (!l->sessionId.empty())
      return fail("the terminal DONE should carry no session id; the header slot was '"
                  + l->sessionId + "'");
    if (!bodyHasNoSessionId(*l))
      return fail("the DONE body must not carry a \"session_id\" member: " + l->body);
  }
  sup.stop();                         // reaps the (now-exited) child
  const int code = sup.exitCode();
  if (code != 0)
    return fail("the worker exited non-zero (" + std::to_string(code) + ", expected a clean 0)");

  std::printf("testworker: PASS -- protocol-check over one `ovgenai worker`: "
              "READY/PONG/ITEM/DONE/CANCEL all framed <cmd>:<session>:<body> with the session id "
              "in the header and no \"session_id\" in any body\n");
  if (o.verbose) sup.dumpLineTypes(std::cerr);
  return 0;
}

}  // close the anonymous namespace HERE so main lives below it (global, external linkage).
// A `main` inside the anonymous namespace above would have internal linkage and the CRT could not
// find it ("undefined reference to 'main'"); it must be in the global namespace.

int main(int argc, char** argv) {
  // Never die from SIGPIPE if the worker dies mid-protocol: a ::write to its now-closed stdin would
  // otherwise take THIS test process down; ignored, the write returns EPIPE and writeLine stops, the
  // reader sees the read end's EOF, and reapChild reaps the (dead/leaving) child -> a clean FAIL.
  ::signal(SIGPIPE, SIG_IGN);
  Opts o;
#ifdef OVGENAI_WORKER_BIN
  o.program = OVGENAI_WORKER_BIN;
#endif
#ifdef OVGENAI_VLM_FIXTURE
  o.model = OVGENAI_VLM_FIXTURE;
#endif

  // Minimal, dependency-free flag parsing: `--key value` or `--key=value`; --verbose / --help are
  // valueless (so they never swallow the following token).
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto p = a.find('=');
    const bool hasEq = (p != std::string::npos);
    std::string key = hasEq ? a.substr(0, p) : a;
    std::string val = hasEq ? a.substr(p + 1) : std::string();
    const bool valueless =
        (key == "--verbose" || key == "--protocol-check" || key == "--help" || key == "-h");
    if (!hasEq && !valueless) {
      if (i + 1 >= argc) { std::fprintf(stderr, "testworker: %s needs a value\n", a.c_str()); return 2; }
      val = argv[++i];
    }
    if (key == "--help" || key == "-h") {
      std::printf(
          "testworker -- drive one `ovgenai worker` subprocess over the worker line protocol "
          "(<command>:<session>:<body>)\n\n"
          "  --program <path>     the worker binary [default: baked OVGENAI_WORKER_BIN]\n"
          "  --model <path>       the model dir [default: baked OVGENAI_VLM_FIXTURE]\n"
          "  --device <d>         [default: CPU]\n"
          "  --pipeline <p>       [default: vlm]\n"
          "  --max-tokens <n>     [default: 8] the tiny-random fills this cap; the run PASSES only if\n"
          "                      every session generated exactly this many tokens\n"
          "  --prompt <text>      [default: \"Hi, who are you?\"] (same for every session)\n"
          "  --sessions <n>       [default: 1] sessions over the ONE subprocess, run in parallel on n\n"
          "                      threads; each gets a unique id + req_id so its answer never crosses,\n"
          "                      and the run passes when every one generated exactly --max-tokens tokens\n"
          "  --protocol-check     instead of the session-count drive, assert the framing end-to-end: the\n"
          "                      worker's READY/PONG/ITEM/DONE/CANCEL each carry the session id in the\n"
          "                      2nd <session> header slot, with no \"session_id\" leaking into any body\n"
          "  --ready-timeout <ms>  [default: 120000]\n"
          "  --req-timeout <ms>    [default: 120000]\n"
          "  --verbose            log every answered line + the worker's captured stderr\n"
          "  -h | --help\n\n"
          "ctest runs --sessions 1 (the single-session smoke), --sessions 50 (a real 50-session run),\n"
          "and --protocol-check (the framing assertions) over the ONE subprocess.  A single background\n"
          "pump in the worker serves all the sessions in parallel (it is NOT a per-session thread) --\n"
          "per-session demux is this test (a unique req_id each), so the multi-session run already passes\n"
          "with the production worker as-is.\n");
      return 0;
    } else if (key == "--program") {
      o.program = val;
    } else if (key == "--model") {
      o.model = val;
    } else if (key == "--device") {
      o.device = val;
    } else if (key == "--pipeline") {
      o.pipeline = val;
    } else if (key == "--max-tokens") {
      o.maxTokens = static_cast<std::int64_t>(std::atoll(val.c_str()));
    } else if (key == "--prompt") {
      o.prompt = val;
    } else if (key == "--sessions") {
      o.sessions = std::atoi(val.c_str());
    } else if (key == "--ready-timeout") {
      o.readyTimeout = std::chrono::milliseconds(std::atoll(val.c_str()));
    } else if (key == "--req-timeout") {
      o.reqTimeout = std::chrono::milliseconds(std::atoll(val.c_str()));
    } else if (key == "--verbose") {
      o.verbose = true;
    } else if (key == "--protocol-check") {
      o.protocolCheck = true;
    } else {
      std::fprintf(stderr, "testworker: unknown option %s\n", key.c_str());
      return 2;
    }
  }
  if (o.sessions < 1) o.sessions = 1;

  // --- SKIP (ctest SKIP_RETURN_CODE 77) on a box that is not provisioned for inference -----------
#if !defined(OVGENAI_WORKER_BIN)
  std::printf("SKIP: OVGENAI_WORKER_BIN was not baked at configure time\n");
  return 77;
#endif
#if !defined(OVGENAI_VLM_FIXTURE)
  std::printf("SKIP: OVGENAI_VLM_FIXTURE was not baked at configure time\n");
  return 77;
#endif
  if (::access(o.model.c_str(), F_OK) != 0) {
    std::printf("SKIP: model dir not accessible: %s\n", o.model.c_str());
    return 77;
  }
  if (!workerCanExec(o.program)) {
    std::printf("SKIP: `ovgenai` is not runnable here (run ctest under `eval $(./setupenv.sh)`) -- "
                "the runtime .so are not on LD_LIBRARY_PATH\n");
    return 77;
  }

  // Build the worker's argv: `worker --pipeline vlm --model <dir> --device CPU --max-tokens N`.
  const std::vector<std::string> args = {
      "worker", "--pipeline", o.pipeline, "--model", o.model,
      "--device", o.device, "--max-tokens", std::to_string(o.maxTokens)};

  // A one-shot supervisor (RAII): built with std::make_unique (no raw new), and the dtor calls the
  // idempotent stop() which reaps the child + joins the threads + closes the fds.
  // Non-const on purpose: the driver calls the Supervisor's mutating members (start / send* /
  // awaitResult / unload / stop) -- a const object would make them inaccessible, and would give
  // awaitResult/expectControl a `const std::mutex` (the unique_lock inside needs a non-const one).
  std::unique_ptr<Supervisor> sup =
      std::make_unique<Supervisor>(o.program, args, o.reapTimeout);

  if (!sup->start()) {
    std::fprintf(stderr, "testworker: FAIL -- could not start the worker subprocess\n");
    return 1;
  }
  if (o.verbose)
    std::fprintf(stderr, "[started] %d session(s) over pid %ld: %s %s\n", o.sessions,
                 static_cast<long>(sup->pid()), sup->program().c_str(), fmtArgs(args).c_str());

  // The --protocol-check mode runs a self-contained framing assertion (it also waits for the READY);
  // when set, take the early return and skip the session-count drive below.
  if (o.protocolCheck)
    return runProtocolCheck(*sup, o);

  // (1) Confirm the worker is up: it emits READY up front (after the model load), before reading the
  // protocol, and we wait for it (a hard gate).  The test stays NAIVE on purpose -- we do NOT pre-feed
  // the request to keep the worker busy.  A worker that cannot stay up / answer while idle is exactly
  // the production bug this test exists to surface; per the user that gets fixed in the worker in a
  // later session, and a naive consumer is precisely what turns that bug into a loud, readable FAIL
  // instead of a silent masking.  `--sessions N` (a later optional extension) reuses this same naive
  // per-session flow on N threads over this ONE subprocess.
  const std::optional<Line> ready = sup->expectControl("READY", o.readyTimeout);
  if (!ready) {
    sup->stop();
    reportFail(*sup, "no READY from the worker within " +
                 std::to_string(o.readyTimeout.count()) +
                 " ms -- it appears to have died before announcing itself");
    return 1;
  }
  if (o.verbose) std::fprintf(stderr, "[wait] saw READY (worker is up)\n");

  // (2) Drive all the session(s) over the ONE worker, each yielding its generated-token count
  //     (-1 if no answer).  sessions==1 is the registered single-session ctest scope (one request, run
  //     inline); sessions>1 fans out N session-threads over this SAME one subprocess -- each with a
  //     UNIQUE session_id + req_id.  The production worker is now idle-/session-safe and its single
  //     background pump serves the whole batch in parallel; this test does the per-session demux
  //     (awaitTurn matches by the SESSION id of each line's header -- a bare per-chunk <item> carries no
  //     req_id, so the multi-session run needs no further
  //     change to the worker.  The counts gather into a plain vector guarded by a mutex (one write per
  //     slot; a std::vector<std::atomic<>> would not even construct).
  const std::size_t n = static_cast<std::size_t>(o.sessions < 1 ? 1 : o.sessions);  // at least one
  std::vector<std::int64_t> results(n, -1);       // -1 = no answer / no count yet
  std::mutex resultsMtx;
  if (n == 1) {
    results[0] = driveOneSession(*sup, "session1", 1, o);
    if (o.verbose)
      std::fprintf(stderr, "[drive] session1 generated %lld token(s)\n", (long long)results[0]);
  } else {
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (std::size_t s = 0; s < n; ++s) {
      const std::string sid = "session" + std::to_string(s + 1);
      const std::int64_t reqId = static_cast<std::int64_t>(s + 1);   // unique per session
      threads.emplace_back([&sup, &results, &resultsMtx, &o, s, sid, reqId]() {
        const std::int64_t tok = driveOneSession(*sup, sid, reqId, o);
        { std::lock_guard<std::mutex> lk(resultsMtx); results[s] = tok; }   // one write per slot
        if (o.verbose)
          std::fprintf(stderr, "[drive] %s generated %lld token(s)\n", sid.c_str(), (long long)tok);
      });
    }
    for (auto& t : threads) t.join();       // every session has now reported (its slot is written)
  }

  // (3) Clean shutdown: a terminal UNLOAD makes the worker stop its pump, emit DONE, exit; a best-
  // effort wait for that DONE (bounded by the reapTimeout) then stop() closes our stdin (clean EOF)
  // and reaps it.  The exit code must be 0.
  sup->unload();
  (void)sup->awaitTurn(std::string(), o.reapTimeout);   // the global UNLOAD <done>: body has no req_id, session "" keys it
  if (o.verbose) std::fprintf(stderr, "[shutdown] sent UNLOAD\n");
  sup->stop();                                // RAII backstop anyway, but stop here so we can report
  const int code = sup->exitCode();
  if (o.verbose) std::fprintf(stderr, "[done] worker exit code = %d (0 expected)\n", code);

  // (4) Success (the user's criterion): every session generated exactly --max-tokens tokens. The
  // tiny-random fills the cap before any EOS, so a finished turn reports that many; a shortfall
  // (0 / -1 = no answer or no metrics, or a count below the cap) is a failure -- either the worker
  // under-produced (a production issue to detect, a good reason to fail) or a correlation/harness
  // issue; reportFail dumps the captured stdout/stderr so a reader can tell which.
  {
    std::size_t ok = 0;
    std::string offenders;
    for (std::size_t s = 0; s < n; ++s) {                            // n / results from (2)
      if (results[s] == o.maxTokens) { ++ok; continue; }
      offenders += (offenders.empty() ? std::string() : ", ") +
                   "session" + std::to_string(s + 1) + "=" + std::to_string(results[s]);
    }
    if (ok != n) {
      reportFail(*sup, std::to_string(n - ok) + " of " + std::to_string(n) +
                  " session(s) did not generate exactly " + std::to_string(o.maxTokens) +
                  " token(s)" + (offenders.empty() ? std::string() : " [" + offenders + "]") +
                  "; worker exit code " + std::to_string(code));
      return 1;
    }
  }
  // A clean run also exits 0 (the UNLOAD -> DONE -> exit path); a non-zero exit is itself a failure.
  if (code != 0) {
    reportFail(*sup, "the worker exited non-zero (" + std::to_string(code) + ", expected a clean 0)");
    return 1;
  }

  // PASS: every session over the one worker generated exactly the --max-tokens we handed it.
  std::printf("testworker: PASS -- %zu session(s) over one `ovgenai worker` (pid %ld), each streamed "
              "%lld token(s): READY -> %zu turns (per-chunk <item>s then a <done>) -> clean exit 0\n",
              n, static_cast<long>(sup->pid()), static_cast<long long>(o.maxTokens), n);
  if (o.verbose) sup->dumpLineTypes(std::cerr);
  return 0;   // the unique_ptr<Supervisor> dtor now runs the idempotent stop() a final time
}
