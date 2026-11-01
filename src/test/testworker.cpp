// testworker -- a standalone C++ driver for the `ovgenai worker` subprocess and its
// line-JSON protocol.  It is the test-side counterpart to the worker (the "consume" side a
// Python consumer would be): it spawns ONE worker, talks to it over a real stdin/stdout, and
// gathers the worker's stdout lines until a request completes.
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
// yields a RESULT line, and the worker shuts down cleanly (exit 0).  Per the user, the worker's
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
// protocol is explicitly NOT finalised (the user will nail it down later), so JSON is matched by
// SUBSTRING against the observed line shapes, not a full grammar.  No new/delete/malloc/free in the
// sense the project forbids: the Supervisor is owned through a std::unique_ptr (built with
// std::make_unique -- a subprocess is uniquely owned) and closed via an idempotent, RAII stop() the
// dtor calls (it joins the helper threads, reaps the child, closes every int fd); no raw new/delete.
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

// One protocol line the worker emitted, classified by the keys we recognise.  The protocol is not
// final, so we only ever read the three keys the lines we care about carry (type / session_id /
// req_id); everything else on the line (including a long `result`) is kept verbatim in `raw`.
struct Line {
  std::string type;        // READY / RESULT / PONG / CANCEL / DONE / ERROR / "" (unrecognised)
  std::string sessionId;   // "" when the line carries none
  std::int64_t reqId = 0;  // 0 when the line carries none (READY, or a no-req control, keep it)
  std::string raw;         // the whole physical line, for diagnostics
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
// order-insensitive (we never assume the grammar).  The value's opening quote sits right AFTER the
// ':' -- NOT at the key's opening quote (an earlier off-by bug read the key text as the value, so a
// line like {"v":1,"type":"READY"} parsed its "type" member to the literal "type" not "READY").
std::optional<std::string> pickString(const std::string& line, const std::string& key) {
  const std::string kq = "\"" + key + "\"";
  std::size_t j = line.find(kq);                 // the member, e.g.  "type"
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

// Pull a numeric member `key` ("key":12) as a signed long.  The FIRST occurrence is used; the keys
// we read (req_id) precede any long `result` text in the worker's field order, so a stray "req_id"
// inside rubbish output cannot shadow the real one.
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

// Classify a physical stdout line; a blank line is nullopt (the worker never emits one).
std::optional<Line> parseLine(const std::string& raw) {
  if (raw.empty()) return std::nullopt;
  Line l;
  l.raw = raw;
  if (auto t = pickString(raw, "type")) l.type = *t;
  if (auto s = pickString(raw, "session_id")) l.sessionId = *s;
  if (auto r = pickInt(raw, "req_id")) l.reqId = *r;
  return l;
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
      // Child: stdin <- our write end (we feed the protocol), stdout/stderr -> our read ends.
      int devnull = ::open("/dev/null", O_RDONLY);  // give the child a sane fd 0 before dup2
      ::dup2(inp[1], STDIN_FILENO);
      ::dup2(outp[1], STDOUT_FILENO);
      ::dup2(errp[1], STDERR_FILENO);
      ::close(devnull);
      for (int fd : {inp[0], inp[1], outp[0], outp[1], errp[0], errp[1]}) ::close(fd);
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

  // Send a GENERATE for (session, req_id) carrying a gen_config (the request's messages + sampling).
  void sendRequest(const std::string& session, std::int64_t reqId, const std::string& genConfig) {
    std::string line = std::string("{\"op\":\"GENERATE\"") +
                       ",\"session_id\":\"" + jsonEscape(session) + "\"" +
                       ",\"req_id\":" + std::to_string(reqId) +
                       ",\"gen_config\":" + genConfig + "}";
    writeLine(line);
  }

  // A bare control op (PING / CANCEL -- with the session/req the worker echoes back).
  void sendControl(const std::string& op, const std::string& session, std::int64_t reqId) {
    std::string line = std::string("{\"op\":\"") + op + "\",";
    line += "\"session_id\":\"" + jsonEscape(session) + "\"";
    line += ",\"req_id\":" + std::to_string(reqId) + "}";
    writeLine(line);
  }

  // The terminal shutdown op: no session/req -> the worker stops its pump, emits a lone DONE, exits.
  void unload() { writeLine("{\"op\":\"UNLOAD\"}"); }

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
// On any failure, point unambiguously at the WORKER (not the test harness): dump what the worker
// itself did.  This test only sends a request and gathers stdout/stderr; if it can't complete, the
// suspect is the worker's behaviour -- i.e. a production issue to fix separately (a later session),
// not something to workaround here.  We print the evidence by DEFAULT (not just --verbose) so a
// reader of the ctest log sees the finding at a glance.
void reportFail(Supervisor& sup, const std::string& why) {
  std::fprintf(stderr, "testworker: FAIL -- %s\n", why.c_str());
  std::fprintf(stderr, "  the worker emitted on stdout:");
  sup.dumpLineTypes(std::cerr);
  std::fprintf(stderr, "\n  captured worker stderr:\n%s", sup.stderrLog().c_str());
  if (sup.stderrLog().empty()) std::fprintf(stderr, "\n");
  std::fprintf(stderr, "  => the worker is the suspect (this test only sends + gathers); this looks like "
                      "a production issue (the worker is not idle-/session-safe) to fix in a later session.\n");
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
};

// Drive ONE session over the (shared) supervisor: send a GENERATE and await the correlated RESULT.
// Returns true iff an answer arrived; never inspects the (rubbish) output, only that the plumbing lit.
bool driveOneSession(Supervisor& sup, const std::string& sessionId, std::int64_t reqId, const Opts& o) {
  // The natural consumer flow on purpose: send the request, then gather stdout until it completes.
  // Deliberately NAIVE (not pre-fed) -- a worker that cannot answer while idle is the production bug
  // this test surfaces; the user fixes that in the worker in a later session, so here it must FAIL, not pass.
  sup.sendRequest(sessionId, reqId, buildGenConfig(o.prompt, o.maxTokens));
  const std::optional<Line> res = sup.awaitResult(reqId, o.reqTimeout);
  if (!res) {
    if (o.verbose) std::fprintf(stderr, "  [session %s] no answer for req_id %lld within timeout\n",
                                 sessionId.c_str(), static_cast<long long>(reqId));
    return false;
  }
  if (o.verbose)
    std::fprintf(stderr, "  [session %s] req_id %lld -> %s req_id=%lld\n", sessionId.c_str(),
                 static_cast<long long>(reqId),
                 res->type.empty() ? "(untyped)" : res->type.c_str(),
                 static_cast<long long>(res->reqId));
  return true;
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
    const bool valueless = (key == "--verbose" || key == "--help" || key == "-h");
    if (!hasEq && !valueless) {
      if (i + 1 >= argc) { std::fprintf(stderr, "testworker: %s needs a value\n", a.c_str()); return 2; }
      val = argv[++i];
    }
    if (key == "--help" || key == "-h") {
      std::printf(
          "testworker -- drive one `ovgenai worker` subprocess over the line-JSON protocol\n\n"
          "  --program <path>     the worker binary [default: baked OVGENAI_WORKER_BIN]\n"
          "  --model <path>       the model dir [default: baked OVGENAI_VLM_FIXTURE]\n"
          "  --device <d>         [default: CPU]\n"
          "  --pipeline <p>       [default: vlm]\n"
          "  --max-tokens <n>     [default: 8] (small; output is not inspected)\n"
          "  --prompt <text>      [default: \"Hi, who are you?\"]\n"
          "  --sessions <n>       [default: 1] sessions over the ONE subprocess, on n threads\n"
          "  --ready-timeout <ms>  [default: 120000]\n"
          "  --req-timeout <ms>    [default: 120000]\n"
          "  --verbose            log every answered line + the worker's captured stderr\n"
          "  -h | --help\n\n"
          "The ctest entry runs --sessions 1 (the for-now scope).  --sessions >1 drives N sessions\n"
          "from N threads over the one subprocess; it is NOT in ctest yet -- that needs the production\n"
          "worker made fully session-/thread-aware (a future session); only this test is delivered now.\n");
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

  // (2) Drive the session(s).  sessions==1 is the registered ctest scope (one request); sessions>1
  // fans out N threads over the one subprocess (prepared -- exercised by hand, not in ctest, because
  // making the production worker fully session-/thread-aware is a separate, future session).
  bool allAnswered = true;
  if (o.sessions <= 1) {
    allAnswered = driveOneSession(*sup, "session1", 1, o);
    if (o.verbose)
      std::fprintf(stderr, "[drive] session1 %s\n", allAnswered ? "answered" : "no-answer");
  } else {
    // Plain int results guarded by a mutex (a std::vector<std::atomic<int>> would not construct,
    // atomic being non-copyable); each thread writes its single slot under that lock.
    std::vector<int> results(static_cast<std::size_t>(o.sessions), 0);
    std::mutex resultsMtx;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(o.sessions));
    for (int s = 0; s < o.sessions; ++s) {
      const std::string sid = "session" + std::to_string(s + 1);
      const std::int64_t reqId = static_cast<std::int64_t>(s + 1);
      threads.emplace_back([&sup, &results, &resultsMtx, &o, s, sid, reqId]() {
        const bool ok = driveOneSession(*sup, sid, reqId, o);
        { std::lock_guard<std::mutex> lk(resultsMtx); results[static_cast<std::size_t>(s)] = ok ? 1 : 0; }
        if (o.verbose)
          std::fprintf(stderr, "[drive] %s %s\n", sid.c_str(), ok ? "answered" : "no-answer");
      });
    }
    for (auto& t : threads) t.join();
    for (int r : results) allAnswered = allAnswered && (r == 1);
  }

  // (3) Clean shutdown: a terminal UNLOAD makes the worker stop its pump, emit DONE, exit; a best-
  // effort wait for that DONE (bounded by the reapTimeout) then stop() closes our stdin (clean EOF)
  // and reaps it.  The exit code must be 0.
  sup->unload();
  (void)sup->awaitResult(0, o.reapTimeout);   // any DONE line carries req_id 0
  if (o.verbose) std::fprintf(stderr, "[shutdown] sent UNLOAD\n");
  sup->stop();                                // RAII backstop anyway, but stop here so we can report
  const int code = sup->exitCode();
  if (o.verbose) std::fprintf(stderr, "[done] worker exit code = %d (0 expected)\n", code);

  if (!allAnswered) {
    reportFail(*sup, "a session never got an answer (no correlated RESULT/DONE line for its request)");
    return 1;
  }
  if (code != 0) {
    reportFail(*sup, "the worker exited non-zero (" + std::to_string(code) + ", expected a clean 0)");
    return 1;
  }

  std::printf("testworker: PASS -- %d session(s) over one `ovgenai worker` (pid %ld): "
              "READY -> RESULT -> clean exit 0\n",
              (o.sessions <= 1 ? 1 : o.sessions), static_cast<long>(sup->pid()));
  if (o.verbose) sup->dumpLineTypes(std::cerr);
  return 0;   // the unique_ptr<Supervisor> dtor now runs the idempotent stop() a final time
}
