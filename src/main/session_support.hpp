// Session support -- the GENERAL half of the former vlm.cpp input handling, pulled out
// of the vlm pipeline so it is not vlm-specific: a `SessionDispatcher` that owns the
// conversation state a run/worker needs and the only thing that reads/feeds input.
//
//   * it keeps a map of every open session -- `std::map<std::string, std::shared_ptr<Session>>`
//     keyed by session id.  The value is a `Session` (a pair of a ChatHistory + a per-session
//     mutex) held behind a shared_ptr: `ChatHistory` has a user-declared destructor (it is a
//     pimpl), so it is neither copyable nor move-constructible and a bare
//     `std::map<std::string, Session>` would not even compile -- a shared_ptr to such a container
//     is the natural way to keep one, and selectSession hands a `Session&` (the dereferenced one)
//     to the strategy, which copies the two shared_ptrs into a per-request record that survives
//     a map re-lookup/erase.
//   * it moves the input readers out of vlm.cpp in as its own private methods
//     (readOneLine / readAllInput / readMailInput / acquireInput / protocolLineLimit),
//     the per-session select/create, and the {worker -> parent} control-line emitters.
//   * instead of a free-function generate callback, the dispatcher now holds a
//     `std::unique_ptr<GenerationStrategy>` -- the per-pipeline driver the binding plugs in.
//     Two concrete strategies live in pipeline_support.cpp: a legacy one (wrapping the monolithic
//     `generate()` -- the SDPA / NPU fallback) and a concurrent one (driving the
//     ContinuousBatchingPipeline `add_request()` + a background `step()`/`read()` pump); they
//     share the engine's per-session history + a per-session lock, and the dispatcher drives both
//     through the SAME run()/worker() loop (one code path for the two engine forms).
//   * a process-global `outMutex()` serializes every physical write to stdout, so the legacy live
//     subword stream and the concurrent pump's emits can never interleave / corrupt each other in
//     one process (different pipelines never coexist, but one process has exactly one dispatcher).
//
// The two forms it exposes, the same two subcommands:
//        run()    the one-shot -- exactly ONE session, given a freshly generated id that is always
//                 the selected one; its input is --input when given, else the whole stdin (a pipe
//                 read to EOF, or a terminal typed to a lone '.').  It enqueues that one request
//                 onto the strategy and then drives it to completion by a manual
//                 `while (hasPending()) stepAndDrain()` loop (no background thread).
//        worker() the long-lived process -- hook stdin and read the line protocol a line at a
//                 time; each GENERATE carries a session_id, looked up in the map (creating a fresh
//                 Session on first sight) and enqueued onto the strategy; the strategy's background
//                 pump (started once up front) does the actual engine work, so multiple GENERATEs
//                 overlap / batch on one pipeline; CANCEL cancels by (session_id, req_id);
//                 PING / UNLOAD are the global control ops (UNLOAD / a clean EOF stops the pump,
//                 emits DONE, and exits).  It suppresses the legacy LIVE subword stream to stdout
//                 (clean JSONL -- one RESULT line per finished request) which the one-shot keeps.
#ifndef OVGENAI_SESSION_SUPPORT_HPP
#define OVGENAI_SESSION_SUPPORT_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "config.hpp"               // ovgenai::config::ConfigError / modelRequestConfigFromJson
#include "model_request_config.hpp" // the full ModelRequestConfig (the dispatcher holds a copy as its per-request base)

// Forward-declared (the full type arrives in the .cpp): the engine's conversation each Session owns.
// A `session history` (a `std::shared_ptr<ChatHistory>`), and a `Session&` (the dereferenced one)
// both name an incomplete type fine, so no header is pulled in here.
namespace ov { namespace genai { class ChatHistory; } }

namespace ovgenai {

// One open conversation: the engine's accumulated `ChatHistory` plus a per-session mutex that
// serializes the cheap history-append / reply-append points (the engine `step()` itself is NOT
// guarded -- different sessions are independent by design; the same session is just best-effort).
// Held behind a shared_ptr in the dispatcher's map (ChatHistory is a pimpl, see the header note);
// the per-request record a running strategy keeps COPIES the two shared_ptrs so it outlives a
// re-lookup or erase.
struct Session {
  std::shared_ptr<ov::genai::ChatHistory> history;
  std::shared_ptr<std::mutex>             lock;
};

// The per-pipeline driver the dispatcher runs its run()/worker() loop against.  Two concrete
// backends implement it (both in pipeline_support.cpp): the legacy monolithic `generate()` and the
// concurrent `add_request()` pump -- so the dispatcher has ONE code path for both (the user's
// "do not maintain two paths for generate() and add_request()").
//   enqueue(sid, s, rc, in, isWorker, reqId) -- translate `in` + `rc`, append the user messages to
//       `s.history` under `*s.lock`, and submit the request; returns 0, or throws a ConfigError /
//       std::exception (the run form lets it propagate to the binding; the worker catches it
//       per-line and keeps running).  The legacy form does the whole thing synchronously here
//       (then hasPending() is false, so run()'s drain loop never runs); the concurrent form just
//       registers a pending request and lets the pump drive it.
//   stepAndDrain() -- advance the engine one step and feed the per-iteration tokens out, finalizing
//       any request that finished (the legacy form is a no-op; it has nothing to drain).
//   hasPending()   -- true while any request still needs draining.
//   startBackground() / stop() -- start / stop the background pump used by worker() (the legacy
//       form has no pump, so both are no-ops); stop() must be idempotent (the dispatcher calls it on
//       UNLOAD and again from the strategy's destructor).
//   cancel(sid, reqId) -- stop a still-running request: a non-negative reqId targets one, a
//       negative / 0 target cancels that session's in-flight requests.
class GenerationStrategy {
 public:
  virtual ~GenerationStrategy() = default;
  virtual int enqueue(const std::string& sid, Session& s,
                      const config::ModelRequestConfig& rc, const std::string& in,
                      bool isWorker, std::int64_t reqId) = 0;
  virtual int  stepAndDrain()                       = 0;
  virtual bool hasPending()                        = 0;
  virtual void startBackground()                   = 0;
  virtual void stop()                              = 0;
  virtual void cancel(const std::string& sid, std::int64_t reqId) = 0;
};

// The process-global stdout mutex: every physical write to stdout (the worker control lines, the
// RESULT lines, the legacy live subword stream) takes this first, so the two emitters in one
// process can never interleave bytes.  Shared by this file (the control lines) AND
// pipeline_support / streaming_support (the RESULT body + the live stream) -- single source.
std::mutex& outMutex();

// Holds open sessions, reads input, and runs one session in the form its subcommand named.
class SessionDispatcher {
 public:
  // `strategy` -- the injected per-pipeline driver (a owning unique_ptr, moved in; the dispatcher
  // owns it and stops it on destruction); `baseReq` -- the CLI per-request config, the per-GENERATE
  // default the worker folds its line's gen_config over (and the one-shot's only request config);
  // `cliInput` -- the one-shot's direct --input (nullopt for the worker, and for a --input-less
  // one-shot, at which point run() reads stdin instead).
  SessionDispatcher(std::unique_ptr<GenerationStrategy> strategy,
                    config::ModelRequestConfig baseReq,
                    std::optional<std::string> cliInput);

  // The one-shot `run` form: one session, a freshly generated id that is always the selected one.
  // Fetches the input (--input, else the whole stdin), enqueues that one request on the strategy,
  // then drains it to completion with a manual `while (hasPending()) stepAndDrain()` loop.
  int run();

  // The long-lived `worker` form: hooks stdin and runs the line protocol.  The wire is one
  // <command>:<sessionId>:<body> line each way (see buildFramedLine / parseFramedLine below) --
  // the session id is a 2nd header slot (NOT inside the JSON body), so a client can take the body
  // and drop it into its own response envelope without first fishing a session id out of JSON.
  //  * there is NO load command: the model / pipeline is already built before this loop starts
  //    (the caller hands every load parameter on the command line), so a stray LOAD is unknown.
  //  * READY is emitted once up front and the strategy's background pump is started; each line is
  //    split for its command (GENERATE / PING / CANCEL / UNLOAD -- GENERATE enqueues on the pump,
  //    so the many of them overlap / batch on one pipeline) and -- for a GENERATE -- its session
  //    id (the 2nd header slot) selects (creating on first sight) the session.  One bad line answers
  //    with an ERROR and the loop keeps running; a clean EOF (or an UNLOAD) stops the pump, emits
  //    DONE, and exits.
  int worker();

 private:
  // --- the input readers (moved in from vlm.cpp, now the dispatcher's own methods) ---

  // Read one (possibly very large) line from `in`, bounded by `limit` bytes; an over-limit
  // line is a ConfigError (the limit is protocolLineLimit() -- large, an inlined-base64
  // conversation is tens of MiB on one line).
  std::string readOneLine(std::istream& in, std::size_t limit);
  // The whole-stream reader the one-shot's piped / redirected (a non-tty) case needs: a
  // multi-line / pretty-printed chat is ONE document, so read it to EOF (no per-line stop).
  std::string readAllInput(std::istream& in, std::size_t limit);
  // The protocol line cap (256 MiB default).
  static std::size_t protocolLineLimit();
  // The terminal (interactive) reader: a line at a time, a lone '.' ends the input (mail(1)).
  std::string readMailInput();
  // The one-shot input surface: --input when given (verbatim, no stdin read); else a terminal
  // (a lone '.') or a pipe (one document read to EOF).  An empty result is fine -- the
  // strategy reports "no message" for an empty input.
  std::string acquireInput();

  // A freshly generated one-shot session id (the run form has exactly one, always selected).
  // Non-static: it draws on the per-dispatch instance counter m_seq below (one fresh
  // id per dispatch -- a one-shot opens exactly one such session).
  std::string newSessionId();

  // Look up the session by id, creating a new empty Session (a fresh ChatHistory + a fresh mutex)
  // on FIRST sight only -- a hit returns the (accumulated) session untouched, so an ongoing
  // conversation is never reset.  The worker's "select, or create if there is no session with this
  // id"; returns the (managed) Session& the strategy drives.
  Session& selectSession(const std::string& sessionId);

  // The {worker -> parent} control lines: READY / PONG / CANCEL / DONE / ERROR, each emitted as one
  // <command>:<sessionId>:<body> line (via buildFramedLine -- the session id in the 2nd header slot,
  // never inside the JSON body).  Each write takes outMutex() so a concurrent pump emit cannot
  // interleave with it.
  void emitWorkerLine(const std::string& type, std::int64_t reqId, const std::string& sessionId);
  void emitWorkerError(const std::string& msg, std::int64_t reqId, const std::string& sessionId);

  // --- state ---
  std::unique_ptr<GenerationStrategy> m_strategy;
  config::ModelRequestConfig m_baseReq;
  std::optional<std::string> m_cliInput;

  // session id -> that session's Session (shared_ptr: ChatHistory is a pimpl, see the header note),
  // keyed so the worker can select / create by id.
  std::map<std::string, std::shared_ptr<Session>> m_sessions;
  std::int64_t m_seq = 0;   // the per-dispatch counter newSessionId() draws on
};

// A JSON string-member escaper -- ", \ and control chars escaped, non-ASCII passed through as
// UTF-8.  Shared by the dispatcher's control / RESULT lines AND the vlm pipeline's RESULT body
// (defined once here; the free function a second TU reuses instead of re-typing the same case
// table -- single source of truth, as the rest of the project favours).
std::string jsonEscape(const std::string& in);

// --- The worker line protocol (the two ends' ONE framing; shared by the emit helpers below, by the
//     vlm body's RESULT line, and by a client on the other side of the pipe) --------------------
// One physical line is exactly  <command>:<sessionId>:<body>   (no leading/trailing whitespace;
// terminated by the newline a reader eats).  The session id is the 2nd header slot, never inside
// the JSON body -- that is the whole point: a client (e.g. a web server) splits on the two colons
// to read the session id, then takes <body> verbatim and drops it into its own response envelope
// (an OpenAI chat-completions "choices", which has no session id) without extracting one from JSON.
//
//   command  -- the first ':'-delimited token: a request word on stdin (GENERATE / PING / CANCEL /
//               UNLOAD -- there is deliberately NO load: the model is already built before the loop)
//               or a message-type word on stdout (READY / PONG / CANCEL / DONE / ERROR / RESULT).
//               Never contains ':'.
//   sessionId -- the 2nd ':'-delimited token; a caller-chosen conversation key that never contains
//               ':' (the delimiter).  May be empty (e.g. the up-front READY / a global control).
//   body     -- everything after the 2nd colon, verbatim: a JSON object that may itself hold colons
//               (taken whole).  Always non-empty (at least "{}") so a reader never meets a blank slot.
//
// buildFramedLine assembles one line (NO newline, NO lock: the caller owns the physical write and,
// when relevant, outMutex); parseFramedLine splits one raw line back into its three slots.  Both
// are free so the wire format has ONE definition the worker, the vlm body, the client and the tests
// all share.  isKnownWorkerOp classifies a (request) command word -- the "no load" refinement lives
// here: a LOAD returns false, so the loop treats it as an unknown op and answers ERROR.
std::string buildFramedLine(const std::string& command, const std::string& sessionId,
                            const std::string& body);
bool parseFramedLine(const std::string& line, std::string& command, std::string& sessionId,
                     std::string& body);
bool isKnownWorkerOp(const std::string& command);

}  // namespace ovgenai

#endif  // OVGENAI_SESSION_SUPPORT_HPP
