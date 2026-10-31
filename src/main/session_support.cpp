#include "session_support.hpp"

#include "config.hpp"                // ConfigError / compactJsonOf
#include "model_request_config.hpp"  // modelRequestConfigFromJson (the worker's per-line sampling)

#include <openvino/genai/chat_history.hpp>   // the full ChatHistory (selectSession creates one)

#include <rapidjson/document.h>                // the worker's per-line JSON parse

#include <exception>
#include <iostream>                          // std::cin
#include <istream>                          // std::istream for the readers
#include <memory>
#include <mutex>
#include <unistd.h>                         // isatty

namespace ovgenai {

// The process-global stdout mutex -- the one lock every physical write to stdout takes first.
// Defined here because this is the lowest "session" TU; pipeline_support (the RESULT body) and
// streaming_support (the legacy live subword stream) pull it in from this same header.
std::mutex& outMutex() {
  static std::mutex m;
  return m;
}

// The JSON escaper (declared in the header): the one case table the dispatcher's control
// lines and the vlm RESULT body both reuse, so it lives in exactly one place.
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
          out += hex[(c >> 4) & 0xF];
          out += hex[c & 0xF];
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

SessionDispatcher::SessionDispatcher(std::unique_ptr<GenerationStrategy> strategy,
                                     config::ModelRequestConfig baseReq,
                                     std::optional<std::string> cliInput)
  : m_strategy(std::move(strategy)),
    m_baseReq(std::move(baseReq)),
    m_cliInput(std::move(cliInput)) {}

// ---------------------------------------------------------------------------
// The input readers -- moved verbatim (as private methods) from vlm.cpp.
// ---------------------------------------------------------------------------

// Read one (possibly very large) line from `in`, bounded by `limit` bytes; an over-limit
// line is a ConfigError.  This is the line-JSON reader a --worker form needs.
std::string SessionDispatcher::readOneLine(std::istream& in, std::size_t limit) {
  std::string s;
  int ch = in.get();
  while (ch != EOF) {
    if (ch == '\n') break;
    if (limit && s.size() + 1 > limit)
      throw config::ConfigError("input line exceeds the " + std::to_string(limit) +
                                 "-byte limit");
    s.push_back(static_cast<char>(ch));
    ch = in.get();
  }
  return s;
}

// The whole-stream reader the one-shot's piped / redirected (a non-tty) case needs: a
// multi-line / pretty-printed chat is ONE document, so read it all the way to EOF (no
// per-line "\n" stop like readOneLine).
std::string SessionDispatcher::readAllInput(std::istream& in, std::size_t limit) {
  std::string s;
  int ch = in.get();
  while (ch != EOF) {
    if (limit && s.size() + 1 > limit)
      throw config::ConfigError("input exceeds the " + std::to_string(limit) +
                                 "-byte limit");
    s.push_back(static_cast<char>(ch));
    ch = in.get();
  }
  return s;
}

std::size_t SessionDispatcher::protocolLineLimit() {
#ifdef OVGENAI_PROTOCOL_LINE_LIMIT
  return static_cast<std::size_t>(OVGENAI_PROTOCOL_LINE_LIMIT);
#else
  return (256u * 1024u * 1024u);
#endif
}

// Read chat input a line at a time from a terminal, the classic mail(1) way: a line that
// is a lone '.' (and nothing else) ends the input and is not part of it; every earlier line
// is kept (each with its trailing newline, so a pretty-printed chat JSON is reassembled
// verbatim).  EOF before a lone '.' also ends the input (whatever was typed is used).
std::string SessionDispatcher::readMailInput() {
  std::string acc;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == ".") break;
    acc += line;
    acc.push_back('\n');
  }
  return acc;
}

// Acquire the one-shot chat input -- the precedent (req 2 + 3):
//   A. --input given  -> return its value verbatim; read NO stdin at all.
//   B. --input absent AND stdin is NOT a tty (a pipe / redirect) -> read the whole stdin
//      to EOF, then parse it: a multi-line / pretty-printed chat is ONE document.
//   C. --input absent AND stdin IS a tty (nothing piped) -> read the chat from the command
//      line until a lone '.', the mail command's way.
std::string SessionDispatcher::acquireInput() {
  if (m_cliInput.has_value()) return *m_cliInput;                      // A: --input given
  if (isatty(STDIN_FILENO)) return readMailInput();                    // C: a terminal
  return readAllInput(std::cin, protocolLineLimit());                  // B: piped -> EOF
}

std::string SessionDispatcher::newSessionId() {
  // The one-shot form has exactly one session, always the selected one; give it a generated
  // id distinct across the few sessions a process could open, without reaching for a crypto RNG
  // (a "run-" prefix + the dispatch's own counter -- good enough to correlate a one-shot's
  // RESULT line on the consumer side).
  return "run-" + std::to_string(++m_seq);
}

// Select the session, creating a NEW empty Session (fresh ChatHistory + fresh mutex) ONLY on a
// first sight -- a hit returns the accumulated session untouched (so an ongoing conversation is
// never reset), the worker's "select, or create if there is no session with this id".
// operator[]-style semantics without a by-value Session (ChatHistory is a pimpl -- see the header
// note); the make_shared on a miss is what a plain map could not do for this type.
Session& SessionDispatcher::selectSession(const std::string& sessionId) {
  auto it = m_sessions.find(sessionId);
  if (it != m_sessions.end())
    return *it->second;
  auto session = std::make_shared<Session>();
  session->history = std::make_shared<ov::genai::ChatHistory>();
  session->lock    = std::make_shared<std::mutex>();
  it = m_sessions.emplace(sessionId, session).first;
  return *it->second;
}

// ---------------------------------------------------------------------------
// The {worker -> parent} control lines (one compact JSON object per physical line).
// A non-empty sessionId is carried on the line so the consumer can correlate it; a control
// op with none (e.g. the up-front READY) carries none, so its shape is unchanged.  Each write
// takes the global outMutex() first so a concurrent pump emit cannot interleave with it.
// ---------------------------------------------------------------------------
void SessionDispatcher::emitWorkerLine(const std::string& type, std::int64_t reqId,
                                       const std::string& sessionId) {
  std::string line = std::string("{\"v\":1,\"type\":\"") + type + "\"";
  if (reqId) line += std::string(",\"req_id\":") + std::to_string(reqId);
  if (!sessionId.empty())
    line += std::string(",\"session_id\":\"") + jsonEscape(sessionId) + "\"";
  line += "}";
  std::lock_guard<std::mutex> lk(outMutex());
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

void SessionDispatcher::emitWorkerError(const std::string& msg, std::int64_t reqId,
                                        const std::string& sessionId) {
  std::string line = std::string("{\"v\":1,\"type\":\"ERROR\",\"error\":\"") +
                     jsonEscape(msg) + "\"";
  if (reqId) line += std::string(",\"req_id\":") + std::to_string(reqId);
  if (!sessionId.empty())
    line += std::string(",\"session_id\":\"") + jsonEscape(sessionId) + "\"";
  line += "}";
  std::lock_guard<std::mutex> lk(outMutex());
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// run() -- the one-shot form.
// ---------------------------------------------------------------------------
int SessionDispatcher::run() {
  // Exactly one session, a freshly generated id that is always the selected one.
  const std::string sessionId = newSessionId();
  Session& s = selectSession(sessionId);
  // The input arrives from --input when given, else the whole stdin (a pipe / a terminal).
  const std::string input = acquireInput();
  // Enqueue the one request on the strategy. The legacy form does the whole work here (then
  // hasPending() is false, so the drain loop never runs); the concurrent form only registers it.
  // Let it throw (a ConfigError / a std::exception); it is caught by the binding that built this
  // dispatcher (the "ovgenai: <pipeline>:" report + a 2 / 1 exit), so a one-shot failure is
  // surfaced with the pipeline's own name, as the old vlm body did.
  int code = m_strategy->enqueue(sessionId, s, m_baseReq, input,
                                 /*isWorker=*/false, /*reqId=*/0);
  // Drain it to completion here: no background thread in the one-shot form, so drive the
  // strategy directly until it reports nothing left to do (-1 = stopped, nothing more to pull).
  while (m_strategy->hasPending()) {
    if (m_strategy->stepAndDrain() < 0)
      break;
  }
  return code;
}

// ---------------------------------------------------------------------------
// worker() -- the long-lived form (one JSON op per stdin line; READY up front).
// A GENERATE carries a session_id; it is looked up (creating a fresh Session on first sight)
// and enqueued onto the strategy, whose background pump (started once up front) does the actual
// engine work -- so the multiple GENERATEs overlap / batch on one pipeline.  (req 1) One bad
// line answers with an ERROR and the loop keeps running; a clean EOF or an UNLOAD exits: the
// UNLOAD path stops the pump then emits DONE (the clean-EOF path just stops silently, matching
// the prior behaviour that emitted no DONE on EOF), both via the idempotent stop().
// ---------------------------------------------------------------------------
int SessionDispatcher::worker() {
  emitWorkerLine("READY", 0, std::string());
  // Start the strategy's background pump ONCE up front; the line loop now only enqueues /
  // cancels and lets the pump advance the engine (legacy strategy: a no-op pump).
  m_strategy->startBackground();
  const std::size_t limit = protocolLineLimit();
  for (;;) {
    std::string line;
    try {
      line = readOneLine(std::cin, limit);
    } catch (const config::ConfigError& e) {
      emitWorkerError(e.what(), 0, std::string());
      continue;
    }
    if (line.empty() && std::cin.eof()) break;   // clean EOF -> exit
    if (line.empty()) continue;                    // blank line: skip
    rapidjson::Document msg;
    if (msg.Parse<0>(line.c_str()).HasParseError()) {
      emitWorkerError("malformed line", 0, std::string());
      continue;
    }
    // The session id (the new selection key) -- read it from the line.  A GENERATE must
    // carry one (it is the conversation the request appends to); a control op may omit it.
    std::string sessionId =
        (msg.HasMember("session_id") && msg["session_id"].IsString())
            ? std::string(msg["session_id"].GetString()) : std::string();
    std::string op =
        (msg.HasMember("op") && msg["op"].IsString())
            ? std::string(msg["op"].GetString()) : "";
    std::int64_t reqId = 0;
    if (msg.HasMember("req_id") && msg["req_id"].IsInt64())
      reqId = msg["req_id"].GetInt64();
    else if (msg.HasMember("request_id") && msg["request_id"].IsInt64())
      reqId = msg["request_id"].GetInt64();

    if (op == "UNLOAD") {
      // Stop the pump (draining any in-flight work first), then announce done last -- the
      // after-loop stop() below is idempotent, so this pre-stop is safe and orders the flush.
      m_strategy->stop();
      emitWorkerLine("DONE", reqId, sessionId);
      break;
    } else if (op == "PING") {
      emitWorkerLine("PONG", reqId, sessionId);
      continue;
    } else if (op == "CANCEL") {
      // Stop the in-flight request(s) for this (session, req_id) before reflecting it back.
      m_strategy->cancel(sessionId, reqId);
      emitWorkerLine("CANCEL", reqId, sessionId);
      continue;
    } else if (op == "GENERATE") {
      // A GENERATE must name its session: it is the conversation the request is appended to, and
      // a missing id has nothing to select/create -- answer an ERROR and keep the loop running.
      if (sessionId.empty()) {
        emitWorkerError("GENERATE requires a 'session_id'", reqId, std::string());
        continue;
      }
      // Sampling for THIS request comes from the gen_config (sent per request); start from the
      // CLI base for any field it omits.
      config::ModelRequestConfig reqConfig = m_baseReq;
      std::string genConfigStr;
      auto gc = msg.FindMember("gen_config");
      if (gc != msg.MemberEnd() && gc->value.IsObject())
        genConfigStr = config::compactJsonOf(gc->value);
      if (!genConfigStr.empty())
        reqConfig = config::modelRequestConfigFromJson(genConfigStr);
      // Select (create on first sight) the session, then ENQUEUE the work onto the strategy
      // (the pump advances it).  Per-request failures are caught here (error line, keep running)
      // so one bad request never takes the whole worker down.
      Session& session = selectSession(sessionId);
      try {
        m_strategy->enqueue(sessionId, session, reqConfig, genConfigStr,
                            /*isWorker=*/true, reqId);
      } catch (const config::ConfigError& e) {
        emitWorkerError(e.what(), reqId, sessionId);
      } catch (const std::exception& e) {
        emitWorkerError(e.what(), reqId, sessionId);
      }
      continue;
    }
    emitWorkerError("unknown op: " + op, reqId, sessionId);
  }
  // A clean EOF (the loop's only other exit) still needs the pump stopped; idempotent when an
  // UNLOAD already stopped it, so the final stop is harmless either way.  No DONE on a clean EOF
  // (matching the prior behaviour -- only UNLOAD announces DONE).
  m_strategy->stop();
  return 0;
}

}  // namespace ovgenai
