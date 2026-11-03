// The pipeline-binding layer (see pipeline_support.hpp): the seam between the subcommand
// front-end (commands.cpp's runModeEntry, which selects a pipeline by --pipeline) and each
// pipeline's body.  This TU holds the vlm body -- the one real body today -- moved IN from the
// former vlm.cpp:
//   * its COMMAND-LINE PARSING (was the vlm-general two-pass boost parse here in vlm.cpp);
//   * its pipeline CREATION (load a VLMPipeline from the resolved config);
//   * its per-request generate -- the vlm-specific "the produce + emit" -- now injected into the
//     session-agnostic SessionDispatcher (session_support.{hpp,cpp}) as the run/worker driver the
//     dispatcher runs against a selected session's ChatHistory.
// It also builds the registry (kPipelineBindings): vlm -> runVlmBody; the 13 not-yet-implemented
// pipelines share the infrastructure-only stub (kept in commands.cpp -- it is front-end infra,
// not a pipeline body -- so the front-end's real --help stays the single source of truth there and
// this TU only references commands::runPipelineStub for it).
//
// `commands::runPipelineStub` (declared in commands.hpp, defined in commands.cpp) is referenced by
// name below; `ovgenai::runVlmBody` is defined just below.
#include "pipeline_support.hpp"
#include "pipeline_detect.hpp"      // detectPipelineFor + isImplementedPipeline + detectPipelineDecline (the auto --pipeline default)
#include "commands.hpp"             // canonicalPipelines + pipelineParamDesc + runPipelineStub (the 13 stubs)
#include "session_support.hpp"      // SessionDispatcher + GenerateRequestFn + the shared ovgenai::jsonEscape (reused below)
#include "config.hpp"               // ConfigError / preProcess / rephraseBoost / joinList / kHelp*
#include "util.hpp"                 // base64_decode / env_or (the image-tensor resolver)
#include "model_load_config.hpp"    // ModelLoadConfig + parseModelLoadConfig
#include "model_runtime_config.hpp" // ModelRuntimeConfig + parseModelRuntimeConfig
#include "model_request_config.hpp" // ModelRequestConfig + parseModelRequestConfig
#include "model_scheduler_config.hpp" // ModelSchedulerConfig + parseModelSchedulerConfig
#include "genai_config.hpp"         // makeGenerationConfig + GenAIPipelineConfig
#include "streaming_support.hpp"    // ContentStreamer / selectModelFamily / makeToolCallParserByName / bridgeParsedMessage

#include <boost/program_options.hpp> // po (the `po` alias is provided, at global scope, by config.hpp)

#include <rapidjson/document.h>

#include <openvino/core/descriptor/tensor.hpp> // ov::Tensor / ov::Shape / ov::element::u8
#include <openvino/genai/generation_config.hpp>        // GenerationConfig
#include <openvino/genai/generation_handle.hpp>        // GenerationFinishReason
#include <openvino/genai/streamer_base.hpp>            // StreamerBase / StreamerVariant
#include <openvino/genai/text_streamer.hpp>            // TextParserStreamer (ContentStreamer's base)
#include <openvino/genai/tokenizer.hpp>               // Tokenizer + apply_chat_template
#include <openvino/genai/json_container.hpp>          // JsonContainer
#include <openvino/genai/chat_history.hpp>            // ChatHistory
#include <openvino/genai/visual_language/pipeline.hpp>     // VLMPipeline + VLMPipelineBase + VLMDecodedResults
#include <openvino/genai/visual_language/perf_metrics.hpp> // VLMPerfMetrics
#include <openvino/genai/visual_language/video_metadata.hpp> // VideoMetadata (the empty {} the history generate takes)
#include <openvino/genai/continuous_batching_pipeline.hpp> // ContinuousBatchingPipeline (add_request / step — the concurrent VLM form)
#include <openvino/genai/scheduler_config.hpp>            // SchedulerConfig (the CB's dedicated scheduler argument)

#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ovgenai {
namespace {

// --- small, self-contained helpers (the vlm-specific ones; the INPUT readers + worker
// control lines moved to session_support; this file keeps the vlm translate / emit / generate).

// --- (The argument parsing used to live here: each body re-parsed the command line via its own
// --help detector + a strip of the general routing flags.  That is now a SINGLE general parse
// in commands.cpp -- runModeEntry, via the free function parsePipelineArgs -- produced ONCE per
// invocation and handed to each body as a PipelineParsed (so the bodies neither re-parse nor
// need the raw argv).  The --help the bodies used to render is likewise the front-end's.  This
// TU therefore keeps only the vlm-specific generate machinery below -- the request translate ->
// normalised ChatHistory -> applied template -> generate -> emit -- which the SessionDispatcher
// runs against a selected session; it no longer parses anything.)
// --- image-part decode -> a placeholder tensor. The engine keeps NO image-format decoder, so a
// file / the built-in data: base64 form is read + (kept raw as) a [1,1,1,3] placeholder tensor,
// enough to keep the image-part / image-tensor count one-to-one (so the engine's vision-index
// invariant holds).  `url` is one of (req 4): "data:<mime>;base64,.." | "file://.." | "~..." |
// a plain path (the tilde / file:// forms fall through to read bytes from a path).
std::string dataUrlBase64(const std::string& url) {
  if (url.rfind("data:", 0) == 0) {
    auto comma = url.find(',');
    if (comma != std::string::npos) return url.substr(comma + 1);
  }
  return url;
}

ov::Tensor makeImageTensorFromBytes(const std::string& bytes) {
  ov::Tensor t(ov::element::u8, ov::Shape{1, 1, 1, 3});
  auto p = t.data<unsigned char>();
  for (size_t i = 0; i < 3; ++i) p[i] = (i < bytes.size()) ? static_cast<unsigned char>(bytes[i]) : 0x00;
  return t;
}

std::string expandTilde(const std::string& path) {
  if (path == "~" || path.rfind("~/", 0) == 0) {
    std::string home = env_or("HOME", std::string());
    if (home.empty())
      throw config::ConfigError("vlm: image '" + path + "' uses ~ but $HOME is not set");
    if (path == "~") return home;
    return home + path.substr(1);                                 // keep the leading '/'
  }
  return path;
}

ov::Tensor imageTensorFromUrlOrPath(const std::string& url) {
  std::string bytes;
  if (url.rfind("data:", 0) == 0) {
    // the built-in base64 form (mime type + base64 payload).
    bytes = base64_decode(dataUrlBase64(url));
  } else {
    // a file form: "file://" / "~" / "~/" / absolute / relative-to-CWD.
    std::string path = (url.rfind("file://", 0) == 0) ? url.substr(7) : url;
    path = expandTilde(path);
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) throw config::ConfigError("vlm: could not open image file '" + url + "'");
    std::stringstream ss; ss << f.rdbuf();
    bytes = ss.str();
  }
  static bool warned = false;                                     // once, regardless of source
  if (!warned) {
    warned = true;
    std::fprintf(stderr,
                 "ovgenai: vlm: image pixel-decoding is not yet wired "
                 "(openvino.genai exports no image-format decoder); "
                 "decoding to a [1,1,1,3] placeholder tensor for now\n");
  }
  return makeImageTensorFromBytes(bytes);
}

// --- the request -> normalised ChatHistory + media translator (the one real bit of new code) ---
struct ParsedRequest {
  std::string normMessagesJson;             // compact JSON array of normalised messages
  std::vector<ov::Tensor> images;
  std::vector<ov::Tensor> videos;
  std::vector<ov::Tensor> audios;
  bool hasTools = false;                    // a *non-empty* tools array is present
  bool hasKwargs = false;                   // a chat_template_kwargs object is present
  bool addGenerationPrompt = true;          // for the (pre-apply) string path
  std::string toolsJson;                    // compact, only when hasTools (for apply_chat_template)
  std::string kwargsJson;                   // compact, only when hasKwargs
};

// Serialise one normalised message as a JSON object fragment, rewriting a multipart (list) *user*
// content into the engine's universal media parts (and collecting the decoded tensors onto `pr`);
// a *non-user* list content is flattened to its joined text so the engine's per-role
// content.get_string() never throws.  Every other member of the original message (role,
// tool_calls, tool_call_id, ...) is carried through verbatim.
std::string serializeMessage(const rapidjson::Value& m, ParsedRequest& pr) {
  bool isUser = false;
  {
    auto it = m.FindMember("role");
    isUser = (it != m.MemberEnd() && it->value.IsString() &&
              std::string(it->value.GetString()) == "user");
  }
  std::string out = "{";
  size_t sep = 0;   // 0 = first member, 1 = already have one (needs a leading ", ")
  auto put = [&](const std::string& frag) {
    if (sep) out += ", ";
    out += frag;
    sep = 1;
  };
  // role
  {
    auto it = m.FindMember("role");
    if (it != m.MemberEnd() && it->value.IsString())
      put(std::string("\"role\":\"") + jsonEscape(it->value.GetString()) + "\"");
  }
  // content
  {
    auto it = m.FindMember("content");
    if (it != m.MemberEnd()) {
      if (it->value.IsString()) {
        put(std::string("\"content\":\"") + jsonEscape(it->value.GetString()) + "\"");
      } else if (it->value.IsArray()) {
        if (isUser) {
          std::string parts = "[";
          size_t ps = 0;
          for (auto pit = it->value.Begin(); pit != it->value.End(); ++pit) {
            const std::string ptype =
                (pit->IsObject() && pit->HasMember("type") && (*pit)["type"].IsString())
                    ? std::string((*pit)["type"].GetString()) : "";
            if (ptype == "text") {
              std::string txt = ((*pit).HasMember("text") && (*pit)["text"].IsString())
                                    ? std::string((*pit)["text"].GetString()) : "";
              if (ps) parts += ",";
              parts += "{\"type\":\"text\",\"text\":\"" + jsonEscape(txt) + "\"}";
              ps = 1;
            } else if (ptype == "image_url" || ptype == "image") {
              if (ps) parts += ",";
              parts += "{\"type\":\"image\"}";
              ps = 1;
              if (pit->IsObject() && pit->HasMember("url") && (*pit)["url"].IsString())
                pr.images.push_back(imageTensorFromUrlOrPath(std::string((*pit)["url"].GetString())));
              else
                pr.images.push_back(makeImageTensorFromBytes(std::string()));
            } else if (ptype == "video_url" || ptype == "video") {
              if (ps) parts += ",";
              parts += "{\"type\":\"video\"}";
              ps = 1;
              pr.videos.push_back(makeImageTensorFromBytes(std::string()));
            } else if (ptype == "audio_url" || ptype == "audio") {
              if (ps) parts += ",";
              parts += "{\"type\":\"audio\"}";
              ps = 1;
              pr.audios.push_back(makeImageTensorFromBytes(std::string()));
            } else {
              // unknown / already-universal part: keep it verbatim (re-serialised).
              if (ps) parts += ",";
              parts += config::compactJsonOf(*pit);
              ps = 1;
            }
          }
          parts += "]";
          put(std::string("\"content\":") + parts);
        } else {
          // non-user list content -> its joined text (the engine reads get_string()).
          std::string joined;
          for (auto pit = it->value.Begin(); pit != it->value.End(); ++pit)
            if (pit->IsObject() && pit->HasMember("text") && (*pit)["text"].IsString())
              joined += (*pit)["text"].GetString();
          put(std::string("\"content\":\"") + jsonEscape(joined) + "\"");
        }
      } else {
        // content is a number/bool/null/not-an-object: re-serialise verbatim.
        put(std::string("\"content\":") + config::compactJsonOf(it->value));
      }
    }
  }
  // carry every other member through verbatim (role + content handled above).
  for (auto it = m.MemberBegin(); it != m.MemberEnd(); ++it) {
    const std::string k = it->name.GetString();
    if (k == "role" || k == "content") continue;
    put(std::string("\"") + jsonEscape(k) + "\":" + config::compactJsonOf(it->value));
  }
  out += "}";
  return out;
}

// Translate a gen_config object (`g`, read-only) into a ParsedRequest: build the normalised
// messages array as a compact JSON string (for from_json_string), and detect the tools/kwargs
// sub-rule + the pre-apply add_generation_prompt.
ParsedRequest translateGenConfig(const rapidjson::Value& g) {
  ParsedRequest pr;
  auto tc = g.FindMember("tools");
  if (tc != g.MemberEnd() && !tc->value.IsNull() && tc->value.IsArray() && !tc->value.Empty()) {
    pr.hasTools = true;
    pr.toolsJson = config::compactJsonOf(tc->value);
  }
  auto kc = g.FindMember("chat_template_kwargs");
  if (kc != g.MemberEnd() && !kc->value.IsNull()) {
    pr.hasKwargs = true;
    pr.kwargsJson = config::compactJsonOf(kc->value);
    auto agp = kc->value.FindMember("add_generation_prompt");
    if (agp != kc->value.MemberEnd() && agp->value.IsBool())
      pr.addGenerationPrompt = agp->value.GetBool();
  }
  auto agpTop = g.FindMember("add_generation_prompt");
  if (!pr.hasKwargs && agpTop != g.MemberEnd() && agpTop->value.IsBool())
    pr.addGenerationPrompt = agpTop->value.GetBool();

  std::string out = "[";
  size_t ms = 0;
  auto mk = g.FindMember("messages");
  if (mk != g.MemberEnd() && !mk->value.IsNull() && mk->value.IsArray()) {
    for (auto it = mk->value.Begin(); it != mk->value.End(); ++it) {
      if (!it->IsObject()) continue;
      if (ms) out += ",";
      out += serializeMessage(*it, pr);
      ms = 1;
    }
  }
  out += "]";
  pr.normMessagesJson = out;
  return pr;
}

// (c) native finish_reason -> the wire field (null for NONE).
std::string finishReasonName(ov::genai::GenerationFinishReason r) {
  switch (r) {
    case ov::genai::GenerationFinishReason::STOP:      return "stop";
    case ov::genai::GenerationFinishReason::LENGTH:    return "length";
    case ov::genai::GenerationFinishReason::TOOL_CALL: return "tool_calls";
    case ov::genai::GenerationFinishReason::NONE:      return std::string();
  }
  return std::string();
}

// One number member of the metrics object, formatted shortest-round (g).
void addNumMember(std::string& out, const std::string& key, double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.6g", x);
  out += "\"";
  out += key;
  out += "\":";
  out += buf;
}

// Emit one line to stdout (the "consume" side a Python / web client reads) as a
//   <type>:<sessionId>:<body> line (see buildFramedLine / parseFramedLine).  The body carries NO
//   session_id -- for the worker the session id is the 2nd header slot (the framing, not the body), so a
//   client takes <body> verbatim into its own response envelope without extracting a session id.
//   * worker -- a framed DONE is the turn's TERMINAL (a streamed turn is <item>* deltas THEN this <done>;
//     `--stream 0` is a lone <done>).  Its body is the turn the CLIENT HAS NOT YET SEEN: a STREAMED turn
//     (`--stream N>0`) already showed its content / reasoning_content / tool_calls on the <item> lines, so
//     the <done> carries only the remaining keys -- finish_reason + metrics (and NOTHING else: no "result"
//     key, no "session_id" -- it is the 2nd header slot -- and no `gen_config` echo, a one-shot-only
//     extra the request already carried in its own body); a turn that did NOT stream (`--stream <=0`, no
//     <item> lines) gets its WHOLE turn in the lone <done>.  So a client keys only on DONE, and a <done>
//     never repeats what a stream already emitted (this is what makes `--stream` behave on the worker
//     exactly as it does on `run`).
//   * one-shot (run) -- the legacy single object, UNCHANGED (session_id leads; result / tool_calls only
//     when not streamed, the per-chunk stream carrying them when it is): a direct `run` output is
//     byte-for-byte what it was.
// The produced content (after the pure-tool-call blank + the res.texts fallback) and hasToolCalls are
// computed ONCE by the caller (vlmFinalizeResult) and passed in -- so this emit and the appended assistant
// reply cannot diverge.
void emitResult(bool isWorker, const std::string& sessionId,
                const std::string& out, bool hasToolCalls,
                const ov::genai::VLMDecodedResults& res,
                const ovgenai::config::ModelRequestConfig& reqConfig,
                bool streamed) {
  std::string line = "{";
  auto first = true;
  auto put = [&](const std::string& frag) {
    if (!first) line += ",";
    line += frag;
    first = false;
  };
  // The one-shot leads with its selection key (a leading `session_id`, the legacy single object).
  // The worker emits NOTHING leading: its <body> starts at the turn's `content` (no req_id on the wire).
  if (!isWorker)
    put("\"session_id\":\"" + jsonEscape(sessionId) + "\"");
  // The terminal line repeats nothing the per-chunk stream already showed: a turn that STREAMED (`--stream
  // N>0`) put its content / reasoning_content / tool_calls out on its per-chunk lines (a worker's <item>
  // lines, a one-shot's bare stream), so its terminal <done> / result carries only the other keys
  // (finish_reason + metrics) -- exactly the one-shot's original `!streamed` gate, so a worker now mirrors
  // `run`.  A turn that did NOT stream (`--stream <=0`, no <item> lines at all) gets its WHOLE turn in the
  // lone terminal line.  The worker calls that field `content` (the key the <item> deltas used), the
  // one-shot keeps its legacy `result`; the one-shot alone leads with the `session_id` selection key (so a
  // worker's body carries no `session_id` -- it is the 2nd header slot).  Only emit it when the turn
  // produced text: an empty `out` (a pure-tool-call turn -- the same case that already suppresses an empty
  // `reasoning_content` just below) must NOT leave an empty "content"/"result" key behind, so the lone
  // `<done>` then carries just tool_calls + finish_reason + metrics.  (`run` always streams live, so this
  // `!streamed` branch never fires there -- a `run` line is byte-identical to before.)
  if (!streamed && !out.empty()) {
    if (isWorker)
      put("\"content\":\"" + jsonEscape(out) + "\"");
    else   // one-shot: the result field (dead here -- a one-shot is always streamed, so this never fires).
      put("\"result\":\"" + jsonEscape(out) + "\"");
  }
  // The worker's <done> also carries reasoning_content when the turn produced one -- the SAME keys the
  // per-chunk <item> deltas used, so a client accumulating them reads the same names.  Gated by `!streamed`:
  // a streamed turn already showed it in the <item> lines, so the <done> omits it (only a `--stream <=0`
  // turn, with no <item> lines, carries the full reasoning in the lone <done>).
  if (isWorker && !streamed && !res.parsed.empty() && res.parsed[0].contains("reasoning_content")) {
    std::string rcr = res.parsed[0]["reasoning_content"].as_string().value_or(std::string());
    if (!rcr.empty())
      put("\"reasoning_content\":\"" + jsonEscape(rcr) + "\"");
  }

  // (b) the structured tool calls, if any -- one compact array so the line stays a single physical line
  // (to_json_string(-1) is the compact dumper).  Same gate now for both forms: a `--stream <=0` turn
  // (not streamed) carries them inline; a `--stream N>0` streamed turn omits them (the per-chunk <item>
  // deltas / bare stream already showed them) -- so the terminal line repeats nothing the stream showed,
  // and the worker's <done> matches the one-shot's result line exactly on this gate.
  if (!streamed && hasToolCalls)
    put(std::string("\"tool_calls\":") + res.parsed[0]["tool_calls"].to_json_string(-1));

  // The PARSER is the authority for a tool-call turn: a non-empty tool_calls wins over the
  // engine's finish, which may read "stop" when the model under-rands its </tool_call> outer tag
  // (we finalize on the inner </function>, so the turn is still reported as a tool call, not a stop).
  std::vector<ov::genai::GenerationFinishReason> fr = res.finish_reasons;
  ov::genai::GenerationFinishReason finish =
      fr.empty() ? ov::genai::GenerationFinishReason::NONE : fr[0];
  std::string frName = hasToolCalls ? std::string("tool_calls") : finishReasonName(finish);
  if (frName.empty())
    put(std::string("\"finish_reason\":null"));
  else
    put("\"finish_reason\":\"" + frName + "\"");

  // metrics -- a modest, consumer-useful subset of the VLM perf metrics.
  std::string m = "\"metrics\":{";
  ov::genai::VLMPerfMetrics pm = res.perf_metrics;   // non-const copy: accessors are non-const
  size_t mfirst = 1;   // 1: `m` already opens with `\"metrics\":{`, so the first metric needs no leading ","
  auto mPut = [&](const std::string& frag) {
    if (mfirst) { m += frag; mfirst = 0; }
    else { m += ","; m += frag; }
  };
  auto mInt = [&](const std::string& k, long long x) {
    mPut("\"" + k + "\":" + std::to_string(x));
  };
  auto mDbl = [&](const std::string& k, double x) {
    char b[64]; std::snprintf(b, sizeof b, "%.6g", x);
    mPut("\"" + k + "\":" + b);
  };
  mInt("num_input_tokens",   static_cast<long long>(pm.get_num_input_tokens()));
  mInt("num_generated_tokens", static_cast<long long>(pm.get_num_generated_tokens()));
  mDbl("load_time_ms",                static_cast<double>(pm.get_load_time()));
  mDbl("mean_ttft_ms",                pm.get_ttft().mean);
  mDbl("mean_tpot_ms",                pm.get_tpot().mean);
  mDbl("mean_throughput_tps",         pm.get_throughput().mean);
  mDbl("mean_vision_encoding_ms",     pm.get_vision_encoding_duration().mean);
  mDbl("mean_text_embedding_ms",      pm.get_text_embedding_duration().mean);
  mInt("total_image_slice_count",     static_cast<long long>(pm.get_total_image_slice_count()));
  m += "}";
  put(m);

  // The one-shot (alone) echoes the gen_config it used (so the consumer can re-derive / log it).  The
  // worker drops it: its turn body is just content / reasoning_content / tool_calls / finish_reason /
  // metrics -- a req_id is never on the wire and the request already carried those knobs in its body.
  if (!isWorker) {
    std::string gcj = config::modelRequestConfigToJson(reqConfig);
    put(std::string("\"gen_config\":") + gcj);
  }
  line += "}";
  // Emit: for the worker the body rides after the framing (<DONE>:<sessionId>:<body>, via the shared
  // builder) -- the turn's terminal (see the worker <done> branch's doc above); for the one-shot it IS the
  // line (the legacy single object).  Either way the caller (vlmFinalizeResult) already holds outMutex() +
  // the session lock, so this only touches stdout.
  const std::string framed =
      isWorker ? buildFramedLine("DONE", sessionId, line) : line;
  std::fputs(framed.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// The model's effective thinking switch for THIS request -- the SINGLE on/off the render and
// the parser agree on.  Precedence: a PER-REQUEST input's chat_template_kwargs enable_thinking
// (narrowest, wins -- a per-request --input/stdin flips it over the global --enable-thinking flag),
// then the global --enable-thinking flag, then NONE.  It reads enable_thinking directly (not the
// rendered prompt) to sidestep the engine's own inject-default (the render alone always looks
// thinking-on for a Qwen model, yet it may never actually reason).
std::optional<bool> resolveEffectiveThinking(const ParsedRequest& pr,
                                            const config::ModelRequestConfig& reqConfig) {
  if (pr.hasKwargs && !pr.kwargsJson.empty()) {
    std::optional<bool> per;
    try {
      auto kc = ov::genai::JsonContainer::from_json_string(pr.kwargsJson);
      if (kc.contains("enable_thinking"))
        per = kc["enable_thinking"].as_bool();
    } catch (...) {}
    if (per.has_value())
      return per;
  }
  return reqConfig.enable_thinking;
}

// The SHARED finalize both engine forms (the synchronous legacy generate and the concurrent pump)
// drive: the per-finished-request "emit the result line + update the session history" step that used
// to live as a capture-lambda inside vlmGenerate, now a free function so the two forms share it byte
// for byte (the user's "one code path").  `out` is the content the request's streamer drained in; we
// apply the pure-tool-call blank + the res.texts fallback ONCE here (the same `o` feeds BOTH emissions
// below, so the wire line -- a DONE for the worker, the legacy single object for the one-shot -- and the
// persisted assistant turn can never diverge): a non-empty
// tool_calls keeps the specials-preserving `out` (NOT the engine's specials-stripped res.texts[0],
// which would re-expose the raw tool-block tags), while a pure tool-call turn blanks a whitespace-only
// lead-in so it reports "" rather than a bare newline.  We then hold BOTH the process-global stdout
// lock AND this session's lock together (outMutex first, then the session lock -- a fixed order, and
// the only other lock hold is one of the two on its own, so they can never deadlock) so the emit and
// the history update are serialized against every other stdout writer and against the (best-effort)
// same-session appends; a CANCELLED request undoes the user message its enqueue appended (so the next
// request on the session starts clean) while a normal finish appends the assistant reply back (so the
// next turn on this session renders it).  Returns 1 for a cancelled request, else 0.
int vlmFinalizeResult(bool isWorker, const std::string& sid,
                      const ov::genai::VLMDecodedResults& res, const std::string& out,
                      const config::ModelRequestConfig& reqConfig,
                      ov::genai::ChatHistory& history, std::mutex& sessionLock,
                      bool cancelled, bool streamed) {
  // (b) The parser is the authority for a tool-call turn: a non-empty tool_calls array wins over the
  // engine's finish_reason (which may read "stop" when the model under-rands its outer tag).
  bool hasToolCalls =
      !res.parsed.empty() && res.parsed[0].is_object() &&
      res.parsed[0].contains("tool_calls") &&
      res.parsed[0]["tool_calls"].is_array() &&
      res.parsed[0]["tool_calls"].size() > 0;
  std::string o = out;
  if (!hasToolCalls && o.empty() && !res.texts.empty()) o = res.texts[0];
  if (hasToolCalls) {
    bool blank = true;
    for (const char c : o)
      if (!std::isspace(static_cast<unsigned char>(c))) { blank = false; break; }
    if (blank) o.clear();
  }
  // outMutex first, then the session lock -- both held over the emit AND the history update (a fixed
  // lock order): the content + reasoning_content (+ a tool-only blank) have either already streamed to
  // stdout per chunk (a live streamer, where `streamed` is true => the terminal line omits the result
  // field) or were SUPPRESSED (a worker streamer, live == false => `streamed` false => the content is
  // carried in the result field), so `streamed` is the caller's call (the live gate).
  std::scoped_lock<std::mutex, std::mutex> lk(outMutex(), sessionLock);
  emitResult(isWorker, sid, o, hasToolCalls, res, reqConfig, streamed);
  if (cancelled) {
    if (!history.empty()) history.pop_back();   // undo the user append this request's enqueue did
  } else {
    ov::genai::JsonContainer reply = ov::genai::JsonContainer::object();
    reply["role"] = std::string("assistant");
    reply["content"] = o;
    history.push_back(reply);
  }
  return cancelled ? 1 : 0;
}

// The core one-request driver -- the per-pipeline `generate()` the SessionDispatcher runs against a
// selected session's ChatHistory, taking the pipeline (captured in the callback) + the per-request
// sampling (the CLI base for a one-shot, the line's gen_config folded over it for a worker
// GENERATE) + the selected session + this request's input (a gen_config chat).
//
// The session is the persistence point (req 1): this request's new (normalised) messages are
// appended to the session's ChatHistory FIRST, so the engine renders the ACCUMULATED conversation
// (a worker's later turns on the same session carry earlier context); the produced assistant reply
// is appended back at the end so the NEXT turn renders it too.  On first contact the session is
// empty, so the append seeds it with exactly this request == the old stateless single-turn behaviour
// (which is also why an empty input still hits the "no message" guard below).  Everything else
// mirrors the old executeRequest verbatim, just reading/writing the (now accumulating) session
// `history` instead of a throwaway local, and threading the sessionId through to the emit.  It now
// takes the whole `Session` (the session's ChatHistory + its per-session lock, the cheap serialize
// point a concurrent enqueue takes while appending this request's messages) and drives the SHARED
// vlmFinalizeResult (the one the concurrent pump also uses) instead of its own capture-lambda, so the
// two engine forms finalize byte-for-byte identically.
int vlmGenerate(const std::shared_ptr<ov::genai::VLMPipelineBase>& pipe,
                ov::genai::Tokenizer tokenizer,
                const std::string& sessionId,
                Session& s,
                const config::ModelRequestConfig& reqConfig,
                const std::string& inputJson,
                bool isWorker) {
  ov::genai::ChatHistory& history = *s.history;   // the session's accumulated conversation (read + appended to)
  std::mutex& lock = *s.lock;                     // the session's cheap serialize point (the finalize takes it)
  // Build the base GenerationConfig from the per-request sampling (the apply_chat_template gate is
  // set per-branch below: the (tools|kwargs) sub-rule decides whether the engine or the CLI applies
  // the model template).
  ov::genai::GenerationConfig cfg = makeGenerationConfig(reqConfig);

  // The one input surface: parse the (possibly multi-line) chat as a gen_config and translate it to
  // a normalised messages array + media tensors + the tools/kwargs sub-rules.  An empty input leaves
  // the messages "[]" so the engine path below reports "no message".
  ParsedRequest pr;
  if (!inputJson.empty()) {
    rapidjson::Document g;
    if (g.Parse<0>(inputJson.c_str()).HasParseError())
      throw config::ConfigError("vlm: input is not valid JSON at position " +
                                 std::to_string(g.GetErrorOffset()));
    pr = translateGenConfig(g);
  }

  // The top-level --chat-template-kwargs flag (a global override for THIS run, parsed in
  // config::ModelRequestConfig) must also force the (tools|kwargs) sub-rule ON here: otherwise
  // the engine path -- which by construction renders with NO kwargs -- would be taken and the flag
  // would never reach apply_chat_template, so an explicit chat_template_kwargs["enable_thinking"]
  // bool could not override the reasoning gate.  Fold it into pr only when the input did not
  // already carry its own chat_template_kwargs (an inline per-request object is more specific and
  // wins); then its JSON becomes both the render's extra_context (so the template's else branch
  // fires) and the override read by resolveEffectiveThinking below.
  if (!pr.hasKwargs && reqConfig.chat_template_kwargs && !reqConfig.chat_template_kwargs->empty()) {
    pr.hasKwargs = true;
    pr.kwargsJson = *reqConfig.chat_template_kwargs;
    rapidjson::Document kd;
    if (!kd.Parse<0>(reqConfig.chat_template_kwargs->c_str()).HasParseError()) {
      auto agp = kd.FindMember("add_generation_prompt");
      if (agp != kd.MemberEnd() && agp->value.IsBool())
        pr.addGenerationPrompt = agp->value.GetBool();
    }
  }

  // (1) Persist the session: append THIS request's new (normalised) messages to the selected
  // session's ChatHistory -- the engine renders the whole accumulated conversation, so a worker's
  // later turns on the same session carry context.  On first contact the session is empty, so this
  // seeds it with exactly this request's messages == the stateless single-turn behaviour.  Build /
  // check pr first so an empty input still hits the "no message" guard the engine path throws.
  if (!pr.normMessagesJson.empty() && pr.normMessagesJson != "[]") {
    // Append under the session lock -- the cheap serialize point (the slow model step is NOT guarded
    // here, so in the concurrent form a single session's turns are best-effort serial while different
    // sessions run freely; in this synchronous legacy form the lock is uncontended and harmless).
    std::lock_guard<std::mutex> lk(lock);
    ov::genai::JsonContainer arr = ov::genai::JsonContainer::from_json_string(pr.normMessagesJson);
    auto& msgs = history.get_messages();
    for (std::size_t i = 0; i < arr.size(); ++i)
      msgs.push_back(arr[i]);
  }

  // The one on/off for the reasoning splitter, shared by BOTH branches below (so they can never
  // disagree): a per-request chat_template_kwargs enable_thinking wins, then the --enable-thinking
  // flag, then the model's own default (nullopt -> off).  It drives the render (pre-apply: projected
  // onto extra_context) AND the parser (the dynamic reasoner's setActive) -- both from this one value.
  const auto effThinking = resolveEffectiveThinking(pr, reqConfig);
  const bool thinking = effThinking.has_value() ? *effThinking : false;
  // The --stream N token-aggregation size (default 1 == today's one line per token; >1 = N tokens
  // per line; <=0 = no mid-stream output, one total line at the end).  Resolved once here and shared
  // by BOTH generate branches (so the pre-apply and the engine path can never disagree).
  const int stream = reqConfig.stream.value_or(1);
  bool useEngineTemplate = !(pr.hasTools || pr.hasKwargs);   // the (tools|kwargs) sub-rule

  // (b) Resolve the model family for this request: an explicit --model-family name wins; else
  // auto-detect from the model's OWN rendered template (the model-intrinsic sentinel); else none.
  // A selected-but-unwired name degrades (instance() splices no tool parser for it) to
  // reasoner-only / plain text; an auto-detected wired format is reported on stderr (never the wire
  // stdout, so the protocol stays one-JSON-per-line).
  std::optional<std::string> detectedName;
  const auto modelFamily = ::ovgenai::toolcall::selectModelFamily(
      reqConfig.model_family, tokenizer, &detectedName);
  if (modelFamily.has_value() && !modelFamily->empty()) {
    std::fprintf(stderr, "ovgenai: vlm: using model family '%s'\n", modelFamily->c_str());
    if (::ovgenai::toolcall::makeToolCallParserByName(*modelFamily).empty())
      std::fprintf(stderr,
          "ovgenai: vlm: --model-family '%s' is advertised but not yet wired in the CLI; "
          "degrading to plain text (no tool_calls)\n", modelFamily->c_str());
  } else if (detectedName.has_value()) {
    std::fprintf(stderr,
        "ovgenai: vlm: auto-detected model family '%s' from the model's chat template\n",
        detectedName->c_str());
  }

  // The produced content (specials preserved) the streamer drains into; computed ONCE below (after
  // the pure-tool-call blank + the res.texts fallback) and fed to BOTH emitResult AND the appended
  // assistant reply so the wire result and the persisted session turn can never diverge.
  std::string acc;
  // The shared finalize (the one the concurrent pump also uses): this legacy path is synchronous and
  // streams live (the streamer's live == true for a one-shot; a worker's per-chunk <item> stream is the
  // ContentStreamer's own `stream > 0` gate), so the terminal line is gate by how the turn streamed; the
  // session is never cancelled here (only the normal-finish append).  vlmFinalizeResult applies the
  // hasToolCalls -> tool_calls authority + the res.texts fallback + the tool-only blank ONCE, then emits
  // its result line (under the process-global stdout lock) and appends the assistant reply to the session
  // (under its lock) -- so the wire + the persisted turn can never diverge.
  auto finalize = [&](const ov::genai::VLMDecodedResults& res) -> int {
    // `streamed` = did the turn emit per-chunk lines?  For the worker that is exactly `stream > 0` (the
    // ContentStreamer's m_live gate: a `--stream 0` turn streams no <item>, so its WHOLE turn must ride
    // the lone <done>; a `--stream N>0` turn does, so the <done> carries only finish_reason + metrics).
    // The one-shot always streams live, so its result line is always gated as streamed (unchanged).
    // `stream` is the value resolved above from `reqConfig` -- the SAME one the streamer used.
    const bool streamed = isWorker ? (stream > 0) : true;
    return vlmFinalizeResult(isWorker, sessionId, res, acc, reqConfig, history, lock,
                             /*cancelled=*/false, streamed);
  };

  if (!useEngineTemplate) {
    // --- (tools | kwargs) pre-apply (string path, apply_chat_template = false).  Mirror the reference parser:
    // apply the template here (the model's own template -- "" = keep the loaded one -- plus tools /
    // extra_context), then hand the resulting string to the string generate with the gate OFF so the
    // engine does not template a second time.  The engine's own normalize_media_tags then maps the
    // (model-native) tags the template emitted onto the images by order.  The HISTORY rendered is the
    // (now accumulated) session -- the pre-apply path renders the same accumulated conversation the
    // engine path would, just hand-templated.
    cfg.apply_chat_template = false;
    std::optional<ov::genai::JsonContainer> tools = pr.hasTools
        ? std::optional<ov::genai::JsonContainer>(ov::genai::JsonContainer::from_json_string(pr.toolsJson))
        : std::nullopt;
    // Project the effective thinking switch onto the render (apply_chat_template's extra_context) so
    // the engine's applied template matches the parser split: enable_thinking == thinking, the SAME
    // effective value -- render + split can never disagree.  This also removes the engine's own
    // render-time inject-default from the pre-apply path (so a model that does not actually reason
    // no longer gets the reasoning-on form).
    rapidjson::Document ek;
    if (pr.hasKwargs && !pr.kwargsJson.empty())
      ek.Parse(pr.kwargsJson.c_str());
    if (!ek.IsObject()) ek.SetObject();
    auto& ekAlloc = ek.GetAllocator();
    ek.RemoveMember("enable_thinking");
    rapidjson::Value ekKey("enable_thinking", ekAlloc);
    rapidjson::Value ekVal; ekVal.SetBool(thinking);
    ek.AddMember(ekKey, ekVal, ekAlloc);
    std::optional<ov::genai::JsonContainer> extra =
      std::optional<ov::genai::JsonContainer>(
        ov::genai::JsonContainer::from_json_string(config::compactJsonOf(ek)));
    std::string rendered = tokenizer.apply_chat_template(
        history, pr.addGenerationPrompt, std::string(), tools, extra);

    // The single DynamicReasoningIncrementalParser is ALWAYS first (instance() puts it at slot 0, the
    // tool parser after) -- the one parser for both sides.  The pipeline sets its on/off via setActive
    // using the SAME effective thinking the render above used (render + split can never disagree), then
    // the ContentStreamer runs over the list.
    auto streamer = ::ovgenai::toolcall::ContentStreamer::instance(
        tokenizer, modelFamily.value_or(std::string()), thinking, acc, stream,
        /*live=*/true, /*isWorker=*/isWorker, sessionId);
    auto res = pipe->generate(rendered, pr.images, cfg, ov::genai::StreamerVariant(streamer));
    // (b) BRIDGE: the VLM pipeline this command drives (unlike the LLM pipeline's
    // run_generate_with_parsers) never copies a TextParserStreamer's get_parsed_message() into
    // res.parsed, so a tool-call turn would leave res.parsed EMPTY and emitResult reported the
    // engine's own finish_reason "tool_calls" yet NO tool_calls array.  The CLI still holds the
    // carrier (a ContentStreamer, a TextParserStreamer), so bridge it: a plain TextStreamer makes
    // the cast miss (no-op).
    ::ovgenai::toolcall::bridgeParsedMessage(streamer, res);
    return finalize(res);
  }

  // --- engine path (default, no tools/kwargs): the HISTORY form, apply_chat_template = true -- the
  // history path always applies the template in the engine (the gate only affects the string form) and
  // splices <ov_genai_image_N>/<ov_genai_video_N>/<ov_genai_audio_N>, which the translator already put
  // in the content parts, so we just pass the (accumulated) history + this request's media.
  cfg.apply_chat_template = true;
  if (history.empty())
    throw config::ConfigError("vlm: no message to generate -- give a chat via --input,"
                              " pipe one on stdin, or type one at a terminal (a lone"
                              " '.' ends the input, as in the mail command)");
  auto streamer = ::ovgenai::toolcall::ContentStreamer::instance(
      tokenizer, modelFamily.value_or(std::string()), thinking, acc, stream,
      /*live=*/true, /*isWorker=*/isWorker, sessionId);
  auto res = pipe->generate(history, pr.images, pr.videos, pr.audios, {}, cfg,
                           ov::genai::StreamerVariant(streamer));
  ::ovgenai::toolcall::bridgeParsedMessage(streamer, res);
  return finalize(res);
}

}  // close the file-local (vlm-specific helpers) anonymous namespace

// ---------------------------------------------------------------------------
// GenerationStrategy backends -- the two engine forms behind ONE dispatcher code path
//
// The user's mandate is "do not maintain two different paths for generate() and add_request()".  We
// honour that at the DISPATCHER level, not the engine level: the SessionDispatcher's run()/worker()
// drive a std::unique_ptr<GenerationStrategy>, and BOTH backends answer the same six virtuals, so the
// dispatcher has exactly one run() (enqueue the one request, then `while (hasPending()) stepAndDrain()`)
// and one worker() (startBackground(), then the line loop that enqueues / cancels).  The two backends
// DIFFER only in HOW they do the engine work:
//   * VlmLegacyStrategy wraps the UNCHANGED monolithic vlmGenerate (the pre-refactor path) byte-for-byte
//     -- it is the FALLBACK backend the one-shot / worker take when a ContinuousBatchingPipeline can be
//     built is refused (the load picked an SDPA / NPU backend that cannot batch, or the CB ctor threw).
//     It is fully synchronous: enqueue() does the whole request, then reports nothing pending, so the
//     dispatcher's drain loop never runs and the pump stays idle.
//   * VlmConcurrentStrategy is the NEW form and the default on the paged-attention backends (x86/arm64
//     / GPU): it drives the engine's ContinuousBatchingPipeline through add_request() (one request at a
//     time, the caller's own unique id flowing through to the handle so read() keys back to it) + a
//     background pump of step() + read() feeding each request's ContentStreamer, so multiple GENERATEs on
//     DIFFERENT sessions overlap / batch on ONE pipeline.  same-session turns are best-effort serial (the
//     per-session lock guards only the cheap append/finalize points, not the slow model step).
// ---------------------------------------------------------------------------

// The FALLBACK / synchronous backend: wrap the monolithic generate as a GenerationStrategy so the
// dispatcher drives it on the same one code path.  enqueue() is the full vlmGenerate (translate + append
// the user messages to the session, render the accumulated history, generate, finalize via the shared
// vlmFinalizeResult); the rest are no-ops (a synchronous request leaves nothing for a pump to drain),
// which is exactly why the dispatcher's drain loop is a no-op here and output is byte-identical to before
// this refactor.
class VlmLegacyStrategy : public GenerationStrategy {
 public:
  VlmLegacyStrategy(std::shared_ptr<ov::genai::VLMPipelineBase> pipe, ov::genai::Tokenizer tok)
      : m_pipe(std::move(pipe)), m_tok(std::move(tok)) {}

  int enqueue(const std::string& sid, Session& s, const config::ModelRequestConfig& rc,
              const std::string& in, bool isWorker) override {
    return vlmGenerate(m_pipe, m_tok, sid, s, rc, in, isWorker);
  }
  int  stepAndDrain() override { return 0; }           // the synchronous path leaves nothing to drain
  bool hasPending() override { return false; }
  void startBackground() override {}                   // no background pump on the monolithic path
  void stop() override {}
  void cancel(const std::string&) override {}   // a synchronous request cannot be mid-flight

 private:
  std::shared_ptr<ov::genai::VLMPipelineBase> m_pipe;  // kept alive so the (multi-turn) session's pipeline outlives the dispatcher
  ov::genai::Tokenizer m_tok;
};

// The CONCURRENT / batched backend (the default, the whole point of option 4).  Holds the engine's
// ContinuousBatchingPipeline + tokenizer + a registry of in-flight requests (cbid -> a PendingRequest
// that COPIES the owning Session's two shared_ptrs, so the record outlives a map re-lookup/erase during
// the long model step) + a mutex + a condition variable a background pump waits on while the registry is
// empty + an atomic stop flag + the bg pump thread.  RAII: the ~dtor joins the pump (stop()).  It is
// idle-aware: while the registry is empty it BLOCKS on the CV (it never steps an empty pipeline), an
// enqueue wakes it to drain a running request, and stop() first drains whatever is in-flight before
// retiring it -- so a worker that simply stays idle takes no wasted steps, and a clean shutdown does
// not leak an in-flight request's blocks.
class VlmConcurrentStrategy : public GenerationStrategy {
 public:
  // One in-flight request the pump owns.  cbid is the unique engine request id WE hand add_request (it
  // flows through to the handle, so read() outputs key back to it and we key the registry on the same id);
  // everything else is the per-request carrier the finalize emits from.
  struct PendingRequest {
    uint64_t cbid = 0;                                  // the engine request id (1-based, monotonic, unique per add_request)
    std::string sid;
    std::shared_ptr<ov::genai::ChatHistory> history;   // COPIED from the session (so it outlives a map erase)
    std::shared_ptr<std::mutex> lock;                  // COPIED from the session (the finalize takes it)
    ov::genai::GenerationHandle handle;                // RAII: the dtor releases the engine's slot
    std::shared_ptr<toolcall::ContentStreamer> streamer;  // the per-request content/reasoning/tool parser (live-gated)
    std::shared_ptr<std::string> acc;                  // the drained content the streamer fills (the <done>'s content; the one-shot's "result")
    std::optional<ov::genai::GenerationFinishReason> lastFinish;  // the last non-NONE finish seen
    config::ModelRequestConfig rc;                     // the per-request sampling (the one-shot echoes it as "gen_config")
    bool isWorker = false;
  };

  VlmConcurrentStrategy(std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipe,
                        ov::genai::Tokenizer tok, double loadTimeMs)
      : m_pipe(std::move(pipe)), m_tok(std::move(tok)), m_loadTimeMs(loadTimeMs) {}
  ~VlmConcurrentStrategy() override { stop(); }        // idempotent: a 2nd stop just re-joins a non-joinable pump
  VlmConcurrentStrategy(const VlmConcurrentStrategy&) = delete;
  VlmConcurrentStrategy& operator=(const VlmConcurrentStrategy&) = delete;

  int enqueue(const std::string& sid, Session& s, const config::ModelRequestConfig& rc,
              const std::string& in, bool isWorker) override {
    ov::genai::GenerationConfig cfg = makeGenerationConfig(rc);

    // Translate the input (a gen_config chat) into the normalised messages + media + the tools/kwargs
    // sub-rules -- the SAME translation vlmGenerate runs, so the two backends agree on what maps to the
    // engine; a parse error is a ConfigError the dispatcher surfaces without ever registering a
    // half-built request (so the registry stays consistent).
    ParsedRequest pr;
    if (!in.empty()) {
      rapidjson::Document g;
      if (g.Parse<0>(in.c_str()).HasParseError())
        throw config::ConfigError("vlm: input is not valid JSON at position " +
                                   std::to_string(g.GetErrorOffset()));
      pr = translateGenConfig(g);
    }
    // A top-level --chat-template-kwargs flag folds into pr when the input did not already carry its own
    // (a more specific inline object wins) -- then it is both the render's extra_context and the reasoning
    // gate resolveEffectiveThinking reads.
    if (!pr.hasKwargs && rc.chat_template_kwargs && !rc.chat_template_kwargs->empty()) {
      pr.hasKwargs = true;
      pr.kwargsJson = *rc.chat_template_kwargs;
      rapidjson::Document kd;
      if (!kd.Parse<0>(rc.chat_template_kwargs->c_str()).HasParseError()) {
        auto agp = kd.FindMember("add_generation_prompt");
        if (agp != kd.MemberEnd() && agp->value.IsBool())
          pr.addGenerationPrompt = agp->value.GetBool();
      }
    }
    // Append THIS request's normalised messages to the session's ChatHistory, under the session lock --
    // the cheap point the pump's finalize also takes (so a same-session append never interleaves a
    // finalize; the slow step below is NOT locked, so different sessions run freely).  The accumulated
    // history is what we then render.  Take the lock only for the (cheap) append, and release it before
    // the render so the per-session lock is held as briefly as possible.
    if (!pr.normMessagesJson.empty() && pr.normMessagesJson != "[]") {
      std::lock_guard<std::mutex> lk(*s.lock);
      ov::genai::JsonContainer arr = ov::genai::JsonContainer::from_json_string(pr.normMessagesJson);
      auto& msgs = s.history->get_messages();
      for (std::size_t i = 0; i < arr.size(); ++i)
        msgs.push_back(arr[i]);
    }
    // Brand-new / message-less session: same "no message" guard the legacy engine path throws -- a Config
    // Error the dispatcher reports and (for a worker) keeps running.
    if (s.history->empty())
      throw config::ConfigError("vlm: no message to generate -- give a chat via --input (run) or a "
                                "messages-bearing GENERATE line (worker)");
    // The CB add_request path carries no audio (its prompt overload is prompt+images+videos); a request
    // with an audio part is a clear usage error -> route it via the legacy (SDPA) monolithic path.
    if (!pr.audios.empty())
      throw config::ConfigError("vlm: audio / Omni parts are not supported on the continuous-batching "
                                "path (the default backend); use --runtime-config "
                                "{\"ATTENTION_BACKEND\":\"SDPA\"} for the legacy monolithic path");

    // Resolve the one on/off for the reasoning splitter + the model family + the --stream N aggregation
    // (shared with the legacy form so the render (pre-applied) and the parser split can never disagree).
    const auto effThinking = resolveEffectiveThinking(pr, rc);
    const bool thinking = effThinking.has_value() ? *effThinking : false;
    std::optional<std::string> detectedName;
    const auto modelFamily = ::ovgenai::toolcall::selectModelFamily(rc.model_family, m_tok, &detectedName);

    // PRE-APPLY the chat template ON the CLI: the CB's internal add_request path calls
    // set_apply_chat_template_status(cfg.apply_chat_template) then normalize_prompt(prompt, 0, 0, 0,
    // images, videos, {}), so with the gate OFF the engine does NOT re-template -- this ONE add_request
    // entry replaces the legacy split (the pre-apply branch AND the engine-template branch).  Project
    // enable_thinking's effective value onto the render's extra_context so the applied template matches
    // the parser split (the same value; this also removes the engine's own inject-default, so a
    // non-reasoning model no longer renders the thinking-on form).  Take the session lock for the
    // (cheap) render too, so a same-session render never interleaves an append.
    cfg.apply_chat_template = false;
    std::optional<ov::genai::JsonContainer> tools = pr.hasTools
        ? std::optional<ov::genai::JsonContainer>(ov::genai::JsonContainer::from_json_string(pr.toolsJson))
        : std::nullopt;
    rapidjson::Document ek;
    if (pr.hasKwargs && !pr.kwargsJson.empty())
      ek.Parse(pr.kwargsJson.c_str());
    if (!ek.IsObject()) ek.SetObject();
    auto& ekAlloc = ek.GetAllocator();
    ek.RemoveMember("enable_thinking");
    rapidjson::Value ekKey("enable_thinking", ekAlloc);
    rapidjson::Value ekVal; ekVal.SetBool(thinking);
    ek.AddMember(ekKey, ekVal, ekAlloc);
    std::optional<ov::genai::JsonContainer> extra =
      std::optional<ov::genai::JsonContainer>(
        ov::genai::JsonContainer::from_json_string(config::compactJsonOf(ek)));
    std::string rendered;
    {
      std::lock_guard<std::mutex> lk(*s.lock);
      rendered = m_tok.apply_chat_template(*s.history, pr.addGenerationPrompt, std::string(), tools, extra);
    }

    // The worker STREAMS by default -- per-chunk <item>:<session>:<body> lines -- and a `--stream N` (this
    // request's gen_config) controls the window exactly like `run`: N>1 -> N tokens per line, N<=0 -> no
    // <item> lines at all (the whole turn then rides the single framed <done>).  The one-shot keeps its
    // legacy live per-token stream.  `live` here is the one-shot's bare-stream gate; the worker's per-chunk
    // stream is gated by `stream>0` inside the streamer (its m_live = isWorker ? stream>0 : live), and the
    // session id is threaded so the streamer frames its <item> lines with it.
    const int stream = rc.stream.value_or(1);
    auto acc = std::make_shared<std::string>();
    std::shared_ptr<toolcall::ContentStreamer> streamer = ::ovgenai::toolcall::ContentStreamer::instance(
        m_tok, modelFamily.value_or(std::string()), thinking, *acc, stream,
        /*live=*/true, /*isWorker=*/isWorker, sid);

    // Hand OUR own unique (1-based, monotonic) id to add_request -- the engine uses it as the request id
    // (it builds the SequenceGroup with it), so the handle's read() outputs key back to it and we key the
    // pump registry on the SAME id.
    const uint64_t cbid = ++m_nextCbid;

    // Which add_request OVERLOAD to call -- the CB decides vlm-vs-llm itself (it fixed its own
    // m_model_input_type, TOKENS or EMBEDDINGS, at construction from the model dir: a vision/text embedder
    // dir -> EMBEDDINGS, a plain-text dir -> the default TOKENS).  So WE make no vlm/llm claim; we only pick
    // the overload the request needs:
    //   * a request that CARRIES image/video media must use the 5-arg (prompt + images + videos) form, which
    //     REQUIRES the model be an embedder -- it asserts m_model_input_type == EMBEDDINGS at
    //     continuous_batching/pipeline_base.cpp:817 -- i.e. a real VLM;
    //   * otherwise the 2-arg (prompt + sampling) form, which is UNIVERSAL: the engine branches on the model's
    //     OWN input type (TOKENS -> it tokenizes the already-rendered prompt and runs the tokens path;
    //     EMBEDDINGS -> the embedder form with empty media -- so even a media-less VLM request runs through
    //     it).  This is exactly "the vlm/llm split is the CB's": the one key we peek (config.json's
    //     vision_config, upstream) only *labels* the run -- it never chooses the overload here.  It is also
    //     the fix for a pure-text LLM (e.g. Qwen3-Coder): routed here as llm, it is now a TOKENS model and
    //     takes the 2-arg form, instead of the 5-arg embedder form that asserted "doesn't support embeddings".
    ov::genai::GenerationHandle handle;
    if (!pr.images.empty() || !pr.videos.empty())
      handle = m_pipe->add_request(cbid, rendered, pr.images, pr.videos, cfg);   // a VLM: carry the media
    else
      handle = m_pipe->add_request(cbid, rendered, cfg);                          // text-only / media-less: the universal form

    auto p = std::make_shared<PendingRequest>();
    p->cbid = cbid;
    p->sid = sid;
    p->history = s.history;            // copy the two shared_ptrs (the record outlives a map re-lookup/erase)
    p->lock = s.lock;
    p->handle = handle;
    p->streamer = streamer;
    p->acc = acc;
    p->rc = rc;
    p->isWorker = isWorker;
    {
      std::lock_guard<std::mutex> lk(m_mtx);
      m_pending[cbid] = std::move(p);   // the registry: a concurrent insert of a DIFFERENT cbid cannot race this
    }
    m_cv.notify_one();                  // the pump may be asleep (registry was empty) -- wake it to drain this one
    return 0;
  }

  int stepAndDrain() override {         // guarded (the run() manual drain loop + the bg pump both call this)
    if (m_stopped.load()) return -1;    // a stopped pump / one-shot has nothing to pull
    return doStepAndDrain();
  }
  bool hasPending() override {
    bool any;
    { std::lock_guard<std::mutex> lk(m_mtx); any = !m_pending.empty(); }
    if (any) return true;
    return m_pipe->has_non_finished_requests();   // engine view (its own lock) -- the last in-flight still running
  }
  void startBackground() override {       // idempotent; only the worker starts the pump (the one-shot drives manually)
    if (m_bg.joinable()) { m_bg.detach(); return; }
    m_bg = std::thread([this] { bgLoop(); });
  }
  void stop() override {                  // idempotent (UNLOAD then the dtor both call it): once stopped, skip the join
    if (m_stopped.exchange(true)) {
      if (m_bg.joinable()) m_bg.join();
      return;
    }
    m_cv.notify_all();
    if (m_bg.joinable()) m_bg.join();            // the pump drains any in-flight request, THEN exits
    // Graceful teardown: the pump above already brought any in-flight request to completion before it
    // exited (so this is normally a no-op); for the one-shot form (no pump was started) it IS the drain
    // -- loop while a request is still queued, finishing it (emit its terminal <done> + free its blocks) before
    // we may be torn down -- so a clean-EOF mid-generation no longer leaks its blocks / drops its result.
    // Keyed on the registry (not the engine's lagging flag) so it cannot busy-spin.
    bool any;
    { std::lock_guard<std::mutex> lk(m_mtx); any = !m_pending.empty(); }
    while (any) {
      doStepAndDrain();
      { std::lock_guard<std::mutex> lk(m_mtx); any = !m_pending.empty(); }
    }
  }
  void cancel(const std::string& sid) override {
    std::lock_guard<std::mutex> lk(m_mtx);
    for (auto& [cbid, p] : m_pending) {
      if (p->sid != sid)
        continue;
      p->handle->cancel();              // flip every in-flight request on this session to CANCEL; the pump's
                                          // next step finalizes it (popping the appended user message) and
                                          // the handle is released
    }
  }

 private:
  void bgLoop() {                               // the single background pump the worker starts
    while (true) {
      bool work;                                // a request is queued/running -> we have something to drain
      bool stop;
      {
        // IDLE-AWARE (the fix): block on the CV while the registry is empty.  An enqueue notifies us
        // (the registry is then non-empty -> `work`); stop() flips m_stopped + notifies (so we still
        // wake, but only to drain any in-flight request, then exit).  We NEVER step an empty pipeline
        // (the empty-poll/step that made a worker with no pending request die is gone).
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv.wait(lk, [&] { return m_stopped.load() || !m_pending.empty(); });
        stop = m_stopped.load();
        work = !m_pending.empty();
      }                                  // lock released at scope end; doStepAndDrain re-takes it
      if (stop && !work)
        break;                                   // stop woke us with nothing queued -> retire the worker
      if (work)
        doStepAndDrain();                       // drains one token; a still-running request stays in the
                                                // registry, so the next round runs again until it finishes
    }
  }

  // The actual one-step pump: snapshot the still-running requests under m_mtx + RELEASE it (the step is
  // lock-free: a handle's SynchronizedQueue makes its read() race-safe with the step that just pushed),
  // step the engine, then feed each request's drained tokens out and finalize any that terminated.
  // m_mtx is held ONLY for the snapshot + the trailing erase (never during the slow step/finalize), and
  // the finalize (vlmFinalizeResult) takes only outMutex + the per-session lock -- so no thread ever holds
  // m_mtx together with either of those, and the only other lock-holder (a line-reader enqueue) takes just
  // one of them at a time -> there is no lock-ordering cycle (deadlock-free).
  int doStepAndDrain() {
    std::vector<std::shared_ptr<PendingRequest>> live;
    {
      std::lock_guard<std::mutex> lk(m_mtx);
      for (auto& [cbid, p] : m_pending)
        if (p->handle->get_status() == ov::genai::GenerationStatus::RUNNING)
          live.push_back(p);                    // copy the shared_ptr -> the PendingRequest survives across the step
    }
    if (live.empty())
      return 0;                                  // idle: never step() the empty pipeline (idle-safe / no churn)
    m_pipe->step();                             // advances the whole batch one token; pulls in any awaiting request
    std::vector<uint64_t> finished;
    for (auto& p : live) {
      while (p->handle->can_read()) {           // the SynchronizedQueue makes this race-safe with step()
        auto outs = p->handle->read();
        for (auto& kv : outs) {
          p->streamer->write(kv.second.generated_ids);     // feed the decoded tokens out to the streamer (live-gated)
          if (kv.second.finish_reason != ov::genai::GenerationFinishReason::NONE)
            p->lastFinish = kv.second.finish_reason;
        }
      }
      if (p->handle->get_status() == ov::genai::GenerationStatus::RUNNING)
        continue;                               // not done yet: leave it for the next step
      p->streamer->end();                       // flush the trailing window + finalize the parser (content + tool_calls)
      ov::genai::VLMDecodedResults res;
      res.parsed.push_back(p->streamer->consolidatedMessage());   // content + reasoning + the one-shot tool_calls array
      res.texts.push_back(*p->acc);                                              // the drained (specials-preserving) content
      ov::genai::VLMPerfMetrics pm = p->handle->get_vlm_perf_metrics();
      pm.load_time = m_loadTimeMs;                             // the CB's get_metrics() carries no load_time
      res.perf_metrics = pm;
      res.finish_reasons.push_back(p->lastFinish.value_or(ov::genai::GenerationFinishReason::NONE));
      const bool cancelled =
          p->handle->get_status() == ov::genai::GenerationStatus::CANCEL;
      // Both a worker and a one-shot finish with a terminal line (a framed <done> for the worker, the
      // bare single object for the one-shot), and `streamed` = did the turn put its content on
      // per-chunk lines?  For the worker that is exactly its resolved `stream > 0` (the ContentStreamer's
      // m_live gate, the SAME value the enqueue used to build it): a `--stream N>0` turn streamed its
      // <item> deltas, so its <done> carries only the other keys (finish_reason + metrics); a `--stream
      // <=0` turn streamed nothing, so its lone <done> carries the WHOLE turn.  `stream` is read straight
      // from this request's carrier (p->rc), not re-resolved, so it cannot disagree with the streamer.
      // A CANCEL does not change that streamed/off decision (it only also pops the session's last user
      // message, the separate `cancelled` argument); the one-shot always streams live, so its line is
      // always gated as streamed (unchanged).
      const bool streamed =
          p->isWorker ? (p->rc.stream.value_or(1) > 0) : true;
      vlmFinalizeResult(p->isWorker, p->sid, res, *p->acc, p->rc,
                        *p->history, *p->lock, cancelled, streamed);
      finished.push_back(p->cbid);
    }
    {
      std::lock_guard<std::mutex> lk(m_mtx);    // the only m_mtx mutation here is a keyed erase (races no insert)
      for (uint64_t cbid : finished)
        m_pending.erase(cbid);                  // the PendingRequest dtor releases the engine's slot
    }
    return 0;
  }

  std::shared_ptr<ov::genai::ContinuousBatchingPipeline> m_pipe;
  ov::genai::Tokenizer m_tok;
  double m_loadTimeMs = 0.0;                    // captured around the ctor (the CB reports no load_time)
  std::atomic<uint64_t> m_nextCbid{0};          // ++ per add_request (unique engine request id); atomic, so an
                                                // enqueue from any thread can't race it (a POD, no lock needed)
  mutable std::mutex m_mtx;                     // guards m_pending (the registry)
  std::condition_variable m_cv;                 // the pump waits on it while the registry is empty
  std::unordered_map<uint64_t, std::shared_ptr<PendingRequest>> m_pending;
  std::atomic<bool> m_stopped = false;                 // the stop flag (set+notify in stop(), polled in the pump + drain)
  std::thread m_bg;
};

// --- The vlm body (the one real body today) -- but it no longer parses the command line -- --------
// It receives the PARSED arguments runModeEntry produced ONCE for the whole subcommand (a
// PipelineParsed: the four typed configs + the run form's --input + the raw model/device the body
// was given) -- so the body owns only "create the pipeline + generate" (the PipelineBinding's job);
// the general argument parsing is the front-end's (commands.cpp runModeEntry -> parsePipelineArgs),
// and the --help this vlm used to render in-body is likewise the front-end's (the front-end
// intercepts --help BEFORE delegating, so a body is never reached for it).  With that, the body:
//   1. reports the mode + the model it was given, as the stubs do (so a failure is still visible);
//   2. CREATE the pipeline (load a VLMPipeline from the resolved load/runtime/scheduler config) --
//      the PipelineBinding's job, so the body, not the front-end, owns the load;
//   3. build a GenerateRequestFn that drives it (capturing the created pipeline + tokenizer) and
//      hand a SessionDispatcher built around that pipeline (its injected run/worker driver);
//   4. call dispatcher.run() (the run one-shot) or .worker() (the long-lived form) on it.
// For a `run` one-shot the dispatcher makes exactly one generated session; for a `worker` it hooks
// stdin and selects / creates a session per incoming line's session_id.  There is no config to MISS
// here (runModeEntry already validated the load set before routing, and handed it in), so a run
// failure is a load / generate throw (exit 1) -- not a usage error.
int runVlmBody(const std::string& prog, const PipelineParsed& args, bool worker) {
  // Report the mode + the *requested* model / device up front (mirrors the other pipelines' stub),
  // so the mode is visible on stderr even if a later step (the load) then fails.
  std::fprintf(stderr, "ovgenai: vlm%s: loading model '%s' on %s\n",
               worker ? " (worker mode)" : " (one-shot)", args.model.c_str(), args.device.c_str());
  // For a resolved ref (an `hf://` -> a snapshot dir) that is not the path actually loaded, report
  // where it resolved so a long load is identifiable.  A plain model path is returned verbatim, so
  // the resolve line is skipped for it.
  if (args.model.rfind("hf://", 0) == 0 && args.model != args.load.model)
    std::fprintf(stderr, "ovgenai: vlm: resolved %s -> %s\n",
                 args.model.c_str(), args.load.model.c_str());

  try {
    // The pipeline create is now the interesting part (behaviour change #4): we PREFER the engine's
    // ContinuousBatchingPipeline -- driven via add_request() + step(), which is the whole point of option
    // 4 (multiple sessions' requests overlap / batch on ONE pipeline) -- and fall back to the legacy
    // monolithic VLMPipeline only when it cannot be built (the load picked an SDPA / NPU backend that
    // cannot batch, or the CB ctor threw).  Which backend is chosen is ONE decision at load time; the two
    // are then driven on the SAME dispatcher code path behind the GenerationStrategy interface (honouring
    // the user's "do not maintain two paths for generate() and add_request()": at the dispatcher level,
    // run()/worker() never know which backend they drive).
    GenAIPipelineConfig pcfg(args.load, args.scheduler, args.runtime);

    // The continuous-batching path, tried first.  A VLMPipeline is a pimpl (neither copyable nor moveable)
    // so a held pipeline is a shared_ptr: here to the engine's ContinuousBatchingPipeline, which auto-builds
    // the VLM embedder when the model dir has a text_embeddings_model.xml (so add_request(images) works for
    // a VLM).  The scheduler config is its DEDICATED arg (the CB ctor does NOT read a "scheduler_config" key
    // out of the AnyMap), so we hand it via resolveSchedulerConfig() and pass the AnyMap with that key
    // removed (toPipelinePropertiesNoScheduler()).  We time the ctor (the CB's get_metrics() reports NO
    // load_time -- only the VLM/LLM pipelines do) so the concurrent path can still stamp one on the metrics.
    // Any genuine engine throw is caught and we fall back; a usage ConfigError is re-thrown so the dispatcher
    // still reports it and exits 2 (it is-a std::exception, so this more-specific catch must come first).
    std::unique_ptr<GenerationStrategy> strategy;
    // A repeatable opt-out (for benchmarking, and for paged-attention hosts that simply should not
    // batch): with OVGENAI_FORCE_LEGACY set we build the legacy (monolithic, non-batching, otherwise
    // byte-identical) VLMPipeline directly, instead of letting the "prefer CB, fall back on a throw"
    // rule pick the backend at load.  The engine's CB ctor rewrites to paged attention UNCONDITIONALLY
    // (an ATTENTION_BACKEND=SDPA in the runtime config is just overwritten), so there is no engine-side
    // lever to opt out -- this is the only clean way to A/B the two backends on one model + one host.
    // Unset (the default) keeps the prefer-CB-at-load decision exactly as before.
    if (std::getenv("OVGENAI_FORCE_LEGACY")) {
      std::fprintf(stderr, "ovgenai: vlm: OVGENAI_FORCE_LEGACY set -> legacy (monolithic, non-batching) path\n");
      std::shared_ptr<ov::genai::VLMPipeline> owned = std::make_shared<ov::genai::VLMPipeline>(
          args.load.model, args.load.device, pcfg.toPipelineProperties());
      ov::genai::Tokenizer tokenizer = owned->get_tokenizer();
      std::shared_ptr<ov::genai::VLMPipelineBase> base = owned;
      strategy = std::make_unique<VlmLegacyStrategy>(std::move(base), tokenizer);
    } else {
    try {
      const auto t0 = std::chrono::steady_clock::now();
      auto cb = std::make_shared<ov::genai::ContinuousBatchingPipeline>(
          args.load.model, pcfg.resolveSchedulerConfig(), args.load.device,
          pcfg.toPipelinePropertiesNoScheduler());
      const double ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0).count();
      ov::genai::Tokenizer tokenizer = cb->get_tokenizer();
      std::fprintf(stderr, "ovgenai: vlm: using the continuous-batching pipeline (add_request / step)\n");
      strategy = std::make_unique<VlmConcurrentStrategy>(std::move(cb), tokenizer, ms);
    } catch (const config::ConfigError&) {
      throw;                                    // a usage error -> let it propagate to the outer catch (exit 2)
    } catch (const std::exception& e) {
      // CB unavailable (e.g. the load picked an SDPA / NPU backend that cannot batch) -> fall back to the
      // legacy monolithic VLMPipeline (the pre-refactor path, byte-identical + serial).  The engine
      // already chose this for an SDPA/NPU load, so the fallback is the "opt out of batching" path.
      std::fprintf(stderr,
          "ovgenai: vlm: continuous-batching unavailable (%s); falling back to the legacy pipeline "
          "(sequential, non-batching)\n", e.what());
      std::shared_ptr<ov::genai::VLMPipeline> owned = std::make_shared<ov::genai::VLMPipeline>(
          args.load.model, args.load.device, pcfg.toPipelineProperties());
      ov::genai::Tokenizer tokenizer = owned->get_tokenizer();
      std::shared_ptr<ov::genai::VLMPipelineBase> base = owned;   // to the public base (virtual dispatch)
      strategy = std::make_unique<VlmLegacyStrategy>(std::move(base), tokenizer);
    }
    }   // close the "else": the prefer-CB branch above (the forced-legacy "if" owns its own scope)

    // Hand the chosen strategy to a fresh SessionDispatcher (which owns it, and stops it on destruction) and
    // run it in the requested form: run() enqueues the one request then drains it to completion with a manual
    // loop; worker() starts the background pump and drives the line loop.  Both drive the strategy through the
    // SAME virtual interface -- the one dispatcher code path for both engine forms.
    SessionDispatcher dispatcher(std::move(strategy), args.request, args.input);
    if (worker)
      return dispatcher.worker();
    return dispatcher.run();
  } catch (const config::ConfigError& e) {
    // A usage error that surfaced mid-run -- notably a bad image part: vlmGenerate throws
    // "vlm: could not open image file '...'" at DECODE time, after the load config is already
    // valid and the load is well under way.  It is still a ConfigError (a usage problem), so it
    // is exit 2 -- and this handler must sit BEFORE the std::exception catch below (a ConfigError
    // is-a std::exception, so the more-specific one must come first, or it would fall through to
    // the run-failure exit 1, which is what an earlier draft of this body got wrong).
    std::fprintf(stderr, "ovgenai: vlm: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    // A run failure (a model-load / generate throw that is NOT a ConfigError) -- not a usage
    // error; the label was already printed, so just report and exit non-zero.  The load-config
    // set itself never reaches here: runModeEntry validated it -- and reported any miss -- before
    // routing; only a LATER mid-run ConfigError (the catch above), or a genuine run failure, does.
    std::fprintf(stderr, "ovgenai: vlm: %s\n", e.what());
    return 1;
  }
}

}  // namespace ovgenai

// --- The pipeline registry --------------------------------------------------------------
// The run/worker binding table, as a file-local `static const` map keyed by the --pipeline name
// (runModeEntry no longer builds or scans it: it parses the one argument set exactly once, then calls
// dispatchPipeline, which looks it up by name -- a single find, not a linear scan over the set).
// Each entry says ONLY which body runs that pipeline -- NOT its description: the descriptions stay in
// the canonical name set (it feeds show-pipelines + the "choose one of" routing message), so this
// table carries no descriptions.  (A name in the table but missing from the canonical set -- or
// vice-versa -- would show up at run time: the everyPipeline test runs every canonical pipeline by
// name and expects exit 0, so the two sets stay in lock-step.)
//
// Each entry's body is set EXPLICITLY (one line each) -- NOT built in a loop with an `if`: `vlm`
// (the one pipeline with a real body today) is its own lambda that drives runVlmBody; the other
// thirteen not-yet-implemented pipelines share one lambda onto runPipelineStub (the infrastructure-only
// stub the front-end owns as the single --help source).  When a pipeline grows a real body, only its
// one entry changes -- no shared stub code, no per-name `if`.
namespace ovgenai {
namespace commands {

// The single shared body for the not-yet-implemented pipelines: runPipelineStub, fed the pipeline name
// it just ran (parsed.pipeline) so the stub can name itself; the rest of its report comes from the
// parsed model / device.  One lambda, referenced by every non-vlm entry below.
static const std::function<int(const std::string&, const PipelineParsed&, bool)>
  runPipelineStubBody =
    [](const std::string& prog, const PipelineParsed& parsed, bool worker) -> int {
      return runPipelineStub(parsed.pipeline, parsed, worker);
    };

// The table itself (the requested "assign as static const ... we don't need a function at all"):
// one explicit entry per canonical pipeline -- `vlm` its own lambda (runVlmBody); the other thirteen
// share runPipelineStub above.  An entry stores only its name + its body (a PipelineBinding is a
// { name, run } pair): the name is the lookup key, run is the body dispatchPipeline forwards to with
// the parsed arguments + the program name + the mode.
static const std::unordered_map<std::string, PipelineBinding> kPipelineBindings = {
  { "llm",            { "llm",            runPipelineStubBody } },
  { "vlm",            { "vlm",            [](const std::string& prog, const PipelineParsed& parsed, bool worker) -> int { return ::ovgenai::runVlmBody(prog, parsed, worker); } } },
  { "whisper",        { "whisper",        runPipelineStubBody } },
  { "asr",            { "asr",            runPipelineStubBody } },
  { "tts",            { "tts",            runPipelineStubBody } },
  { "omni",           { "omni",           runPipelineStubBody } },
  { "embedding",      { "embedding",      runPipelineStubBody } },
  { "text-embedding", { "text-embedding", runPipelineStubBody } },
  { "rerank",         { "rerank",         runPipelineStubBody } },
  { "text2image",     { "text2image",     runPipelineStubBody } },
  { "image2image",    { "image2image",    runPipelineStubBody } },
  { "inpainting",     { "inpainting",     runPipelineStubBody } },
  { "text2video",     { "text2video",     runPipelineStubBody } },
  { "image2video",    { "image2video",    runPipelineStubBody } },
};

// run / worker's single delegated entry point: look the selected --pipeline up by name (a single find
// over kPipelineBindings) and run that binding's body -- so runModeEntry neither looks up nor routes a
// pipeline itself; it parses the one argument set exactly once, then calls this.  If the name is not
// in the table (unknown --pipeline) it is a usage error: named + pointed at show-pipelines (the
// canonical set via pipelineNames(), the same text that listing renders) + exit 2.  A hit runs the
// bound body and its exit code (0 ok / 1 a run failure / 2 a usage failure) is returned.
int dispatchPipeline(const std::string& modeName, const std::string& prog,
                     const PipelineParsed& parsed, bool worker) {
  // The omitted / "auto" case (the whole point of --pipeline being optional): the one front-end parse
  // leaves parsed.pipeline == "auto" when the operator did not pass --pipeline/-p (the option now
  // defaults to "auto").  We auto-detect the family from the model dir's config.json (a pure file/JSON
  // read -- no model load, no engine, no race) and then route: a detected *generation* model (llm or
  // vlm) runs the SINGLE real body here, runVlmBody, which the ContinuousBatchingPipeline inside drives --
  // it is the CB (not us) that decides vlm-vs-llm at runtime (it builds the vision/text embedder iff the
  // dir has one), so our peek at config.json only *labels* the run.  A known non-generation family, or a
  // config.json we cannot place, is a not-yet- IMPLEMENTED body: a graceful usage decline (exit 2) that
  // names what it detected and points at the one implemented pipeline -- never a false pass, and (because
  // the 12 non-gen families all drop to this very same decline) a wrong *label* among them only changes
  // which name is reported, never which body runs.  An *explicit* --pipeline (any valid name, or an
  // unknown one) is unaffected: it falls through to the table lookup unchanged.
  if (parsed.pipeline == "auto") {
    const std::string fam = detectPipelineFor(parsed.load.model);
    if (isImplementedPipeline(fam)) {           // llm|vlm both run the one real body (the CB) -- it decides vlm vs llm
      // Tell the operator what we detected (a pure config.json peek) and that the body is the generation
      // body (runVlmBody / the CB): a detected text LLM and a detected VLM share this path -- the body is
      // named "vlm" (it prints its own label just below), and the CB is what actually decides the embedder.
      std::fprintf(stderr,
          "ovgenai: %s: auto-detected pipeline '%s' from the model dir's config.json; running the "
          "generation body (the continuous-batching pipeline decides vision-language vs text-only)\n",
          modeName.c_str(), fam.c_str());
      return runVlmBody(prog, parsed, worker);   // the single real body; it prints its own "ovgenai: vlm …" label
    }
    std::fprintf(stderr, "ovgenai: %s: %s\n", modeName.c_str(), detectPipelineDecline(fam).c_str());
    return 2;                                   // a not-yet-implemented / unrecognised family -> a usage decline, not a run failure
  }
  const auto it = kPipelineBindings.find(parsed.pipeline);
  if (it == kPipelineBindings.end()) {
    std::fprintf(stderr, "ovgenai: %s: unknown pipeline '%s'; choose one of: %s\n",
                 modeName.c_str(), parsed.pipeline.c_str(),
                 config::joinList(pipelineNames(), ", ").c_str());
    return 2;
  }
  return it->second.run(prog, parsed, worker);
}

}  // namespace commands
}  // namespace ovgenai
