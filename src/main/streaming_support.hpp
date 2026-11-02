// Streaming support -- the single ENTRY POINT for the vlm subcommand's streaming / parsing (the registry
// moved in from the old tool_call_support.{hpp,cpp}, which is removed, plus the engine-side carrier):
//   * the CONTROL half -- the modelFamily registry / detect / select / bridge.  A model's OWN loaded chat
//     template is the source of truth for HOW it states a call, so the CLI derives BOTH "parse or not" and
//     "which parser" by rendering the template and matching a model-intrinsic sentinel (below); a selected
//     family's tool parser(s) are then carried in the ContentStreamer the engine runs.
//   * the CARRY half -- ContentStreamer (the TextParserStreamer whose base keeps the parser loop + content
//     accumulation; the carried parser routes the tool block out into tool_calls; the always-slot-0
//     DynamicReasoningIncrementalParser splits reasoning_content).  ContentStreamer::instance() is THE entry
//     point: it assembles the proper streamer for a modelFamily (the registry picks the tool parser(s)) and
//     returns it for the engine.
//
// Bracket-safety guardrail (shared with the per-model headers): the tag BYTES are built from the
// \u003C / \u003E escape constants in the per-model headers.  The ONLY place an angle byte would be named
// is the per-family reasoning tags -- and those live as LOCAL constants inside ContentStreamer::instance(),
// built from those escape constants (no literal angle-byte pair sits in this source).
#pragma once

#include <openvino/genai/parsers.hpp>          // IncrementalParser / ReasoningIncrementalParser / StreamingStatus
#include <openvino/genai/text_streamer.hpp>     // TextParserStreamer (ContentStreamer's base) / TextStreamer / StreamerBase
#include <openvino/genai/tokenizer.hpp>         // Tokenizer + apply_chat_template / get_original_chat_template
#include <openvino/genai/chat_history.hpp>       // ChatHistory (detectModelFamily renders the loaded template)
#include <openvino/genai/json_container.hpp>     // JsonContainer (the parse / write / probe payloads)

#include "toolcallparsers/qwen35_tool_call_parser.hpp"  // Qwen35ToolCallParser + the LT/GT angle bytes + the kSentfunc/kSentry sentinels

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Forward declaration (the full type arrives via llm_pipeline.hpp, included in the .cpp) -- the
// engine-side result whose `parsed` member the bridge fills.  Forward-declared to keep the header light.
namespace ov { namespace genai { class DecodedResults; } }

namespace ovgenai {
namespace toolcall {

// The engine-side copy the *VLM* pipeline never performs: only the *LLM* pipeline's run_generate_with_parsers
// copies a TextParserStreamer's folded get_parsed_message() into res.parsed[0].  The VLM pipeline that
// `vlm` drives leaves res.parsed EMPTY, so a tool-call turn would otherwise report the engine's own
// finish_reason "tool_calls" (set from the parser's TOOL_CALL_STOP) yet carry NO tool_calls array.  The
// CLI still holds the carrier, so we reproduce the copy: for a TextParserStreamer (our ContentStreamer) we
// copy its folded message -- content PLUS the one-shot tool_calls array the parser wrote into its private
// message -- into res.parsed[0]; a plain TextStreamer is not a TextParserStreamer, so the cast misses and
// res.parsed is left exactly as the engine produced it (graceful fallback: no tool_calls, finish_reason
// stays as the engine's own).
void bridgeParsedMessage(const std::shared_ptr<ov::genai::StreamerBase>& streamer,
                         ov::genai::DecodedResults& res);

// --- the registry + detection / selection (bodies in streaming_support.cpp) ------------
// One source of truth: every advertised (validated) name is an entry; an entry is "wired"
// iff its `make` is non-null.  `sentinels` are the substrings whose presence in the model's
// RENDERED template selects this parser (empty = not auto-detected yet).
struct ModelFamilySpec {
  std::string name;
  std::vector<std::string> sentinels;
  // This format's tool parser(s) only -- a reasoning splitter is NOT in here; a separate, gated
  // ReasoningIncrementalParser is composed beside it by instance() (gated on the rendered
  // chat template).  For qwen35 this returns just the model-specific Qwen35ToolCallParser (it adds
  // the tool_calls key).  An unwired (not-yet-implemented) format carries an empty (null) function.
  std::function<std::vector<std::shared_ptr<ov::genai::IncrementalParser>>()> make;
};

// Every advertised name -> an entry (wired iff `make` is non-null); the registry owns the
// kSentfunc / kSentry sentinels via the included per-model header.
const std::vector<ModelFamilySpec>& knownModelFamilies();
// The flat list of every advertised (validated) --model-family name (the one source the
// CLI-flag validation reads).
std::vector<std::string> advertisedModelFamilies();
// Build the format's tool parser(s) for a (validated) name, or an empty list if advertised-but-unwired.
// For qwen35 this returns just the model-specific Qwen35ToolCallParser.  A reasoning splitter is NOT
// returned here -- a separate, gated ReasoningIncrementalParser is prepended by instance() only
// when the rendered chat template shows thinking is on.
std::vector<std::shared_ptr<ov::genai::IncrementalParser>> makeToolCallParserByName(const std::string& name);
// Render the model's OWN loaded template with a probe tool arg and match that against the
// registry sentinels; nullopt if none match (graceful: no parser, no turn-stop).
std::optional<std::string> detectModelFamily(const ov::genai::Tokenizer& tok);
// The MODEL-FREE core of the auto-detect gate (no model, no render) so the gate is unit
// testable: the entry whose `sentinels` are ALL present in `rendered` wins; nullopt if none.
std::optional<std::string> matchFormat(const std::string& rendered);

// The emitted-line merge: fold a window's raw per-step `tool_calls` fragments (each an OpenAI streaming
// delta) into the meaningful per-call array a --stream N line carries -- one entry per call index, the
// call's sub-keys accumulated WITHIN it.  Generalised over the OpenAI Chat Completions `tool_calls` union:
// the sub-object is keyed on the top-level `type` ( "function" -> function:{ name, arguments }, where
// arguments is a JSON string ;  "custom" -> custom:{ name, input }, where input is the FREE-FORM model
// input ) via the file-local kToolCallTypes table -- the SINGLE extension point for the later
// Responses-API members (deferred, per the user).  Public (not file-local) so it is unit-testable
// model-free with a synthetic fragment sequence per `type`, not only through a live model that only emits
// "function".
ov::genai::JsonContainer mergeToolCallFragments(const ov::genai::JsonContainer& fragments);

// Selection precedence: an explicit --model-family name override wins; else auto-detect
// from the model's OWN loaded template; else no parser (graceful: no stop, finish_reason
// stays stop, no tool_calls -- exactly the no-parser default today).
std::optional<std::string> selectModelFamily(
    const std::optional<std::string>& override_name,
    const ov::genai::Tokenizer& tok,
    std::optional<std::string>* detected_out = nullptr);

// Forward declaration: the slot-0 reasoner the ContentStreamer keeps a handle to, so a runtime on/off
// (the user's one-parser-both-sides design) can be flipped via ContentStreamer without the caller ever
// seeing the reasoner.  Its full definition follows the ContentStreamer class (it befriends it so the
// on/off is a friend call, not a public knob).
class DynamicReasoningIncrementalParser;

// The engine-side content / parser-carrying streamer (a Qwen35 tool-call streamer analog): its
// TextParserStreamer base keeps the parser loop and copies its folded get_parsed_message() into the
// result; this streamer also appends the (specials-preserving) content the caller drains, per chunk.
// The controls the user gets are activateEnableThinking -- a friend delegate to the reasoner's now
// private setActive (the reasoning on/off) -- and the --stream N count (the token-aggregation size:
// 1 == today, one emitted line per token; >1 == N tokens per line; <=0 == no mid-stream output, a single
// total line at end(); for the WORKER -- instance()'s isWorker, threaded through the session below -- the
// per-chunk lines are instead framed <item>:<session>:<body> and a framed <done> is one turn's terminal,
// so `--stream N` controls a streamed worker turn exactly as it controls the one-shot `run`).
// A static instance() factory (declared below) builds + returns the proper streamer
// for a modelFamily: for "qwen35" the always-slot-0 DynamicReasoningIncrementalParser (the qwen35 reasoning
// open / close-of-thinking tags, with expect_open_tag = false and keep_original_content = false -- those
// tags are LOCAL to instance(), not namespace-wide constants) followed by the family's tool parser(s) (the
// model-specific Qwen35ToolCallParser); any other / empty family the registry does not wire gets a plain
// (empty-parser) content streamer.  The pure-virtual write(JsonContainer&) is the one that fires; the
// string-write base drives the parser loop and calls it.  (Method bodies in streaming_support.cpp.)
class ContentStreamer : public ov::genai::TextParserStreamer {
public:
  // (direct / test use) build the streamer with an explicit parser list, binding the out-accumulator the
  // base drains; no reasoner is retained (m_dyn stays null) and activateEnableThinking is then a no-op.
  ContentStreamer(const ov::genai::Tokenizer& tok,
                  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                  std::string& acc);
  // the instance() entry-point ctor: everything the 3-arg ctor does, PLUS retaining a handle to the
  // always-slot-0 reasoner (so activateEnableThinking can flip it), PLUS the --stream N aggregation
  // factor (default 1 == today, one emitted line per token; >1 == N tokens per line; <=0 == no streaming,
  // a single total line at end()).  A defaulted 5th arg so a caller that omits it -- including the
  // 3-arg delegating ctor -- keeps today's per-token behaviour.  The 6th arg `live` (default true) is the
  // one-shot's gate: for the one-shot (isWorker false, below) m_live = live, so only the per-chunk
  // PHYSICAL write is suppressed when live == false (the parser loop + the accumulation + the carried
  // tool parser ALWAYS run, so end()/consolidatedMessage() and the <done> payload are complete either
  // way).  The 7th / 8th args -- isWorker (default false) and the session id the worker routes by
  // (default "") -- are the worker's controls: m_live = isWorker ? (stream > 0) : live, so a worker turn
  // streams by default and `--stream <=0` turns its per-chunk <item> stream off (the whole turn then
  // rides the framed <done> the finalize emits).
  ContentStreamer(const ov::genai::Tokenizer& tok,
                  std::vector<std::shared_ptr<ov::genai::IncrementalParser>> parsers,
                  std::string& acc,
                  const std::shared_ptr<DynamicReasoningIncrementalParser>& dyn,
                  int stream = 1,
                  bool live = true,
                  bool isWorker = false,
                  const std::string& sessionId = std::string());

  // The SOLE public control of the always-slot-0 reasoning splitter -- a delegate to that reasoner's
  // (now private) setActive, reachable because ContentStreamer is its friend.  enableThinking = true turns
  // the reasoning_content split on (the engine's normal split); false (the default) routes a non-thinking
  // model's lead-in to content instead of a spurious reasoning_content.
  void activateEnableThinking(bool activate);

  // THE entry point for the vlm subcommand: build + return the proper streamer for a modelFamily.  For
  // "qwen35" the reasoner (the qwen35 reasoning tags + expect_open_tag = false + keep_original_content =
  // false -- a factory previously named makeDynamicReasoningParser, now folded in here) is ALWAYS at slot 0,
  // followed by the family's tool parser(s) (the model-specific Qwen35ToolCallParser via the registry's
  // makeToolCallParserByName); the reasoner on/off is then flipped via the delegate above per enableThinking.
  // An advertised-but-unwired (or empty) family is not built here -- it gets an empty-parser content
  // streamer (graceful: content streams, finish_reason the engine's own).  (Compose logic previously named
  // composeParserList, now folded in here.)  The trailing isWorker / sessionId (both defaulted) frame the
  // WORKER's per-chunk + terminal lines (<item> / <done>) and thread the session id the finalize's <done>
  // carries in its header slot; a one-shot call omits them (the 6-arg form) and stays exactly as today.
  static std::shared_ptr<ContentStreamer> instance(
      const ov::genai::Tokenizer& tok, const std::string& modelFamily,
      bool enableThinking, std::string& acc, int stream = 1, bool live = true,
      bool isWorker = false, const std::string& sessionId = std::string());

  // Re-export the base write(int64_t) / write(const std::vector<int64_t>&) that the JsonContainer& override
  // below hides (a derived member declaration hides inherited overloads of the same name); the pump feeds
  // decoded token ids via write(const std::vector<int64_t>&), so the base overloads must stay visible.
  using ov::genai::TextStreamer::write;
  ov::genai::StreamingStatus write(ov::genai::JsonContainer& message) override;

  // The terminal end-of-stream hook (overrides TextStreamer::end(), which the engine's generate loop
  // invokes once, idempotently, after the last token -- via ThreadedStreamerWrapper).  It first lets
  // the base flush its trailing decoded text (one more write(JsonContainer) -> accumulates it into the
  // aggregate buffer), THEN calls emitAggregate() to emit whatever is left: a final partial chunk for
  // --stream N>=1, or the WHOLE stream for --stream N<=0 (no mid-stream lines at all).
  void end() override;

  // The FULL, final-passthrough result consumed by the bridge -> res.parsed (the VLM pipeline that
  // `vlm` drives never fills res.parsed on its own -- only the LLM pipeline does).  It is the base's
  // get_parsed_message() (accumulated content + reasoning_content) PLUS the consolidated tool_calls
  // array assembled here from the carried parser's completed_calls() -- every call the turn produced
  // (a multi-call turn is several, stop_after_tool_call == False), each index-free OpenAI shape
  // { id, type, function:{ name, arguments } }.  Assembled at the END (not per-delta), it never has
  // to survive the engine's per-delta concatenate (which throws on a list key re-set across deltas)
  // and can never lose a call after the first.  Empty for a content-only turn / a non-tool family.
  ov::genai::JsonContainer consolidatedMessage() const;

  // The per-step fragment sink the carried parser forwards into (wired via its on_fragment in
  // instance(), reached through a weak_ptr so there is no ownership cycle).  Appended here; the next
  // write() drains it onto stdout as this delta's tool_calls (req 2).  Only the qwen35 family wires
  // it; an unwired/plain streamer's parser simply never calls it.
  void collectFragment(const ov::genai::JsonContainer& fragment);

private:
  std::string& m_acc;                                            // the output the caller drains (specials preserved)
  std::shared_ptr<DynamicReasoningIncrementalParser> m_dyn;       // the slot-0 reasoner (retained for activateEnableThinking; null if none)
  ov::genai::JsonContainer m_tool_calls;                         // this delta's per-step tool_calls fragments (drained in write)
  std::shared_ptr<Qwen35ToolCallParser> m_tool = nullptr;         // the carried tool parser (qwen35 only; null otherwise) -- the consolidated all-calls source

  // --- token aggregation (--stream N): the emitted line carries N tokens, not one ------------
  // m_stream is the requested N from the --stream flag (1 == default == today): the per-emit
  // aggregation size.  m_stream_count counts the output-bearing deltas accumulated since the last
  // emit; a line is emitted when it reaches m_stream.  The three buffers (content / reasoning / the
  // accumulated tool_calls fragments) span the whole current window: an N>=1 window is N deltas, an
  // N<=0 window is the ENTIRE stream (one total at end).
  int m_stream = 1;                                              // the --stream N (default 1 == per-token); <=0 disables mid-stream output
  int m_stream_count = 0;                                        // output-bearing deltas accumulated since the last emit (reset per window)
  std::string m_buf_content;                                     // this window's accumulated content (specials preserved)
  std::string m_buf_reasoning;                                   // this window's accumulated reasoning_content
  // this window's accumulated per-step tool_calls fragments (the gradual tool_calls the carried parser
  // forwards, folded in each delta); at emit they are MERGED per call-index into one entry per call
  // (name / arguments accumulated within it) -- not dumped as one array element per fragment (see
  // MergeToolCallFragments in the .cpp, invoked by emitAggregate)
  ov::genai::JsonContainer m_buf_tools;
  // Gate for the PHYSICAL stdout emit (the per-chunk + end() lines), set in the ctor body as
  //   m_live = isWorker ? (m_stream > 0) : m_live_in;
  // -- for the one-shot (isWorker false) it is the `live` arg (true == the legacy bare per-chunk stream),
  // and for the worker (isWorker true) it is `stream > 0`, so the per-chunk line is framed
  // <item>:<session>:<body> and `--stream <=0` is the worker's "off": no <item> lines, the whole turn
  // riding the framed <done> the finalize emits.  When false the parser loop + the content/reasoning/
  // tool_calls accumulation still run (so end()/consolidatedMessage()/the <done> payload are complete);
  // only the per-chunk emitted line is suppressed.
  bool m_live = true;
  bool        m_worker = false;   // the per-chunk lines are framed <item>:<session>:<body> when true
  std::string m_session;          // the header session slot the worker's <item> lines carry ("" on a one-shot)
  // Build one emitted line from the current window's buffers (omitting empty keys, in the original
  // content -> reasoning_content -> tool_calls order so N==1 is byte-identical to the per-token line),
  // then reset the buffers + the per-window counter.  Called both from write() (the N>=1 window
  // boundary) and from end() (the final chunk, or the whole total for N<=0).
  void emitAggregate();
};

// A runtime on/off on the reasoning splitter, NOT a compose-time gate: one DynamicReasoningIncrementalParser
// is ALWAYS at slot 0 of the parser list the ContentStreamer carries (the tool parser(s) after it) -- one
// parser for BOTH the reasoning_content split AND the tool_calls split.  The old render-based gates
// (reasoningEnabledFromRender / detectReasoningEnabled) are removed because a model may render a reasoning
// template yet not actually reason (the Qwen3.5-0.8B), so the rendered prompt is not a trustworthy on/off.
// Reasoning is instead an EXPLICIT user control -- the --enable-thinking flag / a per-request
// chat_template_kwargs enable_thinking -- applied by ContentStreamer::activateEnableThinking (a friend),
// which sets the (now private) setActive below.  Default OFF: a non-thinking model's lead-in routes to
// content (not a spurious reasoning_content); ON routes reasoning into reasoning_content (the engine's
// normal split).  The qwen35 reasoning tags (formerly the namespace-wide kThinkingOpen / kThinkingClose)
// now live as LOCALS inside ContentStreamer::instance() -- not here, not at namespace scope -- since they
// are specific to the one wired family.
class DynamicReasoningIncrementalParser : public ov::genai::ReasoningIncrementalParser {
 public:
  using ov::genai::ReasoningIncrementalParser::ReasoningIncrementalParser;
  bool isActive() const { return m_active; }
  std::string parse(ov::genai::JsonContainer& dm, std::string& delta_text,
                    const std::optional<std::vector<int64_t>>& delta_tokens) override {
    if (!m_active)
      return delta_text;
    return ov::genai::ReasoningIncrementalParser::parse(dm, delta_text, delta_tokens);
  }
 private:
  // PRIVATE: the on/off the reasoner exposes to NOBODY but ContentStreamer (its friend, via
  // ContentStreamer::activateEnableThinking) -- so flipping reasoning is the streamer's job, not a public
  // knob a caller could reach on a raw parser.
  void setActive(bool active) { m_active = active; }
  bool m_active = false;
  friend class ContentStreamer;
};

}  // namespace toolcall
}  // namespace ovgenai
