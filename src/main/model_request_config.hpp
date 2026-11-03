// --- Per-request generation config ------------------------------------------
//
// For every pipeline subcommand:  populated from CLI flags (parseModelRequestConfig).
// For a `--worker`-mode pipeline (future):  populated from a protocol command per request.
// Maps the per-request generation knobs; the JSON round-trip uses snake_case
// keys (see modelRequestConfigFromJson below), so the wire form is stable.
//
// The 13 fields are kept in *alphabetical order* (so the struct,  the --help
// listing,  and the JSON round-trip all read the same way top-to-bottom).  Five
// carry a non-nil *struct default* (a non-optional field,  emitted even when the
// request omits it);  the other eight are std::optional,  where nullopt means
// "not in this request" and then falls back to the model-load default or,
// failing that,  the engine default.
#ifndef OVGENAI_MODEL_REQUEST_CONFIG_HPP
#define OVGENAI_MODEL_REQUEST_CONFIG_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <boost/program_options.hpp>   // for po::options_description in requestOptionsDesc
#include "config.hpp"  // ConfigError + the parse helpers both sections share

namespace ovgenai {
namespace config {

struct ModelRequestConfig {
  // The 13 per-request knobs, in *alphabetical field order*.  The five that carry
  // a struct default are plain (non-optional) fields and are emitted even when
  // the omits them;  the other eight are std::optional (nullopt == "not in this
  // request").  The comment to the right of each is the knob it is.
  std::optional<double>      assistant_confidence_threshold;   // a drafted token is accepted only when the main model's probability for it is at least this; lower = accept more drafts (faster, riskier). Per-request default; the protocol may override
  std::optional<std::string> chat_template_kwargs;            // chat-template kwargs as a JSON object passed to apply_chat_template, e.g. {"add_generation_prompt": true}; an "enable_thinking" bool there ALSO overrides the vlm reasoning-splitter gate (wire/omit a ReasoningIncrementalParser)
  std::optional<bool>          enable_thinking;              // the vlm reasoning switch (tri-state: absent = the model's own default; true/false enable/disable). Drives BOTH the parser active state (setActive on the always-present DynamicReasoningIncrementalParser) and the render; a per-request chat_template_kwargs enable_thinking wins over it
  std::optional<double>      frequency_penalty;              // penalty by how often a token has already appeared (0 = off; higher penalises frequent repeats)
  std::size_t                max_tokens               = 16384;  // max number of new tokens to generate before stopping (non-optional; struct default when omitted)
  std::optional<std::size_t> num_assistant_tokens;            // draft ('assistant') tokens produced per speculative-decoding step before verification against the main model; used with --assistant-confidence-threshold. Per-request default; the protocol may override
  std::optional<double>      presence_penalty;               // penalty for any token that has already appeared (0 = off; higher penalises reappearing tokens, encouraging variety)
  double                     repetition_penalty     = 1.0;   // logit multiplier over already-generated tokens (1.0 = no penalty, >1.0 penalises repeats, <1.0 favours them; non-optional)
  std::optional<std::size_t> seed;                           // a random seed for reproducible sampling (same model/flags/seed reproduces the same sequence)
  std::optional<int>         stream;                        // --stream N: how many tokens to aggregate into ONE emitted output line (nullopt -> 1).  1 = one line per token (today's behaviour); N>1 = N tokens per line; N<=0 = NO mid-stream output -- everything is aggregated and a single total line (all of reasoning_content / content / tool_calls) is emitted at the end of the stream.
  double                     temperature              = 1.0;  // sampling temperature as a float > 0 (higher = more random, lower = more deterministic; 0 or negative = greedy; non-optional)
  std::optional<std::string> model_family;          // model family (the reasoning / tool-call output format this model declares): qwen35 / hermes / gemma4 / museglimmer -- an explicit --model-family value OVERRIDES the family auto-detected from the rendered chat template (absent / empty = use the detected one, or none); an advertised-but-unwired name degrades to plain text.  No short: kept long-only (the leading _m_ shares the prefix with --max-tokens).
  std::size_t                top_k                    = 50;   // top-k sampling: only the k highest-probability tokens are considered (0 disables; non-optional -- a value is needed to turn it off)
  double                     top_p                    = 1.0;  // top-p (nucleus) sampling threshold 0..1: the smallest set of highest-probability tokens whose cumulative probability reaches p (1.0 = disabled; non-optional)

  bool operator==(const ModelRequestConfig&) const = default;

  // --- The "Model request parameters" boost section this class owns ---------
  //
  // The generation-config flags (the ones the model-load section does not
  // carry), in alphabetical-flag order, each with a single-letter short where
  // its first letter is free (see the note beside the section below and
  // requestOptionsDesc()).  Feeds both usage() and parseModelRequestConfig
  // (spliced alongside the load section the other class owns).
  static po::options_description requestOptionsDesc();

  // This class's slice of the --help listing (its "Model request parameters"
  // section only); the combined view is config.cpp's combinedConfigUsage.
  static std::string usage();
};

// Parses the per-request generation config (for every subcommand) via boost over the
// full argv (its section plus the load + runtime + scheduler sections the other
// classes own, so the flags a caller also passes are "known", not "unknown
// option"); only the generation-config fields are read. Unknown/missing flags
// leave the corresponding field at its struct default (matching pydantic's
// defaults for non-optional fields; nullopt for optional ones). Throws
// ConfigError on an unknown / value-invalid flag.
ModelRequestConfig parseModelRequestConfig(const std::vector<std::string>& args);

// JSON round-trip for the per-request generation config.
// Uses snake_case keys, so the wire form is identical on both sides.
// An optional `base` (defaulting to empty) is the config a key the JSON OMITS falls back to -- so the
// worker can fold a per-request body over its CLI launch config (m_baseReq): a knob the body carries
// wins, any knob it omits keeps `base` (a CLI --stream / --max-tokens / ... then reaches that request).
// An omitted non-optional key keeps `base`'s value (the empty default = the struct default, so a baseless
// call is byte-identical to before).
ModelRequestConfig modelRequestConfigFromJson(const std::string& json,
                                              const ModelRequestConfig& base = ModelRequestConfig{});
std::string modelRequestConfigToJson(const ModelRequestConfig& config);

}  // namespace config
}  // namespace ovgenai

#endif  // OVGENAI_MODEL_REQUEST_CONFIG_HPP
