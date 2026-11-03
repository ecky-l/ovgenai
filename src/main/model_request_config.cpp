// ovgenai's per-request generation config: the "Model request parameters" boost
// section this class owns (requestOptionsDesc / usage), its parser, and its JSON
// round-trip (snake_case keys, so the wire form is identical on both sides).
// Request-only helpers live in this file's anonymous namespace; the argv / boost
// / number / JSON coercions all sections share come from config.hpp (ConfigError
// is one of them); the load + runtime + scheduler section descriptions are
// included so the parser splices them in alongside our own.  Every parallel
// structure (the options,  the parser,  the JSON reader / writer) lists the 12
// fields in the same *alphabetical* order as the header so it reads the same way
// top-to-bottom.
#include "model_request_config.hpp"
#include "config.hpp"
#include "model_load_config.hpp"       // the load section, spliced into our parse
#include "model_runtime_config.hpp"    // the runtime section, spliced into our parse
#include "model_scheduler_config.hpp"  // the scheduler section, spliced into our parse
#include "streaming_support.hpp"        // advertisedModelFamilies() -- one source of truth for accepted --model-family names

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>   // `po` alias is provided by config.hpp

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ovgenai {
namespace config {
namespace {

// The accepted --model-family names are NOT a local list: they come from the model-family
// registry (ovgenai::toolcall::advertisedModelFamilies() -- one source of truth for both this
// CLI-flag validation and the parser actually used at run time in vlm.cpp).  An advertised-but-
// unwired name is still accepted here and simply resolves to no parser at run time, degrading
// gracefully to plain text.

// Parse a double from the CLI; a malformed / trailing-garbage value is a ConfigError.
double parseDouble(const std::string& s, const char* flag) {
  try {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) throw std::invalid_argument("trailing characters");
    return v;
  } catch (...) {
    throw ConfigError(std::string("invalid number for ") + flag + ": '" + s + "'");
  }
}

// Parse a SIGNED base-10 integer for the CLI (unlike config's parseSize, it ALLOWS a leading '-':
// --stream may be 0 or negative to mean "no streaming, emit one total at the end" -- a negative is a
// valid, named value, not an error).  The result is clamped into int range so an absurdly large input
// degrades to "aggregate the whole stream" (a positive INT_MAX) rather than wrapping to a misleading N.
int parseSignedInt(const std::string& s, const char* flag) {
  try {
    size_t pos = 0;
    const long long v = std::stoll(s, &pos);
    if (pos != s.size()) throw std::invalid_argument("trailing characters");
    if (v > std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
    if (v < std::numeric_limits<int>::min()) return std::numeric_limits<int>::min();
    return static_cast<int>(v);
  } catch (...) {
    throw ConfigError(std::string("invalid integer for ") + flag + ": '" + s + "'");
  }
}

// --- JSON field readers (used by modelRequestConfigFromJson) ----------------
// Absent or null → leave the field at its struct default (non-optional) or at
// nullopt (optional). A present non-number / negative int / wrong type is an error.

// Parse a boolean keyword from the CLI; only true/false (case-insensitive) are accepted (so a typo
// is a ConfigError rather than a silent mis-toggle of the reasoning switch).
bool parseBool(const std::string& s, const char* flag) {
  std::string v = s;
  for (auto& c : v)
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  if (v == "true")  return true;
  if (v == "false") return false;
  throw ConfigError(std::string("invalid boolean for ") + flag + ": " + s + " (use true or false)");
}

// Read one optional bool from a JSON object (absent or null -> nullopt; wrong type -> ConfigError).
void readOptBool(rapidjson::Document& doc, const char* key,
                  std::optional<bool>& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;
  if (!it->value.IsBool())
    throw ConfigError(std::string(path) + ": expected a boolean");
  out = it->value.GetBool();
}
// Read one optional int from a JSON object (absent or null → nullopt; negative
// integer or non-integer type → ConfigError).
void readOptInt(rapidjson::Document& doc, const char* key,
                std::optional<std::size_t>& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;
  if (!it->value.IsInt() && !it->value.IsInt64())
    throw ConfigError(std::string(path) + ": expected an integer");
  if (it->value.GetInt64() < 0)
    throw ConfigError(std::string(path) + ": must be non-negative");
  out = static_cast<std::size_t>(it->value.GetUint64());
}

// Like readOptInt but ALLOWS negatives and yields a signed optional: the --stream wire key may be 0 or
// negative to mean "no streaming, one total at end", and it must round-trip across the worker's
// per-request gen_config (so the same "stream" key it reads is the one the writer emits).  Absent or
// null leaves the field at its struct default (nullopt); a value is clamped into int range.
void readOptIntSigned(rapidjson::Document& doc, const char* key,
                      std::optional<int>& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;
  if (!it->value.IsInt() && !it->value.IsInt64())
    throw ConfigError(std::string(path) + ": expected an integer");
  const long long v = it->value.GetInt64();
  if (v > std::numeric_limits<int>::max()) out = std::numeric_limits<int>::max();
  else if (v < std::numeric_limits<int>::min()) out = std::numeric_limits<int>::min();
  else out = static_cast<int>(v);
}

void readOptDouble(rapidjson::Document& doc, const char* key,
                   std::optional<double>& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;
  if (!it->value.IsNumber())
    throw ConfigError(std::string(path) + ": expected a number");
  out = it->value.GetDouble();
}

void readOptStr(rapidjson::Document& doc, const char* key,
                std::optional<std::string>& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;
  if (!it->value.IsString()) throw ConfigError(std::string(path) + ": expected a string");
  out = it->value.GetString();
}

void readDefaultDouble(rapidjson::Document& doc, const char* key,
                       double& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;  // keep struct default
  if (!it->value.IsNumber()) throw ConfigError(std::string(path) + ": expected a number");
  out = it->value.GetDouble();
}

void readDefaultSize(rapidjson::Document& doc, const char* key,
                     std::size_t& out, const char* path) {
  auto it = doc.FindMember(key);
  if (it == doc.MemberEnd() || it->value.IsNull()) return;  // keep struct default
  if (!it->value.IsInt() && !it->value.IsInt64())
    throw ConfigError(std::string(path) + ": expected an integer");
  if (it->value.GetInt64() < 0)
    throw ConfigError(std::string(path) + ": must be non-negative");
  out = static_cast<std::size_t>(it->value.GetUint64());
}

}  // close the anonymous namespace of request-only helpers

// --- The "Model request parameters" boost section this class owns ------------
//
// The generation-config flags (the ones the model-load section does not carry),
// in alphabetical-flag order, each with a single-letter short only where its
// first letter is not already claimed by a longer-name prefix.  Single source of
// truth for both usage() (its contribution to the shared --help) and
// parseModelRequestConfig (spliced alongside the other sections).
po::options_description ModelRequestConfig::requestOptionsDesc() {
  po::options_description s(
      "Model request parameters (per-request generation defaults; nullopt falls back to the model-load, then the engine default)",
      kHelpLineLength, kHelpMinDescriptionLength);
  s.add_options()(
      "assistant-confidence-threshold,a", po::value<std::string>(),
      "Confidence threshold for accepting a speculative (assistant/draft) "
      "token: "
      "a draft is accepted only when the main model's probability for it is at "
      "least this. "
      "Lower = accept more drafts (faster, riskier), higher = more "
      "conservative. "
      "Per-request default (the worker protocol may override per request).")(
      "chat-template-kwargs", po::value<std::string>(),
      "Chat-template kwargs as a JSON object passed to the tokenizer's "
      "apply_chat_template, "
      "e.g. {\"add_generation_prompt\": true}.")(
      "enable-thinking", po::value<std::string>(),
      "The vlm reasoning (thinking) switch: takes true or false (omit for the "
      "model's own default). "
      "true enables the reasoner and sets enable_thinking=true in the applied "
      "chat template; false "
      "disables the reasoner.  A per-request chat_template_kwargs "
      "enable_thinking (e.g. over the "
      "worker stdin) OVERRIDES this flag for that request.  Only meaningful "
      "for vlm.")("frequency-penalty", po::value<std::string>(),
                  "Penalty proportional to how often a token has already "
                  "appeared in the output; "
                  "0 = no penalty, higher = penalise frequently-repeated "
                  "tokens by their frequency.")(
      "max-tokens", po::value<std::string>(),
      "Maximum number of new tokens to generate before stopping (default "
      "16384).")("num-assistant-tokens,n", po::value<std::string>(),
                 "Number of draft ('assistant') tokens produced per "
                 "speculative-decoding step "
                 "before they are verified against the main model (used with "
                 "--assistant-confidence-threshold). Per-request default (the "
                 "worker protocol may override).")(
      "presence-penalty", po::value<std::string>(),
      "Penalty added to the logit of any token that has already appeared in "
      "the output, "
      "regardless of how often; 0 = no penalty, higher = penalise reappearing "
      "tokens (encourages variety).")(
      "repetition-penalty", po::value<std::string>(),
      "A logit multiplier for already-generated tokens; 1.0 = no penalty (the "
      "default), "
      ">1.0 penalises repeated tokens, <1.0 favours them.")(
      "seed", po::value<std::string>(),
      "A random seed for reproducible sampling (the same model / flags / seed "
      "reproduces the same token sequence).")(
      "stream", po::value<std::string>(),
      "Token aggregation for streaming output: how many tokens to fold into "
      "ONE emitted "
      "JSON line (default 1 = one line per token).  N>1 aggregates N tokens "
      "per line; "
      "N<=0 emits NO mid-stream output and instead writes a single total line "
      "(all of content / "
      "reasoning_content / tool_calls that the turn produced) at the end of "
      "the stream.")("temperature", po::value<std::string>(),
                     "Sampling temperature as a float > 0 -- higher = more "
                     "random/uniform, lower = more "
                     "deterministic; 0 or negative = greedy (deterministic) "
                     "decoding. Default 1.0.")
      // Help mirrors the registry (advertisedModelFamilies()); boost's desc arg
      // is a const char*, so the *validated* set comes from the registry and
      // this is a literal kept in sync.
      ("model-family", po::value<std::string>(),
       "Overwrite for the (usually auto-detected) model family, that is used "
       "for the output stream parsing. Possible values are qwen35 / hermes / gemma4 / museglimmer")
    ("top-k", po::value<std::string>(),
        "Top-k sampling: only the k highest-probability tokens are considered; 0 disables "
        "top-k (considers all). Default 50 -- so a value is required to turn it off.")
    ("top-p", po::value<std::string>(),
        "Top-p (nucleus) sampling threshold 0..1: the smallest set of highest-probability "
        "tokens whose cumulative probability reaches p is considered; 1.0 = disabled. Default 1.0.");
  return s;
}

// This class's slice of the --help listing (its "Model request parameters"
// section only); the combined four-section view is config.cpp's
// combinedConfigUsage.
std::string ModelRequestConfig::usage() {
  std::ostringstream os;
  os << ModelRequestConfig::requestOptionsDesc();
  return os.str();
}

// --- Public: parseModelRequestConfig (for ovgenai run) -------------------------
//
// Reads only the generation-config fields (alphabetical). Parsed over the full
// argv (this class's section plus the load + runtime + scheduler sections the
// other classes own) so boost doesn't reject their flags that this function
// simply doesn't read.
ModelRequestConfig parseModelRequestConfig(const std::vector<std::string>& args) {
  const PreProcessed pp = preProcess(args);
  if (pp.positionalCount > 1) {
    throw ConfigError("only one positional (the model path) may be given; found "
                       + std::to_string(pp.positionalCount)
                       + ". ovgenai loads exactly one model -- set it with --model");
  }
  std::vector<std::string> cleanArgs = pp.args;
  cleanArgs.erase(std::remove(cleanArgs.begin(), cleanArgs.end(), std::string("-")),
                  cleanArgs.end());

  // Parse over the full argv: all four sections (our request section plus the
  // load + runtime + scheduler sections the other classes own), so a caller's
  // load / runtime / scheduler flags are "known" (not "unknown option"); we read
  // only the request fields.
  po::options_description desc;
  desc.add(ModelRequestConfig::requestOptionsDesc());
  desc.add(ModelRuntimeConfig::runtimeOptionsDesc());
  desc.add(ModelLoadConfig::loadOptionsDesc());
  desc.add(ModelSchedulerConfig::schedulerOptionsDesc());
  desc.add_options()("help", po::bool_switch(), "Show this help message");

  po::positional_options_description posDesc;
  posDesc.add("model", 1);
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(cleanArgs)
                 .options(desc)
                 .positional(posDesc)
                 .style(po::command_line_style::unix_style)
                 .run(), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    throw rephraseBoost(e);
  }

  // Only read the generation-config fields (alphabetical); the load / runtime /
  // scheduler fields (present in the same vm but for the other parsers) are
  // simply not accessed.
  ModelRequestConfig cfg;  // non-optional fields at struct defaults
  if (vm.count("assistant-confidence-threshold"))
    cfg.assistant_confidence_threshold =
        parseDouble(vm["assistant-confidence-threshold"].as<std::string>(),
                     "--assistant-confidence-threshold");
  if (vm.count("chat-template-kwargs"))
    cfg.chat_template_kwargs = std::string(requireJsonObject(
        vm["chat-template-kwargs"].as<std::string>(),
        "--chat-template-kwargs"));
  if (vm.count("enable-thinking")) {
    cfg.enable_thinking = parseBool(vm["enable-thinking"].as<std::string>(), "--enable-thinking");
  }
  if (vm.count("frequency-penalty"))
    cfg.frequency_penalty = parseDouble(vm["frequency-penalty"].as<std::string>(),
                                        "--frequency-penalty");
  if (vm.count("max-tokens"))
    cfg.max_tokens = static_cast<std::size_t>(
        parseSize(vm["max-tokens"].as<std::string>(), "--max-tokens"));
  if (vm.count("num-assistant-tokens"))
    cfg.num_assistant_tokens = static_cast<std::size_t>(
        parseSize(vm["num-assistant-tokens"].as<std::string>(),
                   "--num-assistant-tokens"));
  if (vm.count("presence-penalty"))
    cfg.presence_penalty = parseDouble(vm["presence-penalty"].as<std::string>(),
                                       "--presence-penalty");
  if (vm.count("repetition-penalty"))
    cfg.repetition_penalty = parseDouble(vm["repetition-penalty"].as<std::string>(),
                                        "--repetition-penalty");
  if (vm.count("seed"))
    cfg.seed = static_cast<std::size_t>(
        parseSize(vm["seed"].as<std::string>(), "--seed"));
  if (vm.count("stream"))
    cfg.stream = parseSignedInt(vm["stream"].as<std::string>(), "--stream");
  if (vm.count("temperature"))
    cfg.temperature = parseDouble(vm["temperature"].as<std::string>(), "--temperature");
  if (vm.count("model-family")) {
    const std::string v = vm["model-family"].as<std::string>();
    if (!v.empty()) cfg.model_family = v;
  }
  if (vm.count("top-k"))
    cfg.top_k = static_cast<std::size_t>(
        parseSize(vm["top-k"].as<std::string>(), "--top-k"));
  if (vm.count("top-p"))
    cfg.top_p = parseDouble(vm["top-p"].as<std::string>(), "--top-p");

  const std::vector<std::string> allowedParsers =
      ::ovgenai::toolcall::advertisedModelFamilies();
  if (cfg.model_family.has_value() && !cfg.model_family->empty() &&
      !inSet(*cfg.model_family, allowedParsers))
    throw ConfigError("--model-family '" + *cfg.model_family +
                      "' must be one of: " + joinList(allowedParsers, ", "));
  return cfg;
}

// --- Public: modelRequestConfigFromJson -------------------------------------
//
// Read each recognised key (alphabetical) as the matching type, folding the JSON OVER `base`: a
// present (non-null) key wins; an absent (or null) one keeps its value in `base` -- so the worker can
// overlay a per-request body on its CLI launch config (any knob the body omits inherits the CLI value).
// With the empty default `base` an omitted key keeps the struct default (so a baseless call is exactly
// what it was); a present value of the wrong type is a ConfigError.
ModelRequestConfig modelRequestConfigFromJson(const std::string& json,
                                              const ModelRequestConfig& base) {
  rapidjson::Document doc;
  if (doc.Parse<0>(json.c_str()).HasParseError())
    throw ConfigError("runtime config is not valid JSON at position " +
                        std::to_string(doc.GetErrorOffset()));
  if (!doc.IsObject()) throw ConfigError("runtime config must be a JSON object");

  // Seed from `base`: each read* below only overrides a PRESENT key, so a key the JSON omits keeps
  // `base`'s value (the fold the worker uses to overlay a per-request body on its CLI launch config);
  // with the empty default `base` an omitted key keeps the struct default, exactly as before.
  ModelRequestConfig cfg = base;
  readOptDouble(doc, "assistant_confidence_threshold", cfg.assistant_confidence_threshold, "runtime_config.assistant_confidence_threshold");
  // chat_template_kwargs: a JSON object; compact it if present.
  {
    auto it = doc.FindMember("chat_template_kwargs");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
      if (!it->value.IsObject())
        throw ConfigError("runtime_config.chat_template_kwargs: expected an object");
      cfg.chat_template_kwargs = std::string(compactJsonOf(it->value));
    }
  }
  readOptBool(doc, "enable_thinking", cfg.enable_thinking, "runtime_config.enable_thinking");
  readOptDouble(doc, "frequency_penalty", cfg.frequency_penalty, "runtime_config.frequency_penalty");
  readDefaultSize(doc,   "max_tokens",    cfg.max_tokens,        "runtime_config.max_tokens");
  readOptInt(doc,    "num_assistant_tokens",  cfg.num_assistant_tokens,   "runtime_config.num_assistant_tokens");
  readOptDouble(doc, "presence_penalty",  cfg.presence_penalty,   "runtime_config.presence_penalty");
  readDefaultDouble(doc, "repetition_penalty", cfg.repetition_penalty, "runtime_config.repetition_penalty");
  readOptInt(doc,    "seed",              cfg.seed,               "runtime_config.seed");
  readOptIntSigned(doc, "stream",          cfg.stream,             "runtime_config.stream");
  readDefaultDouble(doc, "temperature",   cfg.temperature,        "runtime_config.temperature");
  readOptStr(doc,    "model_family",  cfg.model_family,   "runtime_config.model_family");
  readDefaultSize(doc,   "top_k",         cfg.top_k,              "runtime_config.top_k");
  readDefaultDouble(doc, "top_p",         cfg.top_p,              "runtime_config.top_p");
  return cfg;
}

std::string modelRequestConfigToJson(const ModelRequestConfig& c) {
  rapidjson::Document out(rapidjson::kObjectType);
  auto& a = out.GetAllocator();
  auto addInt    = [&](const char* key, std::size_t v) {
    rapidjson::Value k(key, a);
    rapidjson::Value val; val.SetUint64(v);
    out.AddMember(k, val, a);
  };
  // A SIGNED int value (for --stream, which may be <= 0 to mean "no streaming, one total at end" -- a
  // negative/zero a plain addInt/ SetUint64 would mangle).  SetInt64 stores it correctly.
  auto addSignedInt = [&](const char* key, int v) {
    rapidjson::Value k(key, a);
    rapidjson::Value val; val.SetInt64(static_cast<long long>(v));
    out.AddMember(k, val, a);
  };
  auto addDouble = [&](const char* key, double v) {
    rapidjson::Value k(key, a);
    rapidjson::Value val; val.SetDouble(v);
    out.AddMember(k, val, a);
  };
  auto addStr    = [&](const char* key, const std::string& v) {
    rapidjson::Value k(key, a);
    rapidjson::Value val(v.c_str(), v.size(), a);
    out.AddMember(k, val, a);
  };
  auto addBool   = [&](const char* key, bool v) {
    rapidjson::Value k(key, a);
    rapidjson::Value val; val.SetBool(v);
    out.AddMember(k, val, a);
  };
  // Emit fields in alphabetical order.  The non-optional fields (max_tokens /
  // temperature / top_k / top_p / repetition_penalty) carry a real default, so
  // they are always emitted; an optional key is omitted entirely (no "null"
  // placeholder) unless it is set.
  if (c.assistant_confidence_threshold.has_value())
    addDouble("assistant_confidence_threshold", *c.assistant_confidence_threshold);
  if (c.chat_template_kwargs && !c.chat_template_kwargs->empty()) {
    rapidjson::Document tmp;
    rapidjson::Value node;
    if (!tmp.Parse<0>(c.chat_template_kwargs->c_str()).HasParseError())
      node.CopyFrom(tmp, a);
    else node.SetObject();
    out.AddMember(rapidjson::Value("chat_template_kwargs", a), std::move(node), a);
  }
  if (c.enable_thinking.has_value()) addBool("enable_thinking", *c.enable_thinking);
  if (c.frequency_penalty.has_value()) addDouble("frequency_penalty", *c.frequency_penalty);
  addInt("max_tokens",         c.max_tokens);
  if (c.num_assistant_tokens.has_value()) addInt("num_assistant_tokens", *c.num_assistant_tokens);
  if (c.presence_penalty.has_value()) addDouble("presence_penalty", *c.presence_penalty);
  addDouble("repetition_penalty", c.repetition_penalty);
  if (c.seed.has_value()) addInt("seed", *c.seed);
  if (c.stream.has_value()) addSignedInt("stream", *c.stream);
  addDouble("temperature",     c.temperature);
  if (c.model_family && !c.model_family->empty())
    addStr("model_family", *c.model_family);
  addInt("top_k",             c.top_k);
  addDouble("top_p",           c.top_p);

  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  out.Accept(w);
  return std::string(sb.GetString());
}

}  // namespace config
}  // namespace ovgenai
