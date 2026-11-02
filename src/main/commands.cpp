// ovgenai's subcommand front-end -- the ONE place that owns the command surface.
// See commands.hpp for the design: four subcommands -- run / worker / show-config /
// show-pipelines (it was once one per OpenVINO.GenAI pipeline plus a --worker flag).  The
// pipeline is now a
// *general* required parameter, --pipeline/-p, shared by run and worker -- so the
// same one subcommand drives every OpenVINO.GenAI C++ pipeline (the old per-pipeline
// subcommands are gone): run picks the one-shot form, worker the long-lived one, and
// the pipeline name (the --pipeline value) selects which body executes.  `--input`
// is the special one-shot direct chat of the `run` subcommand; `worker` has no
// --input (its stdin IS the protocol).
//
// Each subcommand's own --help is boost-rendered (the general section -- its own
// root options, --pipeline / [--input] / --help -- boost draws first, the four
// named config sections after).  The *top-level* `ovgenai --help` index is
// hand-laid like it was: Commands (run / worker) / a "Known pipelines" line that now
// just points at `show-pipelines` (the full table moved there) / Helper subcommands
// (show-config + show-pipelines).
#include "commands.hpp"

#include "config.hpp"                    // ConfigError / preProcess / rephraseBoost / joinList
                                        // / kHelp* / showConfigJson / the shared parse helpers
#include "model_load_config.hpp"         // parseModelLoadConfig + loadOptionsDesc
#include "model_runtime_config.hpp"      // parseModelRuntimeConfig + runtimeOptionsDesc
#include "model_scheduler_config.hpp"    // parseModelSchedulerConfig + schedulerOptionsDesc
#include "model_request_config.hpp"      // parseModelRequestConfig + requestOptionsDesc
#include "pipeline_support.hpp"      // PipelineParsed + the registry (PipelineBinding + dispatchPipeline): runModeEntry parses once, then defers look-up + execute to it


#include "ovgenai_config.h"              // OVGENAI_VERSION

#include <algorithm>
#include <cstdio>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>    // the `po` alias is provided by config.hpp

namespace ovgenai {
namespace commands {

// --- The canonical pipeline set: the single source of truth every surface reads --
//   * the --pipeline / -p help (pipelineParamDesc) -- now a one-line pointer to the
//     show-pipelines helper (the full table now lives there, listed in one place);
//   * `pipelineNames` (a stable snapshot the tests assert);
//   * the SubcommandTable's top-level "Known pipelines" pointer to show-pipelines;
//   * the run/worker router (a body is bound to each name in the constructor).
// These are non-anonymous (declared in commands.hpp) so vlm.cpp reuses
// pipelineParamDesc for its own in-body --pipeline flag (a pointer to show-pipelines,
// not the full list), keeping the body's parse
// consistent with the help.  Entry order is the display order.
const std::vector<PipelineSpec>& canonicalPipelines() {
  static const std::vector<PipelineSpec> p = {
      {"llm",            "LLMPipeline: causal text generation (text in -> text out)"},
      {"vlm",            "VLMPipeline: vision-language model (images + text in -> text out)"},
      {"whisper",        "WhisperPipeline: speech recognition (audio in -> transcribed text out)"},
      {"asr",            "ASRPipeline: automatic speech recognition (audio in -> text out)"},
      {"tts",            "Text2SpeechPipeline: text-to-speech (text in -> audio out)"},
      {"omni",           "OmniPipeline: audio-visual multi-modal model (audio + text)"},
      {"embedding",      "EmbeddingPipeline: inputs -> dense embedding vectors"},
      {"text-embedding", "TextEmbeddingPipeline: text -> embedding vectors"},
      {"rerank",         "TextRerankPipeline: rerank candidate texts against a query"},
      {"text2image",     "Text2ImagePipeline: text -> image generation (diffusion)"},
      {"image2image",    "Image2ImagePipeline: image + text -> image (img2img, diffusion)"},
      {"inpainting",     "InpaintingPipeline: image + mask + text -> edited image (diffusion)"},
      {"text2video",     "Text2VideoPipeline: text -> video generation (diffusion)"},
      {"image2video",    "Image2VideoPipeline: image + optional text -> video (diffusion)"},
  };
  return p;
}

const std::vector<std::string>& pipelineNames() {
  static const std::vector<std::string> n = [] {
    std::vector<std::string> out;
    for (const auto& e : canonicalPipelines()) out.push_back(e.name);
    return out;
  }();
  return n;
}

// The --pipeline / -p option's description: a single concise line pointing the operator
// at the `show-pipelines` helper, which lists every pipeline with a one-line explanation.
// It no longer enumerates the full table -- that table now lives in `show-pipelines` (so
// the pipelines show in ONE place, not twice: in this --help and in the top-level index).
// As a one-liner it needs no "\n" bookkeeping; the caller (generalOptionsDesc below and
// vlm.cpp) still holds it in a named C-string local passed via .c_str(), because this
// build's options_description_easy_init has only a (const char* name, value,
// const char* description) overload -- no (std::string, value, std::string) form.
std::string pipelineParamDesc() {
  // Optional now (default "auto"): omit it and the registry auto-detects the model's family from its
  // config.json and routes a *generation* model to the (single, real) body.  This one-liner must name
  // the selector + still point at show-pipelines (the full per-pipeline table lives there, never inlined
  // in the --help index) -- so it must contain NO pipeline class name (e.g. not "LLMPipeline").
  return "The OpenVINO.GenAI pipeline to run (optional; default auto). "
         "Omit to auto-detect the model's pipeline family from its config.json and run that body "
         "(`auto` if unsure). Run 'ovgenai show-pipelines' to list and describe the available pipelines.";
}

namespace {  // file-local front-end helpers

// True iff the subcommand's own args ask for help before we touch a value.
bool anyHelp(const std::vector<std::string>& a) {
  for (const auto& x : a)
    if (x == "--help" || x == "-h") return true;
  return false;
}

// Word-wrap `desc` into lines no wider than `lineLen - colStart`, then emit: the
// first wrapped line rides the same physical line as the already-written (padded)
// name; the rest are indented to `colStart`.  Used only by the top-level index (a
// subcommand is a bare word, not flag); boost still renders each subcommand's own
// --help.  The token longer than the column is left intact rather than split.
void renderEntry(std::ostream& os, const std::string& name, const std::string& desc,
                 size_t colStart, size_t lineLen) {
  const size_t blurbWidth = (lineLen > colStart) ? lineLen - colStart : 0;
  std::vector<std::string> lines;
  {
    std::string line;
    std::istringstream ss(desc);
    std::string tok;
    while (ss >> tok) {
      if (!line.empty() && line.size() + 1 + tok.size() > blurbWidth) {
        lines.push_back(line);
        line.clear();
      }
      line += line.empty() ? tok : std::string(" ") + tok;
    }
    if (!line.empty()) lines.push_back(line);
  }
  std::string paddedName(2, ' ');
  paddedName += name;
  if (paddedName.size() < colStart)
    paddedName.append(colStart - paddedName.size(), ' ');
  for (size_t i = 0; i < lines.size(); ++i) {
    os << (i == 0 ? paddedName : std::string(colStart, ' '));
    os << lines[i] << "\n";
  }
  if (lines.empty()) os << paddedName << "\n";  // a blurb-less entry is still one line
}

// One capioned category of the top-level index: a (name | description) pair list,
// rendered with boost's column geometry (the shared kHelp* constants) so an index
// line visually lines up with the boost-rendered section the same subcommand later
// shows in its own --help.  One capter works for {name, description} pairs, which
// is both a subcommand entry (name + blurb) and a pipeline spec (name + blurb).
void renderCategory(std::ostream& os, const std::string& caption,
                    const std::vector<std::pair<std::string, std::string>>& kv) {
  if (kv.empty()) return;                        // an empty category draws nothing
  os << caption << ":\n";
  size_t nameCol = 0;
  for (const auto& kv2 : kv)
    nameCol = std::max(nameCol, kv2.first.size());
  const size_t colStart = 2 + nameCol + 2;       // 2-space indent, then the name column + a gap
  for (const auto& kv2 : kv)
    renderEntry(os, kv2.first, kv2.second, colStart, config::kHelpLineLength);
  os << "\n";
}

// The combined option set a `run` / `worker` parse (and, via subcommandHelp, its
// --help) shares: the four named config sections the class that owns them render,
// plus the GENERAL (untitled-root) section boost draws *first* -- the new
// `--pipeline`/`-p` (always; required) and, for the one-shot `run` form only,
// `--input`/`-i`.  A `worker` form passes isRunForm=false so its --help offers no
// --input (a worker's stdin IS its protocol) -- which is exactly why --input is a
// `run`-only special and a `worker --input ...` is rejected as "unknown option".
//
// The description is returned (not a string) so the SAME object renders the
// --help and parses, which cannot disagree; it is built identically to
// config.cpp's combinedConfigUsage() (untitled root + the four sections + --help)
// plus the general parameters, so a no-pipeline request still parses clean.
po::options_description generalOptionsDesc(bool isRunForm) {
  po::options_description s("", config::kHelpLineLength, config::kHelpMinDescriptionLength);
  s.add(config::ModelLoadConfig::loadOptionsDesc());
  s.add(config::ModelRuntimeConfig::runtimeOptionsDesc());
  s.add(config::ModelSchedulerConfig::schedulerOptionsDesc());
  s.add(config::ModelRequestConfig::requestOptionsDesc());
  // The general parameter: which pipeline this run / worker drives.  OPTIONAL now: omitting --pipeline
  // means "auto" (the operator does not have to name a pipeline for a generation model) -- but this boost
  // build's value<> takes no default argument (only a store pointer), so the "auto" default is applied
  // one layer up, in parsePipelineArgs, where the omitted-branch sets out.pipeline = "auto"; that is why
  // the front-end no longer reports a "missing --pipeline" (no required set is left to enforce) and the
  // registry's dispatchPipeline then auto-detects from the model dir's config.json and routes.  Value
  // taken by -p; help is a one-line pointer to `show-pipelines` (the full table lives there, not inlined)
  // -- a single line, no per-pipeline list.  Name / description are C-strings: this boost build's
  // easy_init only overloads operator() for (const char* name, value, const char* description) (no
  // std::string variant), so the std::string from pipelineParamDesc is held in a named local and passed
  // by .c_str() -- boost copies it inside add_option, so the pointer never dangles.
  const std::string pipelineDesc = pipelineParamDesc();
  s.add_options()
    ("pipeline,p", po::value<std::string>(), pipelineDesc.c_str());
  if (isRunForm)
    s.add_options()
    // The special one-shot surface: only `run` offers it (a worker reads its
    // protocol from stdin).  When given it is used verbatim (no stdin read); when
    // omitted the chat is read from stdin (a pipe => one line; a terminal => a
    // lone '.').  Meaningful for chat-shaped pipelines (e.g. vlm); others ignore it.
    ("input,i", po::value<std::string>(),
        "The 'run' one-shot's direct input: the chat to run, as a gen_config JSON object "
        "({\"messages\":[...], optionally \"tools\" / \"chat_template_kwargs\"}). When given, "
        "no stdin is read. When omitted, the chat is read from stdin: a pipe yields one "
        "(possibly large) line, an interactive terminal is read a line at a time until a "
        "lone '.' (the mail command's way). Meaningful for pipelines that take a direct chat "
        "(e.g. vlm); a worker has no --input -- it reads its line protocol from stdin instead.");
  s.add_options()("help", po::bool_switch(), "Show this help message");
  return s;
}

// `<prog> <run|worker> --help`: a Usage line + a note, then the shared options
// (the general section + the four config sections) rendered by boost -- one source
// for both the print and the parse.  isRunForm gives `run` (offers --input); the
// worker's prose explains why it has none.
std::string subcommandHelp(const std::string& prog, const std::string& modeName,
                           bool isRunForm) {
  std::ostringstream os;
  if (isRunForm) {
    os << "Usage: " << prog << " run --pipeline <p> [model-dir] <options>\n";
    os << "  Choose the pipeline with --pipeline / -p (required); then run it ONCE: the chat\n";
    os << "  comes from --input when given, otherwise from stdin (a pipe -> one (possibly large)\n";
    os << "  line; a terminal -> line-by-line, a lone '.' ending it, as in mail).  The result is\n";
    os << "  one JSON object on stdout.\n\n";
  } else {
    os << "Usage: " << prog << " worker --pipeline <p> [model-dir] <options>\n";
    os << "  Choose the pipeline with --pipeline / -p (required); the model + pipeline are built\n";
    os << "  from the command line (like 'run'), then STAY UP and read/write the line protocol on\n";
    os << "  stdin/stdout: each line is <command>:<sessionId>:<body>.  stdin GENERATE / PING /\n";
    os << "  CANCEL / UNLOAD -> stdout RESULT / PONG / CANCEL / DONE / ERROR (one <type>:<sessionId>:\n";
    os << "  <body> line each).  The session id is the 2nd slot (NOT inside the JSON <body>), so a\n";
    os << "  client reads it then drops <body> straight into its own response envelope.  There is NO\n";
    os << "  LOAD command (the model is already up).  A worker has NO --input: its stdin IS that\n";
    os << "  protocol (a direct --input is the 'run' one-shot's surface, which is why the two are now\n";
    os << "  two subcommands).\n\n";
  }
  os << generalOptionsDesc(isRunForm);          // boost renders the section
  return os.str();
}

// `show-config`'s --help: the four shared config sections + --help, and NOTHING else
// -- it prints the resolved config and loads no model, so it neither picks a pipeline
// (--pipeline) nor takes a one-shot chat (--input) nor is a worker.
std::string showConfigHelp(const std::string& prog) {
  std::ostringstream os;
  os << "Usage: " << prog << " show-config <options>\n";
  os << "  Parse the (unchanged) load config from the command line and print it as JSON, then\n";
  os << "  exit -- no model is loaded, so no --model / --device is required just to see what\n";
  os << "  would resolve.  A helper: it neither selects a pipeline (--pipeline) nor takes the\n";
  os << "  'run' one-shot input (--input), and is not a (--worker) worker.\n\n";
  po::options_description s("", config::kHelpLineLength, config::kHelpMinDescriptionLength);
  s.add(config::ModelLoadConfig::loadOptionsDesc());
  s.add(config::ModelRuntimeConfig::runtimeOptionsDesc());
  s.add(config::ModelSchedulerConfig::schedulerOptionsDesc());
  s.add(config::ModelRequestConfig::requestOptionsDesc());
  s.add_options()("help", po::bool_switch(), "Show this help message");
  os << s;
  return os.str();
}

// The `show-config` helper, kept from the pre-redesign front-end (its
// resolve-and-print behaviour): parse the (unchanged, all-boost) four configs and
// print their JSON, loading nothing.  Only its --help renderer changed (now
// showConfigHelp -- no --pipeline / --input / --worker, since it is a helper).
int runShowConfig(const std::string& prog, const std::vector<std::string> args) {
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the token
  if (anyHelp(sa)) {
    std::printf("%s", showConfigHelp(prog).c_str());
    return 0;
  }
  try {
    config::ModelLoadConfig      loadCfg  = config::parseModelLoadConfig(sa);
    config::ModelRuntimeConfig   rtCfg    = config::parseModelRuntimeConfig(sa);
    config::ModelSchedulerConfig schedCfg = config::parseModelSchedulerConfig(sa);
    config::ModelRequestConfig   reqCfg   = config::parseModelRequestConfig(sa);
    std::printf("%s\n",
                config::showConfigJson(loadCfg, rtCfg, schedCfg, reqCfg).c_str());
    return 0;
  } catch (const config::ConfigError& e) {
    std::fprintf(stderr, "ovgenai: show-config: %s\n", e.what());
    return 2;
  }
}

// `show-pipelines`'s --help: it lists the canonical pipeline set and loads nothing, so it
// offers neither --pipeline nor --input -- it mirrors showConfigHelp's shape (a Usage +
// prose + only a --help switch).  It is the single place the full pipeline table (every
// name + its one-line explanation) is now printed; that is why the top-level --help and
// the run / worker --pipeline help just point here instead of re-spelling it.
std::string showPipelinesHelp(const std::string& prog) {
  std::ostringstream os;
  os << "Usage: " << prog << " show-pipelines\n";
  os << "  List every OpenVINO.GenAI pipeline this build can run -- "
     << canonicalPipelines().size() << " of them -- one line each, a short explanation\n";
  os << "  after every --pipeline token, then exit.  A pure reader: it loads no model, so it\n";
  os << "  takes no --model / --device (and no --pipeline of its own -- the value 'run' /\n";
  os << "  'worker' pick is exactly what it prints).  This is the table the 'run' / 'worker'\n";
  os << "  --pipeline help now points at.\n\n";
  os << "  Drive a pipeline you pick from the list with:\n";
  os << "    " << prog << " run    --pipeline <p> [model-dir] <options>   one-shot\n";
  os << "    " << prog << " worker --pipeline <p> [model-dir] <options>   long-lived worker\n";
  po::options_description s("", config::kHelpLineLength, config::kHelpMinDescriptionLength);
  s.add_options()("help", po::bool_switch(), "Show this help message");
  os << s;
  return os.str();
}

// The `show-pipelines` helper body: the single place the full pipeline table renders.
// A pure reader -- it neither selects a pipeline nor loads a model, so it takes no
// --model / --device (and no --pipeline): it prints the canonical set (the same source
// of truth the --pipeline help now points at, instead of inlining) with each pipeline's
// one-line explanation, then exits 0.  Only its own --help is special; any stray token is
// ignored (it parses no config, so there is nothing to validate).
int runShowPipelines(const std::string& prog, const std::vector<std::string> args) {
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the token
  if (anyHelp(sa)) {
    std::printf("%s", showPipelinesHelp(prog).c_str());
    return 0;
  }
  std::vector<std::pair<std::string, std::string>> kv;
  for (const auto& e : canonicalPipelines())
    kv.push_back({e.name, e.description});
  std::ostringstream os;
  os << prog << " -- OpenVINO/GenAI inference CLI: supported pipelines\n\n";
  os << "  Select a pipeline with --pipeline / -p on " << prog
     << " run (one-shot) or " << prog
     << " worker (long-lived), then load a model for it (see " << prog
     << " show-config).\n\n";
  renderCategory(os, "--pipeline choices (one OpenVINO.GenAI C++ pipeline each)", kv);
  std::printf("%s", os.str().c_str());
  return 0;
}


// --- Argument parsing, made GENERAL -- and lived in the front-end (runModeEntry), not the bodies --
// Two free helpers (used only by runModeEntry, so they stay file-local here):
//   * stripRoutingArgs -- the one thing the four typed section parsers (load / runtime / scheduler /
//     request) do NOT recognise: the general routing flags (--pipeline / -p, and the run form's
//     --input / -i).  Removing them (for the value-taking forms, their one trailing value too, unless
//     that value is itself an option) from a copy is what lets each section parser -- who know only
//     their own flags -- accept the line; a bare positional (it becomes --model inside the parser) and
//     everything after a `--` pass through.  General -- it discards only the general routing flags, so
//     no body is any closer to the parser than the next.
//   * parsePipelineArgs -- THE single general parse: from the one combined `vm` already stored above it
//     lifts --pipeline (to re-confirm) and the run form's --input (nullopt for a worker), and fills the
//     four typed config structs each from a routing-flag-stripped copy of the argv (they each do their
//     own cheap boost pass; the cost is the model LOAD, not the parse).  A missing/invalid model / device
//     surfaces as a ConfigError from the load parser -- runModeEntry reports it (naming the now-known
//     pipeline) and exits 2.
std::vector<std::string> stripRoutingArgs(const std::vector<std::string>& args) {
  static const std::set<std::string> kSingle = {"--input", "-i", "--pipeline", "-p"};
  std::vector<std::string> out;
  bool afterDashDash = false;
  for (std::size_t i = 0; i < args.size(); ) {
    const std::string& tok = args[i];
    if (!afterDashDash && tok == "--") { out.push_back(tok); afterDashDash = true; ++i; continue; }
    if (afterDashDash) { out.push_back(tok); ++i; continue; }  // positionals pass through
    if (tok.rfind("--input=", 0) == 0 || tok.rfind("-i=", 0) == 0 ||
        tok.rfind("--pipeline=", 0) == 0 || tok.rfind("-p=", 0) == 0) {
      ++i;                                                     // the =form is one token
      continue;
    }
    if (kSingle.count(tok)) {
      if (i + 1 < args.size()) {
        const std::string& nx = args[i + 1];
        if (!(nx.size() > 1 && nx[0] == '-')) ++i;             // consume its (space) value
      }
      ++i;
      continue;
    }
    out.push_back(tok);
    ++i;
  }
  return out;
}

// Build a body's PipelineParsed from runModeEntry's one combined parse (vm) + the raw argv (sa):
// the small things a body may log / the run form's direct chat are lifted straight from vm; the four
// typed sections are each re-read once on a routing-flag-stripped copy of the argv.
void parsePipelineArgs(const std::vector<std::string>& sa,
                       const po::variables_map& vm,
                       PipelineParsed& out) {
  // Omitting --pipeline means "auto" (the option is optional now; the default is applied HERE, since this
  // boost build's value<> takes no default arg): the registry's dispatchPipeline then auto-detects.
  out.pipeline = vm.count("pipeline") ? vm["pipeline"].as<std::string>() : std::string("auto");
  out.model    = vm.count("model")    ? vm["model"].as<std::string>()    : std::string();
  out.device   = vm.count("device")   ? vm["device"].as<std::string>()    : std::string();
  out.input    = vm.count("input")    ? std::optional<std::string>(vm["input"].as<std::string>())
                                       : std::nullopt;
  const std::vector<std::string> bodyArgs = stripRoutingArgs(sa);
  out.load      = config::parseModelLoadConfig(bodyArgs);
  out.runtime   = config::parseModelRuntimeConfig(bodyArgs);
  out.scheduler = config::parseModelSchedulerConfig(bodyArgs);
  out.request   = config::parseModelRequestConfig(bodyArgs);
}

int runModeEntry(const std::string& prog, const std::vector<std::string>& args, bool worker) {
  const std::string modeName = worker ? "worker" : "run";
  const std::vector<std::string> sa(args.begin() + 1, args.end());  // drop the run/worker token
  if (anyHelp(sa)) {
    std::printf("%s", subcommandHelp(prog, modeName, /*isRunForm=*/!worker).c_str());
    return 0;
  }

  // One full combined parse over the (untitled) general section + the four named
  // sections: it rejects an unknown/missing-value option here (so a bad flag never
  // reaches a body) and yields the single general parameter --pipeline, REQUIRED,
  // which selects the body (it has no default -- boost has no clean "required"
  // option, so the front-end enforces it the way the model's required set is done).
  po::options_description desc = generalOptionsDesc(/*isRunForm=*/!worker);
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(sa)
                  .options(desc)
                  .style(po::command_line_style::unix_style)
                  .run(),
              vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::fprintf(stderr, "ovgenai: %s: %s\n", modeName.c_str(), config::rephraseBoost(e).what());
    return 2;
  }
  // (No "missing --pipeline" check here: the option now defaults to "auto", so it is always *present* --
  //  omitting it routes to the registry's auto-detect, not a usage error.  The front-end therefore
  //  reports no such thing; it only reports below a config miss of a body that IS valid.)

  // Route to the selected pipeline -- but NOT inline: both the look-up (find in the registry
  // table) and the execute (run that binding's body) now live in the registry layer -- the
  // dispatchPipeline over kPipelineBindings (a file-local static const, in pipeline_support.cpp) --
  // so runModeEntry only does the front-end's job: parse the one combined argument set (the general
  // section + the four typed layers) exactly once, into a PipelineParsed, and hand it to
  // dispatchPipeline (then return its code).  runModeEntry itself can no longer report an *unknown*
  // pipeline -- it cannot know which --pipeline names are valid (that is the registry's job and the
  // test suite keeps the two sets in lock-step, see that .cpp) -- so it reports only what is in its
  // own authority: a config miss of a body that IS valid (through the parse, below); dispatchPipeline
  // names + rejects the unknown ones, and for the now-default "auto" case auto-detects the family.
  PipelineParsed parsed;
  try {
    parsePipelineArgs(sa, vm, parsed);
  } catch (const config::ConfigError& e) {
    // A usage error from a VALID body's argument parse -- e.g. a missing --model a body requires -- on
    // a pipeline dispatchPipeline accepts: name it via the --pipeline value the one parse just lifted
    // (parsePipelineArgs sets parsed.pipeline from the combined vm first, before any section parser can
    // throw, so it is known here even when the load section then fails), and exit 2 (a usage failure).
    std::fprintf(stderr, "ovgenai: %s: %s\n", parsed.pipeline.c_str(), e.what());
    return 2;
  }

  // Hand the (now known, valid) body its program name + the PARSED arguments + the subcommand-derived
  // mode -- the body neither re-parses the command line nor needs the raw argv at all.  dispatchPipeline
  // does the find + the run, over its file-local table; its exit code (0 ok / 1 a run failure / 2 a
  // usage failure) is returned straight back to the caller.
  return dispatchPipeline(modeName, prog, parsed, worker);

}

}  // close the anonymous namespace of this file's front-end helpers

// --- Infrastructure-only body -- the not-yet-implemented pipelines' shared stub ---------
// It receives the PARSED arguments (a PipelineParsed) runModeEntry already produced ONCE for the
// whole invocation (and reported the route + any usage error at the front-end) -- so the stub
// neither parses the command line nor needs the raw argv, and it never even sees a --help
// (runModeEntry intercepts that before delegating).  Its whole job is to reflect, to stderr, that a
// `run` / `worker` was issued for a pipeline that has no real body yet: report the mode plus the
// (already parsed) model / device it was given, as a supervisor would log it (stdout stays reserved
// for the future result / protocol channel), and exit 0.  The real per-pipeline body (load + a
// one-shot / worker loop) lands per pipeline later; nothing here touches OpenVINO.
// It lives here in commands.cpp (not in pipeline_support.cpp) -- it is front-end infrastructure, so
// the front-end's real --help (subcommandHelp) it used to mirror stays the single source of truth in
// THE front-end, and the registry in pipeline_support.cpp merely references commands::runPipelineStub
// for the eleven stub pipelines.  Declared (external linkage), not in the anonymous namespace, so the
// registry in another translation unit can reach it.
int runPipelineStub(const std::string& name,
                    const PipelineParsed& parsed,
                    bool worker) {
  // A supervisor's line -- the pipeline + the form so a run is provably *not* reported as the other.
  // It uses the (already parsed) model / device; the load set is a contract runModeEntry enforced
  // -- and reported -- up-front, before routing, so nothing to re-check here.  stdout stays reserved
  // for the result / protocol channel.
  std::fprintf(stderr,
               "ovgenai: %s%s: not yet implemented (this build is infrastructure only); "
               "would %s model '%s' on '%s'\n",
               name.c_str(),
               worker ? " (worker mode)" : " (one-shot)",
               worker ? "run it as a worker for" : "run a one-shot for",
               parsed.model.c_str(), parsed.device.c_str());
  return 0;
}


// Build the table: the two operator-facing forms -- run (one-shot) and worker
// (long-lived), each routing by the --pipeline value to an unchanged pipeline body --
// then the two helpers, show-config (prints the resolved config) and show-pipelines
// (lists every pipeline with an explanation; the table the --help no longer inlines).
// A body is bound to a pipeline name in the per-entry lambda, as the old per-pipeline
// entries did -- that is the move from "one subcommand per pipeline + a --worker
// flag" to "run / worker subcommand + a --pipeline parameter".
SubcommandTable::SubcommandTable() {
  m_entries.push_back(Subcommand{
      "run", "command",
      "Run one OpenVINO.GenAI pipeline one-shot; the pipeline is selected by the "
      "required --pipeline / -p (a lone <model-dir> stands in for --model).",
      [](const std::string& prog, const std::vector<std::string>& a) {
        return runModeEntry(prog, a, /*worker=*/false);
      }});

  m_entries.push_back(Subcommand{
      "worker", "command",
      "Run one OpenVINO.GenAI pipeline as a long-lived worker (the form the old --worker "
      "flag selected); the pipeline is chosen by the required --pipeline / -p, and it reads "
      "its line protocol (<command>:<sessionId>:<body>) from stdin (so it has no --input).",
      [](const std::string& prog, const std::vector<std::string>& a) {
        return runModeEntry(prog, a, /*worker=*/true);
      }});

  // The first of two helpers: `show-config` (no --pipeline / --input / --worker -- it
  // prints the resolved config and loads nothing).
  m_entries.push_back(Subcommand{"show-config", "helper",
      "Print the resolved load config as JSON and exit (no model is loaded)",
      &runShowConfig});

  // The second helper: `show-pipelines` -- the single place the full pipeline table (all
  // 14: name + one-line explanation) is listed; the top-level --help and the run / worker
  // --pipeline help just point at it instead of re-spelling it.  Like show-config it loads
  // no model -- it neither selects a pipeline nor is a worker.
  m_entries.push_back(Subcommand{
      "show-pipelines", "helper",
      "List every supported OpenVINO.GenAI pipeline with a one-line explanation "
      "(loads no model; the table the run/worker --pipeline help points at).",
      &runShowPipelines});
}

const std::vector<Subcommand>& SubcommandTable::entries() const {
  return m_entries;
}

int SubcommandTable::dispatch(const std::string& prog,
                              const std::vector<std::string>& args) const {
  if (args.empty()) {
    std::printf("%s", usage(prog).c_str());
    return 2;
  }
  const std::string& mode = args[0];
  if (mode == "help" || mode == "-h" || mode == "--help") {
    std::printf("%s", usage(prog).c_str());
    return 0;
  }
  if (mode == "version" || mode == "--version") {
    std::printf("ovgenai %s\n", OVGENAI_VERSION);
    return 0;
  }
  for (const auto& sc : m_entries)
    if (sc.name == mode) return sc.run(prog, args);
  std::fprintf(stderr, "ovgenai: unknown command '%s'\n", mode.c_str());
  std::printf("%s", usage(prog).c_str());
  return 2;
}

std::string SubcommandTable::usage(const std::string& prog) const {
  std::ostringstream os;
  os << prog << "  -- OpenVINO/GenAI inference CLI\n\n";
  os << "USAGE:\n";
  os << "  " << prog << " <subcommand> [model-dir] <options>\n";
  os << "  " << prog << " run      --pipeline <p> [model-dir] <options>   run one pipeline one-shot\n";
  os << "  " << prog << " worker   --pipeline <p> [model-dir] <options>   run one pipeline as a long-lived worker\n";
  os << "  " << prog << " version | --version                            print the version and exit\n";
  os << "  " << prog << " -h | --help                                    show this help\n\n";

  // The two operator-facing surfaces: run / worker (one entry each), then the helpers
  // (show-config / show-pipelines).  Categories carry them; the full pipeline set no
  // longer renders here -- it moved to the `show-pipelines` subcommand, so where the
  // "Known pipelines" table used to be spelled out there is now a one-line pointer.
  std::vector<std::pair<std::string, std::string>> commandSubs, helperSubs;
  for (const auto& sc : m_entries)
    (sc.category == "helper" ? helperSubs : commandSubs).push_back({sc.name, sc.description});
  renderCategory(os, "Commands (the pipeline is chosen with --pipeline / -p)",
                 commandSubs);

  // Where the full pipeline table used to be (a "name = blurb" line per pipeline, shared
  // with the --pipeline help) now sits a pointer instead: `show-pipelines` lists every
  // pipeline with its explanation, so the whole set is shown ONCE -- there -- rather than
  // duplicated across this index and every run / worker / vlm --help.  The caption is kept
  // so the slot reads as a section; show-pipelines also appears below under "Helper
  // subcommands".
  os << "Known pipelines: run '" << prog << " show-pipelines' to list and describe the available pipelines.\n\n";
  renderCategory(os, "Helper subcommands", helperSubs);

  os << "EXAMPLES:\n";
  os << "  " << prog << " run --pipeline llm --model ./model --device CPU\n";
  os << "  echo '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}' | " << prog
       << " run --pipeline vlm --model ./model --device CPU\n";
  os << "  echo 'GENERATE:123:{\"req_id\":1,\"gen_config\":{"
       << "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}}' | " << prog
       << " worker --pipeline vlm --model ./model --device CPU "
       << "(READY/RESULT/PONG/... <type>:<sessionId>:<body> out)\n";
  return os.str();
}

const SubcommandTable& subcommandTable() {
  static SubcommandTable t;
  return t;
}

}  // namespace commands
}  // namespace ovgenai
