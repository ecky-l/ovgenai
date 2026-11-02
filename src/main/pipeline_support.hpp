// The pipeline-binding layer -- the seam between the subcommand front-end (commands.cpp,
// which only selects a pipeline by the --pipeline value) and each pipeline's body.
//
// It owns two things that the design (runModeEntry -> "use pipelineBinding to create
// Pipeline, create sessionDispatcher, then call .run()/ .worker()") gathers OUT of the
// bodies and makes GENERAL:
//
//   * PipelineParsed -- one *parsed* argument set for a single run/worker invocation: the
//     four typed config layers (load / runtime / scheduler / request) the body needs, plus
//     the run-form's direct --input (nullopt for a worker) and the raw model / device /
//     the --pipeline value the body was selected by (so a body can log them).  This is what
//     the argument parsing produces; the parsing itself is the ONE general front-end parse
//     (commands.cpp runModeEntry -> the free function parsePipelineArgs), done ONCE per
//     invocation and HANDED to the -- so that a body neither re-parses the command line nor
//     needs the raw argv at all.  It is pipeline-agnostic: every body receives the same
//     shape; `vlm` consumes them to load + generate, and the 13 not-yet-implemented stubs
//     consume only model / device.
//   * PipelineBinding + the registry -- { name, description, run } per canonical pipeline,
//     in a MAP keyed by name (so run/worker route by a single lookup -- a find -- not a
//     linear scan, the requested change from a vector).  `vlm` has a real body (load a
//     VLMPipeline, bind to a SessionDispatcher, run/worker it); the rest share the
//     infrastructure-only stub (kept in commands.cpp -- front-end infra -- so the
//     front-end's real --help stays the single source of truth, and a future body plugs
//     in here unchanged).
#ifndef OVGENAI_PIPELINE_SUPPORT_HPP
#define OVGENAI_PIPELINE_SUPPORT_HPP

#include <functional>      // std::function (PipelineBinding::run)
#include <optional>        // std::optional (PipelineParsed::input -- nullopt for a worker)
#include <string>
#include <unordered_map>   // std::unordered_map (the registry table; built in pipeline_support.cpp)


#include "model_load_config.hpp"      // config::ModelLoadConfig
#include "model_runtime_config.hpp"   // config::ModelRuntimeConfig
#include "model_scheduler_config.hpp" // config::ModelSchedulerConfig
#include "model_request_config.hpp"   // config::ModelRequestConfig

namespace ovgenai {

// The PARSED arguments for one run / worker invocation -- produced ONCE by runModeEntry (see
// commands.cpp: it parses the combined general + four sections a single time and then runs
// parsePipelineArgs to fold that one parse into this), and handed to the selected pipeline's
// body (so the body neither re-parses the command line nor needs the raw argv).  The four
// typed config layers are the real payload a body consumes; the rest are the small bits a
// body may log (the requested model / device, the run form's direct chat, and the --pipeline
// it was selected by).  It is deliberately GENERAL -- no body's name -- because the argument
// parsing it carries is the one general parse, not vlm-specific.
struct PipelineParsed {
  // The four typed config layers, each parsed from the (general-routing-flag-stripped) argv,
  // in the order a body would consume them (load last into the engine's pipeline properties).
  config::ModelLoadConfig      load;
  config::ModelRuntimeConfig   runtime;
  config::ModelSchedulerConfig scheduler;
  config::ModelRequestConfig   request;

  // The run one-shot's direct chat (its --input): nullopt for a worker, whose stdin IS the
  // line-JSON protocol (so a worker's input is per-line, not here).
  std::optional<std::string>   input;

  // The requested (pre-resolve) model / device -- lifted straight out of the one parse -- so a
  // body can log "loading model '...' on ..." before the load + (for an `hf://` ref) report the
  // resolved path, without re-parsing (a plain path is the same pre-/post-resolve).
  std::string                  model;
  std::string                  device;

  // The --pipeline / -p value this invocation was routed by (a body may name it in its logs /
  // its wire line -- e.g. the vlm result already carries session_id; the name helps correlate).
  std::string                  pipeline;
};

namespace commands {

// One CLI pipeline binding: the operator-facing --pipeline token (the registry's lookup key)
// + the run/worker body.  The body receives the PARSED arguments runModeEntry produced once
// (a PipelineParsed) + the program name + the subcommand-derived mode -- and does NOT parse the
// command line itself.  It carries ONLY its name: the one-line descriptions stay in the canonical
// name set (it feeds show-pipelines + the "choose one of" routing message), so this table is name
// + body only, and does not duplicate them.
struct PipelineBinding {
  std::string name;
  std::function<int(const std::string& /*prog*/,
                   const PipelineParsed& /*parsed*/,
                   bool /*worker*/)> run;
};

// run / worker's single delegated entry point (defined in pipeline_support.cpp): it looks the
// selected --pipeline up by name in the registry's file-local static const table and runs that
// binding's body -- so runModeEntry no longer looks up nor routes a pipeline itself; it parses the one
// argument set exactly once (into a PipelineParsed) and then calls this.  An unknown --pipeline is a
// usage error -- named + the operator pointed at show-pipelines (the canonical set via pipelineNames(),
// the same text that listing renders) + exit 2; a valid one runs the bound body and its exit code (0
// ok / 1 a run failure / 2 a usage failure) is returned straight to the caller.
int dispatchPipeline(const std::string& modeName, const std::string& prog,
                     const PipelineParsed& parsed, bool worker);

}  // namespace commands
}  // namespace ovgenai

#endif  // OVGENAI_PIPELINE_SUPPORT_HPP
