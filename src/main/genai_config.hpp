// ovgenai's engine-config layer for the OpenVINO GenAI backend.
//
// The generation / pipeline structs hold the per-request generation parameters
// and the pipeline-provisioning values respectively; the KV-cache / batching
// knobs are not here -- they are the first-class
// ovgenai::config::ModelSchedulerConfig (model_scheduler_config.hpp).
// GenAIPipelineConfig is the load-time counterpart that provisions a pipeline
// and produces the ov::AnyMap the OpenVINO GenAI LLMPipeline / VLMPipeline
// constructor consumes.
//
// What "truthy means forward" means:  for each size_t field a value forwards
// only when has_value() && *val > 0 (0 is the "leave the engine default"
// sentinel);  for each bool field a value forwards only when has_value() &&
// *val == true.  This preserves the engine defaults for everything that was
// never explicitly set (an operator setting a field to 0 / False is a no-op).
#ifndef OVGENAI_GENAI_CONFIG_HPP
#define OVGENAI_GENAI_CONFIG_HPP

#include <optional>
#include <string>

// The concrete config structs + ConfigError moved to namespace ovgenai::config
// (see config.hpp / model_load_config.hpp / model_request_config.hpp /
// model_scheduler_config.hpp); they are pulled in here and re-exported into
// ovgenai so the GenAI config layer can name them.  The scheduler config -- and
// the former GenAISchedulerConfig it absorbed -- now lives in
// model_scheduler_config and is re-exported too.
#include "config.hpp"
#include "model_load_config.hpp"
#include "model_runtime_config.hpp"
#include "model_request_config.hpp"
#include "model_scheduler_config.hpp"

#include <openvino/core/any.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/llm_pipeline.hpp>     // ov::genai::draft_model, scheduler_config property
#include <openvino/genai/scheduler_config.hpp>

#include <rapidjson/document.h>

namespace ovgenai {

using ovgenai::config::ConfigError;
using ovgenai::config::ModelLoadConfig;
using ovgenai::config::ModelRequestConfig;
using ovgenai::config::ModelRuntimeConfig;               // the runtime config,  first-class here too
using ovgenai::config::ModelSchedulerConfig;             // absorbed the former GenAISchedulerConfig
using ovgenai::config::modelSchedulerConfigFromJson;     // its JSON reader, re-exported for the tests

// ---------------------------------------------------------------------------
// (The typed scheduler config used to live in this file as GenAISchedulerConfig.
//  It is now ovgenai::config::ModelSchedulerConfig in model_scheduler_config.hpp
//  -- the first-class sibling of the load / request configs: the same 10
//  std::optional fields, plus its own CLI section ("Scheduler Options"),
//  toSchedulerConfig(), hasAnySet(), and the JSON readers, re-exported into
//  this namespace above.)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GenAIGenerationConfig — the per-request generation parameters at request time.
//  Fields with C++ defaults match the defaults the engine expects
//  (max_tokens=16384, temperature=1.0, top_k=50, top_p=1.0,
//  repetition_penalty=1.0;  the rest are optional to distinguish "not in the
//  request" from "explicitly zero").  To build the ov::genai::GenerationConfig
//  to hand to a pipeline's generate(), call toGenerationConfig() —  no
//  model-level arguments;  the request is the only source of values.
//  (OpenVINO GenAI's own C++ API has no concept of model-level defaults;
//  in ovgenai the per-request knobs live in ModelRequestConfig, not the load
//  config.)
struct GenAIGenerationConfig {
    std::size_t max_tokens            = 16384;
    float       temperature          = 1.0f;
    std::size_t top_k               = 50;
    float       top_p               = 1.0f;
    float       repetition_penalty  = 1.0f;
    std::optional<std::size_t> seed;
    std::optional<float>       frequency_penalty;
    std::optional<float>       presence_penalty;
    std::optional<std::size_t> num_assistant_tokens;
    std::optional<float>       assistant_confidence_threshold;

    //! Build the ov::genai::GenerationConfig, step by step:
    //   - apply_temperature():  sets temperature; do_sample=false if <= 0
    //     (OpenVINO GenAI rejects non-positive temperature with do_sample=true)
    //   - num_assistant_tokens / assistant_confidence_threshold:  forwarded
    //     directly when set;  when unset (nullopt) the engine's own default
    //     (nullopt for num_assistant_tokens,  0.0f for threshold) is left in place
    //   - seed / frequency_penalty / presence_penalty only forwarded when
    //     has_value AND nonzero
    ov::genai::GenerationConfig toGenerationConfig() const;
};

// ---------------------------------------------------------------------------
// GenAIPipelineConfig — the class that provisions one pipeline.
//
// It holds a ModelLoadConfig (which the CLI parser produces) and, from it,
// exposes:
//   * toPipelineProperties()   — the ov::AnyMap for LLMPipeline / VLMPipeline
//   * loadConfig() / scheduler() / schedulerAuthored() — accessors.
//
// Note:  num_assistant_tokens / assistant_confidence_threshold are
// per-request (ModelRequestConfig) in ovgenai's design;  they do not appear
// here.  The pipeline is configured solely by the scheduler config, the
// runtime properties (which now include CACHE_DIR), and the draft model.
class GenAIPipelineConfig {
    ModelLoadConfig     cfg_;
    ModelSchedulerConfig sched_;          // the CLI / LOAD-resolved scheduler config this pipeline is built with
    ModelRuntimeConfig   runtime_;        // the CLI / LOAD-resolved runtime properties this pipeline is built with
    bool                 schedAuthored_ = false;  // true iff sched_ hasAnySet()

 public:
    //! Construct from a resolved ModelLoadConfig, the ModelSchedulerConfig that
    //! shapes the pipeline,  and the ModelRuntimeConfig of OpenVINO Runtime
    //! properties it is built with (the last defaulting to all-unset).  Both the
    //! scheduler and the runtime are first-class inputs now;  schedAuthored_ is
    //! true iff sched_.hasAnySet().
    GenAIPipelineConfig(const ModelLoadConfig& cfg, const ModelSchedulerConfig& sched,
                        const ModelRuntimeConfig& runtime = ModelRuntimeConfig{});

    //! The ov::AnyMap to pass as `properties` to the LLMPipeline / VLMPipeline
    //! constructor.  It folds four sources together:
    //   (1) each runtime_config entry converted to ov::Any
    //   (2) CACHE_DIR from cache_dir, if set (the path, as a string)
    //   (3) scheduler_config as ov::Any(ov::genai::SchedulerConfig), but
    //       only when the block was authored AND the runtime_config does not
    //       set ATTENTION_BACKEND == "SDPA"
    //   (4) draft_model as ov::Any( the second element of ov::genai::draft_model()
    //       ), when draft_model is set; a failed draft load is a warning, not a
    //       hard failure ("Failed to load draft model: … continuing without
    //       speculative decoding")
    ov::AnyMap toPipelineProperties() const;

    // For the ContinuousBatchingPipeline ctor, which takes a SchedulerConfig as a FIRST-class
    // argument (it does not pull a "scheduler_config" key out of the AnyMap the way the LLM/VLM
    // pimpl ctors do): the user-authored config when any scheduler field was set, else the
    // latency-oriented default the engine's own VLM adapter chooses when the user set none
    // (max_num_batched_tokens unlimited + enable_prefix_caching).
    ov::genai::SchedulerConfig resolveSchedulerConfig() const;
    // toPipelineProperties() MINUS the "scheduler_config" key: that config is handed to the pipeline as
    // its dedicated argument above, so leaving it in the map would be inert (and misleading). What is
    // left (runtime props + CACHE_DIR + draft_model) is still consumed by the CB ctor from the map.
    ov::AnyMap toPipelinePropertiesNoScheduler() const;

    const ModelLoadConfig&     loadConfig()   const { return cfg_; }
    const ModelSchedulerConfig& scheduler()   const { return sched_; }
    const ModelRuntimeConfig&  runtimeConfig() const { return runtime_; }
    bool                 schedulerAuthored() const { return schedAuthored_; }
};

// ---------------------------------------------------------------------------
// makeGenerationConfig — builds the ov::genai::GenerationConfig for a request.
//
// Build the ov::genai::GenerationConfig in two steps:
//   (1) Populate a GenAIGenerationConfig from the ModelRequestConfig fields.
//       Non-optional fields (max_tokens, temperature, top_k, top_p,
//       repetition_penalty) are always set.  Optional fields are forwarded
//       only when has_value() (double → float where needed).
//   (2) Call GenAIGenerationConfig::toGenerationConfig() which applies
//       temperature → do_sample (greedy for <= 0), forwards the optional
//       fields only when non-zero, and leaves num_assistant_tokens /
//       assistant_confidence_threshold at their engine defaults when the
//       request does not set them.
//
// This models the OpenVINO GenAI C++ API directly:  there is no model-level
// default layer;  ModelRequestConfig is the sole source of the per-request
// generation parameters.
ov::genai::GenerationConfig makeGenerationConfig(const ModelRequestConfig& rt);

}  // namespace ovgenai

#endif  // OVGENAI_GENAI_CONFIG_HPP
