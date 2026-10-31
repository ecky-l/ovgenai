// ovgenai's engine-config layer for the OpenVINO GenAI backend (the .cpp half):
// each method owns one piece of the load-time / request-time logic —
//   * build the ov::genai::SchedulerConfig from the scheduler config:
//       ovgenai::config::ModelSchedulerConfig::toSchedulerConfig()
//       (definition moved to model_scheduler_config.cpp);
//   * fold runtime_config / CACHE_DIR / scheduler_config / draft_model into the
//     ov::AnyMap a pipeline is built from:  GenAIPipelineConfig::toPipelineProperties();
//   * build the ov::genai::GenerationConfig for a request:
//       GenAIGenerationConfig::toGenerationConfig()  (makeGenerationConfig then
//       populates it from the per-request config).

#include "genai_config.hpp"
#include "model_runtime_config.hpp"

#include <cstdio>     // fprintf
#include <filesystem>  // (used in GenAIPipelineConfig::toPipelineProperties)
#include <limits>      // (numeric_limits<size_t> for the latency-oriented scheduler default)

#include <rapidjson/document.h>

namespace ovgenai {
namespace {

// The free-form `--runtime-config` dict path used to live here (a rapidjson::Value
// -> ov::Any converter,  a parse-json -> AnyMap reader,  a string-keyed SDPA gate).
// It is gone:  that blob is now the first-class ModelRuntimeConfig
// (model_runtime_config.hpp),  whose toProperties() builds the ov::AnyMap directly;
// the gate below simply keys on that typed config's attention_backend field.
//
// The one gate this layer enforces on an authored scheduler_config block:
//   "if ATTENTION_BACKEND == 'SDPA' in the runtime config and we have an authored
//    scheduler_config block,  drop it and warn (SDPA pipelines cannot take one)."
bool runtimeHasSDPA(const ovgenai::config::ModelRuntimeConfig& runtime)
{
    return runtime.attention_backend.has_value() &&
           *runtime.attention_backend == "SDPA";
}

}  // namespace

// (The KV-cache / batching scheduler config -- its schedulerConfigFromJson /
//  hasAnySet / toSchedulerConfig -- used to be defined here as GenAISchedulerConfig.
//  It is now ovgenai::config::ModelSchedulerConfig in model_scheduler_config.cpp;
//  GenAIPipelineConfig consumes it via its ModelSchedulerConfig member.)

// ---------------------------------------------------------------------------
// GenAIGenerationConfig::toGenerationConfig
//
// Mirrors OVGenAI_LLM.create_generation_config (Python):
//   gc = self.model.get_generation_config() if self.model else GenerationConfig()
//   gc.max_new_tokens        = config.max_tokens
//   apply_temperature(gc, config.temperature)
//   gc.top_k                 = config.top_k
//   gc.top_p                = config.top_p
//   gc.repetition_penalty   = config.repetition_penalty
//   if config.seed:               gc.rng_seed          = config.seed
//   if config.frequency_penalty:  gc.frequency_penalty = config.frequency_penalty
//   if config.presence_penalty:   gc.presence_penalty  = config.presence_penalty
//   if config.num_assistant_tokens:      gc.num_assistant_tokens = ...
//   if config.assistant_confidence_threshold:  gc.assistant_confidence_threshold = ...
//
// We start from the C++ default `GenerationConfig` (no model held at this
// layer).  The key behavioural points:
//   * do_sample = (temperature > 0)  —  OpenVINO GenAI rejects non-positive
//     temperature with do_sample=true,  and that failure would unload the model
//   * seed forwarded only when nonzero  (Python `if config.seed:`  —  0 is falsy)
//   * penalties forwarded only when > 0.0  (Python `if config.X:`  —  0.0 is falsy)
//   * spec-decode params forwarded only when set;  when either is unset
//     (nullopt) the engine's own default is left in place
// ---------------------------------------------------------------------------
ov::genai::GenerationConfig GenAIGenerationConfig::toGenerationConfig() const
{
    ov::genai::GenerationConfig gc;
    //  C++ defaults:  max_new_tokens = SIZE_MAX, top_k = SIZE_MAX, top_p = 1.0,
    //  temperature = 1.0, repetition_penalty = 1.0, do_sample = false,
    //  num_assistant_tokens = nullopt, assistant_confidence_threshold = 0.0f
    gc.max_new_tokens     = max_tokens;
    gc.top_k              = top_k;
    gc.top_p              = top_p;
    gc.repetition_penalty = repetition_penalty;

    //  apply_temperature:  OpenVINO GenAI requires do_sample=true only when
    //  temperature > 0  (non-positive causes a load failure in the Python version).
    gc.temperature = temperature;
    gc.do_sample   = (temperature > 0.0f);

    //  seed: Python `if config.seed:`  —  0 is falsy →  leave engine default
    if (seed.has_value()                     && *seed != 0)
        gc.rng_seed = *seed;

    //  penalties: Python `if config.X:`  —  0.0 is falsy →  leave engine default
    if (frequency_penalty.has_value() && *frequency_penalty != 0.0f)
        gc.frequency_penalty = *frequency_penalty;
    if (presence_penalty.has_value()  && *presence_penalty  != 0.0f)
        gc.presence_penalty  = *presence_penalty;

    //  Spec-decode params:  forward when has_value AND nonzero for threshold
    //  (matching Python `if config.assistant_confidence_threshold:`);
    //  num_assistant_tokens forwarded whenever set (nonzero is the normal case,
    //  but has_value alone is the correct check since 0 would be a meaningful
    //  "no speculative decoding" signal that the engine would handle).
    if (num_assistant_tokens.has_value())
        gc.num_assistant_tokens = *num_assistant_tokens;
    if (assistant_confidence_threshold.has_value()
        && *assistant_confidence_threshold != 0.0f)
        gc.assistant_confidence_threshold = *assistant_confidence_threshold;

    return gc;
}

// ---------------------------------------------------------------------------
// GenAIPipelineConfig
// ---------------------------------------------------------------------------

GenAIPipelineConfig::GenAIPipelineConfig(const ModelLoadConfig& cfg,
                                         const ModelSchedulerConfig& sched,
                                         const ModelRuntimeConfig& runtime)
    : cfg_(cfg), sched_(sched), runtime_(runtime)
{
    //  The scheduler config is a first-class input now (see the header); we no
    //  longer copy the 10 fields out of the (now scheduler-less) load config.
    schedAuthored_ = sched_.hasAnySet();
}

// ---------------------------------------------------------------------------
// GenAIPipelineConfig::toPipelineProperties
//
//  Full C++ port of the pipeline_kwargs construction in
//  OVGenAI_LLM.load_model() + OVGenAI_VLM.load_model():
//
//    pipeline_kwargs = {**(loader.runtime_config or {})}
//    scheduler_config = extract_scheduler_config_from_loader(loader)
//    if loader.cache_dir:
//        pipeline_kwargs['CACHE_DIR'] = loader.cache_dir
//    if draft_model is not None:
//        pipeline_kwargs['draft_model'] = draft_model
//    self.model = LLMPipeline(loader.model, loader.device,
//                             **scheduler_config,  **pipeline_kwargs)
//
//  We fold all three sources into a single ov::AnyMap,  since the C++
//  pipeline constructor takes one AnyMap rather than keyword arguments.
// ---------------------------------------------------------------------------
ov::AnyMap GenAIPipelineConfig::toPipelineProperties() const
{
    //  (1)  the runtime config's typed OpenVINO properties (runtime_.toProperties());
    ov::AnyMap props = runtime_.toProperties();

    //  (2)  scheduler_config,  when authored AND the backend is not SDPA
    //       (the (1) runtime.toProperties() already folds in CACHE_DIR, now a
    //       first-class runtime field  --  so there is no separate step for it)
    //  (3)  scheduler_config,  when authored AND the backend is not SDPA
    if (schedAuthored_ && !runtimeHasSDPA(runtime_)) {
        auto sc = sched_.toSchedulerConfig();
        props["scheduler_config"] = ov::Any(sc);
    } else if (schedAuthored_ && runtimeHasSDPA(runtime_)) {
        std::fprintf(stderr,
                     "ovgenai: refusing to set scheduler_config: model '%s' uses "
                     "ATTENTION_BACKEND=SDPA which cannot accept one\n",
                     cfg_.model.c_str());
    }

    //  (3)  draft_model,  when present
    if (cfg_.draft_model.has_value() && !cfg_.draft_model->empty()) {
        try {
            ov::AnyMap draftProps;
            if (runtime_.cache_dir.has_value() && !runtime_.cache_dir->empty())
                draftProps["CACHE_DIR"] = ov::Any(std::string(*runtime_.cache_dir));
            const std::string dev =
                cfg_.draft_device.has_value() && !cfg_.draft_device->empty()
                    ? *cfg_.draft_device : std::string("CPU");
            auto [name, draftAny] = ov::genai::draft_model(
                std::filesystem::path(*cfg_.draft_model),
                dev, draftProps);
            (void)name;
            std::fprintf(stderr, "ovgenai: loaded draft model from %s on %s\n",
                         cfg_.draft_model->c_str(), dev.c_str());
            props["draft_model"] = draftAny;
        } catch (const ov::Exception& e) {
            std::fprintf(stderr,
                         "ovgenai: failed to load draft model from %s: %s — "
                         "continuing without speculative decoding\n",
                         cfg_.draft_model->c_str(), e.what());
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "ovgenai: failed to load draft model from %s: %s — "
                         "continuing without speculative decoding\n",
                         cfg_.draft_model->c_str(), e.what());
        }
    }

    return props;
}

// ---------------------------------------------------------------------------
// GenAIPipelineConfig::resolveSchedulerConfig + toPipelinePropertiesNoScheduler
//
// The ContinuousBatchingPipeline ctor takes the SchedulerConfig as a dedicated
// first-class argument and does NOT read a "scheduler_config" key out of the
// AnyMap; the legacy VLMPipeline ctor does the opposite. So for the CB path we
// hand an explicit SchedulerConfig (this method) and pass the AnyMap with that
// key erased (the next) — the two no longer share the same blob.
// ---------------------------------------------------------------------------
ov::genai::SchedulerConfig GenAIPipelineConfig::resolveSchedulerConfig() const
{
    // An authored scheduler block wins (routed through the first-class
    // ModelSchedulerConfig); otherwise mirror the engine's latency-oriented
    // internal default: max_num_batched_tokens = SIZE_MAX, enable_prefix_caching.
    if (sched_.hasAnySet())
        return sched_.toSchedulerConfig();

    ov::genai::SchedulerConfig sc;
    sc.max_num_batched_tokens = std::numeric_limits<size_t>::max();
    sc.enable_prefix_caching  = true;
    return sc;
}

ov::AnyMap GenAIPipelineConfig::toPipelinePropertiesNoScheduler() const
{
    // Same AnyMap as toPipelineProperties(), minus the "scheduler_config" key
    // (which the CB ctor no longer reads; resolveSchedulerConfig() carries it).
    ov::AnyMap props = toPipelineProperties();
    props.erase("scheduler_config");
    return props;
}
//
// Build the ov::genai::GenerationConfig in two steps:
//   (1)  Populate a GenAIGenerationConfig from the ModelRequestConfig fields.
//        Non-optional fields (max_tokens, temperature, top_k, top_p,
//        repetition_penalty) are always set.  Optional fields are forwarded
//        only when has_value() (double → float cast where needed).
//   (2)  Call GenAIGenerationConfig::toGenerationConfig() which applies
//        temperature → do_sample (greedy for <= 0), forwards optional fields
//        only when non-zero, and leaves num_assistant_tokens /
//        assistant_confidence_threshold at their engine defaults when unset.
// ---------------------------------------------------------------------------
ov::genai::GenerationConfig makeGenerationConfig(const ModelRequestConfig& rt)
{
    GenAIGenerationConfig gc;
    gc.max_tokens            = rt.max_tokens;
    gc.temperature           = static_cast<float>(rt.temperature);
    gc.top_k               = rt.top_k;
    gc.top_p               = static_cast<float>(rt.top_p);
    gc.repetition_penalty  = static_cast<float>(rt.repetition_penalty);
    gc.seed                = rt.seed;
    gc.frequency_penalty   = rt.frequency_penalty
        ? std::optional<float>(static_cast<float>(*rt.frequency_penalty))
        : std::nullopt;
    gc.presence_penalty    = rt.presence_penalty
        ? std::optional<float>(static_cast<float>(*rt.presence_penalty))
        : std::nullopt;
    gc.num_assistant_tokens   = rt.num_assistant_tokens;
    gc.assistant_confidence_threshold = rt.assistant_confidence_threshold
        ? std::optional<float>(static_cast<float>(*rt.assistant_confidence_threshold))
        : std::nullopt;
    return gc.toGenerationConfig();
}

}  // namespace ovgenai
