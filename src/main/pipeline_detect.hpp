// pipeline_detect.hpp -- auto-detect which OpenVINO.GenAI pipeline a converted model directory is,
// from its config.json -- so the CLI's `--pipeline` / `-p` can be OMITTED (== "auto").
//
// PURE: it only reads config.json -- no model load, no engine, no race -- cheap + deterministic, which
// is why it sits in the front-end (a decision the operator makes once, up front).
//
// It returns ONE of the 14 canonical pipeline names that `pipelineNames()` enumerates (the registry
// keeps the two in lock-step), or the sentinel "unknown".
//
// CORRECTNESS CONTRACT (so the auto path can never SILENTLY run the WRONG body):
//   * config.json is MISSING                  -> "llm"  -- the common minimal LLM/VLM export case;
//                                                  safe: the CB then builds the visual/text embedder
//                                                  from the dir at runtime, or not (text-only).
//   * a model that POSITIVELY looks like a text/vision GENERATION model -> "llm" or "vlm"
//     (a causal/conditional architecture, or a known LLM model_type, or the single *visual* key)
//     -> routed to the (only) real body, runVlmBody, which the ContinuousBatchingPipeline drives.
//   * a KNOWN non-generation family (omni / whisper / asr / tts / embedding / text-embedding /
//     rerank / text2image / image2image / inpainting / text2video / image2video) -> that name
//     -> a graceful "detected, but not implemented in this CLI yet" DECLINE (exit 2).  Best-effort:
//     a wrong guess among these only changes WHICH decline message is shown -- never which body runs
//     (every non-generation family drops to the very same decline; only the *label* differs).
//   * config.json is PRESENT but matches NONE of the 14                  -> "unknown" -> decline.
//
// The user mandated that the vlm-vs-llm split be decided *only* by a single visual config.json key
// (we use hasVision, the very signal the CB itself keys on): so here we read just that key to LABEL
// vlm vs llm; the *actual* vlm/llm choice AT RUNTIME is the ContinuousBatchingPipeline's (it builds the
// embedder iff the dir has one).  Hence an imprecise peek only changes a *report label*, never the run.
#pragma once
#include <string>
namespace ovgenai {
// One of: llm / vlm / the 12 non-generation names / "unknown".
std::string detectPipelineFor(const std::string& modelDir);
// The human stderr line for a declined detection (a non-generation family, or "unknown").
std::string detectPipelineDecline(const std::string& detectedFamily);
// True only for a family with a real body in this CLI today (vlm; llm aliases to it via the CB).
bool isImplementedPipeline(const std::string& name);
}  // namespace ovgenai
