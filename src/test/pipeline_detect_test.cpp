// Unit tests for the pure pipeline-family detector (src/main/pipeline_detect.{hpp,cpp}).
//
// detectPipelineFor reads ONLY the model dir's config.json (a file/JSON read -- no model load, no OpenVINO
// engine, no race) and returns one of the 14 canonical pipeline names, or "unknown".  These tests pin the
// table's BEHAVIOUR:
//   * safe fallbacks -- config.json MISSING -> "llm" (the common minimal export; the CB degrades to
//     text-only at runtime); a PRESENT-but-unmatched or MALFORMED config -> "unknown" (we decline, never
//     silently run a generation body on a model we could not place);
//   * the vlm/llm split the user mandated -- we check the single visual key (a `vision_config` block:
//     present -> vlm, absent -> text-only -> llm) only to *label* the row; the actual vlm-vs-llm choice at
//     run time is the ContinuousBatchingPipeline's (it builds the vision/text embedder iff the dir has one).
//     The subtle case: an `image_token_id` (a visual *token*) with NO vision_config is still "llm" (a stray
//     token, no vision tower) -- we read just the one key, exactly as asked.
//   * the embedding vs text-embedding split (the same `vision_config` key);
//   * the 12 non-generation rows -- each -> that name; a wrong *label* is harmless (every non-generation
//     family drops to the very same "not implemented yet" decline, so it can never run the wrong body).
//
// The synthetic cases build a throwaway temp dir (RAII, removed at scope exit) so they run on ANY box
// without the baked fixtures and without the OpenVINO runtime.  The real baked fixtures are added only when
// present (GTEST_SKIP when a fixture's config.json is gone at runtime -- the same guard the rest of the suite
// uses, so a fixture-less box is never failed).  The file is auto-picked up by the test glob (no CMakeLists
// edit); its baked fixture macro (OVGENAI_VLM_FIXTURE) is the existing per-target one.
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>            // getpid (a unique temp dir per test process)
#include <vector>

#include "../main/pipeline_detect.hpp"

using namespace ovgenai;
namespace fs = std::filesystem;

namespace {
// A throwaway temp dir, created in the ctor and removed best-effort on destruction (RAII -- so a failing
// test still cleans up; std::filesystem owns the bytes, so this is not a memory new/delete).  The dir is
// made immediately; a config.json is written ONLY by writeConfig, so a TempDir that is never written is a
// dir with NO config.json (the literal "config.json missing" case the detector must default to "llm").
struct TempDir {
  fs::path p;
  explicit TempDir(const std::string& tag)
      : p(fs::temp_directory_path() / ("ovgenai_detect_" + std::to_string(::getpid()) + "_" + tag)) {
    std::error_code ec;
    fs::create_directories(p, ec);       // idempotent; tolerate a dir a prior run left behind
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(p, ec);               // best-effort, never throws
  }
  TempDir(const TempDir&) = delete;      // non-copyable (it owns its unique path)
  TempDir& operator=(const TempDir&) = delete;
  void writeConfig(const std::string& body) const {
    std::ofstream((p / "config.json").string(), std::ios::binary | std::ios::trunc) << body;
  }
  std::string dir() const { return p.string(); }
};

// A fixture is runnable only if its config.json is actually on disk (the std::error_code overload of
// exists returns false, not throws, when a path is missing/bad).  The guard that makes a fixture-less box
// SKIP (never fail) the baked-fixture tests -- exactly the pattern the rest of the suite uses.
bool fixtureHasConfig(const fs::path& dir) {
  std::error_code ec;
  return fs::exists(dir / "config.json", ec);
}
}  // namespace

// ---- safe fallbacks (the config.json presence is exactly what's being probed) ---------------------
TEST(ovgenai_pipeline_detect, missingConfigJsonDefaultsToLlm) {
  TempDir t("missing");                  // a dir that exists but has NO config.json
  EXPECT_EQ(detectPipelineFor(t.dir()), "llm");             // the safe generation default (the CB degrades to text-only)
}
TEST(ovgenai_pipeline_detect, malformedConfigDeclinesToUnknown) {
  TempDir t("malformed");
  t.writeConfig("{ this is : not , valid json ");           // present but unparseable
  EXPECT_EQ(detectPipelineFor(t.dir()), "unknown");         // decline, don't guess a generation body for a broken file
}
TEST(ovgenai_pipeline_detect, presentButUnmatchedDeclinesToUnknown) {
  TempDir t("unmatched");
  t.writeConfig("{ \"model_type\": \"totally_unmatched_family\", \"architectures\": [\"UnmatchedModel\"] }");
  EXPECT_EQ(detectPipelineFor(t.dir()), "unknown");         // a present config that names none of the 14 -> decline
}

// ---- the vlm/llm split the user mandated: the Single `vision_config` key --------------------------
TEST(ovgenai_pipeline_detect, visualConfigDecidesVlm) {
  TempDir t("vlm");
  t.writeConfig("{ \"model_type\": \"qwen3_5\", \"architectures\": [\"Qwen3_5ForConditionalGeneration\"], "
                "\"vision_config\": {} }");
  EXPECT_EQ(detectPipelineFor(t.dir()), "vlm");            // a vision_config block IS visual
}
TEST(ovgenai_pipeline_detect, noVisualConfigIsTextOnlyLlm) {
  TempDir t("llm");
  t.writeConfig("{ \"model_type\": \"llama\", \"architectures\": [\"LlamaForCausalLM\"] }");   // a plain causal LLM
  EXPECT_EQ(detectPipelineFor(t.dir()), "llm");
}
// The subtle case the single-key read handles: an `image_token_id` (a visual token) but NO vision_config
// must stay "llm" -- we read just the one key the user asked for, so a stray token (no vision tower) does
// not over-label a text model as vlm.
TEST(ovgenai_pipeline_detect, imageTokenWithoutVisionConfigIsLlm) {
  TempDir t("img_token");
  t.writeConfig("{ \"model_type\": \"qwen3\", \"architectures\": [\"Qwen3ForCausalLM\"], "
                "\"image_token_id\": 151655 }");
  EXPECT_EQ(detectPipelineFor(t.dir()), "llm");
}

// ---- the embedding vs text-embedding split (the same single `vision_config` key) ------------------
TEST(ovgenai_pipeline_detect, embeddingWithoutVisionIsTextEmbedding) {
  TempDir t("text_embedding");
  t.writeConfig("{ \"model_type\": \"qwen3_embedding\" }");            // the name says "embedding"; no vision
  EXPECT_EQ(detectPipelineFor(t.dir()), "text-embedding");
}
TEST(ovgenai_pipeline_detect, embeddingWithVisionIsEmbedding) {
  TempDir t("embedding");
  t.writeConfig("{ \"model_type\": \"qwen3_vl_embedding\", \"vision_config\": {} }");           // "embedding" + a visual block
  EXPECT_EQ(detectPipelineFor(t.dir()), "embedding");
}

// ---- the 12 non-generation families (best-effort: each -> its own name; a wrong label is harmless)
TEST(ovgenai_pipeline_detect, nonGenerationFamiliesLabelThemselves) {
  struct Row { const char* cfg; const char* want; };
  const Row rows[] = {
      { "{ \"model_type\": \"qwen3_omni\" }",   "omni" },
      { "{ \"model_type\": \"whisper\" }",      "whisper" },
      { "{ \"model_type\": \"qwen3_asr\" }",    "asr" },
      { "{ \"model_type\": \"kokoro\" }",       "tts" },
      { "{ \"model_type\": \"reranker\" }",     "rerank" },
      { "{ \"model_type\": \"sdxl\", \"architectures\": [\"SDXLInpaintingPipeline\"] }", "inpainting" },
      { "{ \"model_type\": \"controlnet\" }",   "image2image" },
      { "{ \"model_type\": \"img2video\" }",    "image2video" },
      { "{ \"model_type\": \"cogvideox\" }",    "text2video" },
      { "{ \"model_type\": \"sdxl\" }",         "text2image" },
  };
  for (const auto& r : rows) {
    TempDir t(r.want);
    t.writeConfig(r.cfg);
    EXPECT_STREQ(detectPipelineFor(t.dir()).c_str(), r.want) << "for config.json: " << r.cfg;
  }
}

// ---- the baked real fixtures, when present (GTEST_SKIP when a fixture's config.json is gone) -------
#if defined(OVGENAI_VLM_FIXTURE)
// OVGENAI_VLM_FIXTURE is baked to <.../src/test/fixtures/tiny-random--qwen3.5-ov>; its dir's dirname is
// the fixtures dir, so the other baked fixtures are its siblings (derived, never hard-coded to a CWD).
TEST(ovgenai_pipeline_detect, bakedVlmFixturesAreVlm) {
  {
    const fs::path vlm(OVGENAI_VLM_FIXTURE);                  // a real qwen3_5 VLM config.json
    if (!fixtureHasConfig(vlm))
      GTEST_SKIP() << "baked VLM fixture's config.json is absent at runtime";
    EXPECT_EQ(detectPipelineFor(vlm.string()), "vlm");
  }
  {
    const fs::path dir((fs::path(OVGENAI_VLM_FIXTURE) / ".." / "Qwen3.5-0.8B-int4-ov"));   // a real 0.8B int4 VLM
    if (!fixtureHasConfig(dir))
      GTEST_SKIP() << "Qwen3.5-0.8B-int4-ov fixture's config.json is absent at runtime";
    EXPECT_EQ(detectPipelineFor(dir.string()), "vlm");
  }
}
TEST(ovgenai_pipeline_detect, bakedLlmFixtureIsLlm) {
  const fs::path dir((fs::path(OVGENAI_VLM_FIXTURE) / ".." / "tiny-gpt2-ov"));   // a real plain text LLM (no vision_config)
  if (!fixtureHasConfig(dir))
    GTEST_SKIP() << "tiny-gpt2-ov fixture's config.json is absent at runtime";
  EXPECT_EQ(detectPipelineFor(dir.string()), "llm");
}
#endif  // OVGENAI_VLM_FIXTURE
