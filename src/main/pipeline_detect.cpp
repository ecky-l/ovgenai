#include "pipeline_detect.hpp"

#include <rapidjson/document.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

namespace ovgenai {
namespace {

// The 12 non-generation families, as (name -> the substring keys that name them).  These are
// best-effort on config.json's model_type / architecture: ONLY the resulting *label* matters (a wrong
// guess still drops to the same decline -> exit 2; never which BODY runs).  Ordered most-specific first.
const std::vector<std::pair<std::vector<std::string>, std::string>>& nonGenTable() {
  static const std::vector<std::pair<std::vector<std::string>, std::string>> t = {
      {{"omni"},                              "omni"},
      {{"whisper"},                           "whisper"},
      {{"asr", "qwen_asr", "qwen3_asr", "fun_asr", "fun_asr"}, "asr"},
      {{"tts", "kokoro", "talker", "vocoder", "graphemes", "speech2"}, "tts"},
      {{"rerank", "reranker", "crossencoder", "cross_encoder"}, "rerank"},
      {{"inpaint"},                           "inpainting"},
      {{"image2image", "img2img", "controlnet", "instantid"}, "image2image"},
      {{"image2video", "img2video"},         "image2video"},
      {{"text2video", "t2v", "videocrafter", "cogvideox", "cogvideo", "hunyuanvideo"}, "text2video"},
      {{"flux", "stable_diffusion", "sd3", "sd2", "sdxl", "zimage", "qwen_image",
        "sana", "stable_video", "wan2"}, "text2image"},   // the remaining diffusion -> the catch-all
  };
  return t;
}

// Lower-case + fold every non-alphanumeric to '_' so "Qwen3-ASR"/"qwen3_asr"/"stable-diffusion-3"
// all normalise to the underscore table keys (and a hyphen/period/space can't break a match).
std::string normalize(const std::string& s) {
  std::string r;
  r.reserve(s.size());
  for (char c : s)
    r.push_back(std::isalnum(static_cast<unsigned char>(c))
                     ? static_cast<char>(std::tolower(static_cast<unsigned char>(c)))
                     : '_');
  return r;
}

bool containsAny(const std::string& hay, const std::vector<std::string>& keys) {
  for (const auto& k : keys)
    if (hay.find(k) != std::string::npos)
      return true;
  return false;
}

bool endsWithAny(const std::string& a, const std::vector<std::string>& suffixes) {
  for (const auto& s : suffixes)
    if (a.size() >= s.size() && a.compare(a.size() - s.size(), s.size(), s) == 0)
      return true;
  return false;
}

// Read a whole file into a string (RAII -- ifstream self-closes; returns nullopt if it cannot be opened).
std::optional<std::string> readWhole(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return std::nullopt;
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string stringMember(rapidjson::Value::ConstMemberIterator it,
                         rapidjson::Document::ConstMemberIterator end) {
  if (it != end && it->value.IsString())
    return it->value.GetString();
  return {};
}

}  // namespace

std::string detectPipelineFor(const std::string& modelDir) {
  std::filesystem::path dir(modelDir);
  // config.json MISSING -> the common minimal-LLM/VLM export -> the safe generation default "llm"
  // (the CB builds the visual/text embedder from the dir at runtime, or not -- text-only when absent).
  const auto raw = readWhole((dir / "config.json").string());
  if (!raw)
    return "llm";

  rapidjson::Document doc;
  if (doc.Parse<0>(raw->c_str()).HasParseError())
    return "unknown";                         // a present-but-malformed config.json: decline, don't guess generation

  // The single visual key the user asked us to check: a `vision_config` block IS visual (-> would be vlm);
  // its ABSENCE is the "text-only" signal (-> llm).  It is the very key the continuous-batching pipeline
  // also keys on to decide whether to build a vision embedder -- so here we read it ONLY to label the row;
  // the actual vlm-vs-llm choice at run time is the CB's (it builds the embedder iff the dir carries one).
  // We deliberately probe just this one key (not also image_token_id / video_token_id): per the mandate the
  // detector checks the key that determines visual vs text-only, and a `vision_config` block is the
  // unambiguous one -- present in every real OpenVINO-converted VLM (qwen3_5/...) and absent from a text LLM
  // (e.g. gpt2) -- so it is both the cleanest signal and what the engine itself uses.
  const bool hasVision = doc.HasMember("vision_config");

  // model_type + every architecture -> one normalised haystack for the substring lookups below.
  std::string mt;
  if (doc.HasMember("model_type"))
    mt = stringMember(doc.FindMember("model_type"), doc.MemberEnd());
  std::string archNames;
  if (doc.HasMember("architectures")) {
    const auto a = doc.FindMember("architectures");
    if (a != doc.MemberEnd() && a->value.IsArray())
      for (auto& e : a->value.GetArray())
        if (e.IsString())
          archNames += e.GetString();
  }
  const std::string hay = normalize(mt + archNames);

  // 1) A KNOWN non-generation family first -- so a speech / diffusion / embedding / rerank model is
  //    never "text LLM"-ed.  (The embedding-vs-text-embedding sub-split keys on the visual detector.)
  for (const auto& [keys, name] : nonGenTable())
    if (containsAny(hay, keys))
      return name;
  if (containsAny(hay, {"embedding", "embeddings"}))
    return hasVision ? "embedding" : "text-embedding";

  // 2) Otherwise it is a GENERATION model: the single visual key decides vlm vs llm (the user's thing).
  if (hasVision)
    return "vlm";

  // 3) ...but only auto-route to llm when it POSITIVELY looks like a decoder / known text LLM, so an
  //    *unrecognised present* config.json (some exotic family) falls through to a safe "unknown" decline
  //    rather than being silently run through the generation body.
  {
    static const std::vector<std::string> causalArchSuffix =
        {"forcausallm", "forconditiongeneration", "lmheadmodel", "formodel"},
        knownLlmTypes =
        {"gpt2", "gpt_neox", "llama", "mistral", "mixtral", "qwen3", "qwen3_next", "qwen2_next",
         "qwen2", "gemma", "gemma2", "gemma3", "phi", "falcon", "deepseek", "smollm"};
    if (doc.HasMember("architectures")) {
      const auto a = doc.FindMember("architectures");
      if (a != doc.MemberEnd() && a->value.IsArray())
        for (auto& e : a->value.GetArray())
          if (e.IsString() && endsWithAny(normalize(e.GetString()), causalArchSuffix))
            return "llm";
    }
    if (containsAny(hay, knownLlmTypes))
      return "llm";
  }

  // 4) config.json present but matched NOTHING (an unrecognised family) -> decline; do not assume generation.
  return "unknown";
}

std::string detectPipelineDecline(const std::string& detectedFamily) {
  const std::string head =
      detectedFamily == "unknown"
          ? "auto-detect could not identify the pipeline family: the model dir's config.json had a "
            "model_type / architectures that matched none of the 14 known families. "
          : "auto-detected pipeline '" + detectedFamily + "' (from the model dir's config.json), but '"
            + detectedFamily + "' has no body implemented in this CLI yet (it is a forward-declared stub; "
            "only the continuous-batching generation family is wired today). ";
  return head + "Pass an explicit --pipeline for an implemented one (today only 'vlm'), e.g. "
                 "ovgenai run --pipeline vlm --model <dir>.";
}

bool isImplementedPipeline(const std::string& name) {
  // Only the generation family has a real body today; llm aliases to vlm because the CB serves both.
  return name == "vlm" || name == "llm";
}

}  // namespace ovgenai
