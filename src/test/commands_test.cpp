// Unit tests for ovgenai's subcommand front-end (src/main/commands.{hpp,cpp}) --
// the new surface this build phase introduced: four subcommands (run / worker / the
// show-config / show-pipelines helpers) where the pipeline is now a required, *general*
// parameter shared by
// both run and worker -- --pipeline/-p -- replacing the old "one subcommand per
// pipeline" table and the per-subcommand --worker flag.  The pipeline set (14
// OpenVINO.GenAI C++ pipelines) is now a list the --pipeline option takes; `run`
// runs the selected pipeline one-shot, `worker` runs it as the long-lived piped
// worker, `show-config` prints the resolved config, and `show-pipelines` lists every
// supported pipeline -- neither of the two helpers loads a model.
//
// Covered here, at the in-process unit level (the companion black-box layer is
// worker_cli_test.cpp, which drives the built binary):
//   * the subcommand table's shape -- run / worker (category "command") + the
//     show-config helper (category "helper"), no blank blurb, no duplicate names;
//     the --pipeline set pinned as a snapshot (via canonicalPipelines()) so a
//     dropped / added pipeline is caught, and each pipeline spec carries the class
//     name that backs it (llm -> LLMPipeline, ...);
//   * the top-level `ovgenai --help` index -- the Commands / (Known pipelines pointer to
//     show-pipelines) / Helper subcommands sections + USAGE / EXAMPLES framing, every
//     subcommand named, the full pipeline table no longer inlined (just pointed at),
//     and the --pipeline / -p selector called out;
//   * dispatch() contract -- no args -> 2, help/-h/--help -> 0,
//     version/--version -> 0, an unknown token -> 2 ("unknown command" on stderr);
//   * the new --pipeline behaviour -- required (a run / worker without it exits 2),
//     unknown -> 2 naming it; each known pipeline runs one-shot / worker on the
//     shared load-model surface (vlm, a real body, is exercised end-to-end by the
//     companion subprocess test); a run / worker --help offers --pipeline (now a
//     pointer to show-pipelines, not the full table -- and run also --input; worker
//     offers neither); a helper's --help offers neither.  `show-pipelines` itself
//     lists every pipeline and loads no model.
//
// The bodies write to stdout/stderr via C stdio, so each dispatch is run under an
// in-process fd redirect (dup2 the two stdio streams onto pipes, then drain after
// the call) -- the same capture a subprocess test needs, but without forking, so
// it runs inside the test process itself (which links the OpenVINO runtime, as the
// other unit tests do).  Result codes come from the return value; the mode text is
// asserted on the drained stderr so a `run` one-shot is provably *not* reported as
// a `worker` (the mode now comes from the subcommand, not a misbindable flag).
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

#include "../main/commands.hpp"

using namespace ovgenai::commands;

namespace {

// The 14 OpenVINO.GenAI C++ pipelines -- pinned here as the snapshot the
// --pipeline parameter must offer.  The snapshot test guards the *set* (compared
// order-independent against canonicalPipelines()), while the behavioural tests
// iterate canonicalPipelines() itself, so an implementation that drops or adds a
// pipeline trips both.
const char* const kPipelineNames[] = {
    "llm", "vlm", "whisper", "asr", "tts", "omni",     "embedding", "text-embedding",
    "rerank", "text2image", "image2image", "inpainting", "text2video", "image2video",
};
constexpr std::size_t kNumPipelines = sizeof(kPipelineNames) / sizeof(kPipelineNames[0]);

// Captured outcome of one dispatch(): the exit code plus the drained
// stdout / stderr of the (redirected) call.
struct DispatchResult {
  int rc = -1;              // dispatch()'s return; -1 if capture could not even start
  std::string out;
  std::string err;
};

// Read `fd` to EOF, appending to `*s`.
void readUntil(std::string* s, int fd) {
  char buf[8192];
  ssize_t n;
  while ((n = ::read(fd, buf, sizeof buf)) > 0)
    s->append(buf, static_cast<std::size_t>(n));
}

// Run subcommandTable().dispatch("ovgenai", args) with the process's stdout /
// stderr redirected onto pipes, returning the exit code plus the captured text.
// in-process dup2 (not a fork), so it runs inside the test binary even though that
// binary links the OpenVINO runtime.  The real fds are restored and the pipes
// drained imperatively before return, so both the code and the text are on `r`.
// A subcommand lets a parse failure escape only as its own caught ConfigError -> 2,
// so the dispatch call does not throw for any argv we feed.
DispatchResult runDispatch(std::vector<std::string> args) {
  DispatchResult r;
  int pout[2], perr[2];
  if (::pipe(pout) != 0 || ::pipe(perr) != 0)
    return r;  // rc stays -1: the caller treats that as "could not capture"
  int realOut = ::dup(STDOUT_FILENO);
  int realErr = ::dup(STDERR_FILENO);
  if (realOut < 0 || realErr < 0) {
    for (int fd : {pout[0], pout[1], perr[0], perr[1]}) ::close(fd);
    return r;
  }
  ::dup2(pout[1], STDOUT_FILENO);
  ::dup2(perr[1], STDERR_FILENO);

  int code = -1;
  try {
    code = subcommandTable().dispatch("ovgenai", args);
  } catch (...) {
    code = -1;  // defensive: none of our argvs make a subcommand throw
  }
  r.rc = code;

  // Flush the C-stdio buffers (now sitting on the pipes), hand the readers EOF,
  // restore the real fds, then drain -- all before return (not deferred) so both
  // the code and the text are on `r`.
  std::fflush(nullptr);
  ::close(pout[1]);
  ::close(perr[1]);
  if (realOut >= 0) { ::dup2(realOut, STDOUT_FILENO); ::close(realOut); }
  if (realErr >= 0) { ::dup2(realErr, STDERR_FILENO); ::close(realErr); }
  readUntil(&r.out, pout[0]); ::close(pout[0]);
  readUntil(&r.err, perr[0]); ::close(perr[0]);
  return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// The subcommand table's shape: four entries (run / worker + show-config / show-pipelines).
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, registersRunWorkerAndTheTwoHelpers) {
  const std::vector<Subcommand>& e = subcommandTable().entries();
  // 2 command entries (run / worker) + 2 helpers (show-config / show-pipelines) = 4.
  ASSERT_EQ(e.size(), 4u);
  std::size_t commandCount = 0, helperCount = 0;
  for (const Subcommand& s : e) {
    if (s.name == "show-config" || s.name == "show-pipelines") {
      ++helperCount;
      EXPECT_EQ(s.category, std::string("helper"));
    } else if (s.name == "run" || s.name == "worker") {
      ++commandCount;
      EXPECT_EQ(s.category, std::string("command"));
    } else {
      FAIL() << "unexpected subcommand: " << s.name;
    }
  }
  EXPECT_EQ(commandCount, 2u);
  EXPECT_EQ(helperCount, 2u);
}

// Snapshot of the --pipeline set (one entry per OpenVINO.GenAI C++ pipeline): the
// canonical name-set is exactly the 14 expected tokens.  Order-independent, so this
// guards the *set* -- a dropped or added pipeline fails it -- without being brittle
// to a reordering of canonicalPipelines().
TEST(ovgenai_commands, pipelineNameSetIsEveryGenAIPipeline) {
  std::vector<std::string> actual = pipelineNames();
  ASSERT_EQ(actual.size(), kNumPipelines);
  std::vector<std::string> expected(std::begin(kPipelineNames), std::end(kPipelineNames));
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual, expected);
}

TEST(ovgenai_commands, subcommandsAndPipelinesHaveNoBlankBlurbOrDupName) {
  // The three subcommands: each captioned, no two sharing a name.
  const std::vector<Subcommand>& e = subcommandTable().entries();
  std::vector<std::string> names;
  names.reserve(e.size());
  for (const Subcommand& s : e) {
    ASSERT_FALSE(s.name.empty()) << "a subcommand has a blank name";
    ASSERT_FALSE(s.description.empty()) << "a subcommand " << s.name << " has a blank blurb";
    names.push_back(s.name);
  }
  std::sort(names.begin(), names.end());
  bool dup = false;
  for (std::size_t i = 1; i < names.size(); ++i)
    if (names[i] == names[i - 1]) dup = true;
  EXPECT_FALSE(dup) << "two subcommands share a name";
  // The --pipeline set: each spec captioned, no two pipelines sharing a name.
  for (const PipelineSpec& p : canonicalPipelines()) {
    ASSERT_FALSE(p.name.empty());
    ASSERT_FALSE(p.description.empty()) << "pipeline " << p.name << " has a blank blurb";
  }
  std::vector<std::string> pn;
  for (const PipelineSpec& p : canonicalPipelines()) pn.push_back(p.name);
  std::sort(pn.begin(), pn.end());
  bool pdup = false;
  for (std::size_t i = 1; i < pn.size(); ++i)
    if (pn[i] == pn[i - 1]) pdup = true;
  EXPECT_FALSE(pdup) << "two pipelines share a name";
}

TEST(ovgenai_commands, showConfigIsHelperCategoryAndPipelineCarriesItsClassName) {
  const std::vector<Subcommand>& e = subcommandTable().entries();
  const Subcommand* sc = nullptr;
  for (const Subcommand& s : e)
    if (s.name == "show-config") sc = &s;
  ASSERT_NE(sc, nullptr);
  EXPECT_EQ(sc->category, std::string("helper"));
  EXPECT_NE(sc->description.find("Print the resolved load config"), std::string::npos);
  // a pipeline spec carries the class name that backs it (llm -> LLMPipeline, ...)
  const PipelineSpec* llm = nullptr;
  for (const PipelineSpec& p : canonicalPipelines())
    if (p.name == "llm") llm = &p;
  ASSERT_NE(llm, nullptr);
  EXPECT_NE(llm->description.find("LLMPipeline"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The top-level `ovgenai --help` index: the full pipeline table no longer inlines
// here -- the "Known pipelines" slot now points at the `show-pipelines` helper, so
// this tests that the index points there *and* does NOT enumerate the pipelines.
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, topLevelHelpPointsToShowPipelinesNotTheFullTable) {
  const std::string usage = subcommandTable().usage("ovgenai");
  EXPECT_NE(usage.find("USAGE:"), std::string::npos);
  EXPECT_NE(usage.find("Commands"), std::string::npos);
  EXPECT_NE(usage.find("Known pipelines"), std::string::npos);
  EXPECT_NE(usage.find("Helper subcommands"), std::string::npos);
  EXPECT_NE(usage.find("EXAMPLES:"), std::string::npos);
  // the new --pipeline / -p selector is called out in the index
  EXPECT_NE(usage.find("--pipeline"), std::string::npos);
  EXPECT_NE(usage.find("-p"), std::string::npos);
  // the "Known pipelines" slot now points at the show-pipelines subcommand ...
  EXPECT_NE(usage.find("show-pipelines"), std::string::npos)
      << "the top-level --help must point at show-pipelines:\n" << usage;
  // ...and the full per-pipeline table is gone: a blurb (e.g. LLMPipeline) only ever
  // appears in show-pipelines' own listing, never inlined in the index here.
  EXPECT_EQ(usage.find("LLMPipeline"), std::string::npos)
      << "the full pipeline table must no longer be inlined in --help:\n" << usage;
  // every subcommand (indent + token) is listed, now including show-pipelines.
  EXPECT_NE(usage.find("  run "), std::string::npos);
  EXPECT_NE(usage.find("  worker "), std::string::npos);
  EXPECT_NE(usage.find("  show-config "), std::string::npos);
  EXPECT_NE(usage.find("  show-pipelines "), std::string::npos);
}

// ---------------------------------------------------------------------------
// dispatch()'s top-level contract.
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, noArgsPrintsUsageAndExitsTwo) {
  DispatchResult r = runDispatch({});
  EXPECT_EQ(r.rc, 2);
  EXPECT_NE(r.out.find("USAGE"), std::string::npos);
}

TEST(ovgenai_commands, helpFlagsPrintUsageExitZero) {
  for (const std::string h : {"help", "-h", "--help"}) {
    DispatchResult r = runDispatch({h});
    EXPECT_EQ(r.rc, 0) << "for " << h;
    EXPECT_NE(r.out.find("USAGE"), std::string::npos) << "for " << h;
  }
}

TEST(ovgenai_commands, versionPrintsVersionExitZero) {
  for (const std::string v : {"version", "--version"}) {
    DispatchResult r = runDispatch({v});
    EXPECT_EQ(r.rc, 0) << "for " << v;
    EXPECT_NE(r.out.find("ovgenai"), std::string::npos) << "for " << v;
  }
}

TEST(ovgenai_commands, unknownCommandExitsTwoAndNamesIt) {
  DispatchResult r = runDispatch({"frobnicate"});
  EXPECT_EQ(r.rc, 2);
  EXPECT_NE(r.err.find("unknown command"), std::string::npos);
  // the offending token is named so the operator can see what they mistyped
  EXPECT_NE(r.err.find("frobnicate"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The --pipeline parameter is now OPTIONAL (default "auto"): omitting it auto-detects the model's
// family from its config.json -- a *run* that enters the body, no longer a usage error -- while an
// explicit *unknown* name (the test below) is still a 2.
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, runWithoutPipelineAutoDetects) {
  // --pipeline omitted now means "auto": the registry auto-detects (a phantom /opt/m has no config.json ->
  // the safe "llm" default) and routes a generation model to the (single, real) body, which then fails the
  // (absent) load -- so this is now a *run*-outcome exit 1, NOT the old "missing required option" exit 2.
  DispatchResult r = runDispatch({"run", "--model", "/opt/m", "--device", "CPU"});
  EXPECT_EQ(r.rc, 1) << "run without --pipeline now auto-detects + enters the body (rc):\n" << r.err;
  EXPECT_NE(r.err.find("auto-detected pipeline 'llm'"), std::string::npos)
      << "it should auto-detect the family:\n" << r.err;
  EXPECT_NE(r.err.find("ovgenai: vlm"), std::string::npos)
      << "and route a detected generation model into the (real) vlm body:\n" << r.err;
  EXPECT_NE(r.err.find("/opt/m"), std::string::npos) << "the model it was handed:\n" << r.err;
  EXPECT_EQ(r.err.find("missing required option"), std::string::npos)
      << "the old required --pipeline error is gone:\n" << r.err;
}

TEST(ovgenai_commands, workerWithoutPipelineAutoDetects) {
  // --pipeline omitted now means "auto": auto-detect (the phantom /opt/m -> the safe "llm" default) then
  // enter the (real) vlm body; for /opt/m the CB ctor throws at the load, before the worker's stdin loop,
  // so this dispatch returns without blocking -- a *run*-outcome exit 1, not the old "missing required" 2.
  DispatchResult r = runDispatch({"worker", "--model", "/opt/m", "--device", "CPU"});
  EXPECT_EQ(r.rc, 1) << "worker without --pipeline auto-detects + enters the body (rc):\n" << r.err;
  EXPECT_NE(r.err.find("auto-detected pipeline 'llm'"), std::string::npos)
      << "it should auto-detect the family:\n" << r.err;
  EXPECT_NE(r.err.find("ovgenai: vlm"), std::string::npos) << r.err;
  EXPECT_NE(r.err.find("/opt/m"), std::string::npos) << r.err;
  EXPECT_EQ(r.err.find("missing required option"), std::string::npos)
      << "the old required --pipeline error is gone:\n" << r.err;
}

TEST(ovgenai_commands, unknownPipelineExitsTwoAndNamesIt) {
  DispatchResult r = runDispatch(
      {"run", "--pipeline", "frobnicate", "--model", "/opt/m", "--device", "CPU"});
  EXPECT_EQ(r.rc, 2) << "an unknown --pipeline should exit 2 (stderr):\n" << r.err;
  EXPECT_NE(r.err.find("unknown pipeline"), std::string::npos) << r.err;
  EXPECT_NE(r.err.find("frobnicate"), std::string::npos) << r.err;
  // the choice list a the operator is pointed at names a real pipeline
  EXPECT_NE(r.err.find("llm"), std::string::npos) << "the error should list the choices: " << r.err;
}

TEST(ovgenai_commands, workerOffersNoInputFlagButRunDoes) {
  // --input is a run-only special; the worker subcommand offers no --input (its stdin
  // is the line-JSON protocol).  Match the bracketed OPTION line, not the prose (which
  // for the worker says "NO --input"), so the discriminator is the rendered option.
  DispatchResult wrk = runDispatch({"worker", "--help"});
  ASSERT_EQ(wrk.rc, 0) << "worker --help";
  EXPECT_NE(wrk.out.find("--pipeline"), std::string::npos) << "worker --help offers the general --pipeline";
  EXPECT_EQ(wrk.out.find("[ --input ]"), std::string::npos)
      << "worker --help must not offer --input:\n" << wrk.out;
  DispatchResult run = runDispatch({"run", "--help"});
  ASSERT_EQ(run.rc, 0) << "run --help";
  // both the general --pipeline and the run-only --input, and the --pipeline help now
  // points at show-pipelines (the full table moved there instead of inlining here)
  EXPECT_NE(run.out.find("--pipeline"), std::string::npos) << "run --help offers --pipeline";
  EXPECT_NE(run.out.find("[ --input ]"), std::string::npos) << "run --help offers --input";
  EXPECT_NE(run.out.find("show-pipelines"), std::string::npos)
      << "run --help's --pipeline help should point at show-pipelines: " << run.out;
}

// ---------------------------------------------------------------------------
// Each pipeline's behaviour under run / worker, iterating canonicalPipelines().
// A valid run / worker on the shared load-model surface resolves + exits 0,
// reporting the mode (now from the subcommand) + the model to stderr; a run /
// worker missing the load model or device exits 2.  vlm, the first pipeline with a
// real body (loads a VLMPipeline), cannot take this placeholder /opt/m in a
// universal loop -- its real one-shot flow is the companion subprocess test in
// worker_cli_test.cpp -- so it is skipped here (the shape / snapshot / --help tests
// above still cover it as a --pipeline entry).  A run / worker missing --model /
// --device does NOT reach the real load: the load config throws before it, so the
// (vlm-inclusive) missing-field test below needs no fixture.
// ---------------------------------------------------------------------------
TEST(ovgenai_commands, everyPipelineRunsOneShotAndWorkerWithModelAndDevice) {
  for (const PipelineSpec& p : canonicalPipelines()) {
    if (p.name == "vlm") continue;  // real body needs a fixture; exercised elsewhere
    // `run --pipeline <p>` (one-shot) resolves the model to stderr and exits 0
    DispatchResult os = runDispatch(
        {"run", "--pipeline", p.name, "--model", "/opt/m", "--device", "CPU"});
    EXPECT_EQ(os.rc, 0) << p.name << " run one-shot (stderr):\n" << os.err;
    EXPECT_NE(os.err.find("(one-shot)"), std::string::npos) << p.name << " one-shot label";
    EXPECT_NE(os.err.find("/opt/m"), std::string::npos) << p.name << " logs the model it was given";
    // `worker --pipeline <p>` exits 0 and (crucially) reports *worker* mode, proving
    // the mode comes from the subcommand, not a phantom always-set flag
    DispatchResult wk = runDispatch(
        {"worker", "--pipeline", p.name, "--model", "/opt/m", "--device", "CPU"});
    EXPECT_EQ(wk.rc, 0) << p.name << " worker (stderr):\n" << wk.err;
    EXPECT_NE(wk.err.find("(worker mode)"), std::string::npos) << p.name << " worker label";
    EXPECT_NE(wk.err.find("/opt/m"), std::string::npos) << p.name << " logs the model it was given";
  }
}

TEST(ovgenai_commands, runOrWorkerMissingModelOrDeviceExitsTwo) {
  for (const PipelineSpec& p : canonicalPipelines()) {
    // The front-end routes a known --pipeline to the body, which mirrors the load
    // config's required set (model / device) -> a clean ConfigError (exit 2) before
    // any real load, so even vlm (a real body) yields 2 here without a fixture.
    DispatchResult missModel = runDispatch(
        {"run", "--pipeline", p.name, "--device", "CPU"});
    EXPECT_EQ(missModel.rc, 2) << p.name << " run missing --model";
    EXPECT_NE(missModel.err.find("missing required"), std::string::npos) << p.name;
    DispatchResult missDevice = runDispatch(
        {"run", "--pipeline", p.name, "--model", "/opt/m"});
    EXPECT_EQ(missDevice.rc, 2) << p.name << " run missing --device";
    EXPECT_NE(missDevice.err.find("missing required"), std::string::npos) << p.name;
  }
}

// `show-config` is a helper (no --pipeline / --input / --worker) whose resolve-and-
// print job is to print the JSON of what resolved and exit 0, or exit 2 on a config
// error; its --help shows the four shared config sections but neither the run/worker
// general --pipeline nor the run-only --input.
TEST(ovgenai_commands, showConfigResolvesAndPrintsJsonExitZero) {
  DispatchResult r = runDispatch({"show-config", "--model", "./tiny-gpt2-ov", "--device", "CPU"});
  ASSERT_EQ(r.rc, 0);
  EXPECT_NE(r.out.find("\"model\":\"./tiny-gpt2-ov\""), std::string::npos);
}

TEST(ovgenai_commands, showConfigMissingModelExitsTwo) {
  DispatchResult r = runDispatch({"show-config", "--device", "CPU"});
  EXPECT_EQ(r.rc, 2);
  EXPECT_NE(r.err.find("missing required"), std::string::npos);
}

// A subcommand's --help does not require the model (anyHelp short-circuits before
// parsing); a helper's help offers neither --pipeline (a run/worker general
// parameter) nor --input (a run special) -- match the bracketed option line, not the
// prose, so the answer is "the option is simply absent", not a phrase match.
TEST(ovgenai_commands, showConfigHelpExitsZeroWithoutModelAndWithoutPipelineOrInput) {
  DispatchResult r = runDispatch({"show-config", "--help"});
  ASSERT_EQ(r.rc, 0);
  EXPECT_NE(r.out.find("Model load parameters"), std::string::npos);
  EXPECT_EQ(r.out.find("[ --pipeline ]"), std::string::npos)
      << "a helper's --help offers no --pipeline option:\n" << r.out;
  EXPECT_EQ(r.out.find("[ --input ]"), std::string::npos)
      << "a helper's --help offers no --input option:\n" << r.out;
}

// The full pipeline table now lives in the `show-pipelines` helper (it used to inline in
// the --help): a bare `show-pipelines` is the one place it is listed, and a helper like
// show-config -- it is a pure reader, so it loads no model and offers neither the
// run/worker general --pipeline nor the run-only --input.
TEST(ovgenai_commands, showPipelinesListsEveryPipelineAndExitsZero) {
  DispatchResult r = runDispatch({"show-pipelines"});
  ASSERT_EQ(r.rc, 0) << r.err;
  for (const char* p : kPipelineNames)
    EXPECT_NE(r.out.find(p), std::string::npos) << "show-pipelines missing pipeline: " << p;
  // a blurb is rendered per pipeline (the class name that backs it) -- the table body
  EXPECT_NE(r.out.find("LLMPipeline"), std::string::npos) << r.out;
}

TEST(ovgenai_commands, showPipelinesHelpOffersNeitherPipelineNorInput) {
  DispatchResult r = runDispatch({"show-pipelines", "--help"});
  ASSERT_EQ(r.rc, 0) << r.err;
  // a helper: needs no --model, and offers neither --pipeline nor --input (match the
  // bracketed OPTION line, not any prose).
  EXPECT_EQ(r.out.find("[ --pipeline ]"), std::string::npos)
      << "show-pipelines offers no --pipeline option:\n" << r.out;
  EXPECT_EQ(r.out.find("[ --input ]"), std::string::npos)
      << "show-pipelines offers no --input option:\n" << r.out;
}
