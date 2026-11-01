# ovgenai -- frontend cli tool for openvino/openvino.genai interference
#
# One binary whose subcommand surface (see src/main/commands.{hpp,cpp}) is one
# entry per OpenVINO.GenAI C++ pipeline -- llm / vlm / whisper / asr / tts / omni
# / embedding / text-embedding / rerank / the image- and video-generation
# pipelines -- plus the `show-config` helper, each rendered by
# boost::program_options as two categories in `ovgenai --help`
# (Pipeline subcommands / Helper subcommands). Every pipeline subcommand also
# takes a valueless --worker switch for the long-lived piped worker form -- a
# subprocess that speaks a line-JSON protocol; omit it for a one-shot run.
# This phase is the infrastructure:
# the per-pipeline bodies arrive later, one at a time.
cmake_minimum_required(VERSION 3.28)

project(ovgenai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Binaries (ovgenai + tests) land directly under <build>/ so the integration test
# can reference the worker by a stable, configure-time absolute path.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# find_package(RapidJSON) resolves to the installed config-mode package first:
# rapidjson-devel ships /usr/lib64/cmake/RapidJSON/RapidJSONConfig.cmake exporting
# an INTERFACE target `RapidJSON` (and an alias `rapidjson`). Only when NO such
# config package is found do we include the bundled headers-only fallback finder
# (cmake/FindRapidJSON.cmake) so the project still configures on a box that
# installed RapidJSON without the CMake config. Config mode is searched before
# module mode, so the installed package always wins when present.
find_package(RapidJSON CONFIG QUIET)
if(NOT RapidJSON_FOUND)
  message(STATUS "openvino-cli: no installed RapidJSON config package; using bundled headers-only fallback.")
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindRapidJSON.cmake")
endif()

# Boost.program_options for the shared load-model CLI front-end. Prefer the
# static variant (b2 --layout=system --link=static install, packaged by
# distro as boost-static); setting Boost_USE_STATIC_LIBS=ON selects the .a
# variants when the .so ones would otherwise be chosen. Boost::program_options
# pulls in Boost::container + Boost::headers through the config file, so we
# only need to name program_options here.
set(Boost_USE_STATIC_LIBS ON CACHE BOOL "Prefer static boost libraries" FORCE)
find_package(Boost 1.90 CONFIG REQUIRED COMPONENTS program_options)

option(OVGENAI_BUILD_TESTS  "Build the GoogleTest suite (worker/protocol/config)" ON)

find_package(OpenVINO COMPONENTS Runtime CONFIG REQUIRED)
message(STATUS "OpenVINO: ${OpenVINO_VERSION} (${OpenVINO_DIR})")

find_package(OpenVINOGenAI CONFIG REQUIRED)
message(STATUS "OpenVINO.GenAI: ${OpenVINOGenAI_VERSION} (${OpenVINOGenAI_DIR})")

# Generated constants header consumed by ovgenai_core; build output under
# src/generated/ (gitignored, regenerated on every configure).
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/generated")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ovgenai_config.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/ovgenai_config.h"
  @ONLY)

# --- Engine core (static lib): everything but main.cpp ----------------------
file(GLOB OVGENAI_CORE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/*.cpp"
  # the per-model tool-call parsers live in their own subdir toolcallparsers/ (this phase:
  # qwen35_tool_call_parser.cpp; a later phase adds a sibling file per model there).  One
  # extra glob so the core lib compiles them; the parent glob keeps every other src/main/*.
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/toolcallparsers/*.cpp")
list(REMOVE_ITEM OVGENAI_CORE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/main.cpp")

add_library(ovgenai_core STATIC ${OVGENAI_CORE_SOURCES})
target_include_directories(ovgenai_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_link_libraries(ovgenai_core PUBLIC RapidJSON)
target_link_libraries(ovgenai_core PUBLIC Boost::program_options)
target_link_libraries(ovgenai_core PUBLIC openvino::genai)
target_compile_features(ovgenai_core PUBLIC cxx_std_20)

# --- The ovgenai executable: core + the CLI front-end ------------------------
add_executable(ovgenai "${CMAKE_CURRENT_SOURCE_DIR}/src/main/main.cpp")
target_link_libraries(ovgenai PRIVATE ovgenai_core)
set_target_properties(ovgenai PROPERTIES OUTPUT_NAME ovgenai)

# --- Tests ----------------------------------------------------------------
if(OVGENAI_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)           # installed config exports the lowercase GTest::gtest / GTest::gtest_main / GTest::gmock / GTest::gmock_main
  # The dnf gtest-devel / gmock-devel config ships the LOWERCASE targets (GTestTargets.cmake),
  # but this file links the capital aliases -- so bridge them onto whatever the box actually
  # ships (the same "config wins, alias for compat" move the RapidJSON block makes) and
  # GUARANTEE a GTest::gmock target for the new gmock-based qwen35 tool-call test below.
  if(TARGET GTest::gtest AND NOT TARGET GTest::GTest)
    add_library(GTest::GTest ALIAS GTest::gtest)
  endif()
  if(TARGET GTest::gtest_main AND NOT TARGET GTest::Main)
    add_library(GTest::Main ALIAS GTest::gtest_main)
  endif()
  if(NOT TARGET GTest::gmock AND TARGET GTest::GMock)
    add_library(GTest::gmock ALIAS GTest::GMock)
  endif()

  file(GLOB OVGENAI_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test/*.cpp")

  # testworker is a SEPARATE standalone driver (its own main), so keep it OUT of the gtest glob --
  # that glob feeds ovgenai_tests whose gtest_main.cpp provides the suite's main(), and a second
  # main() in testworker.cpp would clash exactly as main.cpp is removed from ovgenai_core above.
  # (It gets its own executable + add_test further down, after the ovgenai_tests block.)
  list(REMOVE_ITEM OVGENAI_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test/testworker.cpp")

  # The worker's absolute path is a configure-time fact (the exe is emitted to
  # CMAKE_RUNTIME_OUTPUT_DIRECTORY = <build>/), so bake it as a quoted macro.
  add_executable(ovgenai_tests
    ${OVGENAI_TEST_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test/gtest_main.cpp")
  target_link_libraries(ovgenai_tests PRIVATE
    ovgenai_core GTest::GTest GTest::Main GTest::gmock)
  set_target_properties(ovgenai_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(ovgenai_tests PRIVATE
    OVGENAI_WORKER_BIN="${CMAKE_CURRENT_BINARY_DIR}/ovgenai")

  # A real (random-weight) VLM pipeline to exercise the now-genuine `vlm` body
  # end-to-end (the committed tiny-random--qwen3.5-ov fixture): baked absolute,
  # like the worker path, so a subprocess test resolves it regardless of CWD; when
  # the fixture is absent at configure time the macro is simply not defined and the
  # test skips (as ovgenaiRuns() makes it do when the box cannot exec).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test/fixtures/tiny-random--qwen3.5-ov/config.json")
    target_compile_definitions(ovgenai_tests PRIVATE
      OVGENAI_VLM_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/src/test/fixtures/tiny-random--qwen3.5-ov")
  endif()

  # A real, check-in-able JPEG for the image-source test coverage -- baked absolute
  # like the fixture, so a subprocess resolves it regardless of CWD; when it is
  # absent at configure time the macro is not defined and that test file skips.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test/fixtures/dog-237-536x354.jpg")
    target_compile_definitions(ovgenai_tests PRIVATE
      OVGENAI_VLM_IMAGE="${CMAKE_CURRENT_SOURCE_DIR}/src/test/fixtures/dog-237-536x354.jpg")
  endif()

  # The captured real Qwen3.5 / 3.6 / 3.8 tool-call OUTPUTS: one section per model, each a
  # " * " bullet line followed by the result / finish_reason / metrics / gen_config JSON
  # object the live run printed (all four used --tool-call-parser gemma4, the unwired name,
  # so every captured result leaks its raw tags and finishes "stop").  The new
  # qwen35_tool_call_parser_test reads this file and feeds each section `result` -- which
  # carries inter-tag newlines plus a leading close-of-thinking / prose section, exactly as
  # the live model emitted it -- line by line to the parser, then checks the uniform
  # tool_calls + finish_reason output.  Baked absolute (like the other fixtures) so the test
  # reads it regardless of CWD; when the file is absent at configure time the macro is simply
  # not defined and the suite skips.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tool-call-qwen35.txt")
    target_compile_definitions(ovgenai_tests PRIVATE
      OVGENAI_TC_QWEN35_FILE="${CMAKE_CURRENT_SOURCE_DIR}/tool-call-qwen35.txt")
  endif()

  # --- testworker: a SEPARATE standalone driver (its own main, NOT a gtest case) for the long-lived
  # `ovgenai worker` subprocess over its line-JSON protocol.  Excluded from the glob above so its
  # main() does not clash with gtest_main.cpp.  It links nothing beyond the C++ stdlib + Threads
  # (it only forks/execs the worker and pipes stdin/stdout -- no RapidJSON/Boost, so it carries no
  # dependency the not-yet-final protocol will need), bakes the worker path + the VLM fixture (so it
  # resolves regardless of CWD), and is build-ordered after `ovgenai` so the subprocess exists when
  # ctest runs it.  Two ctest entries drive the SAME one subprocess: `testworker` (--sessions 1, the
  # single-session smoke) and `testworker_sessions50` (a real 50-session / 50-thread run) -- both PASS
  # because the worker is now idle- AND session-safe: one background pump serves all sessions in
  # parallel, and the test does the per-session demux (a unique req_id per session), so the multi-
  # session run needs no per-session thread in the worker.
  find_package(Threads REQUIRED)
  add_executable(testworker "${CMAKE_CURRENT_SOURCE_DIR}/src/test/testworker.cpp")
  target_compile_features(testworker PRIVATE cxx_std_20)
  target_link_libraries(testworker PRIVATE Threads::Threads)
  set_target_properties(testworker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(testworker PRIVATE
    OVGENAI_WORKER_BIN="${CMAKE_CURRENT_BINARY_DIR}/ovgenai")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test/fixtures/tiny-random--qwen3.5-ov/config.json")
    target_compile_definitions(testworker PRIVATE
      OVGENAI_VLM_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/src/test/fixtures/tiny-random--qwen3.5-ov")
  endif()
  add_dependencies(testworker ovgenai)        # build the worker before the driver
  # --sessions 1: the registered single-session smoke (one GENERATE per run/worker round).
  add_test(NAME testworker COMMAND $<TARGET_FILE:testworker>
           WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  # --sessions 50: fifty sessions (same prompt, unique id + req_id each) run in PARALLEL over that same
  # one subprocess; the run passes when each generated exactly the --max-tokens (8) tokens.
  add_test(NAME testworker_sessions50 COMMAND $<TARGET_FILE:testworker> --sessions 50
           WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  set_tests_properties(testworker testworker_sessions50 PROPERTIES
    # 77 = SKIP (an unprovisioned box / a missing fixture makes the app exit 77); the outer TIMEOUT is a
    # backstop above the app's own bounded waits (req 120s + shutdown-done 20s + reap 20s << 240; fifty
    # sessions still fit the same single-subprocess pump well within 240 -- a wide margin).
    SKIP_RETURN_CODE 77
    TIMEOUT 240
    ENVIRONMENT "OVGENAI_WORKER_LOGFILE_STUB=/tmp/ovgenai-testworker.log")

  include(GoogleTest)
  gtest_discover_tests(ovgenai_tests
    DISCOVERY_MODE POST_BUILD
    PROPERTIES
      TIMEOUT 120
      ENVIRONMENT "OVGENAI_WORKER_LOGFILE_STUB=/tmp/ovgenai-integration-worker.log")
endif()
