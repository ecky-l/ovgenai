# ovgenai

A command-line inference engine / front-end for
[OpenVINO](https://github.com/openvinotoolkit/openvino) and
[OpenVINO.GenAI](https://github.com/openvinotoolkit/openvino.genai).

`ovgenai` is a small C++ binary that loads a [OpenVINO IR](https://docs.openvino.ai/)
model (produced by OpenVINO's `openvino.convert` / `optimum-intel`) and drives one of
OpenVINO.GenAI's C++ pipelines for inference. It has two major subcommands:

- [`run`](#using-run) — run a pipeline **once**: a chat in on stdin (or `--input`), a
  single JSON result object is streamed out on stdout in several chunks.
- [`worker`](#using-worker) — run a pipeline as a **long-lived process** that reads and
  writes a simple line protocol on stdin / stdout, so it can be supervised and driven by
  another application, such as a server (see [Worker line protocol](#worker-line-protocol)).

The output of both modes is a stream of generated tokens, wrapped in Json objects. Different output classes exist and are parsed from the model's output, depending on the model's capabilities:

* reasoning content is aggregated in the output key *reasoning_content*
* normal content goes to the *content* key
* tool calls are parsed into the json array *tool_calls*

The stream chunk size can be controlled with the `--stream` parameter. With --stream 0 or below 0, no streaming is performed - the output is then collected into a final json object and emitted in one line at the end.

The model to execute is given with the `--model` argument as an OpenVINO IR format directory. If it has the format *hf://<model-name>*, the corresponding snapshot directory inside the huggingface hub cache is resolved and used. Both `run` and `worker` select which pipeline to run with the optional `--pipeline / -p` argument. If omitted, the model's family and suitable pipeline is attempted to be auto-detected from the model's *config.json*, which usually works well for *vlm* and *llm* capable models. More per-request-generation options are available, see `--help`. The --pipeline argument overwrites the auto-detection. See the `show-pipelines` subcommand to get a list of all available pipelines, but note that many of them are not yet implemented. See [OpenVINO.GenAI](https://github.com/openvinotoolkit/openvino.genai) for the model
pipelines and its [C++ API reference](https://openvinotoolkit.github.io/openvino.genai/cpp/index.html)
for what each pipeline does.

---

## Using `run`

`ovgenai run` loads the model and runs the selected pipeline exactly once. The input prompt messages come from the `--input` parameter or from **stdin**: a pipe/redirect is read to EOF (one possibly-large document), an interactive terminal is read line by line until a lone `.` (the `mail(1)` convention). The result is a JSON object on stdout, streamed in chunks, as described above.

```sh
# A VLM text turn, the chat piped on stdin:
echo '{"messages":[{"role":"user","content":"What is in this image?"}]}' \
  | ovgenai run --pipeline vlm --model ./model --device CPU

# The same thing via --input (no stdin is read then):
ovgenai run --pipeline vlm --model ./model --device CPU \
  --input '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'

# Plain text generation (LLM), model dir as a positional, auto-detected pipeline:
echo '{"messages":[{"role":"user","content":"Say hi"}]}' \
  | ovgenai run --model ./model --device CPU
```

The result object contains the turn's `content` and eventually `reasoning_content` and `tool_calls` when the turn produced any. The final line contains a `finish_reason` and a `metrics` object with usage statistics such as `num_generated_tokens`. Useful per-request knobs (all optional, each falling back to the model-load / engine default when omitted) include `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--stream`, and `--enable-thinking`. See `ovgenai run --help` for the full option list.

---

## Using `worker`

`ovgenai worker` behaves like `run` on startup, i.e. the model and pipeline are built from the command line before anything runs. Then the process stays up and reads the [line protocol](#worker-line-protocol) a line at a time on **stdin**, writing matching output lines on **stdout**. The most ergonomic way to drive it manually is with a named pipe (`mkfifo`):

```sh
# --- in one shell: create the pipe and start the worker, attaching its stdin to it ---
mkfifo /tmp/f
ovgenai worker --pipeline vlm --model ./model --device CPU </tmp/f
```

```sh
# --- in another shell: send one request into the pipe ---
echo 'GENERATE:123:{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16}' >/tmp/f
```

The worker first prints `READY`, then, because it streams by default, one `ITEM:123:<delta>` line per chunk of output, and terminates the turn with a single `DONE:123:{...}`. Finish the session and stop the process with an `UNLOAD` (closing the
pipe also exits it):

```sh
# in the other shell:
echo 'UNLOAD:123:{}' >/tmp/f            # -> DONE:123:{} , then the worker exits
```

The `--input` argument is not valid for the worker mode, and a `worker` run shows no direct chat. The most suitable real use case for the worker mode is to drive *ovgenai* as a subprocess from within another application that needs to perform inference tasks.

---

## Build

This is a cmake project, with currently Linux only support. Windows support is upcoming.

### Dependencies

`ovgenai` needs a C++20 toolchain and the CMake/OpenVINO packages below. On a Fedora / RHEL-class box they are all `dnf` packages and any current versions should work. The OpenVINO runtime + OpenVINO.GenAI libraries themselves are downloaded in the next step.

| Dependency               | `dnf` package(s)                 | Used for                                      |
|--------------------------|----------------------------------|-----------------------------------------------|
| CMake                    | `cmake` (≥ 3.28)                 | the build system                              |
| C++20 compiler           | `gcc-c++` (any current)          | compiling the project                         |
| Boost program_options    | `boost-devel`, `boost-static`*   | the CLI argument parser (see `commands.cpp`)  |
| RapidJSON                | `rapidjson-devel`*               | JSON round-trips for the config and protocol  |
| GoogleTest / GoogleMock  | `gtest-devel`, `gmock-devel`     | the test suite (only when building tests)     |
| OpenVINO + OpenVINO.GenAI| *downloaded below*               | the inference engine and pipelines themselves |

\* The build is configured with `Boost_USE_STATIC_LIBS=ON`, so `boost-static` (the static `.a` libraries) is required as well as `boost-devel` (headers + the CMake config). `boost-devel` already pulls in `boost-program-options` among other components that `program_options` links against. A headers-only RapidJSON is bundled with the project as a fallback, so `rapidjson-devel` is optional if you prefer that.

A single line installs the `dnf` side of everything:

```sh
sudo dnf install -y cmake gcc-c++ boost-devel boost-static rapidjson-devel gtest-devel gmock-devel
```

### Steps

1. **Download OpenVINO.GenAI.** The `openvinotoolkit/` directory holds a script that pulls
   the latest **nightly** OpenVINO.GenAI package — which bundles OpenVINO itself — and
   extracts it:

   ```sh
   ./openvinotoolkit/download-latest.sh
   ```

   This updates `openvinotoolkit/latest.txt`, downloads the latest `openvino_genai_*`
   tarball (it greps `latest.txt` for the `rhel9` entry). Re-run it any time you want a
   fresh nightly.

2. **Set the environment.** `setupenv.sh` points the build at the genai SDK the previous
   step just downloaded (it auto-picks the newest `openvino_genai_*` directory) and exports
   the variables CMake and the runtime expect:

   ```sh
   eval "$(./setupenv.sh)"
   ```

   This exports `OpenVINO_DIR` and `OpenVINOGenAI_DIR` (both `.../runtime/cmake`) and
   prepends the SDK's `runtime/lib` and bundled TBB to `LD_LIBRARY_PATH`. Re-evaluate it in
   each new shell. To pin a specific SDK instead of the newest, pre-set `OPENVINO_DIR` /
   `OPENVINO_GENAI_DIR` to the parent of your `runtime/` directory before evaluating.

3. **Configure and build:**

   ```sh
   mkdir build && cd build
   cmake ..
   cmake --build . -j
   ```

   The `ovgenai` binary (plus `ovgenai_tests` and the `testworker` driver) is produced in
   `build/`. To skip the test suite, reconfigure with `-DOVGENAI_BUILD_TESTS=OFF`.

4. **Run the tests** (needs the OpenVINO.GenAI SDK from step 1 to be discoverable):

   ```sh
   cd build
   eval "$(../setupenv.sh)"     # if not already done
   ctest --output-on-failure
   ```

Open the binary directly with, e.g., `./build/ovgenai --help`.

---

## Worker line protocol

This section is for a client that wants to **create and parse** the wire format the `worker`
subcommand speaks. It describes the format only; the C++ side that implements it lives in
`src/main/session_support.cpp` (and the per-chunk streamer in `streaming_support.cpp`).

### Framing

Every physical line on **either** channel is exactly

```
<command>:<sessionId>:<body>
```

— three parts joined by `:`, and terminated by a newline (which the reader consumes).

- **`command`** — the first `:`-delimited token. It is a plain word that never contains `:`.
- **`sessionId`** — the **second** `:`-delimited token. It is the client's own key for a
  conversation, and it never contains `:`. It may be **empty** (e.g. the up-front `READY`, or
  a global control op). The session id is in the header **on purpose**: a client can read it
  here and then take the whole `<body>` verbatim into its own response envelope (an OpenAI
  chat-completions `choices`, which carries no session id) without fishing one out of the
  JSON.
- **`body`** — everything **after the second colon, verbatim**. It is a JSON *object* (at
  least `{}`), and may itself contain further `:` characters freely.

**To parse a line**, split on the **first two** colons only: *everything after the second
colon is the body*. Never assume the body has a fixed number of colons, and never re-split
it. A line with fewer than two `:` (i.e. no `<command>`: `<sessionId>`) is **not** a protocol
line.

On the request side the only words are `GENERATE / PING / CANCEL / UNLOAD`; any other word is an *unknown op* and the worker answers with an `ERROR` line and keeps running. The finished turn is a line starting with `DONE`.

### Lines you send (client → worker, on stdin)

| Line | Body | Meaning |
|------|------|---------|
| `GENERATE:<session>:<config>` | the flat per-request config object | run one completion on this session. The **body *is* the per-request config**: a `messages` array plus whatever sampling knobs you want. A `session` must be non-empty. |
| `PING:<session>:{}` | `{}` | liveness probe; the worker replies `PONG:<session>:{}`. |
| `CANCEL:<session>:{}` | `{}` | stop this session's in-flight work; the worker echoes `CANCEL:<session>:{}`. |
| `UNLOAD:<session>:{}` | `{}` | stop the pump, emit a final `DONE:<session>:…`, and exit. |

A `GENERATE` body is a **flat** JSON object, e.g.

```json
{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":1}
```

Any knob the body omits falls back to the launch config on the `worker` command line, so a CLI `--stream`, `--max-tokens`, `--temperature`, … reaches the request; a knob the body itself carries wins. See the `--help` for more information on model request parameters. The worker can be stopped either with `UNLOAD` line (emits a `DONE`, then exits) or by simply closing
the pipe / stdin.

### Lines the worker sends (worker → client, on stdout)

| Line | Body | When |
|------|------|------|
| `READY::` | `{}` | once, up front, when the worker is ready to accept input. |
| `PONG:<session>:` | `{}` | a reply to `PING`. |
| `CANCEL:<session>:` | `{}` | an echo of a `CANCEL`. |
| `ERROR:<session>:` | `{"error":"<text>"}` | a malformed line or an unknown op; the worker keeps running. |
| `ITEM:<session>:<delta>` | `{"content":"…", "reasoning_content":"…", "tool_calls":[…]}` | **per-chunk stream** while a `GENERATE` runs. Empty contents are omitted for each chunk; `tool_calls` is emitted when the model is calling tools. |
| `DONE:<session>:<body>` | the turn's `finish_reason` + `metrics` (and the full turn when `--stream 0`, see below) | the **terminal** line for a finished `GENERATE`. |

**`--stream N` controls the chunk window** exactly as it does for `run`:

- `N > 1` — pack `N` output tokens into one `ITEM` line.
- `N = 1` (the default) — one token per `ITEM` line.
- `N <= 0` — no `ITEM` lines: the single terminal `DONE` carries the results of the whole turn — `content`, `reasoning_content` (if any), `tool_calls` (if any), `finish_reason`, and `metrics`.

For a streamed turn (`N >= 1`) the `content` / `reasoning_content` / `tool_calls` have already
left on the `ITEM` lines, so the terminal `DONE` body carries only `finish_reason` and
`metrics`. The metrics object is:

```json
"metrics":{
  "num_input_tokens": …,
  "num_generated_tokens": …,
  "load_time_ms": …,
  "mean_ttft_ms": …,
  "mean_tpot_ms": …,
  "mean_throughput_tps": …,
  "mean_vision_encoding_ms": …,
  "mean_text_embedding_ms": …,
  "total_image_slice_count": …
}
```

### A worked round-trip

```text
# on startup the worker reads the pipe, then emits (worker -> client):
#   READY::{}

# client -> worker  (stdin):
GENERATE:123:{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16}

# worker -> client  (stdout):
READY::{}
ITEM:123:{"content":"Hello"}
ITEM:123:{"content":", world"}
ITEM:123:{"content":"!"}
DONE:123:{"finish_reason":"stop","metrics":{"num_input_tokens":2,"num_generated_tokens":16,"mean_ttft_ms":3.1,"mean_tpot_ms":1.2,"mean_throughput_tps":820,"load_time_ms":0,"mean_vision_encoding_ms":0,"mean_text_embedding_ms":0,"total_image_slice_count":0}}

# client -> worker, to end the session / stop the process:
UNLOAD:123:{}
# worker -> client, then exit:
DONE:123:{}
```

(With `--stream 0` the three `ITEM` lines disappear and the single `DONE` instead carries the
whole turn, e.g. `DONE:123:{"content":"Hello, world!","finish_reason":"stop","metrics":{…}}`.)

### A minimal client (pseudocode)

```python
# read one line, split into its three slots, and use the body verbatim
def read_line(f):
    line = f.readline()
    if line == "":                     # EOF: worker went away
        return None
    line = line.rstrip("\n")
    c1 = line.find(":")
    c2 = line.find(":", c1 + 1) if c1 != -1 else -1
    if c1 == -1 or c2 == -1:           # fewer than two ':'  -> not a protocol line
        return None
    command   = line[:c1]
    session   = line[c1+1:c2]
    body      = line[c2+1:]            # everything after the 2nd ':'  (verbatim)
    return command, session, body      # the client parses `body` (JSON) itself
```

A client keys only on the `<command>`; it reads `<session>` from the header, then drops
`<body>` (parse it as its own JSON) straight into its response envelope. It reacts to `READY`
(on startup), collects the `ITEM` deltas for live display, and treats the first `DONE` for a
session as the end of that turn.

---

## Layout

```
CMakeLists.txt              build definition (ovgenai_core static lib + ovgenai + tests)
cmake/FindRapidJSON.cmake   headers-only fallback when no installed RapidJSON config is found
openvinotoolkit/            download-latest.sh pulls + extracts the nightly genai SDK (incl. OpenVINO)
setupenv.sh                 exports OpenVINO_DIR / OpenVINOGenAI_DIR / LD_LIBRARY_PATH (re-eval per shell)
src/main/                   the front-end and engine
  main.cpp                  dispatches argv to the subcommand table
  commands.{hpp,cpp}        the subcommand surface: run / worker / show-config / show-pipelines
  session_support.{hpp,cpp} the worker's session map, input readers, and the line-protocol framing
  streaming_support.{hpp,cpp} the per-chunk ITEM stream + the per-turn DONE
  pipeline_support.{hpp,cpp} the per-pipeline drivers + the result/emit logic
  *_config.{hpp,cpp}        the model-load / runtime / scheduler / per-request config (the CLI options)
src/test/                   gtest suite + testworker (a real over-the-pipe worker driver)
```

## License

See `LICENSE`.
