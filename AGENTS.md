# AGENTS.md - LLMServe

LLMServe is a lean, headless, CUDA-only fork of the Strata LLM engine (upstream: github.com/niko1221/strata, remote `upstream`).
It serves **Qwen3.8-Flash-Next** (GGUF arch `qwen4exp`) over an OpenAI/Anthropic-style HTTP API, launched through Docker.
No desktop app, web UI, TUI, installer, Metal/SYCL/HIP backends, DeepSeek or Qwen3.6 engines. MIT (see LICENSE and
third_party/ggml/LICENSE; keep the model-license notes in docs/).

## Layout
- `src/`, `include/strata/`: C++/CUDA engine. `src/program/generate.cpp` is `main` of the `strata` binary and its `--serve` loop.
  Also `core`, `kernels` (cuda + cpu), `prefill`, `spec`, `artifact`, `plan`, `ngram`, `platform`.
- `CMakeLists.txt`, `cmake/`: build (targets `strata`, `strata_kq`, small CLIs). `third_party/ggml*`: ggml headers/overlay inputs.
- `serve/`: Python HTTP server (`python -m serve.server`). It drives the engine over a stdin/stdout line protocol and does the
  tokenizing, chat template and tool parsing. `strata_control/` (only the modules the server imports) and `strata_cli/client.py`
  are its helpers. `serve/chat_template.jinja` is the fallback template.
- `tools/`: model-prep tools (`iq_pack`, `mtp_fetch`, `mtp_pack`, `mtp_rt`, `draft_vocab`, `strata_pack`, ...) and `tools/vision/`
  (`strata-vision` image encoder, a separate CMake project on llama.cpp's mtmd).
- `data/`: expert profiles, draft vocabularies, GGUF fingerprints. `docs/`: kept reference docs (DETAILS.md, MODELS.md, ...).
- `docker/`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `llmserve/prepare.py`: **the entry points**. `prepare.py` downloads and
  packs the model and writes the run config; the container then runs `python -m serve.server --engine strata --config <cfg.json>`.

## Build (CUDA 13 recommended; sm_120 needs it)
llama.cpp must be at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`third_party/ggml/VERSION.txt`); pass a checkout with
`-DSTRATA_GGML_DIR=<llama.cpp>` or let CMake fetch it. Needs CMake >= 3.24 and Ninja (`requirements-build.txt`).
```
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES="86-real;89-real;120-real;120-virtual" -DSTRATA_GGML_DIR=...
cmake --build build --target strata strata_kq -j"$(nproc)"
```
`libstrata_kq.so` must sit next to `strata`. Use a single arch while iterating (a full arch list takes ~25 min).

## Tests (no GPU needed)
`python -m pytest serve -q` in a venv with `requirements.txt` + `pytest` (server tests use a mock engine / fake engine scripts).
There are no C++ tests in this fork. After pruning or merging, re-run an import check: `python -c "import serve.server"`.

## Syncing from upstream
`git fetch upstream && git merge upstream/main`. Expect modify/delete conflicts in paths removed here (apps, metal, sycl, dsv41, qwen35,
packaging, tests, bench, most of tools/strata_cli/strata_control, docs): resolve by keeping them deleted (`git rm`). Re-check
`CMakeLists.txt` (this fork removed the HIP/SYCL/qwen35/DSV41/test blocks) and the server import closure afterwards.
