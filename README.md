# LLMServe

Run **Qwen3.8-Flash-Next** fast on an NVIDIA GPU, in Docker, and talk to it over an OpenAI- and Anthropic-compatible HTTP API. No UI, no
desktop app: one container, one port.

LLMServe is a lean, headless fork of [**Strata**](https://github.com/niko1221/strata) by Niko1221 and contributors (MIT). The inference
engine (CUDA kernels, hybrid GPU/CPU expert execution, MTP speculative decoding) and the API server are Strata's; this fork removes
everything else (terminal app, desktop app, other models, other GPU vendors) and adds a Docker-first, unattended setup:
`llmserve/prepare.py` replaces Strata's `setup.py` for this one model. The [colibri](https://github.com/JustVugg/colibri) engine
was reviewed for ideas (WSL2 memory and storage advice came from it); none of its code is used.

The model weights are not part of this repository. They are downloaded on first start from Hugging Face
(ISTA-DASLab's GSQ-RCO quantizations of Qwen3.8-Flash-Next, pinned to fixed revisions and SHA-256 checked) and are subject to the
[Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## Quick start (Windows 11, Docker Desktop, NVIDIA GPU)

You need:

- an NVIDIA GPU (RTX 30 series or newer with the default build; RTX 20: set `CUDA_ARCHS="75-real;75-virtual"`) and the **NVIDIA driver 580 or newer** (CUDA 13) installed on Windows. No CUDA toolkit needed.
- **Docker Desktop** with the **WSL2 backend** (Settings > General > "Use the WSL 2 based engine"). Docker Desktop passes the GPU through itself.
- **64 GB of RAM** for the default model (48 GB is the minimum, see the table), about **100 GB of free disk**.
- **Give the WSL2 VM enough RAM.** By default WSL2 (and so Docker Desktop) gets only half of your Windows RAM, and the model needs
  about 48 GB of it *pinned*. Create `%UserProfile%\.wslconfig` (for example `C:\Users\you\.wslconfig`):

  ```ini
  # 64 GB PC
  [wsl2]
  memory=56GB
  ```

  ```ini
  # 128 GB PC (room for IQ3_S / UD-IQ4_XS and long contexts)
  [wsl2]
  memory=112GB
  ```

  then run `wsl --shutdown` in PowerShell and start Docker Desktop again. Details and why: [docs/WINDOWS_DOCKER.md](docs/WINDOWS_DOCKER.md).

Then, in PowerShell in this folder:

```powershell
copy .env.example .env
notepad .env          # optional: pick LLMSERVE_QUANT, LLMSERVE_CONTEXT, set LLMSERVE_API_KEY; uncomment CUDA_ARCHS for your card
docker compose up -d --build
docker compose logs -f
```

- `CUDA_ARCHS` in `.env` cuts the build from 25+ minutes to about 10: `"86-real;86-virtual"` for RTX 30xx, `"89-real;89-virtual"` for RTX 40xx,
  `"120-real;120-virtual"` for RTX 50xx. (`scripts\start.ps1` does the checks and sets this for you.)
- The **first start** builds the engine, downloads ~66 GB (resumable: if it stops, `docker compose up -d` continues), packs the model
  (a few minutes, ~40 GB written for Q2_0 on an AVX-512 CPU) and loads it. Allow one to two hours. The next starts take a few minutes.
- The model lives in the Docker volume `llmserve_llmserve-data` (inside the WSL2 disk, deliberately not in a Windows folder).
- It is ready when the log says `ready: http://127.0.0.1:8080/v1` and `http://localhost:8080/health` answers.

Test it:

```powershell
$body = @{ model = "qwen3.8-flash-next"; max_tokens = 512
           messages = @(@{ role = "user"; content = "Explain what a mixture-of-experts model is in two sentences." }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:8080/v1/chat/completions -Method Post -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:LLMSERVE_API_KEY" } -Body $body   # drop the header if you set no API key
```

```sh
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $LLMSERVE_API_KEY" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Hello!"}],"max_tokens":512,"stream":true}'
```

Connect any OpenAI-compatible client (Open WebUI, Continue, Cline, Aider, the `openai` SDK ...): **base URL `http://localhost:8080/v1`**, API key
= your `LLMSERVE_API_KEY` (anything if none is set), model name anything (the server lists `qwen3.8-flash-next-q2_0`). Anthropic-style
clients: base URL `http://localhost:8080`, endpoint `/v1/messages`. Also served: `/v1/models`, `/v1/responses`, `/health`, `/metrics`.
Qwen3.8 thinks before it answers, and the thinking counts against `max_tokens`: give it room (512 or more).

Stop with `docker compose stop` (it drains running requests, up to two minutes), remove with `docker compose down` (the model stays in the
volume; `docker compose down -v` deletes it). One-command helper that also checks the driver, `.wslconfig` and GPU passthrough:
`powershell -ExecutionPolicy Bypass -File scripts\start.ps1`.

## Which model size

All sizes are Qwen3.8-Flash-Next (a 48-layer, 512-expert MoE with a 262,144-token context). The experts sit in pinned RAM; the GPU holds
the dense weights, the KV cache, the draft layer and a cache of the most-used experts, and the CPU computes the rest, so **RAM decides what
fits and VRAM decides how fast it is** (about 700 more cached experts per extra GB of VRAM). 8 GB of VRAM and up starts; 12 to 16 GB is comfortable.

| `LLMSERVE_QUANT` | Download | RAM asked | Output tok/s (4K / 128K) | Prompt tok/s (4K / 32K) | Notes |
| --- | ---: | ---: | --- | --- | --- |
| **Q2_0** (default) | 66 GB | 48 GB | 93.0 / 73.7 | 1,299 / 2,171 | fastest, good quality |
| IQ2_XS | 68 GB | 48 GB | 78.6 / 62.7 | 1,256 / 2,092 | a little better quality |
| IQ3_XXS | 76 GB | 60 GB | 61.6 / 49.0 | 1,007 / 1,745 | |
| IQ3_S | 84 GB | 62 GB | 53.3 / 45.5 | 913 / 1,624 | best of the small ones, needs a 64 GB PC with little else running |
| IQ1_M (Coder) | 58 GB | 32 GB | 55.1 / 43.0 | 1,583 / 2,177 | half the experts, kept for code, tools and pictures; fits 32 to 40 GB of RAM |
| UD-IQ4_XS | 94 GB | 48 GB, 80+ for all experts | not measured | not measured | Unsloth 4-bit; below ~80 GB of RAM part of the experts is read from the SSD |
| UD-Q4_K_XL | 111 GB | 48 GB, 94+ for all experts | 7 to 8.5 on 64 GB RAM (SSD streaming) | - | Unsloth 4-bit K-quant |
| UD-Q5_K_XL, UD-Q6_K_XL, Q8_0 | 158 to 188 GB | 108 to 136 GB | not measured | not measured | for 128 GB+ machines |

Speeds are Strata's own measurements (`docs/DETAILS.md` upstream): **RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, native Windows,
engine 0.1.26, MTP on**, one code-agent prompt per length. They are not measurements of this container. Newer engines are faster (+16 to 22%
prompts from 0.1.36, +13 to 16% decode from 0.1.42), an RTX 5090 reaches 130 to 190 tok/s, and a community RTX 5060 Ti 16 GB under **WSL2**
(IQ3_S) measured 48 tok/s output and 343 / 598 tok/s prompts at 3.8K / 19K. **Under WSL2 prompts are slower than on native Linux or
Windows** (no KV streaming, see [docs/WINDOWS_DOCKER.md](docs/WINDOWS_DOCKER.md)); decoding is not much affected.

With less RAM than the quant asks for, LLMServe still starts in a *low-RAM mode* that reads experts from the SSD; it works but is much
slower, and the log tells you so.

## Tuning

Everything is a variable in `.env` (documented in `.env.example`); `docker compose up -d` applies a change without downloading anything again.

- **`LLMSERVE_CONTEXT`**: tokens (`65536`, `131072`, `262144`, or `auto`). Longer contexts take VRAM from the expert cache (about 1.7 GB of
  KV per 128K at 8-bit), so output gets slower; Q2_0 loses 93 to 74 tok/s between 4K and 128K. Beyond 262,144 the engine switches on yarn rope scaling.
- **`LLMSERVE_KV`**: `int8` (default), `q4_0` (half the KV memory, a little less exact), `k8v4`.
- **`LLMSERVE_PARALLEL`**: 1 to 8 requests at once. 1 is fastest for a single user. Every slot reserves the whole context of KV cache in VRAM, which
  the expert cache then lacks. An estimate for Q2_0 with the default `int8` KV: about 1.0 GB per slot at 64K (0.6 GB at 32K, 1.8 GB at 128K),
  about 5% of the expert cache of a 24 GB card, so on a 24 GB card 2 slots at 64K, 3 at 48K or 4 at 32K are recommended. On a card that holds only part
  of the experts (12 to 16 GB) slots cost 10 to 25% speed per request. Setup prints the estimate and warns when you ask for more; the engine log prints the exact size.
  Several slots mainly cut the waiting (a second agent no longer waits for the first answer to end); they add little total speed.
- **`LLMSERVE_BATCH_MTP=on`** (needs `LLMSERVE_PARALLEL` of 2 or more and the draft layer): every slot also verifies one draft proposal per step (`--batch-mtp`).
  If the log says `MTP admission ... failed`, turn it off again.
- **`LLMSERVE_BATCH_SWITCHES=on`** (needs `LLMSERVE_PARALLEL` of 2 or more): the engine's opt-in batch switches `STRATA_BATCH_GDN_GROUPS=1`,
  `STRATA_BATCH_ATTN_ROWS=1` (left out when the KV cache is streamed), `STRATA_SLOT_COPY_DIRECT=1`, `STRATA_SLOT_COPY_BATCHSYNC=1`. Each can be
  overridden in `LLMSERVE_ENGINE_ENV`, which always wins.
- **`LLMSERVE_AGENT=on`**: settings for a coding agent with a long system prompt and tool definitions: `STRATA_CACHE_MESSAGE_BOUNDARY=1` (a prompt
  snapshot at the last message, so the next turn re-reads only the new part), with two or more slots `STRATA_ADMIT_SHORT_FIRST=4096` (a short sub-agent
  request is read before a 30K cold prompt; the container entry point sets it for the server), and `--conversation-cache-mib 4096` (RAM for parked conversations; less when
  the RAM left beside the model is short). **`LLMSERVE_CONVERSATION_CACHE_MIB`** sets that size yourself (0 = off) and wins over the agent value.
  `LLMSERVE_EXTRA_ARGS` and `LLMSERVE_ENGINE_ENV` win over all of these. Changing any of them rewrites the run config; nothing is downloaded again.
- **`LLMSERVE_WARMUP=full|short|off`** (default `full`) and **`LLMSERVE_WARMUP_PARKS=4`**: after the model is loaded and *before* the port opens, the server
  sends requests of its own so that your first request does not pay the cold-start costs (thinking off, greedy; the engine keeps none of them except
  the parks): a ~16K-token prefill (code in several languages, prose, a JSON tool call), a ~1K prompt with 768 new tokens (enough for the engine's PCIe-share
  auto-tune, which logs `PCIe share: ... pcie_frac`), N short *different* conversations that the engine does keep, and a ~120-token request. The parks
  touch the engine's RAM conversation cache once at start: parking a conversation into fresh RAM costs about 2.4 s under WSL2 until the cache's 4 places have been used.
  They are the oldest entries, so your own conversations replace them first. `short` runs a 2K prefill, the parks and the short request (about a third of the time);
  `off` skips it. It costs roughly 30 to 60 s on every start (an estimate; the log prints yours), during which the container is `starting` and `/health` does not answer (the port is closed). A step that fails is
  logged and skipped; the server always starts. Not run with a lazy start. Changing either variable rewrites two keys of `/data/cfg.json` (nothing is downloaded or packed again).
  The log shows each step and its duration: see [docs/TTFT_WARMUP.md](docs/TTFT_WARMUP.md).
- **`LLMSERVE_VISION=on`**: the model reads pictures (downloads 0.9 GB more). With under 24 GB of VRAM the reader runs on the CPU.
- **`LLMSERVE_DRAFT_VOCAB=en`**: ~215 MiB less VRAM for the draft head; use it on 12 GB cards or when the start says "the draft head does not fit".
- **`LLMSERVE_VRAM_RESERVE_MIB`**: VRAM left free for Windows and other programs (default 700; try 1500 if the desktop stutters).
- **`LLMSERVE_EXTRA_ARGS`**: any engine argument, e.g. `"--pool-workers 8"`. See what the container would run with
  `docker compose run --rm llmserve prepare --dry-run`.
- **`CUDA_ARCHS`, `PORTABLE`, `WITH_VISION`, `BUILD_JOBS`** (build): which GPU architectures to compile for; `PORTABLE=OFF` builds the CPU kernels with
  `-march=native` for the PC that builds the image (build and run on the same machine); skip the picture reader; limit compiler jobs when the build runs out of memory.
- **`LLMSERVE_API_KEY`**: clients must send it. Set it before changing `LLMSERVE_BIND` from `127.0.0.1` to `0.0.0.0` to open the port to your network.

Where does the time to the first token go? Besides the engine's own `strata serve: prompt ...` line, the server logs one line per request with its phases
(`docker compose logs llmserve | grep "server phases"`):

```
[strata] request 3f9a1c0d2b7e server phases: render 12 ms, encode 340 ms (16512 tokens from 52013 chars, 48,500 tok/s), queue 0 ms, engine prompt 3812 ms, first token 4401 ms after the request arrived (other 237 ms)
```

`render` is the chat template, `encode` the Python tokenizer (with the characters taken from the previous turn's encode when the conversation goes on), `queue` the
wait for the engine's turn, `engine prompt` the engine's own prompt time, and `other` what is left of the time to the first token (HTTP, the first decode window, parsing).
The same numbers are in the response's `timings` object (`server_render_ms`, `server_encode_ms`, `server_encode_tokens`, `server_prompt_chars`,
`server_encode_reused_chars`, `server_queue_ms`, `server_ttft_ms`, next to the existing `prompt_ms` and `predicted_per_second`) and in `/metrics` history rows.
`python tools/tokenizer_bench.py` (in the container: `docker compose exec llmserve python tools/tokenizer_bench.py`) measures the tokenizer alone on the model's own vocabulary:
tokens/s cold and warm on repetitive text and on this repository's sources, the encode time for 32K, 128K and 256K tokens, and, if `pip install tokenizers` is
available, the speed and the exact id equality of Hugging Face's Rust tokenizer on the same text.

## Benchmark: LLMServe vs original Strata

`llmserve/bench.py` measures one server and compares two saved results. It uses only the Python standard library, so it runs inside the
container and on plain Windows Python. It sends streamed chat requests (temperature 0, thinking off) for three cases: `short` (a short
prompt, long answer), `prompt-4k` and `prompt-16k` (a synthetic document of about 4,000 and 16,000 tokens, then "Summarize"). Each request
starts with a unique run id, so no prompt cache can help. It sends one warm-up request that is not counted, then 3 runs per case, and
prints the median and min to max of prompt tok/s, decode tok/s, time to first token, token counts and the MTP draft acceptance rate. Speeds
come from the server's own `timings`; if a server sends none, the client-side numbers are used.

Measure LLMServe (the container is running), then copy the result out:

```powershell
docker exec llmserve python -m llmserve.bench --label llmserve --out /data/bench-llmserve.json
docker cp llmserve:/data/bench-llmserve.json .
```

Measure original Strata running natively on Windows. Stop the container first, because both models will not fit in RAM at once:

```powershell
docker compose stop
# start Strata's server the way you normally do and wait until it is ready; its default port is 8095 (use the port its launch printed)
py llmserve\bench.py --url http://127.0.0.1:8095/v1 --label strata --out bench-strata.json
```

No Python on Windows? Run the same file from the LLMServe image (the container is stopped, so this only starts the benchmark client):

```powershell
docker run --rm --entrypoint python -v ${PWD}:/out llmserve:cuda -m llmserve.bench --url http://host.docker.internal:8095/v1 --label strata --out /out/bench-strata.json
```

Then compare the two files (no server needed; `py llmserve\bench.py --compare ...` works too):

```powershell
docker run --rm --entrypoint python -v ${PWD}:/out llmserve:cuda -m llmserve.bench --compare /out/bench-llmserve.json /out/bench-strata.json
```

### Several requests at once

`--concurrency 1,2,3,4` runs the benchmark with that many clients at the same time (threads released together), `--runs` rounds per level, and
prints one table per level and a summary across levels: aggregate tok/s (completion tokens of the round divided by its wall time), the
median and the slowest stream's decode tok/s, time to first token (p50 and p95), the longest gap between two tokens of a stream, and how many
prompt tokens the server reused. While a round runs it also polls `GET /v1/status` and `/metrics` and saves their slot counters, to show that the slots were really used.

```powershell
docker exec llmserve python -m llmserve.bench --concurrency 1,2,3 --scenario shared-prefix --label parallel2 --out /data/conc-parallel2.json
docker exec llmserve python -m llmserve.bench --concurrency 1,2,3 --scenario join --label parallel2 --out /data/conc-join.json
```

`--scenario` is one of (or a comma list of):

- `independent` (default): every client a different prompt (`--prompt short` or `prompt-<N>k`), nothing reusable.
- `shared-prefix`: every client sends the same system message of about `--prefix-tokens` (default 20000: header and document, like a system prompt with
  tool definitions) and its own user message. The prefix is identical across clients, levels and rounds of one bench run, so the server can reuse
  it; the per-request run id comes after it, in the user message, so the tails never match. A per-run session id at the top makes a second bench run start cold.
  One uncounted request reads the prefix first and its cold TTFT is reported (`--no-prime` skips it).
- `agent`: `shared-prefix`, then `--turns` (default 3) turns per client; each turn appends the answer and a new user message. Shows how well the follow-ups reuse the earlier turns.
- `join`: N-1 clients decode a long answer (`--decode-tokens`, default 2048); `--join-delay` (default 3 s) later one more client sends a cold prompt
  of `--prefix-tokens`. Reports the decoders' longest gap, their tok/s before and while the prompt is read, and the joiner's TTFT. Level 1 is the joiner alone (its TTFT is the baseline).

Keep a setting only if one client gets less than about 8% slower and the 2 to 3 client numbers (aggregate, TTFT p95) improve. Restart the container between arms
(`LLMSERVE_PARALLEL=1`, then `2` with `LLMSERVE_BATCH_MTP=on LLMSERVE_BATCH_SWITCHES=on`, ...), repeat in A, B, A, B order, and compare the files with `--compare` (it
works on concurrency files too, per scenario and level).

The server's own phases (template, tokenizer, queue wait, first token as the server saw it) get a row each in the per-case table and in `--compare` when the server
sends them in `timings` (LLMServe does; a row is left out when neither side has the number): `server: template (ms)`, `server: tokenize (ms)`, `server: chars reused`,
`server: queue wait (ms)`, `engine: prompt (ms)`, `server: TTFT (ms)` and `client TTFT - server (ms)` (the network and HTTP part).

### Does a change alter what the model says? (output identity)

Switching the engine, a setting or a build can change the output without anyone noticing, so `--fixed-id` and `--save-text` check it. With temperature 0 the same prompt should
give the same text; run the identical command against both configurations and compare:

```powershell
# configuration A (e.g. LLMSERVE_EXTRA_ARGS as it is), then change the setting and `docker compose up -d`, then B
docker exec llmserve python -m llmserve.bench --fixed-id identity-1 --save-text --runs 3 --max-tokens 400 --label A --out /data/id-A.json
docker exec llmserve python -m llmserve.bench --fixed-id identity-1 --save-text --runs 3 --max-tokens 400 --label B --out /data/id-B.json
docker exec llmserve python -m llmserve.bench --compare /data/id-A.json /data/id-B.json
```

`--fixed-id STR` replaces the random run id on the first line of the prompt with `STR-r1`, `STR-r2` ... (run number; `STR-warmup` for the warm-up), so run N of A and run N of B
are byte-identical prompts (the saved prompt hash proves it), while the runs of one file still differ from each other and no prompt cache can answer a later run. `--save-text`
stores each run's generated text in the JSON. `--compare` then adds an "output identity" section: for every case and run, `identical` or `DIFFERENT at character N` with the
words in front of the difference and the next words of each side. Use `--max-tokens` of a few hundred so a divergence has room to show, and compare the same `--cases`.
Notes: both runs must have thinking off and temperature 0 (the bench sends both); a run that stops early on one side only is a difference too. Speculative decoding (MTP) and batching
can legitimately change a few tokens on some GPUs (a different kernel, a different rounding at a near tie), and everything after the first changed token follows from it, so a
difference is something to look at, not proof of a bug: see whether it is one word or a different answer, and whether it also happens between two runs of the *same* configuration
(run A twice). Do not assume either way.

If the server needs a key, set `LLMSERVE_API_KEY` (or `STRATA_API_KEY`) or pass `--api-key`. Other options: `--runs`, `--cases`
(`short,prompt-4k,prompt-16k`, or `prompt-32k` and so on), `--max-tokens` (default 512), `--timeout`.

To get a fair result:

- Use the same quant on both (for example Q2_0), the same context size, and MTP on in both.
- One request at a time: nothing else may talk to either server while it runs, and `LLMSERVE_PARALLEL=1`.
- Close other heavy programs (games, browsers with many tabs, a second model). Run one server at a time.
- The first start after a reboot is cold (the model files are not in the page cache yet). Do one full run first, or ignore the first result.
- Check the "completion tokens" row. If a server stops far below `--max-tokens`, its decode speed is less comparable; the script prints a note.
- Strata's docs say KV streaming is off under WSL2, so long prompts may read slower in the container than in native Windows Strata.
  Decode speed should be similar.

## Linux

The same `docker-compose.yml` works on Linux with Docker Engine and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (driver 580+). Native Linux is
faster than WSL2: KV streaming (`--kv-resident`) and the in-RAM n-gram table (`--ple-io ram`, with 90+ GB of RAM) are used, and pinning is not limited
by the WDDM driver. If the data directory must be a host folder, use `volumes: - /srv/llmserve:/data` and `chown 1000:1000` it (the container runs as uid 1000).

## Troubleshooting

- **"no NVIDIA GPU is visible in this container"**, or no `nvidia-smi` output: check Docker passes the GPU through:
  `docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi`. On Windows this needs the WSL2 backend, a current driver, and a Docker Desktop
  restart after driver updates.
- **`unsatisfied condition: cuda>=13.0` / "driver too old"**: update the NVIDIA driver on Windows to 580 or newer (Game Ready or Studio, from nvidia.com).
- **Out of memory / the container is killed (exit 137)**: the WSL2 VM has too little RAM (`.wslconfig`, above), or another program is using it. Check
  `docker stats`. Choose a smaller quant, or a shorter `LLMSERVE_CONTEXT`. For *VRAM* errors: lower the context, `LLMSERVE_DRAFT_VOCAB=en`, `LLMSERVE_VISION=off`, close games.
- **Very slow loading, or "is on a '9p' mount" in the log**: `/data` is on a Windows drive. Use the named volume of `docker-compose.yml`; do not bind-mount a Windows folder.
- **"No space left on device"**: the Docker/WSL2 disk is full. The model needs ~110 GB while installing (66 GB download + packing). `docker system df`,
  `docker builder prune`, and in Docker Desktop: Settings > Resources > Advanced > Disk image location / size.
- **The image build is killed or hangs**: not enough memory. Set `BUILD_JOBS=4` and `CUDA_ARCHS` to your card in `.env`, give the WSL2 VM more RAM.
- **A download keeps stopping**: it resumes from where it stopped on every start. Behind a mirror set `HF_ENDPOINT`; for a gated repository set `HF_TOKEN`.
- **`401 missing or wrong API key`**: send `Authorization: Bearer <LLMSERVE_API_KEY>` (or `x-api-key`).
- **Container shows `starting` for a long time**: normal during the first download and packing (the health check allows two hours); follow `docker compose logs -f`.
  After the model has loaded it stays `starting` for the start-up warm-up (roughly 30 to 60 s; the log prints it) (`LLMSERVE_WARMUP=off` skips it).

## Layout

```
Dockerfile, docker-compose.yml, .env.example   the container
docker/entrypoint.sh                           checks the machine, runs prepare, execs the server
llmserve/prepare.py, catalog.py                download, verify, pack, write /data/cfg.json (pins from Strata's setup.py)
llmserve/bench.py                              stdlib-only speed benchmark, A/B compare, output identity check (see Benchmark)
serve/warmup.py                                the start-up warm-up the server runs before it opens the port (see Tuning)
tools/tokenizer_bench.py                       tokenizer throughput on the model's vocabulary, optional Hugging Face tokenizers comparison
serve/, strata_control/, strata_cli/           Strata's HTTP server (OpenAI / Anthropic API) and what it imports
src/, include/, CMakeLists.txt, third_party/   Strata's C++/CUDA engine
tools/, data/                                  model packing tools, expert profile, draft vocabularies
```

In the volume: `/data/models` (GGUF files), `/data/packs` (packed model), `/data/mtp` (draft layer), `/data/cfg.json` (what the server loads), `/data/logs/engine.log`.

## Status

The Python (prepare, catalog), shell and compose parts are unit-tested (`python -m pytest llmserve/test_prepare.py`). The CUDA image
itself needs a machine with a GPU and several hours of download to test end to end; if something does not start, open an issue with the output
of `docker compose logs`.

## License

MIT, see [LICENSE](LICENSE) (Niko1221 and contributors for Strata). ggml and llama.cpp code: MIT. Model licenses: Qwen Community License 1.0
(Qwen3.8-Flash-Next and the quantizations of it), Apache-2.0 for the Coder release's own changes.
