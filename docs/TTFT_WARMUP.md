# Short-prompt TTFT and cold first requests (RTX 4090, Q2_0, WSL2)

All line numbers are in `src/program/generate.cpp` unless another file is named. **[U]** marks a claim I could not verify here: it is an estimate or needs a run on the GPU.

## 1. Why ~124 tokens take ~2.1 s

### The path a short prompt takes
- `--short-read` defaults to **64** (`:768`, help `:934`). `windows_ok(a,b)` sends a prompt part through the decode verify windows only when `b - a <= short_read` (`:13108-13114`). Each window holds 4 tokens (`S = o.spec`, `:11208`).
- The prompt is cut into parts (`:14073`). The cuts are `{reread_to, root_at, message_at, turn_at, n-1}`. `turn_at` is the last `<|im_start|>` (`:14012-14015`) and it is set whenever `--prompt-cache` > 0, which is the default (`prompt_cache = 6`, `:731`). A chat-templated 124-token prompt is therefore read in two parts:
  1. **[0, turn_at) ≈ 110 tokens.** This is more than 64, so it goes to `read_part`, the **batched prompt path**. Before the read, `lend()` borrows expert-cache slots for the prompt buffers (`:14111-14115`).
  2. A **checkpoint** at `turn_at` (`:14160`). It calls `cudaDeviceSynchronize` and copies the GDN, PLE and index state to the host (`:9727ff`, `src/core/conversation_state.cpp:165`). The code's own note says this takes 17 ms with pooled buffers and **40-60 ms** when fresh pages are faulted in (`:9388-9390`).
  3. **[turn_at, n-1) ≈ 10-13 tokens** (the assistant header plus the empty think block). This part fits under 64, so it goes through the windows. First `refill()` copies the lent slots back (`:14106`).
  4. Last comes the first decode window.
- The engine's own comment on the batched path: *"costs ~300 ms per run however few tokens it has (it streams every expert the chunk routes to that is not in VRAM over PCIe), and it borrows slots it must refill after (~180 ms); a window costs ~16 ms a token, with the misses on the CPU"* (`:13093-13098`). Upstream's benchmark says the same thing: *"short ... its prompt rate is dominated by the fixed cost of a request"* (`upstream/main:strata_cli/bench_standard.py:1094`).

### Why the fixed cost is larger on this machine (~1.5-1.8 s, not ~300 ms) [U]
- The model has 48 layers × 512 experts with 10 active per token (`docs/DETAILS.md:108`), and a Q2_0 expert blob is 1,382,400 B (`src/prefill/prefill.cpp` ring comment). 110 tokens make about 1,100 routes per layer, which touch most of the layer's experts. Every touched expert that is not in VRAM gets streamed. Chunks under 1024 tokens stage only the routed experts and hand some of them to the CPU share (`prefill.cpp:175-230`, on by default; DETAILS "On by default").
- [Corrected by the maintainer] On WSL2 the expert arena *is* pinned: `cudaHostRegister` of the whole arena succeeded on a community WSL2 run (47 GiB, `cudaHostRegister PORTABLE ok`, upstream bench/results/2026-10-09-community-rtx-5060ti-wsl2-iq3_s-0141). The "driver pins only about 1 GB" note in prepare.py is about `cudaHostAlloc` for the KV copy, not the arena. The fixed cost of the batched path is the streaming of every routed expert not cached in VRAM over PCIe, which happens pinned or not. Check the engine's `expert arena:` start-up line to confirm on a given machine.
- Several GB per short prompt is plausible, and that matches ~1.5 s. The `DONE` line's `experts_streamed=` field (`:15443`) shows the exact count per request.
- The 4K prompt costs 1.6 s for 38× the tokens, which also points to a per-run cost that does not depend on length.

### Other per-request costs (all inside `prompt_ms`, which starts at `r0`, `:12545`)
| Cost | Where | Size |
|---|---|---|
| Waiting for the adaptive tier's round in flight | `adapt_tick(true)`, `:13961-13964`; logged at `:15565` | ms to tens of ms |
| Parking the previous conversation (`--conversation-cache-min-tokens` defaults to 0, `:738`) | `park_current`, `:12784-12804`; logs "conversation cache: parked N tokens in X ms" | Small after a short turn. **Large after a 16K turn [U]**, which probably explains why 16K runs 2 and 3 were slower than run 1 |
| Checkpoint at `turn_at` | `:14160` | 17-60 ms |
| Refilling lent slots | `:14106`, `:14171` | ~180 ms (code comment) |
| PLE rows: 124 × 16 ≈ 2K random direct reads | `--ple-io direct` | ~20-100 ms on a VHDX [U]; the start-up line "PLE direct read probe: N rows/s" (`:4066`) shows the rate |
| CUDA graph capture | Verify-window graphs are captured once per width on first use (`:16673`) | First request only |
| Module loading | Already eager: `CUDA_MODULE_LOADING=EAGER` (`:2364`) | Not a cost |

**Server side: negligible.** `StrataEngine.generate` writes `GEN` straight away (`serve/server.py:2184-2203`). The `lines.get(timeout=0.5)` loop returns as soon as a line arrives, so polling adds no latency. The user's numbers support this: TTFT 2.05-2.31 s ≈ `prompt_ms` (124 tokens / 58 tok/s ≈ 2.14 s). The server does not save a session synchronously before `GEN`.

### How to measure each phase
Set `LLMSERVE_ENGINE_ENV="STRATA_TRACE=1 STRATA_PREFILL_TIMING=1 STRATA_DECODE_TIMING=1"`.
- `STRATA_TRACE` prints "read N tokens (batched|windows) in X ms" (`:14123-14127`), "refilled N slots ... in X ms" (`:13358`), the checkpoint times and the request steps (`:11026`).
- `STRATA_PREFILL_TIMING` gives a breakdown per phase: wait copy, gemm, ple and so on (`prefill.cpp:2505-2560`).
- `DONE ... chunk= experts_streamed=` tells you whether a batched part ran.

### Fix design
1. **Config only: `LLMSERVE_EXTRA_ARGS="--short-read 192"`.** All 124 tokens then go through the windows (31 windows). There is no lend, no refill and no streaming. If windows cost ~25-35 ms each [U, derived from decode at 113 tok/s with roughly 3 accepted tokens per window], the prompt takes ~0.8-1.1 s. **Expected TTFT is ~0.9-1.2 s, down from ~2.1 s [U].**
   - Find the crossover with A/B at 128/192/256/384. The batched fixed cost F and the per-token window cost w give the break-even n ≈ F/w, which should be ~200-250 tokens [U].
   - Caveats: rounding differs slightly between the two paths, and #3465 (non-finite logits in prompt windows) falls back by itself (`:9383-9386`, `:13102-13106`).
2. **`STRATA_CKPT_POOL=1`** (`:203-209`) reuses checkpoint buffers. The code's own measurement puts the save at 17 ms against 40-60 ms without it.
3. **Engine change: an automatic short-read.** At `windows_ok` (`:13108`), keep running means of (a) each batched part's ms, fitted as F + b·n, and (b) the windows' ms per token. Send a part through the windows when n·w < F + b·n, with `--short-read` as an upper bound. This is about 40 lines, it adapts to WSL and to cache size, and it starts from today's 64 rule.
4. **Optional:** with an automatic short-read, a part under 1024 tokens could skip the `turn_at` checkpoint sync when it reads only through the windows. A smaller batched-path kernel would not help, because the cost is moving experts, not compute.

### sm_89 / RTX 4090 defaults (coordinator's question)
- `nv_default_env` has tables only for sm_86 and sm_60 (`src/core/nv_defaults.cpp:18-49`). The sm_120 defaults are kernel choices (`strata::sm120_default` in `qsa_decode_attn.cu:856`, `fused_gr.cu:1207`, `verify.cpp:202`, `moe_fused_iq.cu:1294`). The mmvq row table has no sm_89 entry and falls back to the shared table (`native_mmvq.cu:2994`).
- **Every one of these is a decode or kernel speed switch** (ADAPT_LAG/NOWAIT, SIDE_MIXER, ATTN_DAILY, MMVQ rows, and fused IQ prompt experts, which do not cover Q2_0). **None of them touches `short_read`, the turn split, lend/refill or warm-up.** An sm_89 table would not fix the 2.1 s TTFT. It might add a few percent to decode (the sm_86 stack measured +9.6% / +14.6%) [U].
- You can try that by hand: `STRATA_ADAPT_LAG=2 STRATA_ADAPT_NOWAIT=1 STRATA_SIDE_MIXER=2 STRATA_ATTN_DAILY=1 STRATA_B6_MMVQ_ROWS=1 STRATA_QSA_ROWS=1` (all exact on sm_86; not measured on Ada).
- The 24 GB card does hold more experts than 16 GB, which lowers F. The analysis above does not change.

## 2. Cold first requests

### What is cold after a start
| State | Evidence | When it settles |
|---|---|---|
| **PCIe share** (`pcie_frac_auto`, `:685-686`) | `PcieAuto::window` (`:2186-2219`): the first update comes after **48 decode windows** and ≥300 missed experts, then every 96 windows, **at most 3 updates**, each logged "PCIe share: ... -> pcie_frac X" (`:10075`) | ~240 windows ≈ 600-700 decode tokens. The 64-token warm-up never reaches one, which is likely why the first short decode ran at 73 tok/s and later ones at 110 [U] |
| **Adaptive VRAM tier** | Starts from `data/expert-profile.bin`. Swaps up to 96 experts every 4 verify rounds (`:691`, `:727`). The faster warm-up interval is sm_86 opt-in only (`:3228-3235`, `nv_defaults.cpp` `apply_spec_adapt_default`) | Gradual. Not persisted unless `expert_profile_save` is set (`serve/server.py:3196-3215`; prepare.py does not set it) |
| **PLE rows** | Direct I/O bypasses the page cache. Only the 1M-row cache (`:483`) warms. Benchmark runs 2-3 reuse the same document, so their rows are cached; this is a **benchmark artefact** that real traffic does not get [U] | Likely explains 4K run 1 at 1,098 tok/s against 2,559 for runs 2-3, and the 28K first request at 1,378 [U] |
| Verify-window and MTP graphs | Captured on first use (`:16673`, `src/core/mtp.cpp:1403`) | First request |
| Prompt staging and upload buffers | Grow on first use (`:13607-13614`) | First long prompt |
| cuBLAS warm-up | `warm_gemm_libraries` is a **no-op on CUDA** (HIP only, `prefill.cpp:1312-1373`) | First prompt [U, likely tens of ms] |
| Eager modules, callback thread | Already done at start (`:2364`, `:5965` `STRATA_ADAPT_CALLBACK_WARMUP`) | Not a cost |

**No start-up warm-up exists.** I grepped `generate.cpp`, `serve/server.py`, the docker scripts and upstream (`git grep upstream/main`). The only warm-ups are in upstream's benchmark harnesses (`strata_cli/bench_standard.py`, `bench/*`), which discard one request. The HTTP server starts listening only after `StrataEngine()` returns at READY (`serve/server.py:8327` then `serve()` at `:8519`). `/health` answers `ok` from then on (`:6548`).

### Warm-up design
**Where:** in `serve/server.py`, between engine construction (`:8327`) and `httpd = serve(...)` (`:8519`).
- A config key `"warmup"`, written by `prepare.py` and controlled by `LLMSERVE_WARMUP=off|short|full`, default `full`. Skip it when `lazy`.
- Because the port opens only after the warm-up, docker health stays "starting" until it is done. The `--start-period=2h` is enough, so no change is needed in `/health`.
- An engine flag would be harder: the chat template and tokenizer live in the server. The entrypoint is wrong too, because curl would run after the port is already open.

**What to send:** call `engine.generate(ids, max_new, {"strata_checkpoint": False}, cancel)` directly, rendering the prompts with the server's template and thinking off.
1. A ~16K-token synthetic prefill with `max_new=1`. Use workload-like text: code, prose and JSON tool calls, not random tokens, so routing is realistic.
2. A ~1K prompt + **768-token decode** (temperature 0). This gets PcieAuto through all 3 updates and lets the adaptive tier move.
3. One short (~120 token) request, to capture the window graphs for every width and the short path.

**Expected duration [U]:** 16K at 1,400-2,500 tok/s ≈ 7-12 s, decode 768 at 75-110 tok/s ≈ 7-10 s, short ≈ 2 s. **About 20-25 s in total**, small next to model load.

**Keeping it out of the caches:**
- `ckpt=0` (`serve/server.py:1441-1444`, engine `:12341-12344`) writes no checkpoint. The session is "neither continued nor parked", so neither the prompt cache nor the conversation cache keeps any of it.
- The warm-up has no system prompt of 2048+ tokens, so no root checkpoint is pinned (`:14036-14041`).
- The adaptive tier **will** learn from the warm-up text, deliberately. That is why it must look like real traffic. The tier keeps adapting afterwards.
- With `expert_profile_save` on, the warm-up would get saved into the profile. Leave it off, or accept it.

**Cheaper complements that need no code:**
- After one long session, copy the converged `pcie_frac` from the last "PCIe share" log line into `LLMSERVE_EXTRA_ARGS="--pcie-frac X"`. That value always wins, so the first decode is warm from the start.
- Turn on the `expert_profile_save` config key so the learned residency survives restarts.
- To check: compare `DONE experts_streamed=`, the `ple` phase of `STRATA_PREFILL_TIMING` and the PCIe-share lines between the first and the third request after a restart.

## 3. Implemented (server start-up warm-up)

**Where:** `serve/warmup.py`, called from `main()` in `serve/server.py` right before `serve(svc, ...)`. Verified: `serve()` is the call that creates and binds the
`ThreadingHTTPServer` (`Server((host, port), ...)`); `main()` only checks that the port is free (and closes that probe socket) before the minutes of loading. So during the warm-up
nothing listens, `docker/healthcheck.sh`'s `curl` fails, and the container stays `starting` (inside `--start-period=2h`). The ready file and the lifecycle `ready` event come after it too.
`/health` did not need a change. SIGTERM / Ctrl+C during the warm-up cancels the request in flight (the engine gets STOP), closes the engine and exits.

**Config:** `"warmup": "off"|"short"|"full"` and `"warmup_parks": N` in `cfg.json`, written by `llmserve/prepare.py` from `LLMSERVE_WARMUP` (default `full`) and
`LLMSERVE_WARMUP_PARKS` (default 4, 0-64). The same two variables, when set in the server's environment, override the file (so `LLMSERVE_SKIP_PREPARE=1` works too). Neither is in the
prepare fingerprint (changing them redoes no download or pack), but `up_to_date()` compares them with `cfg.json`, and if only they differ `prepare --if-changed` rewrites those two keys.
A wrong value stops the start early (before the model loads). Without the keys (a hand-written config) the server uses `full`, 4. Skipped for a lazy start, replicas and the mock engine.

**Requests** (one user message each, no system prompt, thinking off, greedy: `temperature 0`, `presence_penalty 0` so the model card's 1.5 does not switch the penalty path on),
through `Service.prepare` and `Service.run`, i.e. the template, the tokenizer/PromptEncoder, the queue, `StrataEngine.generate` and its line protocol, as an HTTP request goes:

| # | request | `strata_checkpoint` | mode |
|---|---|---|---|
| 1 | ~16K tokens (python, C++, TypeScript, Rust, Go, SQL, shell, prose, diffs, traces, YAML, JSON tool calls), 1 new token | `false` | full (short: ~2K) |
| 2 | ~1K tokens, up to 768 new tokens (asked to write at least 1500 words so it does not stop early; the log warns if it ends under 90%) | `false` | full |
| 3 | N short different conversations (each starts with its own number), 8 new tokens | not set: the engine keeps and parks them | full, short (if N > 0) |
| 4 | ~120 tokens, 16 new tokens | `false` | full, short |

Request 4 is last on purpose. When the engine starts a request it parks the conversation that is live (if it is `live_ok`). After request 3's last conversation that is conversation N,
so request 4 makes it the Nth park, and being `ckpt=0` it leaves nothing live: the first real request finds a full cache and parks nothing. N parks need N conversations, not N+1.
N is capped at the engine's `--conversation-cache-slots` (default 4) and dropped when the engine reports `conversation_cache_mib=0` or runs batch slots.
The text is deterministic (seeded), so every start routes the same experts. After the warm-up the server puts back its history, totals, latency histograms and `last_timings`, and empties
the PromptEncoder, so `/metrics` starts at zero requests.

**Does `strata_checkpoint: false` really avoid parking?** (`src/program/generate.cpp`, read-only check.) For the request itself: yes. `req_ckpt` is set from `ckpt=0` (`:12375`); then
`turn_at = -1` (`:14035`, no split at the last `<|im_start|>`, so no turn checkpoint), `pp_next_check = INT64_MAX` and `pp_tail_saved = true` (`:13083-13084`, no periodic and no tail checkpoint),
`message_checkpoint_boundary(.., turn_at = -1, ..)` returns -1, and after the request `live_ok = o.prompt_cache > 0 && req_ckpt` is false (`:15300`), so the session is neither continued by the
next request (`:12626`) nor parked (`park_current_body` returns at `!live_ok`, `:9497`). **Two things it does not stop:** (1) the conversation that was live when the request *starts* is still
parked (nothing is live at start-up, so nothing happens; after real traffic a `ckpt=0` request would still park the user's conversation); (2) the system-prompt root checkpoint
(`root_at`, `:14037-14041`) is still taken when the first turn boundary lies at `--prompt-cache-root` (2048) tokens or later and is pinned: a 16K warm-up with a long *system* message would pin
one. The warm-up prompts are one user message, so they never reach it (the test checks that each prompt has exactly one `<|im_start|>` besides the assistant header). Also not stopped: the adaptive
expert tier learns from the text (wanted), and a request that resumes from a checkpoint it matches is unaffected. Under `--batch` the `BGEN` path is separate code that I did not trace.

**Parked warm-up conversations and the cache.** `ConversationCache::put` appends; `make_room` and the RAM-admission loop evict `evict_oldest()` first (`entries_` is "least recently active first"),
so the warm-up entries leave before any later one: the first real park finds the cache full and evicts warm-up conversation 1. Each is a short conversation (~100 tokens): one
running-state snapshot plus one turn checkpoint, ~240 MB, the figure the engine log gives for a short park. Nothing later matches them (a checkpoint matches only when its tokens are a prefix of
the new prompt) and a real prompt of 2K+ tokens parks into a slot freed by one of them. What the pre-touch buys is the *first-touch* cost under WSL2: after the evictions the buffers are freed
(16 MiB segments, returned to the guest kernel) but the pages already belong to the VM. The engine's own log lines ("conversation cache: parked N tokens in X ms") show whether it worked.

**Expected log** (times are examples; the 16K and decode figures are the estimates above, not measurements):

```
[strata] warm-up (full): 7 requests (4 conversations to park); the port opens when it is done (the health check says starting until then)
[strata] warm-up 1/7 prefill ~16,376 tokens, 1 new token: 8.1 s (prompt 16,4xx tokens, engine prompt 7.8 s)
[strata] warm-up 2/7 decode ~1,024-token prompt, up to 768 new tokens: 8.6 s (prompt 1,0xx tokens, engine prompt 0.9 s, 768 tokens at 104 tok/s)
[strata] warm-up 3/7 conversation 1 of 4 (kept by the engine): 2.4 s (prompt 1xx tokens, engine prompt 2.1 s)
[strata] warm-up 4/7 conversation 2 of 4 (kept by the engine): 4.6 s ...
[strata] warm-up 7/7 short ~120-token prompt, 16 new tokens: 3.0 s (prompt 1xx tokens, engine prompt 2.8 s, 16 tokens at 90 tok/s)
[strata] warm-up done in 36.4 s
[strata] warm-up, engine log: PCIe share: ... -> pcie_frac 0.60 (was 0.20)
[strata] warm-up, engine log: 4 conversation(s) parked: parked 1xx tokens in 2169.3 ms; ...
```

Between those lines the server also prints its usual `[strata] done: ...` and `request ... server phases: ...` line per warm-up request. If the engine log has no `PCIe share` line the summary says so.
(`LLMSERVE_ENGINE_ENV="STRATA_TRACE=1 STRATA_PREFILL_TIMING=1"` adds the engine's own phase lines to `engine.log`.)

**Not done / limits.** With `--conversation-cache-min-tokens` above ~150 in `LLMSERVE_EXTRA_ARGS` the engine skips parking the short warm-up conversations (`skip parking (.. tokens, below the .. minimum`), so the parks do nothing (set `LLMSERVE_WARMUP_PARKS=0`). After an idle unload (`idle_unload_s`) or an engine restart the engine is cold again and nothing re-runs the warm-up. Replicas are not warmed. The ~30-60 s total is an estimate until it is
run on the 4090. The warm-up heats the tiers and buffers, it does not fix the per-request costs of section 1 (`--short-read`, the turn checkpoint, lend/refill); those stay for the engine work.

**Per-request phases (same change set).** The server now logs `[strata] request <id> server phases: render .. ms, encode .. ms (N tokens from C chars, [K chars reused | R tok/s]), queue .. ms, engine prompt .. ms, first token .. ms
after the request arrived (other .. ms)` and puts `server_render_ms`, `server_encode_ms`, `server_encode_tokens`, `server_prompt_chars`, `server_encode_reused_chars`, `server_queue_ms`, `server_ttft_ms` into `timings`
(and the `/metrics` history rows). That is what the BENCHMARKS.md trace asked for to locate the 2-3 s gap at 16K. `tools/tokenizer_bench.py` measures the tokenizer alone; see the README.
