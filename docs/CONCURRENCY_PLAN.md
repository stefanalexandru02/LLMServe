# LLMServe concurrency vs vLLM: RTX 4090 24 GB (sm_89, PCIe 4.0 x16), Ryzen 9 9900X, Q2_0, Docker Desktop/WSL2

Labels: **[code]** I read it in this repo. **[doc]** a measured number from docs/ or upstream bench results. **[est]** my estimate, not measured.

## TL;DR

- LLMServe already does iteration-level **continuous batching**. With `parallel: N`, up to 8 rows run in one verify window. A request that is alone runs on the fast solo MTP path, and admissions join or leave between windows. It also has a radix-style **prefix cache** that knows about DeltaNet state, plus RAM "parking" of conversations, prefix-aware slot routing, and cooperative preemption of long prompt reads (`BYIELD`).
- Three things are missing compared with vLLM:
  1. **Paged KV.** Each slot reserves the full `--max-context` K/V up front.
  2. **Prefill mixed with decode in one step.** Here the two take turns: one prompt chunk, then the decoders run for 0.5× that chunk's time.
  3. **A multi-branch prefix store** keyed by token blocks. Today there is one live branch, plus slots and parked snapshots.
- On this hardware, batching buys mostly **latency** (no head-of-line blocking for sub-agents). It buys little throughput. Experts that are not cached run on the CPU, so a batch window costs about the **union** of the experts its rows touch.
- The data that matters most is a 4090 measured upstream (IQ3_XXS, 204K context, slots of 2.88 GiB). `parallel: 2` cost 28% single-stream and gave **0%** with 2 clients. It cut wall time 3-4× with 3-4 clients, by removing queueing.
- Q2_0 at 64K costs about 1.0 GB per slot, a third of that, so the trade should be much better here **[est]**.
- Top actions:
  1. A concurrency mode for `llmserve/bench.py`.
  2. Try `parallel 2-3` with `--batch-mtp` and the batch switches (section 4).
  3. Port the "concurrent prompt read" to one GPU. This is the vLLM chunked-prefill benefit, adapted.
  4. Elastic per-slot KV. This gets the paging benefit without block tables.

---

## 1. What exists now

### 1.1 Server: admission, slots, queue (serve/server.py)

- **Dispatch.** `generate()` sends every request to `generate_batched()` when the engine reports slots (server.py:2184-2193). Otherwise it sends `GEN` on the single engine pipe, so requests run one at a time.
- **How `parallel` reaches the engine.** `parallel_args()` turns cfg `"parallel": N` into `--batch N` (server.py:3133-3152). The server then uses the slot count the engine reports (`INFO batch_slots=N`, server.py:1135-1140; generate.cpp:11138).
- **Control lines are a mutex.** Only one request at a time reads its prompt or is admitted (`_take_control`, server.py:1678-1734). `STRATA_ADMIT_SHORT_FIRST=<tokens>` puts short prompts first, with an anti-starvation timeout `STRATA_ADMIT_MAX_WAIT_S` (server.py:319-322, 1736-1749). It is off by default.
- **Solo-path first.** A request that is alone, with no busy slot and nobody waiting, runs plain `GEN` with full MTP (`--spec 4`) (server.py:1800-1808).
  - When a second request arrives, the first is STOPped and resumes in a slot (`BGEN` with prompt + output). The prompt cache makes that resume free (docs/BATCHING.md:161-166).
  - A request left alone in a slot returns to the solo path, at most twice (`_may_go_solo`, server.py:1751-1759).
- **Cooperative preemption.** A long prompt read gives way at a chunk boundary to a waiting prompt under half its length (`_shorter_waiting` server.py:1761; `BYIELD` server.py:1833, 1954; `YIELDS_MAX=2` server.py:1676). The part already read is parked in a slot (generate.cpp:13464-13470, 13531-13564).
- **Prefix-aware routing already exists.** `pick_slot()` (server.py:2124-2167) picks the free slot whose held stream, or whose turn checkpoint at the last `<|im_start|>`, is the longest prefix of the prompt. Otherwise it takes an empty slot, otherwise the least recently used one. The engine then makes its own search across **all** idle slots (below).
- **Queue bounds.** `--max-queue` / `--queue-timeout` (server.py:3936-3938, 8169-8178).

### 1.2 Engine: how slots decode together

- **Per-slot sessions.** `--batch N` allocates N slot sessions. Each holds its own GDN (DeltaNet) recurrent state, QSA K/V and indexer, and PLE history (generate.cpp:5392-5446).
- **Window size.** One verify window carries one row per active slot, or two with `--batch-mtp`: the token plus one MTP proposal. A window holds at most `kVerifyMaxT = 8` rows (verify_kernels.hpp:21). Slots rotate when there are more (generate.cpp:11564-11585).
- **What batching shares.** Dense weights, the shared expert, the head, and every routed expert two rows share are read once per window (BATCHING.md:3-6).
- **What it does not share.** Rows from different conversations route to different experts. A 4-row window on a 12 GB card read 24 CPU experts per layer against about 8 for one row: 54 ms against about 20 ms (BATCHING.md:452-457) **[doc]**.
- **MTP inside batches.** Without `--batch-mtp`, slots decode **without** MTP drafts. That is why the lone request goes back to solo.
- **Adaptation.** The adaptive VRAM expert tier keeps adapting inside batch windows (BATCHING.md:355-366).

### 1.3 Long prefill arriving while others decode

- **One GPU: chunked and time-sliced, never mixed.** Prompt chunks are the normal `--prefill` chunks, so the arithmetic does not change. After each chunk the slots run `batch_step()` for `STRATA_BATCH_DECODE_SHARE × chunk_time`, default 0.5 (generate.cpp:13452-13460, 13941-13947).
- **Effect.** Decoders get about one third of wall time while a prompt is read. The prompt takes about 1.5× as long.
- **Exception.** A prompt shorter than one chunk is read in one piece, and the slots wait (BATCHING.md:429).
- **Admission cost.** Every prompt is read on the **main** session, produces its first token there, and is then copied into the slot (`copy_to_slot`, generate.cpp:11463). The copy goes through host images by default, or device to device with `STRATA_SLOT_COPY_DIRECT=1` (generate.cpp:11424-11431).
- **Truly concurrent reading exists only for a layer split.** `STRATA_PIPED_CONCURRENT=1` reads the prompt on its own thread and stream while windows keep decoding. It is gated on `piped` (generate.cpp:13583-13606), which needs pipelined groups on a layer split. On 16 GB cards it raised the streams' rate during a read from 0.9 to 52.9 tok/s. The joining prompt's TTFT grew 67-109% (BATCHING.md:254-273) **[doc]**.

### 1.4 KV memory per slot

- **Reservation is fixed: no pool, no paging.** Each slot does `cudaMalloc(session_bytes(g, o.max_context, ...))` (generate.cpp:5403-5406).
- **Elastic K/V is shut off with batching.** `--kv-grow`, the VMM elastic K/V, is explicitly disabled when `o.batch != 0` (generate.cpp:5208-5216).
- **What a slot costs:**
  - About 0.56 GiB at 32K with int8 KV (BATCHING.md:145).
  - Setup's formula is `ctx × 12 × 1056 B + 0.17 GB` (prepare.py:423-425): 0.59 GB at 32K, 0.79 GB at 48K, **1.0 GB at 64K**, 1.83 GB at 128K.
  - With `--kv-resident 32768` a slot stays at 0.59 GB of VRAM. The rest of its K/V then sits in pinned RAM.
- **Not here.** The "shared KV pool" mentioned at BATCHING.md:47 belonged to a contributor's private stack. I found no pool in this code **[code: not present]**.
- **Prices at this scale.** The Q2_0 arena is 34 GB for 24,576 experts (catalog.py:224), about 1.38 MB each, so 1 GB of VRAM ≈ 725 cached experts **[est from catalog]**.
- **What this means on the 4090.** The expert cache is roughly 18-19 GB, about 13K experts or 53% of the model **[est]**. Each 64K slot takes about 5% of the cache.

### 1.5 Prefix / prompt caching, and how DeltaNet is handled

**The core constraint.** The KV cache is "one arena that holds one branch of history at a time" (conv_cache.hpp:7-9). GDN/DeltaNet state exists only at the positions where it was snapshotted. Reuse can therefore resume only at a checkpoint, never at an arbitrary common-prefix length, even though the KV cells for every position are present.

**Checkpoints** (`--prompt-cache N`, default 6, about 118 MB of RAM each: GDN + PLE + QSA indexer tails; generate.cpp:894). They form a chain in which each one is a prefix of `live` (generate.cpp:9370-9387). They are taken at:

- the **root**, i.e. the end of the system prompt + tools (the first `<|im_start|>` after token 0). It is pinned and never evicted (generate.cpp:14029-14042; conv_cache.hpp:11-22);
- the last turn boundary (`turn_at`);
- every 16,384 fresh tokens (generate.cpp:751, 925);
- opt-in: the tail (PROMPT_CACHE_TAIL.md), the message boundary `STRATA_CACHE_MESSAGE_BOUNDARY=1` (only for prefixes ≥8K with a tail ≤1K; MESSAGE_BOUNDARY_CACHE.md), and `pin=N` from the API field `strata_prefix` (server.py:1450-1456, 4898-4970).

**Sharing across slots exists.** At each request the engine scans every idle slot's held ids and turn checkpoints, keeps the longest match, and copies that slot back into the main session (`slot_source`, generate.cpp:12650-12682, 12812-12862). Active slots are scanned too with `STRATA_SLOT_CKPT_MOVE=1`.

**Parked conversations are a RAM swap tier.** They store a full snapshot: K/V up to L, plus checkpoints and the draft K/V. The size is set from free RAM: 40% of (MemAvailable − 2.5 GB), capped at 8 GiB, with 4 entries (conversation_memory.hpp:70-78; generate.cpp:9399-9418).

- `best()` matches the prompt against every parked entry (conversation_cache.hpp:251-266).
- If a **new** conversation matches a checkpoint inside a parked one (a sibling sub-agent sharing the system prompt and tools), that checkpoint is **borrowed**. The donor stays parked (conversation_cache.hpp:274-293; generate.cpp:12726-12730).

**Disk.** Only explicit session SAVE/RESTORE (`slot_action`, server.py:4030; `session_file`, server.py:2359). Nothing is persisted automatically.

**Verdict.** This is a single-branch radix cache plus an LRU of branch snapshots. For one agent with one system prompt it already behaves like automatic prefix caching (APC) for the shared prefix.

---

## 2. vLLM comparison

The regime decides what matters. With experts offloaded, a decode window costs about the union of experts its rows touch, missed experts run on 12 Zen5 cores, and prompt chunks stream the missed experts over PCIe at about 25 GB/s.

| vLLM feature | Strata/LLMServe | Status | Matters here? |
|---|---|---|---|
| Continuous (iteration-level) batching | Slots join and leave between windows; one admission at a time; ≤8 rows | **Equivalent** (small N) | Yes, for latency. Throughput gain is small when experts miss: 12 GB 5070 −11 to −24% total (BATCHING.md:440-457); 4090 IQ3_XXS, 2 clients ±0%, 3-4 clients 3-4× wall time from removed queueing (upstream `bench/results/2026-10-06-community-rtx4090-opt-ins`) **[doc]** |
| PagedAttention | Fixed per-slot max-context buffer; `--kv-grow` disabled with `--batch` | **None** | Moderate. VRAM is the expert cache, so every GB reserved but unused costs about 725 cached experts |
| Automatic prefix caching (block hash) | Root-pinned checkpoint chain + idle-slot scan + RAM parking with borrow + `pin=N` | **Partial** (exact token match at checkpoint boundaries; one live branch) | High for agents, and mostly covered already. The gap is several distinct system prompts alternating beyond 4 parked entries, and divergence points with no checkpoint |
| Chunked prefill mixed with decode, token budget per step | Time-sliced: chunk, then 0.5× its time of decode; concurrent read only on a layer split | **Partial** | High for agents: a sub-agent's 20-40K prompt cuts the other streams to about ⅓ speed for 7-16 s. A vLLM-style small budget (e.g. 512 tokens per step) would be **wrong** here, because each chunk re-streams the missed experts over PCIe (SPEED_PLAN.md:7) |
| Preemption / swap | `BYIELD` preempts prompt reads; idle conversations are parked to RAM; nothing evicts a running decode (it is never needed: memory is reserved up front) | **Partial** | Low |
| Spec decoding with batching | Solo path: `--spec 4` MTP. Batch: `--batch-mtp`, 1 proposal per slot, ≤4 slots per window | **Partial** | Yes. +15.4% on 2×3080 (BATCHING.md:57); +31-39% with 2-4 clients on one RTX PRO 5000 (BATCHING.md:30) **[doc]**. Single-GPU 4090 number: none. On 0.1.40 the 4090's admission crashed with an MMVQ MTP pack (upstream report). Not verified on this fork's pack |
| Multi-LoRA | none | none | Not relevant |

**sm_89 note.** `nv_default_env` has tables only for sm_86 and sm_60 (src/core/nv_defaults.cpp:21-46), and sm_120 has its own `sm120_series.hpp` defaults. The 4090 has none.

- None of the **batch-specific** switches (`STRATA_BATCH_ATTN_ROWS`, `STRATA_BATCH_GDN_GROUPS`, `STRATA_QSA_SLOTS`, `STRATA_BATCH_SELF_COMMIT`, `STRATA_SLOT_COPY_*`) is in any table. So sm_89 loses nothing batch-specific compared with other cards. They are opt-in everywhere.
- The sm_86 decode set (`SIDE_MIXER`, `GU_SWIGLU`, `QSA_ROWS`, ...) is missing on sm_89. Whether those kernels also run in batch windows is **[unverified]**. Note that `STRATA_ADAPT_LAG` explicitly does not apply to batch paths (generate.cpp:1538).

---

## 3. Ranked proposals

Workloads: **(a)** one agentic coding tool. It has a long system prompt + tool definitions (15-30K tokens) and sometimes 2-4 parallel sub-agents. **(b)** 2-3 users chatting.

### P1. `llmserve/bench.py --concurrency` (S, no risk). Do this first; everything below needs it.

**CLI:**

```
--concurrency 1,2,3,4   --scenario independent|shared-prefix|agent|join   --prefix-tokens 20000
--max-tokens 512 --rounds 3 --stagger-ms 0
```

**Clients.** Use `threading.Thread`, one `Client` per thread with its own `http.client` connection. All threads are released by one `threading.Barrier` (plus `i*stagger`). Each request is streamed with `stream_options.include_usage`.

**Scenarios:**

- `independent`: distinct prompts from `make_prompt(case, nonce)`, with the nonce at the start, so nothing is reused.
- `shared-prefix`: an identical synthetic system message of `--prefix-tokens` (`build_document`), then a per-client nonce + task in the user message. This measures APC-like reuse.
- `agent`: one "main" stream that runs 3 turns (each turn's history = the previous prompt + its answer). From turn 2, it fans out N−1 sub-agent requests that share the system prefix.
- `join`: N−1 streams decode a long answer (`--max-tokens 2048`). At t = 3 s one cold request of `--prefix-tokens` joins.

**Per request, from the client clock:**

- TTFT;
- per-stream decode tok/s = (tokens − 1) / (t_last − t_first);
- the **longest gap between tokens** (the stall metric for `join`);
- `usage.prompt_tokens_details.cached_tokens` / `timings.cache_n` (the reused tokens).

**Per level:**

- aggregate tok/s = Σ completion tokens / (last token of the level − barrier release);
- TTFT p50/p95 (nearest rank);
- per-stream tok/s p50/min;
- mean reuse %;
- error count.

**Also record** `GET /v1/status` → `concurrency.serving`, and `/metrics` `live.slots` at mid-level, to confirm the slots were really used.

**Output.** The JSON gets `{"mode": "concurrency", "levels": {N: {...}}}`. `--compare` prints the % change per level and metric. Temperature 0 and seed 1, as now. Note: BDONE figures override timings (server.py:1565-1569), so per-stream engine timings may be missing; rely on the client numbers.

### P2. Settings + prepare.py support (S, low risk). Expected: (a) large latency win, (b) moderate.

prepare.py today:

- recommends **0** at 64K on a 24 GB WSL card. `parallel_recommend(23.99, 34, 65536, int8, streaming=False)` returns 0 because 2 slots leave 16.99 < 17 GB "held". That is a knife-edge.
- recommends 2 at 48K and 3 at 32K, or 3 at any context with KV streaming (I ran prepare.py:423-435);
- warns "10-25% slower" (prepare.py:709-717).

Changes:

1. Add `LLMSERVE_BATCH_MTP=on|off` → append `--batch-mtp` when parallel ≥ 2 and MTP is on.
2. Add `LLMSERVE_BATCH_SWITCHES=on` → cfg `env`: `STRATA_BATCH_ATTN_ROWS=1` (only without KV streaming), `STRATA_BATCH_GDN_GROUPS=1`, `STRATA_SLOT_COPY_DIRECT=1`, `STRATA_SLOT_COPY_BATCHSYNC=1`.
3. Add `LLMSERVE_AGENT=on` → `STRATA_CACHE_MESSAGE_BOUNDARY=1`, `STRATA_ADMIT_SHORT_FIRST=4096`, and an explicit `--conversation-cache-mib` sized from `ram − q.ram_gb − 4`. Do not trust the auto rule on a 58 GB VM.
4. Rewrite the warning text so it reflects the trade on the card at hand. On a 24 GB card it is about 5%/slot, not 10-25%.
5. Echo the slot VRAM cost; this already exists (prepare.py:712).

Tests go in llmserve/test_prepare.py.

### P3. Single-GPU concurrent prompt read: vLLM's "decodes don't stall" (M-L, medium-high risk). Expected: (a) streams keep about 60-80% speed during a sub-agent's prompt read, instead of about 33% **[est from the split numbers]**; the joining TTFT gets worse by +20-60% **[est]**; (b) small.

**Why it fits this regime.** Prefill is GPU and PCIe bound: it streams the missing experts. Decode windows lean on the CPU pool for misses and launch small kernels. The two are complementary, unlike vLLM's compute-bound mix.

**Design:**

- Lift the `piped` gate at generate.cpp:13606 for `stages.empty()` and run the existing `pr_conc` reader thread.
- Use the prompt's own stream at low priority, and keep the windows on a high-priority stream (the CUDA path already exists for splits; BATCHING.md:197-221).
- **Borrowed VRAM slots.** These are lent before the read and refilled after (generate.cpp:13568-13576). The single-GPU prompt borrows a larger share of the cache, so windows would miss more (CPU) during reads.
- **Hazards:** `adapt_tick` must freeze during the read (already does on splits); PLE table locking; the MTP drafter's prompt buffers (`mtp.reserve_prefill`); doorbell threading (#31 hang class).
- **Fallback.** A cheap partial step is a server/env default of `STRATA_BATCH_DECODE_SHARE=1.0` when parallel ≥ 2: the streams then get 50% instead of 33%. It is a pure config change; measure it first.

**Measure** with `bench --scenario join` (longest gap, stream tok/s during the join, joiner TTFT), plus `tools/batch_interleave_test.py`-style token equality (not in this fork: copy from upstream).

### P4. Elastic per-slot K/V: paging without block tables (M, medium risk). Expected: +2-6% single-stream decode with 3 slots at 64K **[est]**.

- Allow `--kv-grow` with `--batch`. Back each slot session's QSA K/V with its own VMM range, reserved for `max_context` and mapped in 16K-cell steps (the `kvg_ensure` path, generate.cpp:7858+). Unmap on BDONE/eviction down to the held length.
- Remove `o.batch == 0` from the gate (generate.cpp:5211).
- The expert cache already gives slots back under VMM (`ExpertCache::set_vmm`), so the cache automatically holds about 725 experts per GB the slots do not use.
- **Gain example.** Sub-agents at about 20-30K tokens instead of 64K reserved would free about 1.2-1.5 GB at 3 slots.
- **Alternatives:**
  - `--kv-resident 32768` on WSL (SPEED_PLAN B1): fixed 0.59 GB per slot. Simpler, untested under Docker Desktop.
  - A real shared block pool (QSA kernels + indexer + graph capture keyed on the per-row layout): **L**, high risk. Not worth it at N ≤ 4.

### P5. Multi-branch prefix store, an "APC-lite" (M, medium risk). Expected: (a) 5-12 s TTFT saved per cold sub-agent **only** when its prefix has fallen out of the 4 parked entries or the root chain; otherwise 0.

**First verify the gap.** Run `bench --scenario agent` and read the engine log's `prompt N tokens = R reused`. The root pin (generate.cpp:14029), the idle-slot scan and borrow already cover "N sub-agents share a system prompt + tools".

**If there is a gap:**

- Add a `PrefixStore` in conversation_cache.hpp holding entries `{hash(ids[0:L]) in 1024-token blocks, KV[0:L) host image, GDN/PLE/tails snapshot at L}`.
- Fill it at root/pin checkpoints from `checkpoint_at` (generate.cpp:9727).
- Look it up in `best()` before parking.
- Use LRU by bytes in pinned RAM.
- Block hashing only speeds up lookups. The real constraint is that DeltaNet state is needed **at L**, so entries can only be created at checkpoint boundaries, never at arbitrary block edges.
- **RAM is the limit.** The WSL VM has 58 GB and Q2_0 asks for 48. A 30K prefix costs about 0.4 GB of KV + 0.12 GB of state.

### P6. Admission tweaks (S, low risk)

Routing to the best-matching slot **already exists** (server.py:2124-2167; generate.cpp:12656-12682).

- Remaining tweak: `STRATA_ADMIT_SHORT_FIRST=4096`. This lets a sub-agent's short follow-up go before a 30K cold read. Today the order is first come, apart from `BYIELD`.
- `STRATA_SLOT_CKPT_MOVE=1` + `STRATA_CKPT_POOL=1` cut the 112 MB checkpoint copy on each admission (BATCHING.md:116-119).
- Gain: TTFT for short requests under load **[doc: 16-user HIP split, TTFT 5.4 → 2.0 s with all switches]**.

### Not recommended

True mixed prefill+decode in a single forward with a small token budget (L, high risk). Each chunk re-streams the non-resident experts (about 15 GB of Q2_0 at about 25 GB/s ≈ 0.6 s **[est]**), so small chunks collapse prompt speed. P3 gets the stall benefit without that cost.

---

## 4. Settings to try today (RTX 4090 24 GB, Q2_0, WSL2)

**How settings flow.**

- `LLMSERVE_PARALLEL` → `--parallel` → cfg `"parallel"` → the server adds `--batch N` (prepare.py:704-717, 750-751; server.py:3133).
- `LLMSERVE_EXTRA_ARGS` is merged last, and it wins (prepare.py:699-702).
- `LLMSERVE_ENGINE_ENV` → cfg `env`. Any `STRATA_*` line in `.env` also reaches the server *and* the engine: compose `env_file` (docker-compose.yml:26), and `child_env` copies `os.environ` (server.py:3480-3491).
- On WSL, prepare never writes `--kv-resident` (prepare.py:663-666).

**VRAM per slot** (int8 KV, no streaming; prepare.py:423-425): 32K 0.59 GB, 48K 0.79 GB, **64K 1.0 GB**, 128K 1.83 GB. Each GB is about 725 fewer cached experts, about 5.5% of the cache. `--batch-mtp` adds a slot drafter per slot: about 50 MiB on a split (BATCHING.md:59), single-GPU size **[unverified]**. The engine prints the exact cost at start (generate.cpp:5422, 5443).

| Arm | `.env` | Slot VRAM | Expected **[est]** |
|---|---|---|---|
| A (baseline) | `LLMSERVE_PARALLEL=1` | 0 | 110-131 tok/s solo; a 2nd agent waits for the whole 1st answer |
| B | `LLMSERVE_PARALLEL=2`, `LLMSERVE_EXTRA_ARGS="--batch-mtp"`, `LLMSERVE_ENGINE_ENV="STRATA_BATCH_GDN_GROUPS=1 STRATA_BATCH_ATTN_ROWS=1 STRATA_SLOT_COPY_DIRECT=1 STRATA_SLOT_COPY_BATCHSYNC=1 STRATA_BATCH_DECODE_SHARE=1.0"` | 2.0 GB @64K | solo −4 to −8%; 2 streams aggregate −5% to +25%; TTFT of the 2nd stream: seconds instead of a full answer |
| C | B + `LLMSERVE_PARALLEL=3`, `LLMSERVE_CONTEXT=49152` | 2.4 GB | for 3 sub-agents; solo −6 to −10% |
| D | B with `LLMSERVE_PARALLEL=4`, `LLMSERVE_CONTEXT=32768` | 2.3 GB | 4 short sub-agents; `--batch-mtp` fits ≤4 slots per window (8 rows) |
| E (try later) | B + `--kv-resident 32768` in EXTRA_ARGS (64K kept, 0.59 GB/slot, +0.9 GB pinned RAM per slot) | 1.2 GB | SPEED_PLAN A2. Untested on Docker Desktop; if the pin is refused the engine exits. Do not combine with `STRATA_BATCH_ATTN_ROWS` (BATCHING.md:485) |

**For agents, add** `STRATA_ADMIT_SHORT_FIRST=4096` and `STRATA_CACHE_MESSAGE_BOUNDARY=1` as `.env` lines, and `--conversation-cache-mib 4096` in EXTRA_ARGS.

**Check the log for:**

- `--batch: N slot sessions ... GiB each` and `--batch-mtp: N slot drafters`. If you see `MTP admission ... failed`, drop `--batch-mtp`.
- `expert cache N slots`
- `conversation cache sized ...`
- `strata batch:` window lines (rows, ms, VRAM/CPU share)
- `prompt N tokens = R reused`

**A/B method.** Restart per arm in A, B, A, B order, with ≥3 rounds each:

```
python -m llmserve.bench --concurrency 1,2,3 --scenario shared-prefix
python -m llmserve.bench --concurrency 1,2,3 --scenario join
```

Keep a setting only if the 1-client loss is under about 8% **and** 2-3-client TTFT p95 or aggregate improves. Also try the sm_86 set (SPEED_PLAN A6) as a **separate** arm: it targets the solo decode path, and its effect in batch windows is unknown.
