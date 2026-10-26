# Engine changes in this fork (C++/CUDA)

Changes to the engine itself (`src/`, `include/`, `CMakeLists.txt`), made for an RTX 4090 (sm_89) on Docker Desktop / WSL2.
Each entry says why (with the measurement), how to turn it off, and what to look for in the engine log
(`docker exec llmserve grep -E "<pattern>" /data/logs/engine.log`).

Status: all four compile cleanly (CUDA 13.0.2, `-DCMAKE_CUDA_ARCHITECTURES=89-real`, 8 min 42 s at `-j4`, no new warnings) and
`strata --help` starts. None of it has run on a GPU yet. The one piece that can run without a GPU, the conversation pool,
has a host test (see 2). Everything below that says "expected" is expected, not measured.

## 1. sm_89 (Ada) default speed set

**What.** `src/core/nv_defaults.cpp`: a card of compute capability 8.9 gets the same nine switches the engine already turns on by
itself for sm_86 cards: `STRATA_ADAPT_NOWAIT=1 STRATA_ADAPT_LAG=2 STRATA_CPU_IQ_ALL_EXACT1=1 STRATA_SIDE_MIXER=2 STRATA_ATTN_DAILY=1
STRATA_GR_UP75=1 STRATA_GU_SWIGLU=1 STRATA_QSA_ROWS=1 STRATA_B6_MMVQ_ROWS=1`. A switch you set yourself (to anything, including 0)
is never overridden.

**Why.** [BENCHMARKS.md](BENCHMARKS.md), one RTX 4090 + Ryzen 9 9900X under WSL2, engine 0.1.51, Q2_0, MTP on, 1 engine start
per arm, 3 runs per case: with the set, decode was +8% / +13% / +25% at ~124 / 4.3K / 16.9K prompt tokens (set alone) and
+7% / +13% / +31% (set + `STRATA_CKPT_POOL=1` + `--short-read 192`), prompt speed unchanged (+1% / -2%).

**Not yet known.**
- Output identity on sm_89. On sm_86 the set was gated IDENTICAL against the default; the 4090 bench compared speeds only.
  Until a token-identity check is run (same prompts, temperature 0, with and without `STRATA_SM89_PERF=0`), treat the log
  line below as "faster, probably the same tokens".
- The short-prompt TTFT was *worse* with the set in that run (2.20 s -> 3.20 s). The trace in BENCHMARKS.md attributes most of the
  short-prompt time to parking the previous conversation (item 2), not to the switches, but that was not separated.
- One card, one start per arm. Keep the opt-out handy.

**Left sm_86-only on purpose** (nothing was measured on Ada for them): `dev_class()` in `native_mmvq.cu` (one-row Q8_0 MMVQ
tables), the wide-dequant choice in `dequant_bf16.cu`, and `apply_spec_adapt_default` (`STRATA_SPEC_ADAPT_SM86`, which is opt-in
even on sm_86 since the owner's round 9 A/B did not reproduce its gain). `arch_defaults.cpp` is keyed on AMD `gcnArchName`
and is never consulted by a CUDA build, so it does not interact.

**Turn off.** `STRATA_SM89_PERF=0` (via `LLMSERVE_ENGINE_ENV`). If you had put the nine variables in `LLMSERVE_ENGINE_ENV` for
the benchmark you can remove them; leaving them is harmless (your setting is kept).

**Look for** (start-up):

```
strata: sm_89 (Ada): 9 (output identity not yet verified on Ada) speed switches on by default (STRATA_SM89_PERF=0 turns them off; a switch you set is kept): ADAPT_NOWAIT ADAPT_LAG ...
```

The number is lower than 9 for each of the nine you set yourself. The line is absent with `STRATA_SM89_PERF=0` and on other
cards (sm_86 prints `sm_86 (Ampere)`, sm_60 `sm_60 (Pascal)`; the old code printed "sm_86" for every compute capability 8.x).
Also the engine log says these switches turn off "adaptive overlap"; that is `STRATA_ADAPT_NOWAIT/ADAPT_LAG`, and is what was measured.

## 2. Conversation cache: no first-touch allocation on the request path

**What.** Parked conversations are copied into `ConversationBuffer` segments (<= 16 MiB). Before, each segment was its own heap
allocation, freed to the kernel on eviction, so every park faulted fresh pages in. Now (Linux):

- `include/strata/core/conversation_pool.hpp`, `src/core/conversation_pool.cpp`: one `mmap`'d region of address space
  (cache budget + 25%, at most +1 GiB, untouched pages cost nothing) cut into 256 KiB slots. A segment of >= 1 MiB takes a run of
  slots (a fully resident run preferred); a freed segment gives its slots back *with the pages still resident*. Segments below
  1 MiB, and any segment the region cannot serve, are plain heap allocations as before.
- A background thread (`conv-prefault`, `SCHED_IDLE` + nice 19, 1 MiB at a time, 0.3 ms apart, `MADV_POPULATE_WRITE` with a
  write-zeros fallback) makes the first K MiB resident at start. It claims only free slots, so a park never waits for it. It
  stops by itself if available RAM falls under the parking floor + 512 MiB.
- `ConversationBuffer` (`conversation_buffer.hpp`) holds its segments as a small block type instead of `std::vector<uint8_t>`;
  with the pool off it is `operator new`/`delete` of exactly the requested capacity, as the vector was.
- **RAM admission stays correct.** Evicting a parked conversation now returns RAM to the pool, not to `MemAvailable`, so
  `conversation_memory_admit()` (`conversation_memory.hpp`) first subtracts the pool's free *resident* bytes from the
  allocation and judges only the remainder against `MemAvailable - floor`; an allocation the pool covers whole needs no free
  RAM. Capacities are rounded up to 256 KiB while the pool is on, and the admission estimate (`allocation_peak`) rounds the
  same way. When a request finds no run (full or fragmented) the pool first hands *all* its idle resident slots back to the
  kernel (`madvise(MADV_DONTNEED)`) and then allocates from the heap, so the process never grows by more than the admission
  assumed (growth <= allocation - credit). The "RAM floor after capture" check is skipped only when the pool covered the
  whole snapshot (nothing of ours was requested from the OS).
- Cost: the pool keeps its high-water mark of conversation memory (up to the cache budget, which the cache could hold anyway)
  resident instead of returning it to the VM after each eviction. The default pre-fault is `min(budget, 1024)` MiB.

**Why.** BENCHMARKS.md ("where the short-prompt TTFT goes"): parking the previous conversation took 2.2-2.5 s for ~240 MB while
the cache's 4 places were first being filled (first-touch allocation in the WSL2 VM at ~100 MB/s) and 34-207 ms once it reused
memory. It is counted in the next request's prompt time, ~2 s of a ~2.8 s short-prompt TTFT.

**Settings.**

| | |
|---|---|
| `STRATA_CONV_POOL=0` | pool off: plain heap segments, behaviour as before this change |
| `STRATA_CONV_PREFAULT_MIB=N` | pre-fault N MiB in the background at start (default `min(cache budget, 1024)`; `0` = none, pages are then faulted at the first park and kept) |

**Look for.**

```
strata serve: conversation pool: 5120 MiB of address space reserved for parked conversations (256 KiB slots); pre-faulting 1024 MiB in the background (...)
strata serve: conversation pool: pre-faulted 1024 MiB in 1.8 s (570 MB/s)          <- seconds after start; at ~100 MB/s it takes ~10 s
```

and after each park (the existing `conversation cache: parked N tokens in X ms` line is unchanged), a pool line:

```
strata serve: conversation pool: 240 MiB in use, 780 MiB free and resident, ... ; 61 segments from the pool, 0 from the heap, 0 trims; pre-fault done
```

Expect the first park after start to be fast (tens to a few hundred ms instead of 2.2-2.5 s) as long as the pre-fault had finished and
the snapshot fits in it. "from the heap" > 0 together with `no free run of N MiB ... heap instead` lines means the region was full or
fragmented (the cache budget is smaller than what is being parked, or checkpoints/disk sessions hold segments too); the
pool then hands its idle memory back, which is safe but gives up the benefit for that park. `conversation pool: off` or
`not available on this system` means the plain path is in use.

**Tested** (no GPU): `src/core/conversation_pool_test.cpp`, standalone, not in CMake:
`g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -Iinclude src/core/conversation_pool_test.cpp src/core/conversation_pool.cpp -pthread && ./a.out`.
It covers: pool-off capacities equal the estimate; pre-fault accounting; admission credit; content integrity through resize,
shrink, copy and move; region exhaustion; the fragmentation trim; 6 threads allocating and freeing; and that a second park
takes no new RSS while a trim returns it (cold 74 ms vs warm 51 ms for 48 MiB on this sandbox, where faults are cheap; the VM
case is the 100 MB/s one).

## 3. `--kv-resident`: fall back instead of exiting when the pin is refused

**What.** `src/program/generate.cpp` (session creation) and `src/core/layer.cpp`. With `--kv-resident N` the K/V host copy is pinned
(`cudaHostRegister` under `/dev/dxg`, else `cudaHostAlloc`). If that is refused, `session_init` failed and the engine exited
with `session_init failed`. Now, unless `STRATA_KV_RESIDENT_STRICT=1`, it logs why, releases the copies the earlier layers did
get (`qsa_kv_host_release()`: they were never freed), the half-built session and its rope-table registration, sets
`--kv-resident` to 0 for the rest of the run, and builds the session again with the whole K/V in VRAM, as if the flag had not
been given. Streaming counts as requested only when the resident cells are fewer than the context (as `kv_plan` decides).

**VRAM accounting.** The expert cache is sized after the session from `cudaMemGetInfo`, so it sees the larger resident session on
the fallback path by itself. `qsa_set_kv_resident(0)` makes every later `session_bytes()` / `kv_plan()` (the `--batch` slots, the
drafter, later stages, the "no VRAM left" message) plan the resident layout, and `o.kv_resident = 0` keeps the conversation
cache's config fingerprint and the `INFO ... kv_resident=` line truthful. If the resident session does not fit in the free VRAM the
engine says how many GiB it needs and how many are free, and exits.

**Not covered.** A layer split (`--layer-split`) priced its stages with the streamed session; if CUDA0 falls back, a later
stage whose resident session no longer fits still exits with `layer split, CUDAn: the session state failed`. The drafter already
had its own fallback (`no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM`). `--kv-grow` stays off after a fallback.

**Turn off.** `STRATA_KV_RESIDENT_STRICT=1` restores the old behaviour (exit with `session_init failed`).

**Look for.** On success, unchanged: `KV streaming: the host copy is pinned with cudaHostRegister` and
`KV streaming: N of M cells per QSA layer in VRAM, the K/V in X GiB of pinned RAM`. On a refused pin:

```
strata: KV streaming: cannot pin ... GiB of RAM for a layer's KV copy ...
strata generate: WARNING: KV streaming (--kv-resident 32768) could not start: the pinned RAM for the K/V host copy was refused (X GiB pinned before the refusal).  Falling back to the whole K/V in VRAM ...
strata generate: KV streaming off: X GiB of half-pinned host copies released; the session now takes Y GiB of VRAM (Z GiB free)
```

(no `KV streaming:` line follows, and the `INFO` line reports `kv_resident=0`). This also lets the config in SPEED_PLAN A2/B1 be
tried on WSL2 without a risk of a start that does not come up.

## 4. Build

`strata` and `strata_kq` build with the Dockerfile's recipe for `89-real` (CUDA 13.0.2, cmake 4.4.3, ninja 1.13.2,
llama.cpp `3cf03257f2`): 164 steps, 8 min 42 s wall at `-j4` on 4 cores / 15 GB. The only warnings are the existing ones
(`native_mmvq.cu`, `iq_kernels.cu`, `layer.cpp` misleading indentation / unused variables, `generate.cpp:12068`); none are in code
these changes added. `src/core/conversation_pool.cpp` is a new line in the `strata_engine` source list in `CMakeLists.txt`.
