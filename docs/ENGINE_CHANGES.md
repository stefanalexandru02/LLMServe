# Engine changes in this fork (C++/CUDA)

Changes to the engine itself (`src/`, `include/`, `CMakeLists.txt`), made for an RTX 4090 (sm_89) on Docker Desktop / WSL2.
Each entry says why (with the measurement), how to turn it off, and what to look for in the engine log
(`docker exec llmserve grep -E "<pattern>" /data/logs/engine.log`).

Status: both pieces that remain (sections 1 and 3) compile cleanly (CUDA 13.0.2, `-DCMAKE_CUDA_ARCHITECTURES=89-real`; details in
section 4) and `strata --help` starts. Neither has run on a GPU yet. Everything below that says "expected" is expected, not
measured.

An earlier version of this change also added a resident conversation-cache segment pool; it was removed after review, see
section 2.

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

## 2. Conversation cache segment pool: removed

An earlier version of this change replaced the heap segments of the conversation cache (`ConversationBuffer`) with slots of one
`mmap`'d region whose freed pages stayed resident, pre-faulted in the background, to remove the 2.2-2.5 s first-touch cost of the
first parks under WSL2. A code review found OOM risks in its RAM admission accounting, so **the pool was removed entirely** and
conversation buffers and RAM admission are exactly as before it (`conversation_buffer.hpp`, `conversation_memory.hpp`,
`CMakeLists.txt` and the conversation-cache code in `generate.cpp` are identical to the commit before the pool; its sources and
the `STRATA_CONV_POOL` / `STRATA_CONV_PREFAULT_MIB` settings are gone). The problem, in short: admission subtracted the pool's free
resident bytes from every allocation, but

- the pool could serve a request from cold slots while the credited resident holes stayed resident and stayed credited, so the
  process grew by the whole allocation although admission assumed it would not;
- a large part of every park and of every session save/read (the GDN/PLE running state and checkpoint copies, segments under
  1 MiB, the save's write buffer) is plain heap memory the pool never serves, yet the credit was applied to it too;
- when the credit covered the request, admission returned true with no `MemAvailable` check, and the "RAM floor after capture"
  check was skipped as well, also on a reuse-park that needed no new K/V at all;
- evicted K/V went back to the pool and not to the kernel, so the eviction loops could no longer raise `MemAvailable`.

**What addresses the first-park cost instead.** The start-up warm-up now parks a few short conversations before the port opens
(`LLMSERVE_WARMUP_PARKS`, default 4; [TTFT_WARMUP.md](TTFT_WARMUP.md)), which takes the first-touch faults of the heap segments
at start-up instead of on a request's path. Like the pool's pre-fault this only helps while the freed memory has not been
returned to the VM, which the plain heap does not guarantee; TTFT_WARMUP.md says what is estimated and what is measured.

## 3. `--kv-resident`: fall back instead of exiting when the pin is refused

**What.** `src/program/generate.cpp` (session creation) and `src/core/layer.cpp`. With `--kv-resident N` the K/V host copy is pinned
(`cudaHostRegister` under `/dev/dxg`, else `cudaHostAlloc`). If that is refused, `session_init` failed and the engine exited
with `session_init failed`. Now, unless `STRATA_KV_RESIDENT_STRICT=1`, it logs why, releases the pinned memory of the half-built
session (below), its rope-table registration, sets `--kv-resident` to 0 for the rest of the run, and builds the session again
with the whole K/V in VRAM, as if the flag had not been given.

**When the fallback applies (narrowed after review).** Only when `qsa_state_init` reports `QsaInitFailure::KvHostCopy`
(`layer.hpp`): `kv_host_alloc` was refused both ways (`cudaHostRegister` and `cudaHostAlloc`) for a layer that `kv_plan` really
streams. That is decided in the layer, not guessed from `--kv-resident` in `generate.cpp`, so a context that fits in the resident
cells, or `STRATA_KV_MAIN_OFF` (which makes `kv_plan` keep every layer resident), can never look like "streaming was asked". Any
other `session_init` failure is `QsaInitFailure::Other`: the pinned `host_step`/`host_pos` staging refused,
`cudaHostGetDevicePointer` on the host copy failed, or the elastic K/V could not be mapped. Without streaming it would fail the
same way again, so the fallback is not tried: `qsa_state_init` names the real cause on stderr, `generate.cpp` prints
`session_init failed (not a refusal to pin the K/V host copy ...; see the line above)` and exits as before.

**What is released, and only that.** Each streamed `QsaState` records its own K/V host copy (`kv_host_p/bytes/registered`);
`qsa_state_release_host(QsaState&)` gives back that state's copy (`cudaHostUnregister` + `munmap`, or `cudaFreeHost` through the
pin budget) and its pinned `host_step`/`host_pos` staging, and touches nothing else. `qsa_state_init` calls it for its own state
when it fails (so the failing layer leaks nothing), `session_release_pinned(SessionState&)` calls it for every state a session
built, and the fallback uses that. The earlier version of this change kept a process-wide list of copies and freed all of them,
which would have unmapped memory that other live states (a later stage, a `--batch` slot, the drafter) still use as soon as a
second caller existed; the staging buffers of the half-built states were not freed at all. The drafter's own fallback in `mtp.cpp`
(which calls `qsa_state_init` too) stops leaking its first attempt's staging for the same reason: the failed call releases it.

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

(no `KV streaming:` line follows, and the `INFO` line reports `kv_resident=0`). An unrelated `session_init` failure instead
prints its own cause (`the pinned step/position staging buffers were refused (not the KV host copy)`, or
`cudaHostGetDevicePointer on a layer's KV host copy failed: ...`) followed by `session_init failed (not a refusal to pin the K/V
host copy, so the --kv-resident fallback does not apply; ...)`. This also lets the config in SPEED_PLAN A2/B1 be
tried on WSL2 without a risk of a start that does not come up.

## 4. Build

`strata` and `strata_kq` build with the Dockerfile's recipe for `89-real` (CUDA 13.0.2, cmake 4.4.3, ninja 1.13.2,
llama.cpp `3cf03257f2`): 163 steps, 7 min 27 s wall at `-j4` on 4 cores / 15 GB, and `strata --help` prints its usage in the CUDA
container (no GPU there). The only warnings are the existing ones (`native_mmvq.cu`, `iq_kernels.cu`, `layer.cpp` misleading
indentation / unused variables, `generate.cpp` unused `w`); none are in code these changes added or touched.
