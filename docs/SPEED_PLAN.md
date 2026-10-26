# LLMServe speed plan: Q2_0 on a 16 GB card, Docker Desktop / WSL2

## 1. Diagnosis: where the time goes

**Measured:** prompt 28,340 tok in 20 s (1,378 tok/s). Decode 23,022 tok in 203 s (113 tok/s), with the context growing from 28K to 51K.

**Decode is already ahead of Strata's own table.** The table shows Q2_0 at 81.8 tok/s at 32K and 76.2 at 64K (DETAILS.md:65-71). That was a 12 GB RTX 5070 on engine 0.1.26. This fork is newer (route tail skip measured +13% on an A4000 with Q2_0, DETAILS.md:1257-1258; PCIe share auto, ibid.) and has about 4 GB more VRAM, which is about 2,800 more cached experts at ~700/GB (DETAILS.md:85). Thinking text also gets high MTP acceptance. Decode headroom is about +5-25%. It depends on the CPU (cache misses run on it) and on the VRAM levers below.

**Prompt speed is the gap.** Strata's table gives 2,171 tok/s at 32K. Interpolated to 28K that is about 2,050, so the user is about 33% below it. A 28K prompt is read in 4 chunks of up to 8,192 tokens, about 5.8 s per chunk. Each chunk touches nearly every expert, so about 26-28 GB of Q2_0 experts that the cache does not hold cross PCIe per chunk. Prompt speed is therefore bound by the larger of GPU MMQ compute and H2D bandwidth. The likely causes, in order:

1. **The GPU class (most likely).** A 4060 Ti 16G or 5060 Ti has 34-36 SMs on a PCIe x8 link; the 5070 has 48 SMs. The community 5060 Ti on Gen3 x8 got 1,549 tok/s at 32K with Q2_0 (upstream `bench/results/2026-10-09-community-rtx-5060-ti-e5-2680-v4`). 1,378 tok/s at 28K is in the same family. On a 4070 Ti S, 5070 Ti, 4080 or 5080 the expected figure is 2,500-3,500. Community examples: 5070 Ti IQ3_S 3,487 at 41K; 4070 Ti S about 2,500 (`bench/results/COMMUNITY.md`). On those cards, 1,378 points to a fault.
2. **The PCIe link.** Gen or width can be downgraded, as on two 5080s running at x8 in community reports.
3. **The prompt chunk.** It may have shrunk below 8192 because the VRAM is full.
4. **n-gram (PLE) row reads.** These are `--ple-io direct` random reads from the WSL2 vhdx.

**WSL2 itself costs little DMA bandwidth from registered memory.** The startup probes measured 28.9 GB/s on a 5080 at Gen5 x8 and 14.1 GB/s on a 5060 Ti at Gen4 x8, both about 90% of the link (`bench/results/2026-10-07-community-rtx-5080-docker-wsl2`, `…-5060ti-wsl2-iq3_s-0141`). A suggestive but confounded comparison: a 5080 on Windows native (Coder IQ1_M, 9900X3D) got 3,175-4,001 / 86-117 tok/s, against 3,000-3,300 / 72-80 tok/s for a 5080 under WSL2 with a 9950X. That hints WSL costs 10-20%, but the CPU, context, KV streaming setting and engine version all differ.

**Assumptions:** the shipped expert profile is in use, vision is off, the GPU may drive the Windows desktop, and the CPU model is unknown.

## 2. The WSL KV-streaming restriction is stale, but small at 64K

- **The engine already handles WSL2.** `kv_host_alloc()` (src/core/layer.cpp:576-626) registers anonymous memory Portable|Mapped with `cudaHostRegister` when `/dev/dxg` exists, and falls back to `cudaHostAlloc` (`STRATA_KV_HOST_REGISTER=1|0` forces the order).
  - Commit c41c0ebb ("KV streaming under WSL", 2026-10-07) measured this on WSL2 with an RTX 5090: "the K/V in 12.38 GiB of pinned RAM", decode 99-124 tok/s at 0.5-1M context. Its message says: "Not tested: … Docker Desktop. setup.py still never writes --kv-resident on WSL."
  - Our prepare.py copied that ban: prepare.py:535-536 and 663-666, and `LLMSERVE_KV_STREAMING=on` is ignored on WSL.
  - The docs repeat the stale "1 GB" claim: DETAILS.md:204-206 and 216, WINDOWS_DOCKER.md:55, and .env.example:44. DETAILS.md:206 also says experts are "copied from unpinned RAM" on WSL, which is wrong: the arena logs `cudaHostRegister PORTABLE ok` (pinned.cu:449; 47 GiB pinned in a community WSL run).
- **What it buys.** About 13.7 KB of KV per token (prepare.py:375-377). At 64K with `--kv-resident 32768` it frees only ~0.45 GB of VRAM, about +300 slots on a cache of ~5,500.
  - Upstream measured +6% decode at 128K and +23% at 262K with Q2_0 (upstream setup.py:9596). At 64K expect roughly +1-3%.
  - Cells beyond 32K are then read over PCIe; the engine logs `KV streaming: X% of block reads hit VRAM`.
  - The tuner forbids other values below the context: 22-34% slower (tune_knobs.py:164). 32768 is setup's own value and is exempt (tune_knobs.py:297-301).
- **It is safe to enable, but a failed pin is fatal today.** If both allocation methods are refused, `qsa_state_init` returns 0 (layer.cpp:905-916) and the engine exits with "session_init failed" (generate.cpp:~5225). It needs a fallback (code change B1).
- **Real value:** it makes 128K-262K context usable on WSL without giving up experts. At 64K it is a minor gain.

## 3. Every `under_wddm()` / `/dev/dxg` gate, judged

| Site | Behaviour under WSL2 | Verdict |
|---|---|---|
| generate.cpp:4101 | Skips the rotational-disk warning, because the vhdx reports itself as rotational | Correct |
| generate.cpp:5818-5831 | Caps arena pinning at 8 GiB, but only for multi-GPU or remote experts | Not hit with one GPU; fine |
| layer.cpp:595-599 | Registers the KV host copy first | Correct (it is the WSL fix) |
| prepare.py:535, 663 (not the engine) | Bans `--kv-resident` and `--ple-io ram` | KV ban stale. `ple-io ram` correctly off: RAM is short (tuner `ram_bound` −61%, tune_knobs.py:162) |

The reverse problem exists too: WDDM rules that WSL2 needs but does not get. Both are `#ifdef _WIN32`:
- `kVramSafeMib` is 1024 on Windows and 512 elsewhere (generate.cpp:1445-1450).
- The note "little room for the verify windows under WDDM" (generate.cpp:11043-11048) is Windows-only.

WSL2 VRAM is managed by Windows' video memory manager, so an over-full card can page. These should use `under_wddm()`. It affects warnings only.

## 4. Ranked changes

### (a) Config-only, try now

Put each in `.env`, then run `docker compose up -d`. One change per A/B.

| # | Change | Expected gain and evidence | Risk |
|---|---|---|---|
| A0 | **Diagnose first** (protocol in section 5). Read the PCIe probe, the prompt chunk, the pool workers and the PLE probe. | Decides whether A3, A5 or A6 apply | none |
| A1 | `LLMSERVE_DRAFT_VOCAB=en` | 110-215 MiB VRAM back, so +80-150 slots; English decode +1-2% (DETAILS.md:309; .env.example:49) | CJK answers lose drafts |
| A2 | `LLMSERVE_EXTRA_ARGS="--kv-resident 32768"`. merge_args appends it (prepare.py:464-468, 699-702) and nothing strips it on WSL. | +0.45 GB, decode about +1-3% at 64K; the main win comes at 128K+ (section 2). Check the log for `KV streaming: the host copy is pinned with cudaHostRegister` and `the K/V in … pinned RAM`. | If the pin is refused the engine exits; remove the flag |
| A2' | Alternative to A2: `--kv-grow`. It needs CUDA VMM; the engine prints why when it stays off (generate.cpp:5203-5216). | Only the cells in use take VRAM. Measured on an emulated 16 GB card at 262K: 3,396 → 5,719 slots, chat 100 → 115 tok/s, a 32K prompt 7.44 → 6.36 s (commit 9e3dbfe7). At 64K the gain is ≤0.45 GB, as A2. | VMM under WSL is untested. Cannot be combined with A2. |
| A3 | `--prefill auto:16384` (the engine shrinks the chunk until its loan fits, generate.cpp:8684) | −3 to +8% at 20K on 12-16 GB cards (DETAILS.md:1260). Larger when PCIe-bound: a 28K prompt is 2 chunks instead of 4, so half the expert streaming. | Not bit-exact on CUDA (DETAILS.md:1080); borrows more slots during the prompt |
| A4 | Per request, no restart: body `"strata_tune": {"spec_min_p": 0.7}` or `{"pcie_frac": 0.2\|0.35}` (server.py:1425-1440; the body is passed as sampling, server.py:7161) | The tuner tries 0.7/0.5 and 0.2/0.35/0 (tune_knobs.py:125-135). The tuned community 5060 Ti Q2_0 config used spec-min-p 0.70 and pcie-frac 0.20. Typically ±3-10%. | none. Then set it permanently: `--spec-min-p` / `--pcie-frac` in EXTRA_ARGS. Never use pcie-frac ≥0.75 (tune_knobs.py:165) |
| A5 | **If the CPU is an Intel hybrid:** `--pool-workers <P-cores−1>` | WSL2 hides the hybrid topology: pool.cpp:488-503 needs `cpu_capacity` or `/sys/devices/cpu_atom`, so the default becomes all cores −1, E-cores included. Measured 4 vs 15 workers: **+28.7% decode** (4080S + i7-13790F, `bench/results/2026-10-04-community-rtx-4080s-iq3s`). | Do not go above the engine default (−36%, tune_knobs.py:160) |
| A5' | AMD: try `--pool-workers <default−1>` to leave a core for Windows and Hyper-V's GPU paravirtualization | Not measured | low |
| A6 | **If RTX 40 (sm_89):** `LLMSERVE_ENGINE_ENV="STRATA_SIDE_MIXER=2 STRATA_ATTN_DAILY=1 STRATA_GR_UP75=1 STRATA_GU_SWIGLU=1 STRATA_QSA_ROWS=1 STRATA_B6_MMVQ_ROWS=1"` | sm_89 has no per-arch default set; sm_86 and sm_120 do (nv_defaults.hpp:1-13, DETAILS.md:1058-1060). The sm_86 set measured +10-14% decode on 8/8 pairs and is bit-exact (DETAILS.md:1260). | Unmeasured on sm_89. Same tokens, so the only risk is a hang. On RTX 50 these are already on. |
| A7 | `STRATA_QSA_SPLIT_ATTN=1` (decode attention split; matters at 30K+ context) | +2.8 / +3.5% after a 30K prompt (DETAILS.md:1263) | Rounding-level (KL 0.0016 vs a 0.0012 floor) |
| A8 | Learned profile: `LLMSERVE_EXTRA_ARGS="--expert-profile-save /data/profile-code.bin --expert-profile-save-every 10"`. After a day of reviews, also add `--expert-profile /data/profile-code.bin` (user args win, prepare.py:464). | Helps the first requests after each restart; the adaptive tier relearns within a session anyway. The only magnitude measured is SYCL: +21% mean (commit 8cb26e60). | Specialises the profile; keep one file per workload |
| A9 | VRAM reserve. Read `N MiB of VRAM free with everything loaded`. Lower `LLMSERVE_VRAM_RESERVE_MIB` only by (N − 1024), and only if the display is not on this GPU. | +~0.7 slots per MiB freed (the ~700/GB rule) | The 5080 WSL run ended with 178 MiB free (`…-5080-docker-wsl2`); WDDM pages instead of failing (DETAILS.md:366-388). If the card drives the desktop, rather raise it to 1000-1500. |
| A10 | `STRATA_EMB_REUSE_ACCOUNT=1` | Prompt +3.9%, chunk 6400 → 6656 (DETAILS.md:1277) | Rounding |

**Do not use:**
- `STRATA_ADAPT_OVERLAP=1`: −21% on an RTX 5090 Laptop under WSL2 (DETAILS.md:1273).
- `--lookup-chain`.
- `--spec` ≥6 or ≤2 (tune_knobs.py:166-167).
- `--ple-io ram` or `mmap`.
- `--kv q4_0` at 64K: it saves only ~0.45 GB and costs quality.
- `--max-context` below 64K: this workload reaches 51K.

### (b) Code changes for us

| # | Change | Gain | Effort |
|---|---|---|---|
| B1 | **Enable KV streaming on WSL** with a fallback. prepare.py:535-536 and 663-666: treat WSL like Linux (honour `LLMSERVE_KV_STREAMING`). In generate.cpp, before `session_bytes`/`cudaMalloc(sbuf)` (~5220): pre-flight `kv_host_alloc(total)` and free it. On refusal, `qsa_set_kv_resident(0)` with one log line instead of exiting. Fix .env.example:44, WINDOWS_DOCKER.md:55, DETAILS.md:204-206 and 216. | +1-3% at 64K, +6% at 128K, +23% at 262K | S-M |
| B2 | **WSL hybrid-CPU worker default.** In pool.cpp's Linux topology, detect Intel hybrid without sysfs (CPUID.7:EDX[15] plus leaf 1Ah per pinned vCPU, or the brand string), or have prepare.py write `--pool-workers` for known hybrid SKUs. Log a hint when WSL and hybrid are both true. | Up to ~+29% decode on Intel 12th-14th gen and Core Ultra | M |
| B3 | Expose the learned profile: `LLMSERVE_EXPERT_PROFILE_SAVE` → `cfg["expert_profile_save"]` in prepare.py:739-747. server.py:3196-3215 already swaps the file in on restart. | Faster warm-up per workload | S |
| B4 | Use `under_wddm()` for `kVramSafeMib` (generate.cpp:1445) and the low-room note (11043). | Diagnoses sysmem paging (speed collapse) | S |
| B5 | Cold start and memory pressure: in `load_experts_ranges` (pinned.cu:850-930, Linux branch), call `posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED)` after each layer. Alternatively make `STRATA_UNBUFFERED_LOAD` work on Linux with O_DIRECT; today it returns "buffered (not Windows)", pinned.cu:827. | Without it, ~34 GB of shard data sits in the VM page cache (the user's 22 GB buff/cache) and pushes the VM to its 58 GB cap, which pressures Windows. Restarts lose the warm cache. Speed gain is indirect. | S |
| B6 | An sm_89 bitwise default table (nv_defaults), once A6 is measured | +5-14% decode on RTX 40, if it holds | S after measuring |

**Build-level, measured to buy little or nothing for Q2_0:**
- `PORTABLE=OFF`: Q2_0 CPU experts run on Strata's own runtime-dispatched AVX-512/AVX2 kernels (`q2_rows_any`, expert_layout.cpp:357-361; pool.cpp:1219-1240, "ggml-cpu has only a scalar one"). PORTABLE only changes ggml-cpu's flags (CMakeLists.txt:390-412).
- A single CUDA arch: build time only; the runtime picks the matching cubin.
- `-O3`: Release already uses it. LTO is negligible because the hot loops are per-file-flagged intrinsics.
- CUDA 13.x: unmeasured.
- The AVX-512 Q2_0 repack is already automatic when `/proc/cpuinfo` shows F/BW/VL/VNNI/VBMI (prepare.py:185-190, 607); confirm "AVX-512 yes" in the prepare log (prepare.py:1268).
- hugetlbfs: THP=always already backs the arena (pinned.cu:281-297). `vm.nr_hugepages` is VM-wide (docker-desktop distro), not per container. Low value.

## 5. Measurement protocol

**Startup lines** (`docker exec llmserve grep -E "<pattern>" /data/logs/engine.log`):
- `PCIe probe: X GB/s host->device … -> pcie_frac` (generate.cpp:3593). Expect about 13-14 at Gen4 x8, 24-26 at Gen4 x16, 28-50 at Gen5. Also run `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv` during a prompt.
- `this CPU has no AVX-512` (generate.cpp:3636), if present.
- `expert arena: cudaHostRegister PORTABLE ok …` and `loaded X GiB at Y GiB/s` (generate.cpp:5939-5940).
- `CPU pool: N workers by default (P physical cores, L logical processors; bounded by …)` and `N pool workers on logical processors …` (5985, 6003). Any `hybrid CPU detected` line.
- `expert cache N slots, X GiB of VRAM` (6421) and `pre-filled …` (6504).
- `PLE direct read probe: N rows/s` or `--ple-io direct measures slow` (4067-4088).
- `prompt chunk auto: N tokens, a R-slot ring` and `the prompt path borrows N … slots` (8675, 8771).
- `N MiB of VRAM free with everything loaded` (11041-11056).
- With A2: `KV streaming: … cells per QSA layer in VRAM, the K/V in …` (5229).

**Per request:**
- `prompt N tokens = R reused + M read in X ms (Y tok/s), G generated in Z ms (W tok/s), drafts accepted A of B` (15559).
- `decode expert cache hit rate: … ; K more read by the GPU over PCIe (P% …)` (15578).
- `PCIe share: the CPU pool takes … us per missed expert … -> pcie_frac` (10075).
- `KV streaming: X% of block reads hit VRAM`.
- `GET /metrics`: hit_rate, pcie_share, drafts.

**Breakdown runs** (diagnosis only, slows things down):
- `LLMSERVE_ENGINE_ENV="STRATA_DECODE_TIMING=1 STRATA_PREFILL_TIMING=1"`, which gives the `strata decode timing:` line and the prompt path's GPU time by phase.
- One run with `STRATA_VERIFY_PROFILE=1`.

**A/B method:**
1. Use `docker exec llmserve python -m llmserve.bench --cases short,prompt-28k --runs 3 --max-tokens 1024 --label <arm> --out /data/bench-<arm>.json`, then `--compare A.json B.json`. It uses temperature 0, a unique nonce per run so no cache reuse, and records the engine's timings.
2. Add one real thinking request (the 28K review, `max_tokens` 4096, seed fixed) for MTP acceptance.
3. Restart per arm (A, B, A, B); at least 3 runs each. Keep a change only if it wins >3% in every pair (the tuner's rule, tune_knobs.py:121-123). Use per-request `strata_tune` for spec_min_p and pcie_frac without restarts.
4. Order: A0 → A5/A6 (the biggest, if applicable) → A1 → A2/A2' → A3 → A4 → A7-A10.

## 6. Questions for the user

1. **GPU model**, and whether it drives the Windows display. This decides whether 1,378 tok/s is expected (4060 Ti/5060 Ti) or a fault (4070 Ti S and up), and whether A6 applies.
2. **CPU model.** Intel hybrid → A5/B2 is the top lever. Zen4/5 → AVX-512 path; confirm the repack.
3. The startup log lines listed above, especially PCIe GB/s, pool workers, the prompt chunk, the PLE probe and free VRAM.
4. The motherboard slot: chipset or CPU lanes? Is an NVMe sharing lanes with the GPU slot?
5. Is 64K enough, or do they want 128K? At 128K, B1 is worth +6%.
6. Mostly English/code output? (A1)
7. Are other GPU apps or browsers open while serving, and is Docker Desktop's Resource Saver on? What is the Windows power plan?
