# LLMServe benchmarks

Measured with `llmserve/bench.py` (streaming chat completions, thinking off, temperature 0, a unique run id at the start of
every prompt so no prompt cache is reused, one excluded warm-up request, then 3 runs per case, max_tokens 512). Speeds are the
engine's own `timings`; TTFT is the client's clock. The table shows medians; the min-max ranges are in the run logs below.

## Machine

| | |
|---|---|
| GPU | RTX 4090 24 GB (sm_89), PCIe 4.0 x16 |
| CPU | Ryzen 9 9900X (12 cores, Zen 5, AVX-512) |
| RAM | 64 GB; WSL2 VM 58 GB |
| Host | Windows, Docker Desktop (WSL2 backend), model on a Docker named volume (ext4) |
| Model | Qwen3.8-Flash-Next, ISTA-DASLab Q2_0, MTP on (`--spec 4 --spec-min-p 0.5`), 128K context, KV int8, 1 slot |
| Engine | Strata 0.1.51 as built by this repo's Dockerfile (CUDA 13.0.2) |

## 2026-10-26: baseline vs the sm_86 speed switches on a 4090

`sm86set` = `LLMSERVE_ENGINE_ENV=STRATA_ADAPT_NOWAIT=1 STRATA_ADAPT_LAG=2 STRATA_CPU_IQ_ALL_EXACT1=1 STRATA_SIDE_MIXER=2
STRATA_ATTN_DAILY=1 STRATA_GR_UP75=1 STRATA_GU_SWIGLU=1 STRATA_QSA_ROWS=1 STRATA_B6_MMVQ_ROWS=1` (the set the engine turns on
by itself only for sm_86 cards, `src/core/nv_defaults.cpp`). One engine start per arm.

| case | metric | baseline | sm86set | diff |
|---|---|---:|---:|---:|
| short (~124 tok) | decode tok/s | 110.4 | 119.6 | +8% |
| | TTFT s | 2.20 | 3.20 | **+45% (worse)** |
| | MTP draft acceptance | 59.5% | 62.5% | |
| prompt-4k (4,253 tok) | prompt tok/s | 2,559 | 2,584 | +1% |
| | decode tok/s | 116.0 | 131.5 | +13% |
| | TTFT s | 1.69 | 1.67 | |
| | MTP draft acceptance | 76.1% | 72.2% | |
| prompt-16k (16,867 tok) | prompt tok/s | 2,461 | 2,423 | -2% |
| | decode tok/s | 123.7 | 154.4 | **+25%** (every sm86set run beat the best baseline run) |
| | TTFT s | 6.88 | 6.99 | |
| | MTP draft acceptance | 72.8% | 70.1% | |

Reading it:

- Decode gains grow with context: the switches mostly speed up attention and the adaptive expert tier.
- The short-prompt TTFT regression is in the batched prompt path, which a ~124-token templated prompt takes (`--short-read`
  is 64 by default; see [TTFT_WARMUP.md](TTFT_WARMUP.md)). Which switch causes it is not known yet.
- Not checked: whether the outputs are token-identical (Strata measured the set as identical on sm_86; the bench does not
  compare texts yet).
- The first run of each case is slow in both arms (e.g. 4K: 1,098 and 898 tok/s): the engine is cold after a start (no warm-up yet).

For reference, Strata's own published Q2_0 numbers (RTX 5070 12 GB, Ryzen 5 7600, native Windows, engine 0.1.26,
[DETAILS.md](DETAILS.md)): decode 87-93 tok/s at 1-4K, prompt 1,299 tok/s at 4K and 2,171 at 32K. Different GPU, so this is
context, not an A/B.

Earlier single request on the same machine (baseline, first request after a restart): a 28,340-token code review prompt read
at 1,378 tok/s, then 23,022 tokens generated at 113 tok/s.

### Run logs

baseline:

```
short       prompt 121/127/124 tok   55.4/55.6/61.0 tok/s   decode 73.2/110.4/112.9   TTFT 2.20/2.31/2.05 s
prompt-4k   prompt 4253/4257/4252    1097.9/2559.1/2695.9   decode 100.1/116.0/126.2  TTFT 4.21/1.69/1.60 s
prompt-16k  prompt 16869/16868/16866 3151.6/2284.3/2460.7   decode 116.6/123.7/131.4  TTFT 5.40/7.43/6.88 s
```

sm86set:

```
short       prompt 122/126/122 tok   38.3/36.4/41.5 tok/s   decode 89.5/119.6/122.8   TTFT 3.20/3.48/2.96 s
prompt-4k   prompt 4254/4255/4250    898.2/2583.8/2686.0    decode 110.5/131.5/138.5  TTFT 5.02/1.67/1.60 s
prompt-16k  prompt 16867/16866/16866 2833.4/2112.0/2423.3   decode 151.6/154.4/163.5  TTFT 6.00/8.02/6.99 s
```

## 2026-10-26: sm86set + `--short-read 192` + `STRATA_CKPT_POOL=1` ("combo192")

Same machine and model; `LLMSERVE_ENGINE_ENV` = the sm86set list plus `STRATA_CKPT_POOL=1`, `LLMSERVE_EXTRA_ARGS=--short-read 192`.

| case | metric | baseline | sm86set | combo192 |
|---|---|---:|---:|---:|
| short (~124 tok) | decode tok/s | 110.4 | 119.6 | 118.5 |
| | TTFT s | 2.20 | 3.20 | 2.84 |
| prompt-4k | prompt tok/s | 2,559 | 2,584 | 2,605 |
| | decode tok/s | 116.0 | 131.5 | 130.9 |
| | TTFT s | 1.69 | 1.67 | 1.66 |
| prompt-16k | prompt tok/s | 2,461 | 2,423 | 2,637 |
| | decode tok/s | 123.7 | 154.4 | **161.6** |
| | TTFT s | 6.88 | 6.99 | 6.43 |

Reading it:

- `--short-read 192` did **not** fix the short-prompt TTFT. The engine's own note (`src/program/generate.cpp`, before
  `windows_ok`) puts the verify windows at ~16 ms per prompt token, so ~124 tokens cost ~2 s that way too. Both prompt
  paths land at 2-3 s for a ~124-token prompt while a 4,253-token prompt is read in ~1.6 s, so most of the short-prompt
  time is a per-request cost outside the read itself. Next step: an `STRATA_TRACE=1 STRATA_PREFILL_TIMING=1` run.
- Decode with the switches holds (+7% short, +13% 4K, +31% 16K over baseline); this is one engine start per arm.
- The first 4K run after a start was the slowest yet (568 tok/s, TTFT 7.94 s): the cold-start cost a warm-up should remove.

combo192 run log:

```
short       prompt 124/126/123 tok   40.2/44.7/45.7 tok/s   decode 80.1/118.5/129.7   TTFT 3.10/2.84/2.71 s
prompt-4k   prompt 4252/4251/4253    568.3/2604.8/2944.9    decode 119.5/130.9/157.4  TTFT 7.94/1.66/1.47 s
prompt-16k  prompt 16867/16868/16868 3275.9/2352.3/2636.7   decode 153.4/164.1/161.6  TTFT 5.21/7.20/6.43 s
```
