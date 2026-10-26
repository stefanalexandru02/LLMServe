# LLMServe on Windows: Docker Desktop, WSL2 and the NVIDIA GPU

The longer notes behind the README's quick start. Sources: Strata's `docs/DOCKER.md` and `docs/DETAILS.md` (WSL section), a community WSL2 benchmark
in Strata's repository (RTX 5060 Ti 16 GB, Ryzen 7 3700X, 128 GB RAM, IQ3_S), and the colibri project's WSL2 notes. Where a statement is general
Docker/WSL2 knowledge that nobody measured with this engine, it is marked *(not measured here)*.

## What runs where

```
Windows 11 ── NVIDIA driver 580+ (WDDM) ──┐
Docker Desktop ── WSL2 backend ── one Linux VM (kernel "…microsoft-standard-WSL2")
                                   └─ your llmserve container   ← the GPU appears as /dev/dxg, libcuda comes from the Windows driver
                                   └─ the named volume llmserve_llmserve-data (an ext4 file system inside the VM's virtual disk)
```

- The GPU is passed through by Windows' WDDM layer (`/dev/dxg`), not as a normal Linux `/dev/nvidia*`. Docker Desktop installs the plumbing
  (the NVIDIA container runtime); you need only a current Windows driver. **Docker Desktop passes through NVIDIA GPUs only** (no Radeon or Arc).
- The container sees the WSL2 kernel, so the engine recognises "WSL" (kernel release contains `microsoft`, or `/dev/dxg` exists) and applies the WDDM rules below.
- All containers share one VM. Its RAM, CPU count and disk are set for the whole VM, not per container.

## RAM: the one setting that decides everything

The engine **pins** the model's experts (34 GB for Q2_0, up to 50 GB for IQ3_S) in RAM so the GPU can read them over PCIe at full speed, and the
CPU computes the experts the GPU's cache does not hold. If the VM has too little RAM the engine falls back to *low-RAM mode* (experts read from the SSD through the
page cache): it works, but output can drop several times in speed.

WSL2 gives its VM **half of the Windows RAM** by default (Microsoft's default, *not measured here*; on a 64 GB PC that is 32 GB, less than Q2_0 asks for).
Set it in `%UserProfile%\.wslconfig`, then `wsl --shutdown` (PowerShell) and restart Docker Desktop:

| Windows RAM | Suggested `memory=` | What fits |
| ---: | ---: | --- |
| 32 GB | 24GB | IQ1_M (the Coder) in low-RAM mode; everything else is too big |
| 48 GB | 40GB | IQ1_M comfortably; Q2_0 and IQ2_XS only in low-RAM mode |
| **64 GB** | **56GB** | Q2_0, IQ2_XS, IQ3_XXS; IQ3_S only with nothing else running |
| 96 GB | 84GB | all the 2- and 3-bit sizes with 128K context, UD-IQ4_XS with every expert in RAM |
| **128 GB** | **112GB** | all of those, and UD-Q4_K_XL with every expert in RAM |

```ini
[wsl2]
memory=56GB
# processors=12      # optional: the default is all logical CPUs, which is right for this engine (CPU experts use them)
# swap=0             # optional: do not page the VM to disk (an out-of-memory then kills the process instead of freezing the PC)
```

Leave Windows 8 GB (an eighth of the RAM on big PCs). If the Windows desktop and browser are open, the VM's memory competes with them. The `scripts\start.ps1` helper reads
your `.wslconfig` and the Windows RAM and warns when the setting is too low for the quant in `.env`.

**Do not put a memory limit on the container** (`mem_limit`, `--memory`): the engine plans from the machine's RAM, and the entrypoint warns if it sees a cgroup limit.

## Pinned memory, `memlock` and what WDDM changes

- `ulimits: memlock: -1` (in `docker-compose.yml`) lets the engine lock memory. Without it pinning fails or falls back to slow paths. For `docker run`: `--ulimit memlock=-1:-1`.
- Strata's engine pins its expert arena with `cudaHostRegister`; on WSL2 a community run pinned 47 GiB this way successfully ("cudaHostRegister PORTABLE ok"). Transparent huge
  pages replaced explicit huge pages there (`vm.nr_hugepages=0`), which is fine.
- **KV streaming is off under WSL2** (`--kv-resident` is not written): it needs a pinned RAM copy of the KV cache, and the WDDM driver pins only about 1 GB of RAM for it. The KV cache stays
  in VRAM, so long contexts leave fewer experts cached and **prompts are slower than on native Linux** (the community RTX 5060 Ti run: 343 / 598 tok/s at 3.8K / 19K with IQ3_S, decode 48 tok/s).
- `--ple-io ram` (the 28.8 GB n-gram table in RAM) is not used on WSL2 either, and WSL2's virtual disk reports itself as rotational, which setup rules on other platforms would misread; LLMServe skips that check on WSL.
- With a multi-GPU Windows machine the WDDM driver pins at most 8 GiB of the arena (Strata's `MULTI_GPU.md`); use one card (`LLMSERVE_GPU=0`).
- Keep some VRAM for Windows itself: the desktop and browsers use 1 to 2 GB. If the screen stutters or goes black once the model loads, set `LLMSERVE_VRAM_RESERVE_MIB=1500` (Strata's advice for a card that drives a display).

## Where the model lives

Use the **named volume**, which is what `docker-compose.yml` does. The model is ~110 GB while installing (66 GB download + packing) and the engine streams parts of it.

- A volume sits on the VM's ext4 disk. A *bind mount of a Windows folder* (`- C:\models:/data`) goes through a 9p/virtiofs bridge. That bridge is several times slower
  for large sequential reads and much worse for the random reads the n-gram table needs *(general Docker Desktop knowledge, not measured here)*. A community WSL2 run that served the model from a Windows NVMe over virtiofs measured 1.1 GB/s sequential
  and loaded at 1.75 GiB/s, so it works but is not what you want. The container entrypoint warns when `/data` is on a `9p`, `drvfs`, `virtiofs` or network mount, or under `/mnt/<drive>`
  (the check is the one colibri uses for WSL).
- Docker Desktop keeps all volumes in one virtual disk (`docker_data.vhdx` or the WSL `ext4.vhdx`). It grows when the model is written and **does not shrink by itself** when you delete it.
  Where it lives and how big it may grow: Docker Desktop > Settings > Resources > Advanced > "Disk image location" (move it to a bigger drive) and "Disk image size". Free space after
  `docker compose down -v` with `docker system prune`, and `wsl --shutdown` + Optimize-VHD if you want the file smaller *(general WSL knowledge)*.
- Don't browse the volume from Windows Explorer. To get files in or out use a helper container.

**Bring GGUF files you already have** (the right files only: the exact ISTA-DASLab files for your quant, names as on Hugging Face). The prepare step finds them, checks their SHA-256 and skips the download:

```powershell
docker volume create llmserve_llmserve-data
docker run --rm -v llmserve_llmserve-data:/data -v D:\gguf\Q2_0:/src:ro ubuntu:24.04 sh -c "mkdir -p /data/models/Q2_0 && cp -v /src/*.gguf /data/models/Q2_0/ && chown -R 1000:1000 /data"
```

(Copying through the Windows share is slow, once.) For Unsloth quants the folder is `unsloth-UD-Q4_K_XL`, for the Coder `coder-IQ1_M`.

## Building the image on Windows

`docker compose up -d --build` compiles the CUDA engine inside the VM. Memory and time:

- Set `CUDA_ARCHS` in `.env` to your card (`"89-real;89-virtual"` for RTX 40xx ...). The default builds four targets and takes 25+ minutes.
- nvcc jobs are memory-hungry. If the build is killed (`Killed`, exit 137, or Docker Desktop restarts), set `BUILD_JOBS=4` (or 2) and give the VM more RAM (`.wslconfig`).
- `PORTABLE=ON` (default) builds the CPU kernels for any AVX2 PC. `PORTABLE=OFF` uses `-march=native` of the machine that builds; use it only if you build and run on the same PC (fine for a typical single-PC setup, and a little faster).
- `WITH_VISION=0` skips the picture reader (about 10 minutes).
- BuildKit's cache keeps the compiled objects between builds: changing `serve/` or `llmserve/` never rebuilds the engine, changing the engine sources rebuilds only what changed.

## Several requests at once (24 GB cards)

Because KV streaming is off here, every slot of `LLMSERVE_PARALLEL=2` or more keeps its whole KV cache in VRAM. An estimate for Q2_0 with `int8` KV: about 1.0 GB per slot at
64K, 0.8 GB at 48K, 0.6 GB at 32K, about 5% of the expert cache of a 24 GB card. Setup recommends 2 slots at 64K, 3 at 48K and 4 at 32K there, and says "estimate" because the engine
log (`--batch: N slot sessions ... GiB each`) has the exact size. Settings to try, one at a time: `LLMSERVE_PARALLEL=2`, then `LLMSERVE_BATCH_MTP=on`, then `LLMSERVE_BATCH_SWITCHES=on`;
for a coding agent also `LLMSERVE_AGENT=on`. Measure each with `python -m llmserve.bench --concurrency 1,2,3` (README, "Several requests at once"). Do not combine `--kv-resident`
(not written on WSL2 anyway) with `STRATA_BATCH_ATTN_ROWS`; setup leaves that switch out when the KV cache is streamed.

## First start timeline

1. Build (10 to 30 minutes, once). 2. Download 66 GB (resumable; at 100 MB/s about 11 minutes, at 20 MB/s an hour). The hashes are checked once (a few minutes).
3. Pack: on a CPU with AVX-512 the Q2_0 experts are repacked for the fast kernel (about 40 GB written, 2 to 5 minutes); other CPUs pack in seconds.
4. MTP draft layer: ~5 GB of tensors by range reads, then packing. 5. Load: a few minutes, pinning the RAM.
`docker compose logs -f` shows each step with a `[prepare]` prefix. The container stays `health: starting` meanwhile (the health check waits two hours before it counts failures).

## Troubleshooting

**The GPU is not visible in the container.** Test: `docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi`. If that fails:
Docker Desktop must use the WSL2 engine (Settings > General); update the NVIDIA driver on Windows (not inside WSL: never install a Linux NVIDIA driver in WSL); restart Docker Desktop
and run `wsl --shutdown`; run `wsl --update`. `nvidia-smi` on Windows should work first.

**Driver too old for CUDA 13.** The image uses CUDA 13.0 (CUDA 12.8 builds crash on long prompts with Blackwell cards, per Strata). The driver must be 580 or newer; the error looks like
`unsatisfied condition: cuda>=13.0` when the container starts, or the entrypoint says "NVIDIA driver … is too old".
RTX 20 series: the default build omits sm_75; set `CUDA_ARCHS="75-real;75-virtual"`.

**Out of memory.**
- *Exit code 137 / "Killed"*: RAM. `docker stats`, `.wslconfig` above; shorter `LLMSERVE_CONTEXT`; smaller quant; close memory-hungry Windows programs.
- *CUDA out of memory / "no VRAM is left for the expert cache" / "the draft head does not fit"*: lower `LLMSERVE_CONTEXT`, `LLMSERVE_DRAFT_VOCAB=en`, `LLMSERVE_VISION=off`, `LLMSERVE_KV=q4_0`, `LLMSERVE_MTP=off` (saves ~1 GB VRAM, output slower), and close games or browsers using the GPU.

**Slow.** Check in the log: "low-RAM mode" (not enough RAM), "is on a '9p' mount" (bind mount of a Windows folder), `Docker Desktop` Resource Saver / Windows power plan on "Power saver", antivirus scanning `\\wsl$`, a laptop on battery. Expected for WSL2: prompts slower than native. In Windows' Task Manager the GPU
"Dedicated memory" should be nearly full when the model is loaded (the expert cache fills what is left).

**Downloads.** Behind a proxy or mirror: set `HF_ENDPOINT` (and Docker Desktop's proxy settings). For a gated or rate-limited download set `HF_TOKEN`. Every shard is checked against a pinned SHA-256; a wrong file is deleted and fetched again.

**Reach the API from another PC.** Set `LLMSERVE_BIND=0.0.0.0` and `LLMSERVE_API_KEY=<long random>` in `.env`, and allow the port in Windows Defender Firewall for the network profile you use *(general knowledge)*. The API has no TLS; do not forward it to the internet without a reverse proxy.

**Stopping and updating.** `docker compose stop` waits up to two minutes for the model to drain and unload (`stop_grace_period`). To update the engine: `git pull`, `docker compose up -d --build`. The model volume is untouched, and `cfg.json` is rewritten only if the settings or the machine's RAM/VRAM changed.
