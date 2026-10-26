#!/usr/bin/env python3
"""LLMServe prepare: get Qwen3.8-Flash-Next ready for the engine, headless, idempotent and resumable.

This is the container's replacement for the Qwen3.8 path of Strata's setup.py (MIT, Niko1221 and contributors):
same files, same pins, same engine arguments, but no questions, no TUI, no engine download and no compiling.

    python -m llmserve.prepare --quant Q2_0 --data-dir /data --context 131072 [--parallel N]
                               [--vision off|on|gpu|cpu] [--extra-args "--pool-workers 8"] [--force] [--dry-run]

Every option can also come from an LLMSERVE_* environment variable (the container passes its settings that way).

What it does, in order (each step writes a marker under <data-dir>/state/ and is skipped when its marker is there):

  1. look at the machine: RAM (/proc/meminfo and the cgroup limit), GPU (nvidia-smi), WSL, AVX-512, disk space;
     say plainly whether the chosen quant fits, and what to do when it does not;
  2. download the pinned GGUF shards (and the 0.9 GB picture reader with --vision) from Hugging Face: resumable,
     SHA-256 checked against the pins in llmserve/catalog.py, a finished and verified file is never hashed again;
  3. pack the model for the engine (tools/iq_pack.py; for Q2_0 on an AVX-512 CPU tools/strata_pack.py + pack_index.py,
     a one-time ~40 GB repack that is the fast path);
  4. the MTP draft layer (tools/mtp_fetch.py, mtp_pack.py, mtp_rt.py, ~5 GB of tensors by HTTP range reads) and the
     draft vocabulary subset;
  5. write <data-dir>/cfg.json, the file `python -m serve.server --engine strata --config` loads.

--dry-run prints the machine it sees, the plan and the exact engine arguments, and downloads nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import http.client
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:                                  # `python -m llmserve.prepare` (the container) and `import llmserve.prepare` (tests)
    from . import catalog as C
except ImportError:                   # run as a script from inside the package folder
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llmserve import catalog as C

GIB = 2 ** 30
FORMAT_VERSION = 1                    # bump when the plan or the cfg.json layout changes: forces a rewrite of cfg.json

# ------------------------------------------------------------------------------------------- engine constants
# Strata setup.py: RAM beside the experts for the OS, the engine's other buffers and the server
LOW_RAM_HEADROOM_GB = 10
UNSLOTH_RAM_LEFT_GB = 24              # RAM beside a Unsloth RAM budget: the OS, the engine, the file cache
UNSLOTH_RESIDENT_BESIDE_GB = 16       # all experts in RAM only with this much RAM beside them
MTP_VRAM_GB = 1.0                     # the draft layer's share of VRAM
KV_CELL_BYTES = {"q4_0": 576, "k8v4": 816}        # int8 is 1056
KV_CHOICES = ("int8", "q4_0", "k8v4", "fp16")
VISION_GPU_MIN_VRAM_GB = 23.5         # "on" reads pictures on the GPU from 24 GB of VRAM, else on the CPU
VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700}, "cpu": {"max_tokens": 300, "reserve_mib": 0}}
VISION_THREADS_CAP = 16
PARALLEL_MAX = 8
# parallel slots (an estimate, see parallel_recommend): the slots together may take PARALLEL_SHARE of the expert cache, and the
# cache must hold at least PARALLEL_MIN_CACHE of the experts before any slot is carved out of it.
PARALLEL_SHARE, PARALLEL_MIN_CACHE = 0.13, 0.40
PARALLEL_REC_MAX = 4                  # --batch-mtp fits 4 slots in an 8-row verify window; more slots take turns
# LLMSERVE_BATCH_SWITCHES (parallel >= 2): the engine's opt-in batch-window switches.  Every name is read by src/ (verify.cpp for
# the first two, generate.cpp for the others).  STRATA_BATCH_ATTN_ROWS is left out with KV streaming (verify.cpp: no streamed KV).
BATCH_SWITCH_ENV = {"STRATA_BATCH_GDN_GROUPS": "1", "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_SLOT_COPY_DIRECT": "1",
                    "STRATA_SLOT_COPY_BATCHSYNC": "1"}
# LLMSERVE_AGENT: STRATA_CACHE_MESSAGE_BOUNDARY is read by the engine (generate.cpp), so it goes into the run config's "env".
# STRATA_ADMIT_SHORT_FIRST is read by the Python server (serve/server.py) from ITS environment, which the run config does not
# reach: docker/entrypoint.sh exports it when parallel >= 2.
AGENT_ENGINE_ENV = {"STRATA_CACHE_MESSAGE_BOUNDARY": "1"}
AGENT_ADMIT_SHORT_FIRST = 4096
AGENT_CONVERSATION_CACHE_MIB = 4096
CONVERSATION_CACHE_RAM_LEFT_GB = 4    # RAM beside the quant and the parked conversations: the OS, the server, the file cache
SLOW_MOUNT_TYPES = {"9p", "drvfs", "virtiofs", "fuse.grpcfuse", "grpcfuse", "fakeowner", "cifs", "smb3", "smbfs",
                    "nfs", "nfs4", "fuse.osxfs", "fuse.sshfs"}
OLD_DRAFT_VOCABS = {"369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"}   # Strata <= 0.1.26


# ------------------------------------------------------------------------------------------- output
def say(msg: str = "") -> None:
    print(msg, flush=True)


def ok(msg: str) -> None:
    say(f"[prepare] ok    {msg}")


def info(msg: str) -> None:
    say(f"[prepare]       {msg}")


def warn(msg: str) -> None:
    say(f"[prepare] WARN  {msg}")


def step_line(n: int, total: int, msg: str) -> None:
    say(f"[prepare] ---- step {n}/{total}: {msg}")


class Fail(Exception):
    """A step failed in a way the person can fix; main() prints it and exits non-zero."""

    def __init__(self, msg: str, hint: str = "", code: int = 1):
        super().__init__(msg)
        self.hint, self.code = hint, code


# ------------------------------------------------------------------------------------------- the machine
@dataclasses.dataclass
class Hardware:
    ram_gib: float = 0.0              # what the engine can use: the smaller of the host's RAM and the cgroup limit
    host_ram_gib: float = 0.0
    cgroup_gib: float | None = None   # a container memory limit, if there is one
    gpu_name: str = ""
    vram_gib: float = 0.0             # 0: no GPU seen
    compute_cap: str = ""
    driver: str = ""
    gpu_index: int = 0
    gpu_count: int = 0
    wsl: bool = False
    avx2: bool = False
    avx512: bool = False
    cpu_name: str = ""
    cpu_threads: int = 1
    rotational: str | None = None     # name of a rotational disk under the data dir (Linux, not WSL)
    data_fs: str = ""                 # file system type of the data dir's mount
    notes: list = dataclasses.field(default_factory=list)

    def facts(self) -> dict:
        """The part of the machine that changes the plan, rounded so that tiny changes do not rewrite cfg.json."""
        return {"ram_gib": round(self.ram_gib), "vram_gib": round(self.vram_gib), "gpu_count": self.gpu_count,
                "wsl": self.wsl, "avx512": self.avx512, "rotational": bool(self.rotational)}


def read_meminfo_gib(text: str) -> float:
    for line in text.splitlines():
        if line.startswith("MemTotal"):
            try:
                return int(line.split()[1]) * 1024 / GIB
            except (IndexError, ValueError):
                break
    return 0.0


def cgroup_limit_gib(v2_text: str | None, v1_text: str | None) -> float | None:
    """A container memory limit in GiB from cgroup v2 memory.max or v1 memory.limit_in_bytes; None if unlimited."""
    for text in (v2_text, v1_text):
        if text is None:
            continue
        t = text.strip()
        if not t or t == "max":
            continue
        try:
            n = int(t)
        except ValueError:
            continue
        if n <= 0 or n >= 2 ** 60:       # v1 reports ~2**63 for "no limit"
            continue
        return n / GIB
    return None


def parse_nvidia_smi(text: str) -> list[dict]:
    """`nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits`."""
    found = []
    for line in text.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            found.append({"index": int(parts[0]), "name": parts[1], "vram_gib": float(parts[2]) / 1024.0,
                          "cc": parts[3], "driver": parts[4] if len(parts) > 4 else ""})
        except ValueError:
            continue
    return found


def detect_wsl(release: str | None = None, dxg_exists: bool | None = None) -> bool:
    """Docker Desktop's containers run on the WSL2 kernel: its release says "microsoft", and /dev/dxg is the GPU."""
    release = platform.uname().release if release is None else release
    if "microsoft" in release.lower():
        return True
    return os.path.exists("/dev/dxg") if dxg_exists is None else dxg_exists


def parse_cpuinfo(text: str) -> tuple[str, bool, bool]:
    """(model name, avx2, avx512): avx512 means everything the engine's fast kernels use (F, BW, VL, VNNI, VBMI)."""
    m = re.search(r"^flags\s*:\s*(.*)$", text, re.M)
    flags = set(m.group(1).split()) if m else set()
    name = re.search(r"^model name\s*:\s*(.*)$", text, re.M)
    return (name.group(1) if name else "unknown CPU", "avx2" in flags,
            {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags)


def mount_fs_type(path: str, mounts_text: str) -> str:
    """The file system type of the mount that holds `path` (longest matching mount point in /proc/mounts)."""
    best, fstype = "", ""
    path = os.path.abspath(path)
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mp = parts[1].replace("\\040", " ")
        if (path == mp or path.startswith(mp.rstrip("/") + "/") or mp == "/") and len(mp) >= len(best):
            best, fstype = mp, parts[2]
    return fstype


def rotational_disk(path: str) -> str | None:
    """Linux: the disk's name when `path` is on a rotational disk (sysfs queue/rotational), else None."""
    try:
        st = os.stat(path)
        p = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
        for q in (p, p.parent):
            f = q / "queue" / "rotational"
            if f.exists():
                return q.name if f.read_text().strip() == "1" else None
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def detect_hardware(data_dir: Path | None = None, *, ram_gib: float | None = None, vram_gib: float | None = None,
                    wsl: bool | None = None, avx512: bool | None = None, gpu_pick: int | None = None) -> Hardware:
    """Look at the machine.  Never raises: a missing GPU or an unreadable /proc gives zeros and a note.
    The keyword arguments replace what was detected (a "what if" plan, and the tests)."""
    hw = Hardware()
    host = read_meminfo_gib(_read("/proc/meminfo") or "")
    cg = cgroup_limit_gib(_read("/sys/fs/cgroup/memory.max"), _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    hw.host_ram_gib, hw.cgroup_gib = host, cg
    hw.ram_gib = min(host, cg) if cg is not None and host else (cg or host)
    if cg is not None and host and cg < host - 0.5:
        hw.notes.append(f"this container has a memory limit of {cg:.0f} GiB (the host has {host:.0f}); the engine "
                        "plans from the smaller one. Remove the limit (docker-compose: no mem_limit)")
    if ram_gib is not None:
        hw.ram_gib = hw.host_ram_gib = ram_gib
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        gpus = parse_nvidia_smi(smi.stdout) if smi.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        gpus = []
    hw.gpu_count = len(gpus)
    if gpus:
        if gpu_pick is not None:
            g = next((x for x in gpus if x["index"] == gpu_pick), None)
            if g is None:
                raise Fail(f"there is no GPU {gpu_pick}: " + ", ".join(f"{x['index']} = {x['name']}" for x in gpus))
        else:
            g = max(gpus, key=lambda x: (round(x["vram_gib"]), -x["index"]))
        hw.gpu_name, hw.vram_gib, hw.compute_cap, hw.driver, hw.gpu_index = \
            g["name"], g["vram_gib"], g["cc"], g["driver"], g["index"]
    if vram_gib is not None:
        hw.vram_gib = vram_gib
    hw.wsl = detect_wsl() if wsl is None else wsl
    hw.cpu_name, hw.avx2, avx = parse_cpuinfo(_read("/proc/cpuinfo") or "")
    hw.avx512 = avx if avx512 is None else avx512
    try:
        hw.cpu_threads = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        hw.cpu_threads = os.cpu_count() or 1
    if data_dir is not None:
        probe = data_dir
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        hw.data_fs = mount_fs_type(str(probe), _read("/proc/mounts") or "")
        if sys.platform.startswith("linux") and not hw.wsl:
            hw.rotational = rotational_disk(str(probe))
    return hw


# ------------------------------------------------------------------------------------------- settings
@dataclasses.dataclass
class Settings:
    quant: str = C.DEFAULT_QUANT
    data_dir: Path = Path("/data")
    app_dir: Path = dataclasses.field(default_factory=lambda: Path(os.environ.get("LLMSERVE_APP_DIR")
                                                                   or Path(__file__).resolve().parent.parent))
    engine_dir: Path | None = None    # holds `strata`, `libstrata_kq.so`, `strata-vision`; default <app_dir>/bin
    context: int | str = 65536        # tokens, or "auto"
    kv: str = "int8"
    kv_streaming: str = "auto"        # auto | on | off (KV streaming is never used on WSL)
    parallel: int | None = None
    vision: str = "off"               # off | on | gpu | cpu
    vision_tokens: int | None = None
    mtp: bool = True
    draft_vocab: str | None = None    # cjk (default) | en | cyrillic | fr
    low_ram: str = "auto"             # auto | off | on | resident | mmap
    resident_budget_gib: float | None = None   # Unsloth quants: GiB of experts kept in RAM
    vram_reserve_mib: int | None = None
    rope_scaling: str | None = None   # yarn | linear | none (default: yarn past 262144 tokens)
    rope_scale: float | None = None
    gpu: int | None = None
    port: int = 8080
    extra_args: tuple = ()
    engine_env: dict = dataclasses.field(default_factory=dict)
    hf_token_env: str = "HF_TOKEN"
    hf_endpoint: str = ""
    force: bool = False
    batch_mtp: bool = False           # parallel >= 2: --batch-mtp (each slot also verifies one MTP proposal per window)
    batch_switches: bool = False      # parallel >= 2: the STRATA_BATCH_* / STRATA_SLOT_COPY_* switches (BATCH_SWITCH_ENV)
    agent: bool = False               # agent-friendly defaults (message-boundary cache, short-first admission, parked chats)
    conversation_cache_mib: int | None = None   # --conversation-cache-mib (default: the engine's own sizing, or 4096 with agent)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.app_dir = Path(self.app_dir)
        self.engine_dir = Path(self.engine_dir) if self.engine_dir else Path(
            os.environ.get("LLMSERVE_ENGINE_DIR") or self.app_dir / "bin")

    def fingerprint_items(self) -> dict:
        return {"quant": self.quant, "context": str(self.context), "kv": self.kv, "kv_streaming": self.kv_streaming,
                "parallel": self.parallel, "vision": self.vision, "vision_tokens": self.vision_tokens,
                "mtp": self.mtp, "draft_vocab": self.draft_vocab, "low_ram": self.low_ram,
                "resident_budget_gib": self.resident_budget_gib, "vram_reserve_mib": self.vram_reserve_mib,
                "rope_scaling": self.rope_scaling, "rope_scale": self.rope_scale, "gpu": self.gpu,
                "port": self.port, "extra_args": list(self.extra_args), "engine_env": dict(self.engine_env),
                "engine_dir": str(self.engine_dir), "data_dir": str(self.data_dir),
                "batch_mtp": self.batch_mtp, "batch_switches": self.batch_switches, "agent": self.agent,
                "conversation_cache_mib": self.conversation_cache_mib}


def parse_context(value: int | str) -> int | str:
    if isinstance(value, int):
        return value
    v = str(value).strip().lower()
    if v in ("", "auto"):
        return "auto"
    m = re.fullmatch(r"(\d+)\s*k", v)
    if m:
        return int(m.group(1)) * 1024
    if v.isdigit():
        return int(v)
    raise ValueError(f"cannot read the context {value!r}: use a number of tokens, 64k, 128k ... or auto")


def parse_vision(value: str) -> str:
    v = (value or "off").strip().lower()
    if v in ("", "off", "no", "none", "false", "0"):
        return "off"
    if v in ("on", "yes", "true", "1", "auto"):
        return "on"
    if v in ("gpu", "cpu"):
        return v
    raise ValueError(f"vision must be off, on, gpu or cpu, not {value!r}")


# ------------------------------------------------------------------------------------------- engine rules (setup.py)
def kv_cell(kv: str) -> int:
    return KV_CELL_BYTES.get(kv, 1056)


def low_ram_needed(q: C.Quant, ram: float) -> bool:
    """The experts do not fit the RAM with room left for the rest: they are mapped from the pack's experts.bin."""
    return ram < q.arena_gb + LOW_RAM_HEADROOM_GB


def low_ram_wanted(q: C.Quant, ram: float, choice: str = "auto") -> bool:
    if q.budget:                       # Unsloth's files always use their own RAM-budget mode
        return False
    return choice in ("on", "resident", "mmap") or (choice == "auto" and low_ram_needed(q, ram))


def low_ram_gpu_gb(q: C.Quant, vram_gb: float, ctx: int, kv: str, mtp: bool) -> float:
    """About how many GB of the experts the GPU's cache holds: VRAM less ~5 GB (dense weights, buffers, a 32K
    context), less the KV cache of a longer context, less ~1 GB for the draft layer."""
    longer = max(0, ctx - 32768) * 13 * kv_cell(kv) / 1e9
    return max(0.0, min(q.arena_gb, vram_gb - 5 - (MTP_VRAM_GB if mtp else 0.0) - longer))


def low_ram_resident(q: C.Quant, ram: float, vram_gb: float, ctx: int, kv: str, mtp: bool) -> bool:
    """The experts the GPU does not hold fit the RAM beside the usual room: copy them in (--resident-experts)."""
    return ram >= q.arena_gb - low_ram_gpu_gb(q, vram_gb, ctx, kv, mtp) + LOW_RAM_HEADROOM_GB


def kv_streaming_ram_gb(ctx: int, kv: str) -> float:
    """RAM of a streamed KV cache: ~13.7 KB per context token with 8-bit KV (1.7 GB at 128K)."""
    return ctx * 13 * kv_cell(kv) / 1e9


def resident_budget_default(q: C.Quant, ram: float, kv_ram_gb: float = 0.0) -> int:
    """Unsloth quants: GiB of experts the engine keeps in RAM: the RAM less 24 for the OS and the file cache that
    reads the other experts, less a streamed KV cache; at most all of them, at least 8.  64 GB: 40."""
    gib = round(ram) - UNSLOTH_RAM_LEFT_GB - math.ceil(kv_ram_gb)
    return max(8, min(gib, int(q.arena_gb / 1.073741824)))


def unsloth_budget_mode(q: C.Quant, ram: float, asked: float | None) -> bool:
    """Unsloth: the RAM budget mode when asked, or when the experts and 16 GiB more do not fit the RAM."""
    return q.budget and (asked is not None or ram < q.arena_gb / 1.073741824 + UNSLOTH_RESIDENT_BESIDE_GB)


def ctx_ram_need(q: C.Quant, ctx: int, low_ram: bool) -> float | None:
    """Setup's estimate of the RAM for a long context with IQ3_XXS / IQ3_S: experts + 8-bit KV + 24 GB of room."""
    if q.name not in ("IQ3_XXS", "IQ3_S") or low_ram:
        return None
    return q.arena_gb + ctx * 13 * 1056 / 1e9 + 24


def recommended_context(q: C.Quant, hw: Hardware, low_ram: bool, in_budget: bool) -> int:
    vram = hw.vram_gib or 24.0                         # unknown GPU: do not shrink the plan
    rec = 32768 if vram < 14 else 65536 if vram < 20 else 131072
    if in_budget:
        rec = 8192 if vram < 14 else 32768
    ram_cap = max(c for c in C.CONTEXTS if c <= 131072 or
                  (c != 204800 and (ctx_ram_need(q, c, low_ram) or 0) <= hw.ram_gib))
    return min(rec, ram_cap)


def resolve_rope(ctx: int, scaling: str | None, scale: float | None, trained: int = C.TRAINED_CONTEXT):
    """(method, factor) for the context served; (None, None) inside the trained range with no request."""
    if ctx <= trained:
        if scale is not None and scaling in (None, "none"):
            raise Fail("--rope-scale needs --rope-scaling linear or yarn (the context fits the trained 262144)")
        if scaling in (None, "none"):
            return None, None
        return scaling, scale if scale is not None else max(1.0, ctx / trained)
    if scaling == "none":
        raise Fail(f"a {ctx // 1024}K context is past the model's trained 262144 and --rope-scaling none keeps the "
                   "stock angles there; use yarn or linear, or a context of 262144 or less")
    return scaling or "yarn", scale if scale is not None else max(1.0, ctx / trained)


def parallel_slot_gb(ctx: int, kv: str, streaming: bool) -> float:
    kv_tok = 12 * (576 if kv == "q4_0" else 1056)
    return (min(ctx, 32768) if streaming else ctx) * kv_tok / 1e9 + 0.17


def parallel_cache_gb(vram_gb: float) -> float:
    """About how much VRAM the expert cache gets: the card less ~5 GB (dense weights, buffers, a 32K context)."""
    return max(0.0, vram_gb - 5)


def parallel_recommend(vram_gb: float, arena_gb: float, ctx: int, kv: str, streaming: bool) -> int:
    """The most slots (2-4) worth running on this card, 0 when none.  An ESTIMATE, from two rules:

    1. the expert cache (VRAM less ~5 GB) must hold at least PARALLEL_MIN_CACHE (40%) of the experts: a card that already holds
       less loses much more per slot (docs/BATCHING.md: 10-25% per request on a 12 GB card), so no recommendation;
    2. all slots together take at most PARALLEL_SHARE (13%) of that cache.

    Rule 2 compares with the cache, not with a fixed number of GB left over, so a card that reports 23.99 GiB instead of 24 does not
    flip the answer.  Q2_0 on a 24 GB card (cache ~19 GB, 34 GB of experts): 2 slots at 64K (1.0 GB each), 3 at 48K, 4 at 32K;
    3 at 64K or 2 at 128K (1.8 GB each) are not recommended.  With KV streaming a slot keeps at most 32K of KV in VRAM."""
    cache_gb = parallel_cache_gb(vram_gb)
    if cache_gb < PARALLEL_MIN_CACHE * arena_gb:
        return 0
    slot = parallel_slot_gb(ctx, kv, streaming)
    best = 0
    for n in range(2, PARALLEL_REC_MAX + 1):
        if n * slot <= PARALLEL_SHARE * cache_gb:
            best = n
    return best


def unsloth_env(q: C.Quant, in_budget: bool) -> dict:
    """The run config's "env" for Unsloth's files (setup.py unsloth_default_env, NVIDIA)."""
    env = {}
    if q.family != "unsloth":
        return env
    if q.name in ("UD-Q5_K_XL", "UD-Q6_K_XL"):
        env["STRATA_MMQ_KQ"] = "1"
    if q.name.startswith("UD-"):
        env["STRATA_HC_Q8"] = "1"
    if not in_budget:
        env["STRATA_ROUTE_TAIL_SKIP"] = "0"
    return env


# the engine arguments the person may override with --extra-args (strata_control/user_args.py semantics)
def arg_groups(args) -> list[list[str]]:
    out: list[list[str]] = []
    for a in args:
        a = str(a)
        if a.startswith("--") or not out:
            out.append([a])
        else:
            out[-1].append(a)
    return out


def merge_args(args: list[str], user: list[str]) -> list[str]:
    ug = arg_groups(user)
    mine = {g[0] for g in ug if g[0].startswith("--")}
    kept = [g for g in arg_groups(args) if g[0] not in mine]
    return [t for g in kept + ug for t in g]


def arg_after(args: list[str], flag: str):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


# ------------------------------------------------------------------------------------------- the plan
@dataclasses.dataclass
class Plan:
    quant: C.Quant
    family: C.Family
    ctx: int
    kv: str
    hw: Hardware
    settings: Settings
    shards: list                      # Paths
    mmproj: Path
    pack: Path
    mtp_dir: Path
    cfg_path: Path
    args: list
    cfg: dict
    notes: list                       # (level, text) level: ok | info | warn
    low_ram: bool = False
    resident: bool = False
    budget: float | None = None
    kv_streaming: bool = False
    vision: str = "off"
    pack_variant: str = "iq"          # iq (iq_pack.py) | q2_avx512 (strata_pack.py + pack_index.py)
    need_experts_bin: bool = False
    download_gb: float = 0.0
    disk_gb: float = 0.0
    fingerprint: str = ""

    @property
    def rt(self) -> Path:
        return self.mtp_dir / "rt"

    def warnings(self) -> list[str]:
        return [t for lvl, t in self.notes if lvl == "warn"]


def build_plan(s: Settings, hw: Hardware, *, ple_gb: float | None = None, files_present: bool = False) -> Plan:
    """Everything decided from the settings and the machine, with no disk or network access.  This is Strata's
    setup.py (steps 5 to 7 for Qwen3.8-Flash-Next) as one function; the numbers and the order of the engine arguments
    are the same.  `ple_gb`: size of the n-gram table when known (the catalog's 28.8 GB otherwise)."""
    q = C.resolve_quant(s.quant)
    fam = C.family_of(q)
    notes: list[tuple[str, str]] = []

    def ok_(t):
        notes.append(("ok", t))

    def info_(t):
        notes.append(("info", t))

    def warn_(t):
        notes.append(("warn", t))

    ram = hw.ram_gib
    vram = hw.vram_gib or 24.0                 # no GPU seen (dry run on a machine without one): assume a 24 GB card
    for t in hw.notes:
        warn_(t)
    if not hw.gpu_count:
        warn_("no NVIDIA GPU was found (nvidia-smi shows none): the plan assumes a 24 GB card; the container needs "
              "`gpus: all` and, on Windows, a current NVIDIA driver (docs/WINDOWS_DOCKER.md)")
    if hw.wsl:
        ok_("WSL2 (Docker Desktop): KV streaming and --ple-io ram stay off, the driver pins little there")

    # ---- the quant against the RAM
    in_budget = unsloth_budget_mode(q, ram, s.resident_budget_gib)
    low_ram = low_ram_wanted(q, ram, s.low_ram)
    if q.budget:
        if ram < q.ram_gb:
            have = resident_budget_default(q, ram)
            total = int(q.arena_gb / 1.073741824)
            warn_(f"{q.name} asks for {q.ram_gb} GB of RAM or more; this machine has {ram:.0f} GiB: it keeps {have} GiB of "
                  f"its {total} GiB of experts in RAM and reads the other {total - have} GiB from the SSD while it "
                  "answers (single-digit tokens/s on most SSDs). Choose Q2_0, IQ2_XS or IQ3_XXS, or give the "
                  "machine more RAM" + (" (.wslconfig memory=, docs/WINDOWS_DOCKER.md)" if hw.wsl else ""))
        elif in_budget:
            info_(f"{q.name}: a RAM budget of the most-used experts, the rest read from the model files on the SSD")
        else:
            ok_(f"all {q.arena_gb:.0f} GB of {q.name}'s experts fit in RAM")
    elif low_ram:
        smaller = " Coder IQ1_M needs ~32 GB." if q.name != "IQ1_M" else ""
        warn_(f"{ram:.0f} GiB of RAM is less than {q.name}'s experts ({q.arena_gb:.0f} GB) plus ~{LOW_RAM_HEADROOM_GB} GB "
              "for the system: low-RAM mode, the experts are read from the model folder on the SSD instead of pinned "
              "in RAM. It runs but is slower; more RAM (or a smaller quant) is better." + smaller
              + (" Raise the WSL2 VM's memory in %UserProfile%\\.wslconfig (docs/WINDOWS_DOCKER.md)." if hw.wsl else ""))
    elif ram and ram < q.ram_gb - 4:
        warn_(f"{q.name} asks for {q.ram_gb} GB of RAM and this machine has {ram:.0f} GiB: it may page")
    else:
        ok_(f"{q.name} fits: {q.arena_gb:.0f} GB of experts pinned in {ram:.0f} GiB of RAM")

    # ---- context, rope, KV
    ctx = s.context
    if ctx == "auto":
        ctx = recommended_context(q, hw, low_ram, in_budget)
        ok_(f"context: {ctx} tokens (auto: from the GPU's VRAM and the RAM)")
    else:
        ctx = int(ctx)
        if ctx < 2048:
            raise Fail(f"context {ctx} is too small")
        ok_(f"context: {ctx} tokens")
    need = ctx_ram_need(q, ctx, low_ram)
    if need is not None and ram < need and ctx > 131072:
        warn_(f"{ctx // 1024}K with {q.name} needs ~{need:.0f} GB of RAM by setup's estimate; this machine has "
              f"{ram:.0f}. It may be slower or run out of RAM under load")
    scaling, rope_scale = resolve_rope(ctx, s.rope_scaling, s.rope_scale)
    if scaling:
        ok_(f"rope scaling: {scaling}, factor {rope_scale:g}")
    kv = "fp16" if ctx <= 8192 else s.kv
    if kv not in KV_CHOICES:
        raise Fail(f"--kv must be one of {', '.join(KV_CHOICES)}, not {kv!r}")

    # ---- vision
    vision = s.vision
    if vision != "off" and not q.vision:
        warn_(f"pictures are not available with {q.name}: off")
        vision = "off"
    if vision == "on":
        vision = "gpu" if vram >= VISION_GPU_MIN_VRAM_GB else "cpu"
    if vision != "off":
        ok_(f"pictures: on (reader on the {vision.upper()}; the 0.9 GB reader file is downloaded with the model)")

    # ---- paths
    data = s.data_dir
    models_dir = data / "models" / q.tag
    shards = [models_dir / n for n in C.shard_names(q)]
    mmproj = data / "models" / C.MMPROJ
    pack = data / "packs" / q.tag.lower()
    mtp_dir = data / "mtp"
    rt = mtp_dir / "rt"
    cfg_path = data / "cfg.json"
    ple = shards[(q.ple_shard or len(shards)) - 1]

    # ---- how the pack is made
    pack_variant = "q2_avx512" if (q.name == "Q2_0" and fam.key == "qwen" and hw.avx512) else "iq"
    need_experts_bin = low_ram and pack_variant != "q2_avx512"

    # ---- low-RAM variant
    resident = False
    if low_ram:
        arena = q.arena_gb
        v = vram - (VISION[vision]["reserve_mib"] / 1024 if vision != "off" else 0)
        held = low_ram_gpu_gb(q, v, ctx, kv, s.mtp)
        resident = s.low_ram == "resident" or (s.low_ram != "mmap" and low_ram_resident(q, ram, v, ctx, kv, s.mtp))
        if resident:
            ok_(f"low-RAM mode: the GPU holds ~{100 * held / arena:.0f}% of {q.name}'s experts and the other "
                f"~{arena - held:.0f} GB stay in RAM, read once from a copy in the pack folder")
        else:
            ok_(f"low-RAM mode: the experts ({arena:.0f} GB) are read from the pack folder through the OS file cache; "
                f"the GPU holds ~{100 * held / arena:.0f}% of them")
            if held / arena < 0.6:
                warn_("most of the experts are read from the SSD while it answers: expect it to be much slower "
                      "than with enough RAM")

    # ---- the engine arguments, in setup.py's order
    profile = s.app_dir / "data" / fam.profile
    args: list[str] = ["--pack", str(pack), "--native", str(shards[0])]
    if len(shards) <= 2:
        args += ["--ple-gguf", str(ple)]               # a 3+ shard file: the engine finds the table from shard 1
    args += ["--expert-profile", str(profile), "--expert-cache", "auto", "--prefill", "auto", "--spec", "4"]
    if s.mtp:
        args += ["--spec-min-p", "0.5", "--mtp", str(rt)]
    args += ["--max-context", str(ctx)]
    if scaling:
        args += ["--rope-scaling", scaling, "--rope-scale", f"{rope_scale:g}"]
    if ctx > 8192:
        args += ["--kv", kv]
    if low_ram:
        args += ["--resident-experts" if resident else "--mmap-experts"]

    table_gb = ple_gb or 28.8
    if hw.rotational and not hw.wsl:
        if ram >= q.ram_gb + table_gb + 4:
            args += ["--ple-io", "ram"]
            ok_(f"the model is on a rotational disk ({hw.rotational}): the {table_gb:.0f} GB n-gram table is kept in RAM")
        else:
            warn_(f"the data folder is on a rotational disk ({hw.rotational}): the n-gram table is read from it at "
                  f"random, which can stall prompts for minutes. Use an SSD, or ~{table_gb:.0f} GB more RAM")
    elif not low_ram and not hw.wsl and sys.platform.startswith("linux"):
        if ram >= q.ram_gb + table_gb + 16:
            args += ["--ple-io", "ram"]
            ok_(f"{ram:.0f} GiB of RAM: the {table_gb:.0f} GB n-gram table is kept in RAM (--ple-io ram)")

    # KV streaming: from 64K up the whole KV cache lives in RAM; the VRAM it frees holds more experts
    kv_ram = kv_streaming_ram_gb(ctx, kv)
    stream_fits = ram >= q.ram_gb + kv_ram + 1
    budget = None
    if in_budget:
        budget = s.resident_budget_gib if s.resident_budget_gib is not None else resident_budget_default(q, ram)
    streaming = False
    if hw.wsl and ctx >= 65536:
        ok_("WSL2: KV streaming off (the driver pins only about 1 GB of RAM); the KV cache stays in VRAM")
        if s.kv_streaming == "on":
            warn_("--kv-streaming on is ignored on WSL2: it cannot pin the RAM copy")
    elif ctx >= 65536 and s.kv_streaming == "off":
        ok_("KV streaming off, as you chose: the KV cache stays in VRAM")
    elif ctx >= 65536 and (stream_fits or s.kv_streaming == "on"):
        args += ["--kv-resident", "32768"]
        streaming = True
        ok_(f"KV streaming on: the KV cache lives in RAM ({kv_ram:.1f} GB), more experts fit in VRAM")
        if not stream_fits:
            warn_(f"KV streaming needs ~{kv_ram:.1f} GB of RAM beside the ~{q.ram_gb} GB {q.name} uses; this machine "
                  f"has {ram:.0f}. Kept as you chose: it may page or run out of RAM under load")
        if in_budget and s.resident_budget_gib is None:
            budget = resident_budget_default(q, ram, kv_ram)
    elif ctx >= 65536:
        ok_(f"KV streaming off: it needs ~{kv_ram:.1f} GB of RAM beside the ~{q.ram_gb} GB {q.name} uses, and this "
            f"machine has {ram:.0f}; the KV cache stays in VRAM")
    if budget is not None:
        args += ["--resident-budget-gib", f"{budget:g}"]

    if vision != "off":
        args += ["--vision"]
        if VISION[vision]["reserve_mib"]:
            args += ["--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
    if s.vram_reserve_mib is not None:
        if "--vram-reserve-mib" in args:
            args[args.index("--vram-reserve-mib") + 1] = str(s.vram_reserve_mib)
        else:
            args += ["--vram-reserve-mib", str(s.vram_reserve_mib)]
        ok_(f"VRAM kept free for other programs: {s.vram_reserve_mib} MiB")
    if hw.vram_gib and hw.vram_gib < 14 and not s.draft_vocab and s.mtp:
        info_(f"tip for a {hw.vram_gib:.0f} GB card: LLMSERVE_DRAFT_VOCAB=en uses ~215 MiB less VRAM for the draft "
              "head (English and code answers); it is the fix when the start says \"the draft head does not fit\"")

    # ---- parallel requests (cfg "parallel": the server adds --batch N)
    parallel = s.parallel
    if parallel is not None and parallel > PARALLEL_MAX:
        warn_(f"the engine runs at most {PARALLEL_MAX} requests at once; --parallel {parallel} becomes {PARALLEL_MAX}")
        parallel = PARALLEL_MAX
    batching = bool(parallel and parallel >= 2)

    # ---- settings that add engine arguments / engine environment (the person's --extra-args / --engine-env win over them)
    gen_env: dict = {}
    if s.batch_mtp:
        if not batching:
            info_("LLMSERVE_BATCH_MTP=on does nothing with LLMSERVE_PARALLEL=1: it needs two or more slots")
        elif not s.mtp:
            warn_("LLMSERVE_BATCH_MTP=on needs the draft layer (LLMSERVE_MTP=on); --batch-mtp left out")
        else:
            args += ["--batch-mtp"]
            if parallel > 4:
                info_(f"batch MTP: a window holds 8 rows, 2 per slot, so {parallel} slots take turns (4 at a time)")
            ok_("batch MTP: every slot also verifies one draft proposal per window (--batch-mtp). If the start log says "
                "\"MTP admission ... failed\", set LLMSERVE_BATCH_MTP=off")
    cache_mib = s.conversation_cache_mib
    if s.agent and cache_mib is None:
        room_mib = int((ram - q.ram_gb - CONVERSATION_CACHE_RAM_LEFT_GB) * 1024)
        cache_mib = min(AGENT_CONVERSATION_CACHE_MIB, room_mib)
        if cache_mib < 1024:
            warn_(f"agent mode: only ~{max(room_mib, 0) // 1024} GiB of RAM is left beside {q.name} ({ram:.0f} GiB, it asks "
                  f"{q.ram_gb}); conversation parking is left to the engine's own sizing")
            cache_mib = None
    if cache_mib is not None:
        args += ["--conversation-cache-mib", str(cache_mib)]
        ok_(f"parked conversations: up to {cache_mib} MiB of RAM (--conversation-cache-mib)"
            + ("" if s.conversation_cache_mib is not None else " (agent mode)"))
    if s.agent:
        gen_env.update(AGENT_ENGINE_ENV)
        ok_("agent mode: prompt snapshot at the message boundary (STRATA_CACHE_MESSAGE_BOUNDARY=1)"
            + (f"; short prompts are admitted first (STRATA_ADMIT_SHORT_FIRST={AGENT_ADMIT_SHORT_FIRST}, exported by the "
               "container entry point for the server)" if batching else ""))

    # ---- --extra-args on top (their value wins for the same flag)
    user = list(s.extra_args)
    if user:
        args = merge_args(args, user)
        ok_("extra engine arguments: " + shlex.join(user))

    if s.batch_switches:
        if not batching:
            info_("LLMSERVE_BATCH_SWITCHES=on does nothing with LLMSERVE_PARALLEL=1: it needs two or more slots")
        else:
            sw = dict(BATCH_SWITCH_ENV)
            if "--kv-resident" in args:                   # a streamed KV cache: one launch per row stays (verify.cpp)
                del sw["STRATA_BATCH_ATTN_ROWS"]
                info_("batch switches: STRATA_BATCH_ATTN_ROWS left out (the KV cache is streamed, --kv-resident)")
            gen_env.update(sw)
            ok_("batch switches: " + " ".join(f"{k}={v}" for k, v in sw.items()))

    if batching:
        rec = parallel_recommend(vram, q.arena_gb, ctx, kv, streaming)
        slot = parallel_slot_gb(ctx, kv, streaming)
        cache = parallel_cache_gb(vram)
        share = f" = ~{100 * slot / cache:.0f}% of the ~{cache:.0f} GB expert cache" if cache > 0 else ""
        ok_(f"parallel requests: {parallel} at once (estimate: each takes ~{slot:.1f} GB of VRAM{share}; "
            "the engine log prints the exact size)")
        if not rec:
            if cache < PARALLEL_MIN_CACHE * q.arena_gb:
                warn_("parallel > 1 costs about 10-25% speed per request on a card that holds only part of the experts "
                      "(estimate, docs/BATCHING.md); kept as you chose")
            else:
                warn_(f"estimate: {parallel} slots at this context would take ~{100 * parallel * slot / cache:.0f}% of the expert "
                      f"cache, more than the {100 * PARALLEL_SHARE:.0f}% LLMServe recommends; on a card this size each slot "
                      "costs roughly 5% of the single-request speed. Kept as you chose; a shorter context makes slots cheaper")
        elif parallel > rec:
            warn_(f"recommended for this card at this context: {rec} (estimate: more slots take over "
                  f"{100 * PARALLEL_SHARE:.0f}% of the expert cache); kept as you chose")

    # ---- disk
    shard_gb = sum(f.size for f in C.shard_files(q)) / 1e9
    download_gb = shard_gb + (C.mmproj_file(q).size / 1e9 if vision != "off" else 0.0)
    disk_gb = download_gb + (40 if pack_variant == "q2_avx512" else 0) + (q.arena_gb + 1 if need_experts_bin else 0) \
        + (C.MTP_DOWNLOAD_GB if s.mtp else 0)
    if not files_present:
        info_(f"disk: ~{disk_gb:.0f} GB needed in {data} ({download_gb:.0f} GB download"
              + (", +40 GB for the AVX-512 repack of the Q2_0 experts" if pack_variant == "q2_avx512" else "") + ")")
    if hw.data_fs in SLOW_MOUNT_TYPES:
        warn_(f"{data} is on a '{hw.data_fs}' mount (a Windows drive or a network share): reading tens of GB of "
              "model files through it is slow and loading takes much longer. Keep /data in the named Docker volume "
              "(docs/WINDOWS_DOCKER.md)")
    if hw.wsl and str(data).startswith("/mnt/"):
        warn_(f"{data} is on a Windows drive (/mnt/...): much slower than the WSL2 disk; use a Docker named volume")

    # ---- the run config
    lib_dirs = [d for d in ("/usr/local/cuda/lib64", "/usr/lib/wsl/lib") if os.path.isdir(d)] or ["/usr/local/cuda/lib64"]
    cfg: dict = {"exe": str(s.engine_dir / "strata"), "args": args, "cwd": str(data),
                 "tokenizer": str(pack / "tokenizer"), "model_name": f"{fam.model_name}-{q.name.lower()}",
                 "log": str(data / "logs" / "engine.log"), "lib_dirs": lib_dirs, "port": s.port}
    env = {**unsloth_env(q, in_budget), **gen_env, **s.engine_env}
    if env:
        cfg["env"] = env
    if hw.gpu_count > 1 or s.gpu is not None:
        cfg["gpu"] = hw.gpu_index if s.gpu is None else s.gpu
        cfg["gpus_asked"] = True
    if s.draft_vocab:
        cfg["draft_vocab"] = s.draft_vocab
    if not s.mtp:
        cfg["mtp"] = False
        ok_("the draft layer (MTP) is off: ~5 GB less to download, slower words per second")
    if parallel is not None:
        cfg["parallel"] = parallel if parallel >= 2 else 1
    if vision != "off":
        vt = s.vision_tokens or VISION[vision]["max_tokens"]
        cfg["vision"] = {"exe": str(s.engine_dir / "strata-vision"), "mmproj": str(mmproj), "model": str(shards[0]),
                         "gpu": vision == "gpu", "max_tokens": vt}
        if vision == "cpu":
            cfg["vision"]["threads"] = max(1, min(hw.cpu_threads // 2, VISION_THREADS_CAP))
    if user:
        cfg["user_args"] = user

    plan = Plan(quant=q, family=fam, ctx=ctx, kv=kv, hw=hw, settings=s, shards=shards, mmproj=mmproj, pack=pack,
                mtp_dir=mtp_dir, cfg_path=cfg_path, args=args, cfg=cfg, notes=notes, low_ram=low_ram,
                resident=resident, budget=budget, kv_streaming=streaming, vision=vision, pack_variant=pack_variant,
                need_experts_bin=need_experts_bin, download_gb=download_gb, disk_gb=disk_gb)
    plan.fingerprint = fingerprint_of(s, hw, q, ctx)
    return plan


def fingerprint_of(s: Settings, hw: Hardware, q: C.Quant, ctx: int) -> str:
    blob = json.dumps({"v": FORMAT_VERSION, "settings": s.fingerprint_items(), "hw": hw.facts(), "ctx": ctx,
                       "rev": C.HF_REVISIONS[C.family_of(q).repo]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------- markers and tools
def marker(data_dir: Path, name: str) -> Path:
    return data_dir / "state" / f"{name}.done"


def marker_key(data_dir: Path, name: str) -> str | None:
    """The key a step's marker was written with, or None when there is no (readable) marker."""
    try:
        return json.loads(marker(data_dir, name).read_text()).get("key", "")
    except (OSError, ValueError, AttributeError):
        return None


def marker_ok(data_dir: Path, name: str, key: str = "") -> bool:
    return marker_key(data_dir, name) == key


def mark_done(data_dir: Path, name: str, key: str = "") -> None:
    p = marker(data_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    os.replace(tmp, p)


def unmark(data_dir: Path, name: str) -> None:
    marker(data_dir, name).unlink(missing_ok=True)


_child: subprocess.Popen | None = None
_stopping = False


def _on_term(signum, _frame):
    """docker stop: end the tool we started and leave; every step is resumable."""
    global _stopping
    _stopping = True
    say(f"[prepare] stop requested (signal {signum}); what is downloaded is kept, run again to continue")
    if _child is not None and _child.poll() is None:
        try:
            _child.terminate()
        except OSError:
            pass
    raise SystemExit(143)


def run_tool(cmd: list[str], env: dict, what: str, cwd: Path | None = None, check: bool = True) -> int:
    global _child
    say(f"[prepare] run   {what}: {shlex.join(cmd[:1] + [Path(cmd[1]).name] + cmd[2:]) if len(cmd) > 1 else cmd[0]}")
    t0 = time.time()
    _child = subprocess.Popen(cmd, env=env, cwd=cwd)
    try:
        rc = _child.wait()
    finally:
        _child = None
    if rc != 0 and check:
        raise Fail(f"{what} failed (exit {rc})", "the lines above say why; run the container again: finished steps are kept")
    if rc == 0:
        info(f"{what} done in {time.time() - t0:.0f} s")
    return rc


def tool_env(s: Settings) -> dict:
    env = dict(os.environ)
    env.setdefault("STRATA_GGUF_PY", str(s.app_dir.parent / "gguf-py"))
    env["PYTHONUNBUFFERED"] = "1"
    if s.hf_endpoint:
        env["HF_ENDPOINT"] = s.hf_endpoint
    return env


def tool(s: Settings, name: str) -> str:
    p = s.app_dir / "tools" / name
    if not p.exists():
        raise Fail(f"{p} is missing: this is not a complete LLMServe install")
    return str(p)


# ------------------------------------------------------------------------------------------- downloads
def hash_file(path: Path, size: int, label: str) -> str:
    h = hashlib.sha256()
    done, last = 0, time.time()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
            done += len(b)
            if time.time() - last > 20:
                last = time.time()
                info(f"checking {label}: {done / 1e9:.1f} / {size / 1e9:.1f} GB")
    return h.hexdigest()


def verified(path: Path, rf: C.RemoteFile) -> bool:
    """A finished file whose marker says it matched this SHA-256 (the marker is written only after the hash)."""
    m = path.with_name(path.name + ".done")
    try:
        return path.exists() and path.stat().st_size == rf.size and f"sha256 {rf.sha256}" in m.read_text()
    except OSError:
        return False


def verify_file(path: Path, rf: C.RemoteFile) -> None:
    if verified(path, rf):
        return
    size = path.stat().st_size if path.exists() else -1
    if size != rf.size:
        path.unlink(missing_ok=True)
        raise Fail(f"{path.name} is {size:,} bytes, not the pinned {rf.size:,}: deleted, run again to download it anew")
    info(f"checking {path.name} (SHA-256, {rf.size / 1e9:.1f} GB) ...")
    digest = hash_file(path, rf.size, path.name)
    if digest != rf.sha256:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".done").unlink(missing_ok=True)
        raise Fail(f"{path.name} does not match the pinned SHA-256 ({digest[:12]}... instead of {rf.sha256[:12]}...): "
                   "deleted, run again to download it anew")
    path.with_name(path.name + ".done").write_text(f"sha256 {rf.sha256} {time.strftime('%Y-%m-%d %H:%M')}")


class _NoAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Hugging Face redirects to a CDN; the token must not follow it to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_NoAuthOnRedirect)


def download_urllib(url: str, part: Path, total: int, label: str, token: str = "", attempts: int = 30,
                    wait: float = 10.0) -> None:
    """One stream with a Range resume from the size of `part`; a resumed answer counts only when it is a 206 that starts
    at the byte asked for, anything else starts the file over."""
    part.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:
            return
        headers = {"User-Agent": "llmserve-prepare", "Range": f"bytes={have}-"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with _opener.open(urllib.request.Request(url, headers=headers), timeout=60) as r:
                if have:
                    m = re.match(r"\s*bytes\s+(\d+)-(\d+)/(\d+)", r.headers.get("Content-Range") or "")
                    if r.status != 206 or not m or int(m.group(1)) != have:
                        have = 0                          # the server ignored the range: start over
                elif r.status not in (200, 206):
                    raise OSError(f"HTTP {r.status}")
                last = 0.0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(8 << 20)
                        if not b:
                            break
                        if total and have + len(b) > total:
                            raise OSError("the server sent more bytes than the file has")
                        f.write(b)
                        have += len(b)
                        if time.time() - last > 30:
                            last = time.time()
                            info(f"downloading {label}: {have / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * have / max(total, 1):.0f}%)")
                    f.flush()
                    os.fsync(f.fileno())
            if not total or have >= total:
                return
        except (OSError, http.client.HTTPException) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise Fail(f"{label}: HTTP {e.code} from {urllib.parse.urlparse(url).netloc}",
                           "a gated or renamed file? set HF_TOKEN in .env if the repository asks for a login") from None
            warn(f"download of {label} interrupted ({e}); retrying in {wait:.0f} s")
            time.sleep(wait)
    raise Fail(f"could not finish downloading {label}", "check the connection and run again: it resumes")


def _poll_progress(stop: threading.Event, root: Path, total: int, label: str) -> None:
    last_bytes, last_t = 0, time.time()
    while not stop.wait(30):
        size = 0
        for p in root.rglob("*.incomplete"):
            try:
                size += p.stat().st_size
            except OSError:
                pass
        now = time.time()
        rate = (size - last_bytes) / max(now - last_t, 1e-6) / 1e6
        last_bytes, last_t = size, now
        if size:
            info(f"downloading {label}: {size / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * size / max(total, 1):.0f}%) {rate:.0f} MB/s")


def download_hf(rf: C.RemoteFile, dst: Path, token: str, endpoint: str) -> bool:
    """huggingface_hub's resumable download (hf_xet when installed: several connections).  False: not usable here."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    tmp_root = dst.parent / ".hf-tmp"
    stop = threading.Event()
    th = threading.Thread(target=_poll_progress, args=(stop, tmp_root, rf.size, dst.name), daemon=True)
    th.start()
    try:
        for attempt in range(1, 6):
            try:
                got = hf_hub_download(repo_id=rf.repo, filename=rf.path, revision=rf.revision, local_dir=str(tmp_root),
                                      token=token or None, **({"endpoint": endpoint} if endpoint else {}))
                os.replace(got, dst)
                return True
            except TypeError:
                return False                      # an API change: use the built-in downloader
            except Exception as e:                # network errors, 5xx, a stalled xet session: retry, it resumes
                name = type(e).__name__
                if name in ("GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError"):
                    raise Fail(f"{dst.name}: {name} from Hugging Face ({e})",
                               "set HF_TOKEN in .env if the repository asks for a login") from None
                warn(f"download of {dst.name} interrupted ({name}: {str(e)[:120]}); attempt {attempt} of 5, retrying in 15 s")
                time.sleep(15)
        raise Fail(f"could not finish downloading {dst.name}", "check the connection and run again: it resumes")
    finally:
        stop.set()
        if dst.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)


def find_twin(models_root: Path, rf: C.RemoteFile) -> Path | None:
    """The same file (name, size, hash) already verified in another model folder: the n-gram table is shared."""
    for p in models_root.glob(f"*/{rf.name}"):
        if verified(p, rf):
            return p
    return None


def fetch(rf: C.RemoteFile, dst: Path, s: Settings, backend: str = "auto") -> None:
    """Make `dst` the pinned file: skip when verified, link a twin, else download (resume) and verify."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if verified(dst, rf):
        ok(f"{dst.name} already downloaded and verified")
        return
    if dst.exists() and dst.stat().st_size == rf.size:
        verify_file(dst, rf)
        ok(f"{dst.name} present, verified")
        return
    twin = find_twin(dst.parent.parent, rf) if dst.parent.parent.exists() else None
    if twin is not None and twin != dst:
        try:
            os.link(twin, dst)
            dst.with_name(dst.name + ".done").write_text(f"sha256 {rf.sha256} linked from {twin.parent.name}")
            ok(f"{dst.name} shared with {twin.parent.name} (identical file, no second copy)")
            return
        except OSError:
            pass
    token = os.environ.get(s.hf_token_env, "")
    endpoint = (s.hf_endpoint or os.environ.get("HF_ENDPOINT", "")).strip().rstrip("/")
    info(f"downloading {dst.name} ({rf.size / 1e9:.1f} GB) from {endpoint or 'huggingface.co'} at revision {rf.revision[:8]}")
    t0 = time.time()
    done = False
    if backend in ("auto", "hf"):
        done = download_hf(rf, dst, token, endpoint)
    if not done:
        part = dst.with_name(dst.name + ".part")
        download_urllib(rf.url(endpoint or "https://huggingface.co"), part, rf.size, dst.name, token)
        os.replace(part, dst)
    info(f"downloaded {dst.name} in {time.time() - t0:.0f} s")
    verify_file(dst, rf)
    ok(f"{dst.name} downloaded and verified")


# ------------------------------------------------------------------------------------------- draft vocab
def refresh_draft_vocab(app_dir: Path, rt: Path, choice: str) -> None:
    """Copy data/draft_vocab*.bin into the MTP folder: when missing, or when it holds another shipped subset (a file
    made by hand is kept)."""
    new, dst = app_dir / "data" / C.DRAFT_VOCABS.get(choice, "draft_vocab.bin"), rt / "draft_vocab.bin"
    if not new.exists() or not rt.is_dir():
        return
    if dst.exists():
        old = hashlib.sha256(dst.read_bytes()).hexdigest()
        shipped = OLD_DRAFT_VOCABS | {hashlib.sha256((app_dir / "data" / f).read_bytes()).hexdigest()
                                      for f in C.DRAFT_VOCABS.values() if (app_dir / "data" / f).exists()}
        if old not in shipped or old == hashlib.sha256(new.read_bytes()).hexdigest():
            return
    shutil.copyfile(new, dst)
    ok(f"draft layer: token subset '{choice}'")


# ------------------------------------------------------------------------------------------- the steps
def check_environment(plan: Plan) -> None:
    s, hw = plan.settings, plan.hw
    if not hw.gpu_count:
        raise Fail("no NVIDIA GPU is visible in this container",
                   "run with `gpus: all` (docker compose) / `--gpus all`, with a current NVIDIA driver on Windows; "
                   "check with: docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi", 3)
    exe = s.engine_dir / "strata"
    if not exe.exists():
        raise Fail(f"the engine is not at {exe}", "build the image (docker compose build) or set LLMSERVE_ENGINE_DIR", 3)
    if plan.vision != "off" and not (s.engine_dir / "strata-vision").exists():
        raise Fail(f"pictures were asked for but {s.engine_dir / 'strata-vision'} is missing",
                   "build the image with --build-arg WITH_VISION=1, or set LLMSERVE_VISION=off", 3)


def free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def disk_needed_gb(plan: Plan) -> float:
    """What the steps still have to write: finished and partial downloads, and finished conversions, count as done."""
    have = 0.0
    for rf, dst in plan_files(plan):
        for f in (dst, dst.with_name(dst.name + ".part")):
            if f.is_file():
                have += min(f.stat().st_size, rf.size) / 1e9
                break
    todo = max(0.0, plan.download_gb - have)
    if plan.pack_variant == "q2_avx512" and not (plan.pack / "experts.bin").exists():
        todo += 40
    if plan.need_experts_bin and not (plan.pack / "experts.bin").exists():
        todo += plan.quant.arena_gb + 1
    if plan.settings.mtp and not (plan.rt / "experts.bin").exists():
        todo += C.MTP_DOWNLOAD_GB
    return todo


def plan_files(plan: Plan) -> list[tuple[C.RemoteFile, Path]]:
    pairs = list(zip(C.shard_files(plan.quant), plan.shards))
    if plan.vision != "off":
        pairs.append((C.mmproj_file(plan.quant), plan.mmproj))
    return pairs


def step_download(plan: Plan, backend: str) -> None:
    for rf, dst in plan_files(plan):
        fetch(rf, dst, plan.settings, backend)


def step_pack(plan: Plan) -> None:
    s, q = plan.settings, plan.quant
    pack, shard1 = plan.pack, plan.shards[0]
    key = plan.pack_variant + ("+bin" if plan.need_experts_bin else "") + ":" + shard1.name
    name = f"pack-{q.tag}"
    has_tok = (pack / "tokenizer" / "vocab.json").exists()
    if plan.pack_variant == "q2_avx512":
        complete = (pack / "index.txt").exists() and (pack / "experts.bin").exists() and has_tok
    else:
        complete = (pack / "native_experts.txt").exists() and has_tok and \
            (not plan.need_experts_bin or (pack / "experts.bin").exists())
    old = marker_key(s.data_dir, name)
    if complete and not s.force and old in (None, key):
        ok(f"model pack present: {pack}")
        mark_done(s.data_dir, name, key)
        return
    if complete and old not in (None, key) and not s.force:
        info(f"the pack was made another way ({old.split(':')[0]}), now {plan.pack_variant}: making it again")
        shutil.rmtree(pack, ignore_errors=True)
    env = tool_env(s)
    py = sys.executable
    if plan.pack_variant == "q2_avx512":
        if pack.exists():
            shutil.rmtree(pack, ignore_errors=True)
        info("converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min)")
        run_tool([py, tool(s, "strata_pack.py"), "build", "--gguf", str(shard1), "--out", str(pack), "--skip-hash",
                  "--force"], env, "strata_pack build")
        run_tool([py, tool(s, "pack_index.py"), "--pack", str(pack)], env, "pack_index")
        run_tool([py, tool(s, "strata_tokenizer.py"), "--gguf", str(shard1), "--out", str(pack)], env, "tokenizer")
    else:
        if s.force and pack.exists():
            shutil.rmtree(pack, ignore_errors=True)
        extra = list(plan.family.pack_args)
        if not ((pack / "native_experts.txt").exists() and has_tok):
            run_tool([py, tool(s, "iq_pack.py"), "--gguf", str(shard1), "--out", str(pack), *extra], env, "iq_pack")
        if plan.need_experts_bin and not (pack / "experts.bin").exists():
            info(f"writing the experts into one file for the low-RAM mode (one time, {q.arena_gb:.0f} GB)")
            run_tool([py, tool(s, "iq_pack.py"), "--gguf", str(shard1), "--out", str(pack), "--experts-bin", *extra],
                     env, "iq_pack --experts-bin")
    mark_done(s.data_dir, name, key)
    ok(f"model pack ready: {pack}")


def step_mtp(plan: Plan) -> None:
    s = plan.settings
    if not s.mtp:
        ok("MTP draft layer off: nothing to fetch")
        return
    mtp, rt = plan.mtp_dir, plan.rt
    env = tool_env(s)
    py = sys.executable
    if s.force:
        for n in ("mtp-fetch", "mtp-pack", "mtp-rt"):
            unmark(s.data_dir, n)
        shutil.rmtree(rt, ignore_errors=True)
        (mtp / "mtp-q2_0.gguf").unlink(missing_ok=True)
    if (rt / "experts.bin").exists() and marker_key(s.data_dir, "mtp-rt") in (None, C.MTP_REVISION):
        ok(f"MTP draft layer present: {rt}")                 # (no marker: made by hand or by Strata; kept)
        mark_done(s.data_dir, "mtp-rt", C.MTP_REVISION)
    else:
        if not marker_ok(s.data_dir, "mtp-fetch", C.MTP_REVISION):
            info("the MTP draft layer (about 2x faster output) comes from the original Qwen checkpoint: only its "
                 f"~{C.MTP_DOWNLOAD_GB:.0f} GB of MTP tensors are downloaded")
            for n in range(1, 4):
                rc = run_tool([py, tool(s, "mtp_fetch.py"), "fetch", "--out", str(mtp)], env, "mtp_fetch", check=False)
                if rc == 0:
                    break
                warn(f"the draft layer's download stopped (exit {rc}); trying again, it continues where it stopped ({n} of 3)")
                time.sleep(5)
            else:
                raise Fail("the draft layer's download keeps stopping",
                           "run the container again (what is downloaded is kept), or set LLMSERVE_MTP=off")
            rc = run_tool([py, tool(s, "mtp_fetch.py"), "verify", "--out", str(mtp)], env, "mtp_fetch verify", check=False)
            if rc != 0:
                raise Fail("some MTP tensors do not match the pinned checkpoint (a mirror that ignored range requests?)",
                           "delete /data/mtp and run again")
            mark_done(s.data_dir, "mtp-fetch", C.MTP_REVISION)
        gguf = mtp / "mtp-q2_0.gguf"
        if not (gguf.exists() and marker_ok(s.data_dir, "mtp-pack", C.MTP_REVISION)):
            run_tool([py, tool(s, "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0", "--out", str(gguf)], env, "mtp_pack")
            mark_done(s.data_dir, "mtp-pack", C.MTP_REVISION)
        run_tool([py, tool(s, "mtp_rt.py"), "--gguf", str(gguf), "--out", str(rt)], env, "mtp_rt")
        mark_done(s.data_dir, "mtp-rt", C.MTP_REVISION)
        ok(f"MTP draft layer ready: {rt}")
    refresh_draft_vocab(s.app_dir, rt, s.draft_vocab or C.DEFAULT_DRAFT_VOCAB)


def measure_ple_gb(plan: Plan) -> float | None:
    """Size of per_layer_token_embd.weight when the GGUF reader and the shards are there (None: use the catalog's)."""
    try:
        sys.path.insert(0, str(plan.settings.app_dir / "tools"))
        from gguf_reader import GGUFFile                      # noqa: E402
        for sh in plan.shards:
            for t in GGUFFile(sh).tensors:
                if t.name == "per_layer_token_embd.weight":
                    n = getattr(t, "expected_bytes", lambda: None)()
                    return n / 1e9 if n else None
    except Exception:
        return None
    return None


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def step_config(plan: Plan, ple_gb: float | None) -> Plan:
    s = plan.settings
    if ple_gb is not None:                                     # the exact table size may change --ple-io
        plan = build_plan(s, plan.hw, ple_gb=ple_gb, files_present=True)
    Path(plan.cfg["log"]).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(plan.cfg_path, plan.cfg)
    write_json_atomic(s.data_dir / "state" / "prepare.json", {
        "fingerprint": plan.fingerprint, "format": FORMAT_VERSION, "quant": plan.quant.name, "context": plan.ctx,
        "settings": s.fingerprint_items(), "hardware": plan.hw.facts(), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine_args": plan.args})
    ok(f"run config written: {plan.cfg_path}")
    return plan


def up_to_date(plan: Plan) -> bool:
    """cfg.json exists, was made from these settings on this kind of machine, and its files are all still there."""
    try:
        st = json.loads((plan.settings.data_dir / "state" / "prepare.json").read_text())
        cfg = json.loads(plan.cfg_path.read_text())
    except (OSError, ValueError):
        return False
    if st.get("fingerprint") != plan.fingerprint or st.get("format") != FORMAT_VERSION:
        return False
    need = [plan.shards[0], plan.pack / "tokenizer" / "vocab.json"]
    if plan.settings.mtp:
        need.append(plan.rt / "experts.bin")
    return all(p.exists() for p in need) and cfg.get("exe") == plan.cfg["exe"]


# ------------------------------------------------------------------------------------------- reporting
def print_plan(plan: Plan, dry: bool) -> None:
    hw, q, s = plan.hw, plan.quant, plan.settings
    say(f"[prepare] LLMServe prepare{' (dry run: nothing is downloaded or written)' if dry else ''}")
    gpu = f"{hw.gpu_name} {hw.vram_gib:.1f} GiB sm_{hw.compute_cap.replace('.', '')} driver {hw.driver}" if hw.gpu_count \
        else "none found"
    info(f"RAM:  {hw.ram_gib:.1f} GiB usable (host {hw.host_ram_gib:.1f}"
         + (f", container limit {hw.cgroup_gib:.1f}" if hw.cgroup_gib else "") + ")")
    info(f"GPU:  {gpu}" + (f" ({hw.gpu_count} cards, using {hw.gpu_index})" if hw.gpu_count > 1 else ""))
    info(f"CPU:  {hw.cpu_name}, {hw.cpu_threads} threads, AVX2 {'yes' if hw.avx2 else 'no'}, "
         f"AVX-512 {'yes' if hw.avx512 else 'no'}")
    info(f"WSL2: {'yes' if hw.wsl else 'no'}    data dir: {s.data_dir} ({hw.data_fs or 'file system unknown'})")
    info(f"model: {plan.family.title} {q.name}: {q.about}")
    info(f"       download {plan.download_gb:.1f} GB, experts in RAM {q.arena_gb:.1f} GB, RAM asked {q.ram_gb} GB")
    for lvl, text in plan.notes:
        (warn if lvl == "warn" else ok if lvl == "ok" else info)(text)
    say("[prepare] plan")
    for i, line in enumerate(plan_steps(plan), 1):
        info(f"{i}. {line}")
    say("[prepare] engine arguments:")
    say("  " + shlex.join(plan.args))
    if plan.cfg.get("parallel", 1) >= 2:
        say(f"  (the server adds: --batch {plan.cfg['parallel']})")
    extra = {k: v for k, v in plan.cfg.items() if k not in ("args",)}
    say("[prepare] run config (cfg.json, without args):")
    say(json.dumps(extra, indent=2))


def plan_steps(plan: Plan) -> list[str]:
    s = plan.settings
    out = []
    names = [dst.name + (" (done)" if verified(dst, rf) else "") for rf, dst in plan_files(plan)]
    out.append("download (resumable, SHA-256 checked): " + ", ".join(names))
    if plan.pack_variant == "q2_avx512":
        out.append("pack: tools/strata_pack.py build + pack_index.py (AVX-512 repack of the Q2_0 experts, ~40 GB) + tokenizer")
    else:
        out.append("pack: tools/iq_pack.py" + (" + --experts-bin (low-RAM mode)" if plan.need_experts_bin else ""))
    out.append("MTP draft layer: mtp_fetch.py (~5 GB by range reads), mtp_pack.py, mtp_rt.py" if s.mtp
               else "MTP draft layer: off")
    out.append(f"write {plan.cfg_path}")
    return out


# ------------------------------------------------------------------------------------------- command line
def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_int(name: str):
    v = os.environ.get(name, "").strip()
    return int(v) if v.lstrip("-").isdigit() else None


def truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def parse_env_pairs(text: str) -> dict:
    out = {}
    for tok in shlex.split(text or ""):
        if "=" not in tok:
            raise ValueError(f"engine env entries look like KEY=VALUE, not {tok!r}")
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m llmserve.prepare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Every option has an LLMSERVE_* environment variable (LLMSERVE_QUANT, "
                                        "LLMSERVE_CONTEXT, LLMSERVE_PARALLEL, LLMSERVE_VISION, LLMSERVE_EXTRA_ARGS ...).")
    e = os.environ.get
    ap.add_argument("--quant", default=e("LLMSERVE_QUANT", C.DEFAULT_QUANT), help="one of: " + ", ".join(C.QUANTS))
    ap.add_argument("--data-dir", default=e("LLMSERVE_DATA_DIR", "/data"), help="models, packs, cfg.json (default /data)")
    ap.add_argument("--context", default=e("LLMSERVE_CONTEXT", "65536"), help="tokens: 65536, 128k ... or auto")
    ap.add_argument("--parallel", type=int, default=env_int("LLMSERVE_PARALLEL"), help="requests answered at once (1-8)")
    ap.add_argument("--vision", default=e("LLMSERVE_VISION", "off"), help="off | on | gpu | cpu (pictures; +0.9 GB)")
    ap.add_argument("--vision-tokens", type=int, default=env_int("LLMSERVE_VISION_TOKENS"))
    ap.add_argument("--kv", default=e("LLMSERVE_KV", "int8"), choices=KV_CHOICES, help="KV cache format")
    ap.add_argument("--kv-streaming", default=e("LLMSERVE_KV_STREAMING", "auto"), choices=("auto", "on", "off"))
    ap.add_argument("--mtp", default=e("LLMSERVE_MTP", "on"), choices=("on", "off"), help="draft layer (speculative decoding)")
    ap.add_argument("--batch-mtp", default=e("LLMSERVE_BATCH_MTP", "off"), choices=("on", "off"),
                    help="with --parallel >= 2: every slot also verifies one draft proposal per window (--batch-mtp)")
    ap.add_argument("--batch-switches", default=e("LLMSERVE_BATCH_SWITCHES", "off"), choices=("on", "off"),
                    help="with --parallel >= 2: the engine's opt-in batch-window switches (" + " ".join(BATCH_SWITCH_ENV) + ")")
    ap.add_argument("--agent", default=e("LLMSERVE_AGENT", "off"), choices=("on", "off"),
                    help="coding-agent defaults: STRATA_CACHE_MESSAGE_BOUNDARY=1, STRATA_ADMIT_SHORT_FIRST=%d (parallel >= 2), "
                         "--conversation-cache-mib %d" % (AGENT_ADMIT_SHORT_FIRST, AGENT_CONVERSATION_CACHE_MIB))
    ap.add_argument("--conversation-cache-mib", type=int, default=env_int("LLMSERVE_CONVERSATION_CACHE_MIB"),
                    help="RAM for parked conversations (0 = off; default: the engine sizes it, or %d with --agent on)"
                         % AGENT_CONVERSATION_CACHE_MIB)
    ap.add_argument("--draft-vocab", default=e("LLMSERVE_DRAFT_VOCAB") or None, choices=list(C.DRAFT_VOCABS))
    ap.add_argument("--low-ram", default=e("LLMSERVE_LOW_RAM", "auto"), choices=("auto", "off", "on", "resident", "mmap"))
    ap.add_argument("--resident-budget-gib", type=float, default=(float(e("LLMSERVE_RESIDENT_BUDGET_GIB")) if e("LLMSERVE_RESIDENT_BUDGET_GIB") else None))
    ap.add_argument("--vram-reserve-mib", type=int, default=env_int("LLMSERVE_VRAM_RESERVE_MIB"))
    ap.add_argument("--rope-scaling", default=e("LLMSERVE_ROPE_SCALING") or None, choices=("yarn", "linear", "none"))
    ap.add_argument("--rope-scale", type=float, default=(float(e("LLMSERVE_ROPE_SCALE")) if e("LLMSERVE_ROPE_SCALE") else None))
    ap.add_argument("--gpu", type=int, default=env_int("LLMSERVE_GPU"), help="the card to use, as nvidia-smi numbers it")
    ap.add_argument("--port", type=int, default=env_int("LLMSERVE_PORT") or 8080)
    ap.add_argument("--extra-args", default=e("LLMSERVE_EXTRA_ARGS", ""), help="extra engine arguments, e.g. \"--pool-workers 8\"")
    ap.add_argument("--engine-env", default=e("LLMSERVE_ENGINE_ENV", ""), help="engine environment, e.g. \"STRATA_X=1 STRATA_Y=2\"")
    ap.add_argument("--hf-token-env", default=e("LLMSERVE_HF_TOKEN_ENV", "HF_TOKEN"), help="name of the variable that holds a Hugging Face token")
    ap.add_argument("--hf-endpoint", default=e("HF_ENDPOINT", ""), help="a Hugging Face mirror")
    ap.add_argument("--download-backend", default=e("LLMSERVE_DOWNLOADER", "auto"), choices=("auto", "hf", "urllib"))
    ap.add_argument("--app-dir", default=e("LLMSERVE_APP_DIR") or None)
    ap.add_argument("--engine-dir", default=e("LLMSERVE_ENGINE_DIR") or None)
    ap.add_argument("--force", action="store_true", default=truthy(e("LLMSERVE_FORCE", "")),
                    help="redo the pack, the MTP layer and cfg.json (downloads that verified are kept)")
    ap.add_argument("--if-changed", action="store_true", help="do nothing when cfg.json is current for these settings")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the engine arguments; change nothing")
    # what-if overrides for planning on another machine
    ap.add_argument("--ram-gib", type=float, default=None, help="plan for this much RAM instead of the detected")
    ap.add_argument("--vram-gib", type=float, default=None, help="plan for this much VRAM instead of the detected")
    ap.add_argument("--wsl", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--avx512", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--disk", choices=("auto", "ssd", "hdd"), default="auto", help="what the data folder's disk is")
    return ap


def settings_from(a: argparse.Namespace) -> Settings:
    try:
        ctx = parse_context(a.context)
        vision = parse_vision(a.vision)
        eenv = parse_env_pairs(a.engine_env)
        extra = tuple(shlex.split(a.extra_args or ""))
    except ValueError as e:
        raise Fail(str(e), code=2) from None
    if a.parallel is not None and a.parallel < 1:
        raise Fail("--parallel must be 1 or more", code=2)
    if a.conversation_cache_mib is not None and a.conversation_cache_mib < 0:
        raise Fail("--conversation-cache-mib must be 0 or more", code=2)
    try:
        q = C.resolve_quant(a.quant)
    except ValueError as e:
        raise Fail(str(e), code=2) from None
    return Settings(quant=q.name, data_dir=Path(a.data_dir), app_dir=Path(a.app_dir) if a.app_dir else
                    Path(os.environ.get("LLMSERVE_APP_DIR") or Path(__file__).resolve().parent.parent),
                    engine_dir=Path(a.engine_dir) if a.engine_dir else None, context=ctx, kv=a.kv,
                    kv_streaming=a.kv_streaming, parallel=a.parallel, vision=vision, vision_tokens=a.vision_tokens,
                    mtp=a.mtp == "on", draft_vocab=a.draft_vocab, low_ram=a.low_ram,
                    resident_budget_gib=a.resident_budget_gib, vram_reserve_mib=a.vram_reserve_mib,
                    rope_scaling=a.rope_scaling, rope_scale=a.rope_scale, gpu=a.gpu, port=a.port, extra_args=extra,
                    engine_env=eenv, hf_token_env=a.hf_token_env, hf_endpoint=a.hf_endpoint.strip().rstrip("/"),
                    force=a.force, batch_mtp=a.batch_mtp == "on", batch_switches=a.batch_switches == "on",
                    agent=a.agent == "on", conversation_cache_mib=a.conversation_cache_mib)


def tri(v: str):
    return None if v == "auto" else v == "on"


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        s = settings_from(a)
        hw = detect_hardware(s.data_dir, ram_gib=a.ram_gib, vram_gib=a.vram_gib, wsl=tri(a.wsl), avx512=tri(a.avx512),
                             gpu_pick=s.gpu)
        if a.disk != "auto":
            hw.rotational = "hdd" if a.disk == "hdd" else None
        plan = build_plan(s, hw)
        if a.dry_run:
            print_plan(plan, dry=True)
            return 0
        signal.signal(signal.SIGTERM, _on_term)
        s.data_dir.mkdir(parents=True, exist_ok=True)
        (s.data_dir / "state").mkdir(parents=True, exist_ok=True)
        lock = open(s.data_dir / "state" / "prepare.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Fail("another prepare is already running on this data folder") from None
        if a.if_changed and not s.force and up_to_date(plan):
            ok(f"cfg.json is current for these settings ({plan.quant.name}, {plan.ctx} tokens): nothing to do")
            return 0
        print_plan(plan, dry=False)
        check_environment(plan)
        free = free_gb(s.data_dir)
        need = disk_needed_gb(plan)
        if free < need:
            raise Fail(f"not enough free disk space in {s.data_dir}: need ~{need:.0f} GB more, {free:.0f} GB free",
                       "free up space (on Docker Desktop: grow the WSL2 disk or prune with `docker system prune`)")
        if free < need + 8:
            warn(f"little free disk space: {free:.0f} GB free for about {need:.0f} GB of files")
        total = 4
        step_line(1, total, "model files")
        step_download(plan, a.download_backend)
        step_line(2, total, "pack for the engine")
        step_pack(plan)
        step_line(3, total, "MTP draft layer")
        step_mtp(plan)
        step_line(4, total, "run config")
        step_config(plan, measure_ple_gb(plan))
        say("[prepare] ready. Engine arguments:")
        say("  " + shlex.join(json.loads(plan.cfg_path.read_text())["args"]))
        return 0
    except Fail as e:
        say(f"[prepare] ERROR {e}")
        if e.hint:
            say(f"[prepare]       {e.hint}")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
