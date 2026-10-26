#!/usr/bin/env bash
# LLMServe container entry point.
#
#   docker compose up -d            start the API (the default command: "serve")
#   docker compose run --rm llmserve prepare --dry-run     show the plan and the engine arguments, change nothing
#   docker compose run --rm llmserve bash                  a shell in the image
#
# What "serve" does, in order:
#   1. looks at the machine and says plainly what is wrong (no GPU, old driver, memlock, too little RAM, /data on a
#      Windows drive) - warnings never stop it, a missing GPU does;
#   2. runs `python -m llmserve.prepare --if-changed`: downloads the model, packs it, writes /data/cfg.json. Every step
#      is resumable and skipped when done, so a restart goes straight on;
#   3. execs the OpenAI/Anthropic-compatible server, so tini hands SIGTERM (docker stop) straight to it.
#
# Settings are the LLMSERVE_* variables in .env (see .env.example); HF_TOKEN for gated downloads.
set -u

APP="${LLMSERVE_APP_DIR:-/opt/llmserve}"
DATA="${LLMSERVE_DATA_DIR:-/data}"
PORT="${LLMSERVE_PORT:-8080}"
cd "$APP" || exit 1

say()  { printf '[llmserve] %s\n' "$*"; }
warn() { printf '[llmserve] WARN  %s\n' "$*" >&2; }
die()  { printf '[llmserve] ERROR %s\n' "$*" >&2; exit "${2:-1}"; }

# ---- other commands: `prepare`, `bash`, `python -c ...`
case "${1:-serve}" in
  serve) [ "$#" -gt 0 ] && shift ;;
  prepare) shift; exec python -m llmserve.prepare "$@" ;;
  *) exec "$@" ;;
esac

# ---- settings (the same defaults as .env.example)
export LLMSERVE_QUANT="${LLMSERVE_QUANT:-Q2_0}"
export LLMSERVE_CONTEXT="${LLMSERVE_CONTEXT:-65536}"
export LLMSERVE_PARALLEL="${LLMSERVE_PARALLEL:-1}"
export LLMSERVE_VISION="${LLMSERVE_VISION:-off}"
export LLMSERVE_DATA_DIR="$DATA"
export LLMSERVE_PORT="$PORT"
# empty values from an unfilled .env line must not reach the tools as "set but empty"
for v in LLMSERVE_EXTRA_ARGS LLMSERVE_KV LLMSERVE_KV_STREAMING LLMSERVE_MTP LLMSERVE_DRAFT_VOCAB LLMSERVE_LOW_RAM \
         LLMSERVE_RESIDENT_BUDGET_GIB LLMSERVE_VRAM_RESERVE_MIB LLMSERVE_ROPE_SCALING LLMSERVE_ROPE_SCALE LLMSERVE_GPU \
         LLMSERVE_ENGINE_ENV LLMSERVE_VISION_TOKENS HF_TOKEN HF_ENDPOINT LLMSERVE_BATCH_MTP LLMSERVE_BATCH_SWITCHES \
         LLMSERVE_AGENT LLMSERVE_CONVERSATION_CACHE_MIB; do
  [ -z "${!v:-}" ] && unset "$v"
done

# LLMSERVE_AGENT=on with several slots: the Python server (not the engine) orders prompt reads short-first.  It reads this from
# its own environment, which llmserve.prepare's cfg.json "env" does not reach.  A STRATA_ADMIT_SHORT_FIRST line in .env wins.
case "$(printf '%s' "${LLMSERVE_AGENT:-off}" | tr '[:upper:]' '[:lower:]')" in
  on)
    if [ "${LLMSERVE_PARALLEL:-1}" -ge 2 ] 2>/dev/null; then
      export STRATA_ADMIT_SHORT_FIRST="${STRATA_ADMIT_SHORT_FIRST:-4096}"
    fi ;;
esac

# ---- minimum RAM per quant (GB): the catalog's "ram_gb" (llmserve/catalog.py; a test keeps the two in step)
quant_ram_gb() {
  case "$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')" in
    Q2_0|IQ2_XS) echo 48 ;;
    IQ3_XXS) echo 60 ;;
    IQ3_S) echo 62 ;;
    IQ1_M) echo 32 ;;
    UD-IQ4_XS|UD-Q4_K_XL) echo 48 ;;
    UD-Q5_K_XL) echo 108 ;;
    UD-Q6_K_XL) echo 118 ;;
    Q8_0) echo 136 ;;
    *) echo 0 ;;
  esac
}

is_wsl() { grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -e /dev/dxg ]; }

# ---- 1. the machine
say "LLMServe: quant $LLMSERVE_QUANT, context $LLMSERVE_CONTEXT, parallel $LLMSERVE_PARALLEL, pictures $LLMSERVE_VISION"

# GPU: nvidia-smi is injected by the NVIDIA container runtime (Docker Desktop: --gpus all / deploy.devices)
if [ "${LLMSERVE_SKIP_GPU_CHECK:-0}" != 1 ]; then
  if smi=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null) && [ -n "$smi" ]; then
    while IFS= read -r line; do say "GPU: $line"; done <<<"$smi"
    drv=$(printf '%s\n' "$smi" | head -1 | awk -F', ' '{print $NF}')
    major=${drv%%.*}
    if [ -n "$major" ] && [ "$major" -lt 580 ] 2>/dev/null; then
      die "NVIDIA driver $drv is too old: the engine is built with CUDA 13, which needs driver 580 or newer. Update the NVIDIA driver on the host (on Windows: the Game Ready or Studio driver from nvidia.com or GeForce Experience; Docker Desktop passes it through)." 3
    fi
  elif compgen -G "/dev/nvidia[0-9]*" >/dev/null || [ -e /dev/dxg ]; then
    warn "a GPU device is present but nvidia-smi does not run (NVIDIA_DRIVER_CAPABILITIES must include 'utility'). Going on."
  else
    die "no NVIDIA GPU is visible in this container. Start it with 'gpus: all' / deploy.resources.reservations.devices (docker-compose.yml does), or 'docker run --gpus all'. Test with: docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi   (Windows: NVIDIA driver 580+ and Docker Desktop with the WSL2 backend; docs/WINDOWS_DOCKER.md)" 3
  fi
fi

# memlock: the engine pins tens of GB of RAM
lim=$(ulimit -l 2>/dev/null || echo unknown)
if [ "$lim" != unlimited ] && [ "$lim" != unknown ]; then
  hard=$(ulimit -Hl 2>/dev/null || echo "$lim")
  if [ "$hard" = unlimited ] && ulimit -l unlimited 2>/dev/null; then
    lim=unlimited
  elif [ "$lim" -lt 67108864 ] 2>/dev/null; then
    warn "the locked-memory limit (ulimit -l) is $lim KB; the engine pins tens of GB of RAM and will be slow or fail. Start with: ulimits: memlock: {soft: -1, hard: -1} (docker-compose.yml has it) or docker run --ulimit memlock=-1:-1"
  fi
fi

# RAM against the quant (the WSL2 VM gets half of the Windows RAM unless .wslconfig says otherwise)
mem_gb=$(awk '/^MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
need_gb=$(quant_ram_gb "$LLMSERVE_QUANT")
say "RAM visible in the container: ${mem_gb} GiB (the quant asks for ${need_gb} GB)"
if [ "$need_gb" -gt 0 ] && [ "$mem_gb" -gt 0 ] && [ "$mem_gb" -lt "$need_gb" ]; then
  if is_wsl; then
    warn "only ${mem_gb} GiB of RAM for ${LLMSERVE_QUANT} (needs ${need_gb}). WSL2 gives its VM half of the Windows RAM by default: put 'memory=...GB' under [wsl2] in %UserProfile%\\.wslconfig, run 'wsl --shutdown' and restart Docker Desktop (docs/WINDOWS_DOCKER.md). It will fall back to slow low-RAM mode."
  else
    warn "only ${mem_gb} GiB of RAM for ${LLMSERVE_QUANT} (needs ${need_gb}): it falls back to low-RAM mode (slower). Choose a smaller quant or add RAM."
  fi
fi
if cg=$(cat /sys/fs/cgroup/memory.max 2>/dev/null) && [ "$cg" != max ] && [ -n "$cg" ]; then
  warn "this container has a memory limit ($((cg / 1073741824)) GiB). Remove mem_limit / --memory: the engine plans from the whole machine's RAM and pins tens of GB."
fi
shm_kb=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')
if [ -n "${shm_kb:-}" ] && [ "$shm_kb" -lt 1000000 ] 2>/dev/null; then
  warn "/dev/shm is only $((shm_kb / 1024)) MiB; the picture reader uses it. Start with shm_size: 2gb"
fi

# /data: writable, and not on a Windows drive
mkdir -p "$DATA" 2>/dev/null
if ! mkdir -p "$DATA/state" "$DATA/logs" "$DATA/home" 2>/dev/null || ! touch "$DATA/state/.w" 2>/dev/null; then
  die "$DATA is not writable by uid $(id -u). Use the named volume in docker-compose.yml, or for a bind mount: chown -R $(id -u) on the host folder (or run with 'user: root')." 4
fi
rm -f "$DATA/state/.w"
mounts=$(cat /proc/mounts 2>/dev/null)
fstype=$(printf '%s\n' "$mounts" | awk -v d="$DATA" '
  { mp = $2; gsub(/\\040/, " ", mp); pre = (mp == "/") ? "/" : mp "/"
    if (d == mp || index(d "/", pre) == 1) { if (length(mp) >= best) { best = length(mp); t = $3 } } }
  END { print t }')
case "$fstype" in
  9p|drvfs|virtiofs|fuse.grpcfuse|grpcfuse|fakeowner|cifs|smb3|smbfs|nfs|nfs4|fuse.osxfs|fuse.sshfs)
    warn "$DATA is on a '$fstype' mount (a Windows drive or a network share). Reading tens of GB of model through it is many times slower than the WSL2 disk, and the engine streams from it. Use the named volume (docker-compose.yml) instead of a bind mount." ;;
esac
case "$DATA" in /mnt/[a-zA-Z]|/mnt/[a-zA-Z]/*)
  warn "$DATA looks like a Windows drive seen from WSL (/mnt/<drive>): much slower than the Linux disk. Keep the model in a Docker volume." ;;
esac
if ! printf '%s\n' "$mounts" | awk -v d="$DATA" '$2 == d {found = 1} END {exit !found}'; then
  warn "$DATA is not a volume: the model (tens of GB) is lost when this container is removed. Mount a volume there (docker-compose.yml: llmserve-data:/data)."
fi

# ---- the API key (the server reads STRATA_API_KEY; an EMPTY one is an error there, so only export a real one)
key="${LLMSERVE_API_KEY:-}"
if [ -z "$key" ] && [ -n "${LLMSERVE_API_KEY_FILE:-}" ]; then
  [ -r "$LLMSERVE_API_KEY_FILE" ] || die "LLMSERVE_API_KEY_FILE=$LLMSERVE_API_KEY_FILE cannot be read." 2
  key=$(tr -d '\r\n' < "$LLMSERVE_API_KEY_FILE")
fi
unset LLMSERVE_API_KEY
if [ -n "$key" ]; then
  [ "${#key}" -lt 16 ] && warn "the API key is short (${#key} characters); 'openssl rand -hex 24' makes a good one."
  export STRATA_API_KEY="$key"
else
  unset STRATA_API_KEY
  warn "no LLMSERVE_API_KEY set: the API needs no key. That is fine while the port is published on 127.0.0.1 only (docker-compose.yml does); set a key before you expose it to the network."
fi

# ---- 2. prepare (download, pack, MTP, cfg.json) - skipped when cfg.json is already current for these settings
prep_args=()
[ "${LLMSERVE_REPREPARE:-0}" != 1 ] && prep_args+=(--if-changed)
[ "${LLMSERVE_FORCE:-0}" = 1 ] && prep_args+=(--force)
if [ "${LLMSERVE_SKIP_PREPARE:-0}" = 1 ] && [ -f "$DATA/cfg.json" ]; then
  say "LLMSERVE_SKIP_PREPARE=1: using $DATA/cfg.json as it is"
else
  say "preparing the model (first start: tens of GB to download; follow the lines below, a restart resumes)"
  python -m llmserve.prepare "${prep_args[@]}" &
  prep_pid=$!
  # docker stop while preparing: pass TERM on, wait for it to stop cleanly (it keeps what it has downloaded)
  trap 'kill -TERM "$prep_pid" 2>/dev/null; wait "$prep_pid" 2>/dev/null; exit 0' TERM INT
  wait "$prep_pid"
  rc=$?
  trap - TERM INT
  [ "$rc" -eq 0 ] || die "prepare stopped with code $rc (the lines above say why). Finished steps are kept; fix the cause and start again." "$rc"
fi
[ -f "$DATA/cfg.json" ] || die "$DATA/cfg.json was not written." 5

# keep the engine log from growing without bound across restarts
log="$DATA/logs/engine.log"
if [ -f "$log" ] && [ "$(stat -c %s "$log" 2>/dev/null || echo 0)" -gt 104857600 ]; then
  mv -f "$log" "$log.1"
fi

# ---- 3. the server (exec: tini's SIGTERM reaches it; it drains and stops the engine)
server_args=(--engine strata --config "$DATA/cfg.json" --host 0.0.0.0 --port "$PORT"
             --shutdown-timeout "${LLMSERVE_SHUTDOWN_TIMEOUT:-30}")
if [ -n "${LLMSERVE_SERVER_ARGS:-}" ]; then
  read -r -a extra <<<"$LLMSERVE_SERVER_ARGS"
  server_args+=("${extra[@]}")
fi
say "starting the server on port $PORT (loading the model takes a few minutes; /health answers when it is ready)"
exec python -m serve.server "${server_args[@]}"
