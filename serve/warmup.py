"""serve/warmup.py - the start-up warm-up (docs/TTFT_WARMUP.md section 2).

The first requests after an engine start pay for things a running engine has already done: CUDA graphs captured on first
use, prompt staging buffers that grow on first use, the PCIe share that `--pcie-frac auto` tunes only after ~240 decode
windows, the adaptive expert tier that starts from the profile, and (under WSL2) the fresh RAM the conversation cache
touches for the first time when it parks a conversation.  The server runs a few requests of its own after the engine says
READY and before the HTTP port opens, through the path real requests take (template, tokenizer, Service.run, the engine's
line protocol), so a user's first request finds all of that done.  The port is closed meanwhile, so the container's
health check (docker/healthcheck.sh: GET /health) reports "starting" until the warm-up is over.

THE STEPS (mode "full"; "short" leaves out the two long ones):
  1. prefill   a ~16K-token prompt (code in several languages, prose, a JSON tool call), thinking off, greedy, 1 new token
  2. decode    a ~1K-token prompt and up to 768 greedy new tokens: enough decode windows for the engine's PCIe-share
               auto-tune to run its updates (it logs "PCIe share: ... -> pcie_frac X")
  3. parks     N short, different conversations that the engine DOES keep (parks N-1 of them, the last one below):
               the conversation cache faults its RAM in now instead of 2.4 s per park on the user's requests
  4. short     a ~120-token prompt, 16 new tokens (the short path, the window graphs).  It is the last step on purpose:
               it carries "strata_checkpoint": false, so the engine parks conversation N of step 3 when it starts and
               keeps nothing of itself: the first real request finds all N places filled, nothing live to park.

WHAT "strata_checkpoint": false DOES (engine: ckpt=0, src/program/generate.cpp `req_ckpt`): no checkpoint at the prompt's
last turn boundary, none every --prompt-cache-every tokens, no tail checkpoint, and after the request the session is neither
continued nor parked (live_ok = false).  It does NOT stop (a) the engine from parking the conversation that was live when
the request STARTED - nothing is live at start-up, so there is nothing - nor (b) the system-prompt root checkpoint, which a
prompt gets when its first turn ends at --prompt-cache-root tokens (2048) or later; the warm-up prompts are one user
message with no system prompt, so they never reach it.

The warm-up never stops the server from starting: a step that fails is logged and the next one runs; an engine that dies
ends the warm-up and the first real request starts it again as any request does.  Not run for a lazy start (nothing is
loaded), replicas (one engine would be warmed) or the mock engine.
"""
from __future__ import annotations

import os
import random
import threading
import time

MODES = ("off", "short", "full")
DEFAULT_MODE = "full"
DEFAULT_PARKS = 4
MAX_PARKS = 64
ENV_MODE = "LLMSERVE_WARMUP"
ENV_PARKS = "LLMSERVE_WARMUP_PARKS"

PREFILL_TOKENS = {"short": 2048, "full": 16384}        # step 1
DECODE_PROMPT_TOKENS, DECODE_NEW_TOKENS = 1024, 768    # step 2
SHORT_PROMPT_TOKENS, SHORT_NEW_TOKENS = 120, 16        # step 4
PARK_PROMPT_TOKENS, PARK_NEW_TOKENS = 100, 8           # step 3
GREEDY = {"temperature": 0, "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1}
KWARGS = {"enable_thinking": False}


# ------------------------------------------------------------------------------------------- settings
def parse_mode(value, where: str = "warmup") -> str:
    """"off" | "short" | "full" (true/false read as full/off); ValueError names the setting."""
    if value is True:
        return "full"
    if value is False or value is None:
        return "off" if value is False else DEFAULT_MODE
    v = str(value).strip().lower()
    if v in ("", "default"):
        return DEFAULT_MODE
    if v in MODES:
        return v
    if v in ("on", "yes", "true", "1"):
        return "full"
    if v in ("no", "false", "0", "none"):
        return "off"
    raise ValueError(f'{where}: "off", "short" or "full", not {value!r}')


def parse_parks(value, where: str = "warmup_parks") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where}: a whole number 0..{MAX_PARKS}, not {value!r}")
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{where}: a whole number 0..{MAX_PARKS}, not {value!r}") from None
    if not 0 <= n <= MAX_PARKS:
        raise ValueError(f"{where}: a whole number 0..{MAX_PARKS}, not {value!r}")
    return n


def settings(cfg: dict, environ=None) -> tuple[str, int]:
    """-> (mode, parks).  The environment (LLMSERVE_WARMUP, LLMSERVE_WARMUP_PARKS) overrides the run config's "warmup" and
    "warmup_parks"; neither given: full, 4.  ValueError for a value that is none of the above."""
    environ = os.environ if environ is None else environ
    mode, parks = DEFAULT_MODE, DEFAULT_PARKS
    if cfg.get("warmup") is not None:
        mode = parse_mode(cfg["warmup"], 'config "warmup"')
    if cfg.get("warmup_parks") is not None:
        parks = parse_parks(cfg["warmup_parks"], 'config "warmup_parks"')
    if (environ.get(ENV_MODE) or "").strip():
        mode = parse_mode(environ[ENV_MODE], ENV_MODE)
    if (environ.get(ENV_PARKS) or "").strip():
        parks = parse_parks(environ[ENV_PARKS], ENV_PARKS)
    return mode, parks


# ------------------------------------------------------------------------------------------- the synthetic text
_WORDS = ("cache request token buffer window expert layer queue stream handler parser config tensor index shard batch "
          "checkpoint session scheduler allocator kernel profile latency throughput retry timeout fallback contract "
          "schema payload context prefix suffix boundary iterator registry resolver adapter plugin watcher").split()
_PROSE = (
    "The change keeps the old behaviour for callers that pass no options, and only the new code path reads the extra field. "
    "We measured the {a} on a warm {b} and it stayed within noise, but the {c} case still allocates once per call, which is "
    "why the {d} is hoisted out of the loop. If the {e} fails we retry with a smaller {f}; after three attempts the error "
    "is returned to the caller unchanged. Reviewers should look at the ordering of the {g} and the {h} first, because that "
    "is where the previous version went wrong.",
    "This module owns the {a}: every {b} goes through it, and nothing else may write to the {c}. The {d} used to be rebuilt on "
    "each {e}; it is now built once at start-up and shared, so a {f} that needs a different {g} has to ask the {h} for a copy "
    "instead of changing it in place.",
    "TODO before merging: the {a} test is flaky when the {b} is slow, because it waits for the {c} with a fixed sleep. Replace "
    "that with a poll on the {d}, add a regression test for the {e} path, and document why the {f} must not be reordered "
    "after the {g}. Also check that the {h} survives a restart.",
)


def _name(rng: random.Random, k: int = 2) -> str:
    return "_".join(rng.choice(_WORDS) for _ in range(k))


def _camel(rng: random.Random) -> str:
    return "".join(w.capitalize() for w in (rng.choice(_WORDS), rng.choice(_WORDS)))


def _python(rng, i):
    f, a, b = _name(rng), _name(rng, 1), _name(rng, 1)
    return (f"def {f}_{i}({a}: dict, {b}: int = {rng.randint(1, 512)}) -> list[str]:\n"
            f"    \"\"\"Collect the {a} entries that are older than {b} seconds.\"\"\"\n"
            f"    out: list[str] = []\n"
            f"    for key, value in sorted({a}.items()):\n"
            f"        if value.get(\"age\", 0) > {b} and not value.get(\"pinned\"):\n"
            f"            out.append(f\"{{key}}:{{value['age']}}\")\n"
            f"        elif value.get(\"retries\", 0) >= {rng.randint(2, 9)}:\n"
            f"            raise RuntimeError(f\"{{key}} exceeded its retry budget\")\n"
            f"    return out\n\n")


def _cpp(rng, i):
    f, a = _name(rng), _name(rng, 1)
    return (f"static int64_t {f}_{i}(const std::vector<int32_t>& {a}, int64_t limit) {{\n"
            f"    int64_t total = 0;\n"
            f"    for (size_t k = 0; k < {a}.size(); ++k) {{\n"
            f"        if ({a}[k] % {rng.randint(2, 13)} == 0) continue;\n"
            f"        total += static_cast<int64_t>({a}[k]) * {rng.randint(3, 97)};\n"
            f"        if (total > limit) return -1;  // {rng.choice(_WORDS)} overflow guard\n"
            f"    }}\n"
            f"    return total;\n}}\n\n")


def _typescript(rng, i):
    c, a = _camel(rng), _name(rng, 1)
    return (f"export async function load{c}{i}(id: string, opts: {{ retries?: number }} = {{}}): Promise<{c}> {{\n"
            f"  const retries = opts.retries ?? {rng.randint(1, 5)};\n"
            f"  for (let attempt = 0; attempt <= retries; attempt++) {{\n"
            f"    const res = await fetch(`/api/{a}/${{id}}?attempt=${{attempt}}`);\n"
            f"    if (res.ok) return (await res.json()) as {c};\n"
            f"    if (res.status === {rng.choice([404, 409, 429])}) break;\n"
            f"  }}\n"
            f"  throw new Error(`could not load {a} ${{id}}`);\n}}\n\n")


def _rust(rng, i):
    c, a = _camel(rng), _name(rng, 1)
    return (f"impl {c} {{\n"
            f"    pub fn {a}_{i}(&mut self, input: &[u8]) -> Result<usize, Error> {{\n"
            f"        if input.len() > {rng.randint(64, 4096)} {{\n"
            f"            return Err(Error::TooLarge(input.len()));\n"
            f"        }}\n"
            f"        let n = self.buf.extend_from_slice(input);\n"
            f"        self.pending += n as u64 * {rng.randint(2, 9)};\n"
            f"        Ok(input.len())\n    }}\n}}\n\n")


def _go(rng, i):
    c, a = _camel(rng), _name(rng, 1)
    return (f"func (s *{c}) {a.title().replace('_', '')}{i}(ctx context.Context, key string) ([]byte, error) {{\n"
            f"\ts.mu.RLock()\n\tdefer s.mu.RUnlock()\n"
            f"\tif v, ok := s.items[key]; ok && time.Since(v.at) < {rng.randint(1, 90)}*time.Second {{\n"
            f"\t\treturn v.data, nil\n\t}}\n"
            f"\treturn nil, fmt.Errorf(\"{a}: %q not cached\", key)\n}}\n\n")


def _sql(rng, i):
    t, a = _name(rng, 1), _name(rng, 1)
    return (f"SELECT {a}.id, {a}.created_at, COUNT(e.id) AS events_{i}\n"
            f"FROM {t}s AS {a}\nLEFT JOIN events e ON e.{t}_id = {a}.id AND e.kind = '{rng.choice(_WORDS)}'\n"
            f"WHERE {a}.created_at > now() - interval '{rng.randint(1, 30)} days'\n"
            f"GROUP BY {a}.id ORDER BY events_{i} DESC LIMIT {rng.randint(10, 200)};\n\n")


def _shell(rng, i):
    return (f"for f in $(git diff --name-only HEAD~{rng.randint(1, 9)} -- '*.{rng.choice(['py', 'cpp', 'ts', 'rs'])}'); do\n"
            f"  wc -l \"$f\" | awk '{{ if ($1 > {rng.randint(100, 900)}) print \"large:\", $2 }}'\n"
            f"done  # step {i}\n\n")


def _tool_call(rng, i):
    return ('{"name": "read_file", "arguments": {"path": "src/%s/%s.%s", "offset": %d, "limit": %d}}\n'
            '{"name": "grep", "arguments": {"pattern": "%s", "path": "src", "context": %d}}\n\n'
            % (rng.choice(_WORDS), _name(rng), rng.choice(["py", "cpp", "ts"]), rng.randint(0, 900), rng.randint(40, 400),
               _name(rng), rng.randint(0, 5)))


def _prose(rng, i):
    fill = {k: rng.choice(_WORDS) for k in "abcdefgh"}
    return f"Note {i}: {_PROSE[i % len(_PROSE)].format(**fill)}\n\n"


def _diff(rng, i):
    a, b = _name(rng, 1), rng.randint(1, 60)
    return (f"--- a/src/{a}.py\n+++ b/src/{a}.py\n@@ -{b},6 +{b},8 @@ def {_name(rng)}():\n"
            f"     value = load({a!r})\n-    return value\n+    if value is None:\n+        raise KeyError({a!r})\n"
            f"+    return value  # {rng.choice(_WORDS)}\n\n")


def _trace(rng, i):
    return (f"Traceback (most recent call last):\n  File \"app/{_name(rng, 1)}.py\", line {rng.randint(10, 900)}, in {_name(rng)}\n"
            f"    result = self.{_name(rng)}(payload)\n  File \"app/{_name(rng, 1)}.py\", line {rng.randint(10, 900)}, in {_name(rng)}\n"
            f"    raise ValueError(\"unexpected {rng.choice(_WORDS)} at index {rng.randint(0, 5000)}\")\n"
            f"ValueError: unexpected {rng.choice(_WORDS)}\n\n")


def _yaml(rng, i):
    return (f"service_{i}:\n  image: registry.example.com/{_name(rng, 1)}:{rng.randint(1, 9)}.{rng.randint(0, 40)}\n"
            f"  environment:\n    - {_name(rng, 1).upper()}={rng.randint(1, 9999)}\n  ports: [\"{rng.randint(1024, 9999)}:80\"]\n\n")


_BLOCKS = (_python, _cpp, _prose, _typescript, _tool_call, _rust, _sql, _go, _diff, _shell, _trace, _yaml)


def blocks(seed: int = 1):
    """An endless, deterministic stream of realistic mixed text blocks (code in Python, C++, TypeScript, Rust, Go, SQL
    and shell; prose; diffs; stack traces; YAML; JSON tool calls).  The same seed gives the same text byte for byte, so
    every start routes the same experts; the names and numbers vary from block to block, so no block repeats."""
    rng = random.Random(seed)
    i = 0
    while True:
        yield _BLOCKS[i % len(_BLOCKS)](rng, i)
        i += 1


def text_of_tokens(tok, tokens: int, seed: int = 1) -> str:
    """About `tokens` tokens of block text under `tok`: the chars-per-token ratio is measured on a first sample of
    blocks, then blocks are added until that many characters are there; the last block is cut at a line end, so a
    prompt of 100 tokens is a few lines of one snippet and a long one holds only whole blocks."""
    gen = blocks(seed)
    sample = "".join(next(gen) for _ in range(len(_BLOCKS) * 2))
    per_token = max(0.5, len(sample) / max(1, len(tok.encode(sample))))
    want = max(40, int(tokens * per_token))
    gen, out, size = blocks(seed), [], 0
    while size < want:
        b = next(gen)
        out.append(b)
        size += len(b)
    text = "".join(out)
    if len(text) > want * 1.1:                       # over by more than a tenth: cut the last block at a line end
        cut = text.rfind("\n", 0, want)
        text = text[:cut + 1] if cut > want * 0.5 else text[:want]
    return text


# ------------------------------------------------------------------------------------------- the prompts
def prefill_messages(tok, tokens: int) -> list:
    body = text_of_tokens(tok, max(64, tokens - 80), seed=11)
    return [{"role": "user", "content":
             "Below is an excerpt of a large repository: sources in several languages, review notes, diffs, logs and tool "
             "calls from an earlier session.\n\n" + body + "\nReply with the single word OK."}]


def decode_messages(tok, tokens: int) -> list:
    body = text_of_tokens(tok, max(64, tokens - 120), seed=23)
    return [{"role": "user", "content":
             "Review the following code and notes. For every function explain what it does, list at least three risks or "
             "edge cases with concrete examples, and propose a refactoring with code. Be exhaustive and do not stop "
             "early; write at least 1500 words.\n\n" + body}]


def short_messages(tok, tokens: int) -> list:
    body = text_of_tokens(tok, max(16, tokens - 70), seed=31)
    return [{"role": "user", "content": body + "\nIn one sentence: what does this do?"}]


def park_messages(tok, k: int, n: int, tokens: int) -> list:
    """Conversation k of n: a prompt of its own (it starts with its number), so none of them is a prefix of another or of
    a real request's prompt."""
    body = text_of_tokens(tok, max(16, tokens - 70), seed=100 + k)
    return [{"role": "user", "content": f"[warm-up conversation {k} of {n}]\n{body}\nIn one sentence: what does this do?"}]


# ------------------------------------------------------------------------------------------- running
def fmt_s(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 600 else f"{seconds / 60:.1f} min"


class _Isolated:
    """The warm-up's requests go through Service.run, which records them in the history, the totals and the latency
    histograms the Monitor and /metrics read, and keeps their prompts in the PromptEncoder: put all that back as it was."""

    def __init__(self, svc):
        self.svc = svc

    def __enter__(self):
        s = self.svc
        self.saved = (list(s.history), dict(s.totals), s.latencies, s.last_timings, s.last_request_at)
        from serve.prometheus import Latencies
        s.latencies = Latencies()
        return self

    def __exit__(self, *exc):
        s = self.svc
        history, totals, latencies, last_timings, last_at = self.saved
        with s.status_lock:
            s.history.clear()
            s.history.extend(history)
            s.totals.clear()
            s.totals.update(totals)
            s.latencies, s.last_timings, s.last_request_at = latencies, last_timings, last_at
        if s.prompts is not None:
            with s.prompts.lock:
                s.prompts.entries = []
        s.phases.d = None
        return False


def _ask(svc, messages: list, max_new: int, sampling: dict, cancel: threading.Event) -> dict:
    """One request through the path HTTP requests take: prepare (template + tokenizer + context fit) then Service.run."""
    svc.phase_begin()
    ids, thinking, max_new = svc.prepare(messages, None, dict(KWARGS), max_new, cancel=cancel)
    t0 = time.perf_counter()
    done = None
    for kind, x in svc.run(ids, thinking, None, max_new, dict(sampling), cancel):
        if kind == "done":
            done = x
    done = done or {}
    return {"prompt_tokens": len(ids), "completion_tokens": done.get("completion_tokens", 0),
            "finish": done.get("finish"), "timings": done.get("timings") or {}, "seconds": time.perf_counter() - t0}


def _log_offset(engine) -> tuple[str | None, int]:
    path = getattr(engine, "log_path", None)
    try:
        return path, os.path.getsize(path) if path else 0
    except OSError:
        return path, 0


def _engine_log_lines(path, offset: int, needles: tuple, limit: int = 2 << 20) -> list[str]:
    """The engine log's lines written since `offset` that hold one of `needles` (the PCIe-share updates, the parks)."""
    if not path:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(limit)
    except OSError:
        return []
    return [ln.strip() for ln in data.decode("utf-8", "replace").splitlines() if any(n in ln for n in needles)]


def run(svc, mode: str, parks: int, *, log=print, cancel: threading.Event | None = None) -> dict:
    """Run the warm-up of `mode` ("short" | "full") with `parks` conversations to park.  Never raises for a failing
    step.  -> {"steps": [(name, ok, seconds)], "seconds": total, "aborted": bool}."""
    from serve import server as S
    cancel = cancel or threading.Event()
    engine, tok = svc.engine, svc.tok
    result = {"steps": [], "seconds": 0.0, "aborted": False}
    if mode == "off":
        return result
    ctx = int(getattr(engine, "max_context", 0) or getattr(engine, "known_ctx", 0) or 0)
    batched = bool(getattr(engine, "batch", 0))
    slots = S.conversation_slots(engine) or 4
    cache_mib = (getattr(engine, "info", None) or {}).get("conversation_cache_mib")
    n_parks = min(parks, slots)
    why_no_parks = ("conversation cache off" if isinstance(cache_mib, int) and cache_mib == 0 else
                    "batch slots keep their own conversations" if batched else None)
    if why_no_parks:
        n_parks = 0
    plan = [("prefill", None)] + ([("decode", None)] if mode == "full" else [])
    plan += [("park", k) for k in range(1, n_parks + 1)] + [("short", None)]
    log(f"[strata] warm-up ({mode}): {len(plan)} requests"
        + (f" ({n_parks} conversations to park)" if n_parks else f" (no conversation parks: {why_no_parks})" if why_no_parks and parks
           else "") + "; the port opens when it is done (the health check says starting until then)")
    path, offset = _log_offset(engine)
    started = time.perf_counter()

    def fits(prompt_tokens: int, new: int) -> bool:
        return ctx <= 0 or prompt_tokens + new + S.CTX_SLACK + 64 <= ctx

    with _Isolated(svc):
        for number, (kind, k) in enumerate(plan, 1):
            if cancel.is_set():
                result["aborted"] = True
                log("[strata] warm-up stopped")
                break
            label, ask = kind, None
            try:
                if kind == "prefill":
                    want = PREFILL_TOKENS[mode]
                    want = min(want, max(0, ctx - S.CTX_SLACK - 1024)) if ctx > 0 else want
                    label = f"prefill ~{want:,} tokens, 1 new token"
                    if want < 512:
                        raise _Skip(f"the context ({ctx}) is too small")
                    ask = (lambda w=want: _ask(svc, prefill_messages(tok, w), 1, {**GREEDY, "strata_checkpoint": False}, cancel))
                elif kind == "decode":
                    label = f"decode ~{DECODE_PROMPT_TOKENS:,}-token prompt, up to {DECODE_NEW_TOKENS} new tokens"
                    if not fits(DECODE_PROMPT_TOKENS, DECODE_NEW_TOKENS):
                        raise _Skip(f"the context ({ctx}) is too small")
                    ask = (lambda: _ask(svc, decode_messages(tok, DECODE_PROMPT_TOKENS), DECODE_NEW_TOKENS,
                                        {**GREEDY, "strata_checkpoint": False}, cancel))
                elif kind == "park":
                    label = f"conversation {k} of {n_parks} (kept by the engine)"
                    ask = (lambda k=k: _ask(svc, park_messages(tok, k, n_parks, PARK_PROMPT_TOKENS), PARK_NEW_TOKENS,
                                            dict(GREEDY), cancel))
                else:
                    label = f"short ~{SHORT_PROMPT_TOKENS}-token prompt, {SHORT_NEW_TOKENS} new tokens"
                    ask = (lambda: _ask(svc, short_messages(tok, SHORT_PROMPT_TOKENS), SHORT_NEW_TOKENS,
                                        {**GREEDY, "strata_checkpoint": False}, cancel))
                r = ask()
            except _Skip as e:
                log(f"[strata] warm-up {number}/{len(plan)} {label}: skipped ({e})")
                result["steps"].append((kind, None, 0.0))
                continue
            except S.EngineDied as e:
                log(f"[strata] warm-up {number}/{len(plan)} {label}: the engine stopped ({e}); the rest is skipped, "
                    "the first request starts it again")
                result["steps"].append((kind, False, 0.0))
                result["aborted"] = True
                break
            except Exception as e:          # noqa: BLE001 - the warm-up never keeps the server from starting
                log(f"[strata] warm-up {number}/{len(plan)} {label}: failed ({type(e).__name__}: {e}); continuing")
                result["steps"].append((kind, False, 0.0))
                continue
            t = r["timings"]
            extra = [f"prompt {r['prompt_tokens']:,} tokens"]
            if t.get("prompt_ms") is not None:
                extra.append(f"engine prompt {t['prompt_ms'] / 1000:.1f} s")
            if r["completion_tokens"] > 1 and t.get("predicted_per_second"):
                extra.append(f"{r['completion_tokens']} tokens at {t['predicted_per_second']:.0f} tok/s")
            if kind == "decode" and r["completion_tokens"] < 0.9 * DECODE_NEW_TOKENS:
                extra.append(f"WARNING only {r['completion_tokens']} of {DECODE_NEW_TOKENS} tokens: the PCIe share may not "
                             "have tuned fully")
            if r["finish"] == "error":
                extra.append("finished with an error")
            log(f"[strata] warm-up {number}/{len(plan)} {label}: {fmt_s(r['seconds'])} ({', '.join(extra)})")
            result["steps"].append((kind, True, r["seconds"]))
    result["seconds"] = time.perf_counter() - started
    log(f"[strata] warm-up done in {fmt_s(result['seconds'])}" + (" (stopped early)" if result["aborted"] else ""))
    seen = _engine_log_lines(path, offset, ("PCIe share", "conversation cache: parked", "conversation cache: evicted"))
    if path:
        shares = [ln for ln in seen if "PCIe share" in ln]
        parked = [ln for ln in seen if "conversation cache: parked" in ln]
        for ln in shares[-3:]:
            log("[strata] warm-up, engine log: " + ln[ln.find("PCIe share"):][:200])
        if not result["aborted"] and mode == "full" and not shares:
            log("[strata] warm-up: the engine logged no \"PCIe share\" update (a fixed --pcie-frac, or too few decode "
                "windows): nothing was tuned")
        if parked:
            log(f"[strata] warm-up, engine log: {len(parked)} conversation(s) parked: " +
                "; ".join(ln[ln.find("parked "):].split(";")[0][:60] for ln in parked[-n_parks or None:][:6]))
    return result


class _Skip(Exception):
    """A step that cannot run here (the context is too small): logged, not an error."""


def run_blocking(svc, mode: str, parks: int, *, log=print) -> dict:
    """run() on a thread of its own, so Ctrl+C / SIGTERM (a KeyboardInterrupt in the main thread) can cancel the request in
    flight: the engine is told to STOP, the thread given a moment to see it, and the interrupt goes on."""
    cancel = threading.Event()
    out: dict = {}

    def work():
        try:
            out.update(run(svc, mode, parks, log=log, cancel=cancel))
        except BaseException as e:          # noqa: BLE001 - last line of defence: the server still starts
            log(f"[strata] warm-up failed ({type(e).__name__}: {e}); the server starts without it")

    t = threading.Thread(target=work, name="strata-warmup", daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.5)
    except KeyboardInterrupt:
        cancel.set()
        t.join(30)
        raise
    return out
