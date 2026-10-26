#!/usr/bin/env python3
"""A/B speed benchmark for an OpenAI-compatible chat server (LLMServe, original Strata, llama.cpp ...).

Standalone and stdlib-only (Python 3.9+): it imports nothing from this repository, so the same file runs on Windows
(`py bench.py`) and inside the container (`python -m llmserve.bench`).

    python -m llmserve.bench --label llmserve --out /data/bench-llmserve.json     # measure one server
    py bench.py --url http://127.0.0.1:8095/v1 --label strata --out bench-strata.json
    python bench.py --compare bench-llmserve.json bench-strata.json               # side by side, % difference

Concurrency mode (`--concurrency 1,2,3,4 --scenario independent|shared-prefix|agent|join`, see --help) runs N clients at once
and reports aggregate tok/s, per-stream tok/s, TTFT p50/p95, the longest gap between tokens and the reused prompt tokens per
level. `--compare` takes two concurrency files as well.

Every request is a streamed /v1/chat/completions call (temperature 0, seed 1, thinking off) with a unique run id on the first
line, so a prompt cache cannot reuse an earlier run. One warm-up request is sent first and not counted.

Per request it records the server's own `timings` (llama.cpp names: prompt_n, prompt_per_second, predicted_n,
predicted_per_second, cache_n, draft_n, draft_n_accepted) when the last chunk carries them, plus client-side numbers: time to
first token (TTFT), wall time and a decode rate = (completion_tokens - 1) / (last token time - first token time). The client
decode rate is the fallback when the server sends no `timings`; the first token is excluded because it arrives at the start of
the interval being timed.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import random
import re
import socket
import statistics
import sys
import threading
import time
import urllib.parse
import uuid

DEFAULT_URL = "http://127.0.0.1:8080/v1"
DEFAULT_CASES = "short,prompt-4k,prompt-16k"
RESULT_VERSION = 1
WARMUP_MAX_TOKENS = 64
CHARS_PER_TOKEN = 4                    # the rough size of the synthetic documents: 1 token ~ 4 characters
DOC_SEED = 1234
CONN_HINT = ("is the server running? on Docker Desktop, from inside a container use "
             "http://host.docker.internal:PORT/v1")

SHORT_PROMPT = (
    "You are a senior systems engineer. Write a long, detailed technical explanation of how a modern mixture-of-experts "
    "language model is served on a single consumer GPU: cover expert routing, where the weights and the KV cache are "
    "placed in memory, batching, speculative decoding, quantization trade-offs, and latency versus throughput. Use "
    "numbered sections with concrete examples, and do not stop early."
)
SUMMARIZE = "Summarize the document above in detail."


class BenchError(Exception):
    """A failure to report to the user without a traceback."""


# ----------------------------------------------------------------------------------------------------------------------
# the synthetic documents

_SUBJECTS = ["The scheduler", "A worker thread", "The storage layer", "Our caching proxy", "The routing table",
             "Each replica", "The build system", "The monitoring agent", "A long-running job", "The network driver",
             "The configuration loader", "A client library", "The expert cache", "The tokenizer", "The deployment pipeline"]
_VERBS = ["rebalances", "records", "validates", "compresses", "forwards", "retries", "schedules", "drops", "merges",
          "serializes", "prefetches", "audits", "throttles", "indexes", "replays"]
_OBJECTS = ["pending requests", "the incoming batches", "stale entries", "every partition", "the shared buffers",
            "a rolling window of samples", "the ordered log", "all open connections", "idle sessions",
            "the dependency graph", "small writes", "the oldest checkpoints", "temporary files", "hot keys"]
_TAILS = ["before the next heartbeat arrives", "whenever memory pressure rises above the soft limit",
          "so that readers never observe a partial update", "while holding only a shared lock",
          "unless the operator has disabled the feature", "to keep tail latency predictable",
          "and reports the outcome to the central dashboard", "using a fixed budget of {n} milliseconds",
          "after at most {n} attempts", "in groups of {n} items", "because the previous run left {n} gaps",
          "which usually finishes within {n} seconds"]
_PROSE = [
    "In practice the hardest part of the migration was not the code but agreeing on what counted as a complete record.",
    "Early measurements suggested that the slowest requests were dominated by waiting, not by computation.",
    "The committee met on {day} to review the budget, and most of the discussion concerned maintenance rather than growth.",
    "A careful reader will notice that the two reports disagree about the number of incidents in {month}.",
    "When the river floods, the old bridge is closed and traffic is redirected through the northern valley.",
    "Researchers described the new alloy as unusually light, yet it held its shape at temperatures above {n} degrees.",
    "The recipe calls for slow heat, patient stirring, and a pinch of salt added only at the very end.",
    "Several users reported that the settings page loaded slowly, so the team added a progress indicator first.",
    "Nothing in the contract prevents either party from ending the arrangement with {n} days of written notice.",
    "The museum reopened in {month} after a long renovation, with a quieter entrance and better lighting.",
    "Although the forecast was uncertain, the crew decided to leave at dawn and take the longer coastal route.",
    "To reduce waste, the kitchen now plans each week's menu around what is already in the pantry.",
    "The orchestra rehearsed the second movement {n} times before the conductor was satisfied with the balance.",
    "Historians still debate whether the letter was written in {month} or several weeks later.",
    "Storage costs fell steadily, which made it practical to keep {n} years of raw data instead of a summary.",
    "A good interface hides complexity without lying about it, and that balance is harder than it sounds.",
]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
           "December"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_NAMES = ["parse_header", "load_shard", "flush_queue", "merge_runs", "hash_key", "encode_block", "next_token",
          "pick_expert", "write_page", "read_chunk", "update_stats", "plan_batch"]
_CODE = [
    "def {f}(buf, offset={n}):\n    return buf[offset:offset + {m}]",
    "for i in range({n}):\n    total += values[i] * {m}  # weight {k}",
    "if (count > {n}) {{ retries = {m}; backoff_ms = {k}; }}",
    "const {v} = await {f}(session, {{ limit: {n}, timeout: {m} }});",
    "static int {f}(struct ctx *c, size_t len) {{ return len > {n} ? -{m} : 0; }}",
    "SELECT id, {v}, COUNT(*) FROM events WHERE ts > {n} GROUP BY id, {v} LIMIT {m};",
    "# TODO({v}): revisit the {n} byte limit once {f} is faster",
    "let {v} = items.iter().filter(|x| x.size > {n}).map(|x| x.id * {m}).sum::<u64>();",
]
_VARS = ["offset", "limit", "score", "latency", "window", "index", "cursor", "budget", "ratio", "depth"]


def _fill(rng: random.Random, template: str) -> str:
    return template.format(n=rng.randint(2, 9999), m=rng.randint(2, 999), k=rng.randint(1, 64),
                           f=rng.choice(_NAMES), v=rng.choice(_VARS), day=rng.choice(_DAYS), month=rng.choice(_MONTHS))


def _sentence(rng: random.Random) -> str:
    if rng.randint(0, 2) == 0:
        return _fill(rng, rng.choice(_PROSE))
    tail = _fill(rng, rng.choice(_TAILS))
    return "%s %s %s %s." % (rng.choice(_SUBJECTS), rng.choice(_VERBS), rng.choice(_OBJECTS), tail)


def build_document(target_tokens: int, seed: int = DOC_SEED) -> str:
    """A deterministic document of about `target_tokens` tokens (1 token ~ 4 characters): prose and code-like lines,
    varied by a seeded generator so it is not one repeated sentence. The same text on every machine and every call
    (for the same seed; the concurrency scenarios use other seeds to get different documents)."""
    rng = random.Random(seed)
    target_chars = target_tokens * CHARS_PER_TOKEN
    parts = []
    size = 0
    section = 1
    while size < target_chars:
        if rng.randint(0, 4) < 3:
            block = " ".join(_sentence(rng) for _ in range(rng.randint(2, 5)))
        else:
            block = "\n".join(_fill(rng, rng.choice(_CODE)) for _ in range(rng.randint(2, 4)))
        block = "[%d] %s" % (section, block)
        section += 1
        parts.append(block)
        size += len(block) + 2
    return "\n\n".join(parts)


def case_target_tokens(case: str):
    """None for `short`; N*1000 for `prompt-Nk` (prompt-4k, prompt-16k, and any other N)."""
    if case == "short":
        return None
    m = re.fullmatch(r"prompt-(\d+)k", case)
    if not m or int(m.group(1)) < 1:
        raise BenchError("unknown case %r (use short, prompt-4k, prompt-16k, or prompt-<N>k)" % case)
    return int(m.group(1)) * 1000


_DOC_CACHE = {}


def case_body(case: str) -> str:
    """The prompt text of a case without the run id line."""
    target = case_target_tokens(case)
    if target is None:
        return SHORT_PROMPT
    if target not in _DOC_CACHE:
        _DOC_CACHE[target] = build_document(target)
    return _DOC_CACHE[target] + "\n\n" + SUMMARIZE


def make_prompt(case: str, nonce=None) -> str:
    """`Run id: <uuid4>` on the very first line (so no prompt cache can match a previous run), then the fixed text."""
    return "Run id: %s\n\n%s" % (nonce or uuid.uuid4(), case_body(case))


# ----------------------------------------------------------------------------------------------------------------------
# HTTP

def normalize_url(url: str) -> str:
    """Base URL ending in /v1, whether the user typed it with or without it (and with or without a trailing slash)."""
    url = (url or "").strip()
    if not url:
        raise BenchError("empty --url")
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def root_url(base: str) -> str:
    return base[:-3] if base.endswith("/v1") else base


class Client:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 1800.0):
        self.base = normalize_url(base_url)
        self.api_key = api_key or ""
        self.timeout = timeout
        parts = urllib.parse.urlsplit(self.base)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise BenchError("bad --url %r" % base_url)
        self.scheme, self.host, self.port = parts.scheme, parts.hostname, parts.port
        self.prefix = parts.path.rstrip("/")                  # ".../v1"
        self.root_prefix = self.prefix[:-3] if self.prefix.endswith("/v1") else self.prefix

    def _conn(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def _headers(self, body: bool):
        h = {"Accept": "application/json, text/event-stream", "User-Agent": "llmserve-bench"}
        if body:
            h["Content-Type"] = "application/json"
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def _wrap(self, exc):
        if isinstance(exc, (socket.timeout, TimeoutError)):
            return BenchError("timed out after %g s waiting for %s:%s (raise --timeout)" % (self.timeout, self.host, self.port or "default"))
        if isinstance(exc, ConnectionRefusedError):
            return BenchError("connection refused by %s:%s - %s" % (self.host, self.port or "default", CONN_HINT))
        if isinstance(exc, (ConnectionError, socket.gaierror, OSError, http.client.HTTPException)):
            return BenchError("cannot talk to %s:%s (%s: %s) - %s" % (self.host, self.port or "default", type(exc).__name__, exc, CONN_HINT))
        return exc

    def get_json(self, path: str, root: bool = False, timeout=None):
        """GET and parse JSON. Raises BenchError on HTTP errors and connection problems."""
        full = (self.root_prefix if root else self.prefix) + path
        conn = self._conn()
        if timeout:
            conn.timeout = timeout
        try:
            conn.request("GET", full, headers=self._headers(False))
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise BenchError("GET %s -> HTTP %d %s: %s" % (full, resp.status, resp.reason, data.decode("utf-8", "replace")[:2000]))
            try:
                return json.loads(data.decode("utf-8", "replace"))
            except ValueError:
                raise BenchError("GET %s: not JSON: %r" % (full, data[:200]))
        except BenchError:
            raise
        except Exception as exc:                                # noqa: BLE001
            raise self._wrap(exc)
        finally:
            conn.close()

    def stream_chat(self, payload: dict) -> dict:
        """POST a streaming chat completion; returns the parsed stream (see parse_stream)."""
        full = self.prefix + "/chat/completions"
        conn = self._conn()
        try:
            t0 = time.perf_counter()
            conn.request("POST", full, body=json.dumps(payload).encode("utf-8"), headers=self._headers(True))
            resp = conn.getresponse()
            if resp.status != 200:
                body = resp.read().decode("utf-8", "replace")
                raise BenchError("POST %s -> HTTP %d %s: %s" % (full, resp.status, resp.reason, body[:4000]))
            return parse_stream(resp, t0)
        except BenchError:
            raise
        except Exception as exc:                                # noqa: BLE001
            raise self._wrap(exc)
        finally:
            conn.close()


def parse_stream(resp, t0: float, clock=time.perf_counter) -> dict:
    """Read SSE `data:` lines from a file-like `resp` (readline) until [DONE] or EOF.

    Returns {ttft_s, wall_s, first_s, last_s, usage, timings, finish_reason, content_chars, deltas, t0, times, text}. Times are
    seconds since `t0` (`times`: one per delta with text). The first/last token times are those of the first and last delta
    with text (content or reasoning); `text` is the content only."""
    first = last = None
    usage = timings = finish = None
    chars = deltas = 0
    times, parts = [], []                 # per text delta: seconds since t0; the answer text (content only)
    while True:
        raw = resp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("error"):
            err = obj["error"]
            raise BenchError("server error in stream: %s" % (err.get("message") if isinstance(err, dict) and err.get("message") else json.dumps(err)[:2000]))
        now = clock() - t0
        if isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
        if isinstance(obj.get("timings"), dict):
            timings = obj["timings"]
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            text = ""
            for key in ("content", "reasoning_content", "reasoning"):
                v = delta.get(key)
                if isinstance(v, str) and v:
                    text += v
            if text:
                if first is None:
                    first = now
                last = now
                chars += len(text)
                deltas += 1
                times.append(now)
                if isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {"ttft_s": first, "wall_s": clock() - t0, "first_s": first, "last_s": last, "usage": usage, "timings": timings,
            "finish_reason": finish, "content_chars": chars, "deltas": deltas, "t0": t0, "times": times,
            "text": "".join(parts)}


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def cached_tokens(stream: dict):
    """Prompt tokens the server reused: llama.cpp's `timings.cache_n` and/or `usage.prompt_tokens_details.cached_tokens` (the
    larger one when a server sends both; None when it sends neither)."""
    found = []
    v = (stream.get("timings") or {}).get("cache_n")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        found.append(v)
    details = (stream.get("usage") or {}).get("prompt_tokens_details")
    v = details.get("cached_tokens") if isinstance(details, dict) else None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        found.append(v)
    return max(found) if found else None


def run_metrics(stream: dict) -> dict:
    """One request's metrics from a parsed stream: server `timings` first, client-side numbers as the fallback."""
    usage = stream.get("usage") or {}
    timings = stream.get("timings") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None and timings:
        prompt_tokens = (timings.get("prompt_n") or 0) + (timings.get("cache_n") or 0) or None
    if completion_tokens is None:
        completion_tokens = timings.get("predicted_n") if timings.get("predicted_n") is not None else stream.get("deltas")
    ttft = stream.get("ttft_s")
    first, last = stream.get("first_s"), stream.get("last_s")
    client_decode = None
    if completion_tokens and completion_tokens > 1 and first is not None and last is not None and last > first:
        client_decode = (completion_tokens - 1) / (last - first)
    prompt_tps, prompt_src = _num(timings.get("prompt_per_second")), "server"
    if prompt_tps is None:
        prompt_src = "client"
        prompt_tps = (prompt_tokens / ttft) if prompt_tokens and ttft else None       # includes the first token and the network
    decode_tps, decode_src = _num(timings.get("predicted_per_second")), "server"
    if decode_tps is None:
        decode_src, decode_tps = "client", client_decode
    draft_n, draft_ok = _num(timings.get("draft_n")), _num(timings.get("draft_n_accepted"))
    return {
        "prompt_tps": prompt_tps, "prompt_tps_source": prompt_src,
        "decode_tps": decode_tps, "decode_tps_source": decode_src,
        "client_decode_tps": client_decode,
        "ttft_s": ttft, "wall_s": stream.get("wall_s"),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "cache_n": cached_tokens(stream),
        "draft_n": draft_n, "draft_n_accepted": draft_ok,
        "draft_accept": (draft_ok / draft_n) if draft_n and draft_ok is not None else None,
        "finish_reason": stream.get("finish_reason"),
        "timings": stream.get("timings"), "usage": stream.get("usage"),
    }


def chat_payload(model: str, prompt: str, max_tokens: int) -> dict:
    return chat_payload_messages(model, [{"role": "user", "content": prompt}], max_tokens)


def chat_payload_messages(model: str, messages: list, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "seed": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def run_request(client: Client, model: str, case: str, max_tokens: int) -> dict:
    stream = client.stream_chat(chat_payload(model, make_prompt(case), max_tokens))
    if stream["first_s"] is None:
        raise BenchError("the server streamed no text for case %s (finish_reason=%s)" % (case, stream.get("finish_reason")))
    return run_metrics(stream)


# ----------------------------------------------------------------------------------------------------------------------
# statistics and tables

# key, label, format, direction (+1 higher is better, -1 lower is better, 0 informational)
METRICS = [
    ("prompt_tps", "prompt tok/s", "%.1f", 1),
    ("decode_tps", "decode tok/s", "%.1f", 1),
    ("client_decode_tps", "client decode tok/s", "%.1f", 1),
    ("ttft_s", "TTFT (s)", "%.2f", -1),
    ("wall_s", "wall time (s)", "%.2f", -1),
    ("prompt_tokens", "prompt tokens", "%.0f", 0),
    ("completion_tokens", "completion tokens", "%.0f", 0),
    ("cache_n", "cached prompt tokens", "%.0f", 0),
    ("draft_accept", "draft acceptance", "pct", 1),
]


def fmt(value, f: str) -> str:
    if value is None:
        return "-"
    if f == "pct":
        return "%.1f%%" % (value * 100)
    return f % value


def summarize(runs: list) -> dict:
    """{metric: {median, min, max, n}} over the runs that have the metric."""
    out = {}
    for key, _label, _f, _d in METRICS:
        vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        if vals:
            out[key] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}
    return out


def render_case(case: str, runs: list, max_tokens: int) -> str:
    s = summarize(runs)
    lines = ["", "== %s: %d run(s), max_tokens %d ==" % (case, len(runs), max_tokens),
             "%-24s %12s   %s" % ("metric", "median", "min - max")]
    for key, label, f, _d in METRICS:
        if key not in s:
            continue
        m = s[key]
        lines.append("%-24s %12s   %s - %s" % (label, fmt(m["median"], f), fmt(m["min"], f), fmt(m["max"], f)))
    srv_p = sum(1 for r in runs if r.get("prompt_tps_source") == "server")
    srv_d = sum(1 for r in runs if r.get("decode_tps_source") == "server")
    lines.append("source: prompt/decode tok/s from server timings in %d/%d and %d/%d runs (the rest are client estimates)"
                 % (srv_p, len(runs), srv_d, len(runs)))
    notes = case_notes(runs, max_tokens)
    lines.extend("note: " + n for n in notes)
    return "\n".join(lines)


def case_notes(runs: list, max_tokens: int) -> list:
    notes = []
    comp = [r["completion_tokens"] for r in runs if isinstance(r.get("completion_tokens"), (int, float))]
    if comp and statistics.median(comp) < 0.5 * max_tokens:
        notes.append("completion tokens (median %.0f) are far below max_tokens %d: the model stopped early, so decode "
                     "numbers are less comparable between servers." % (statistics.median(comp), max_tokens))
    cached = [r["cache_n"] for r in runs if isinstance(r.get("cache_n"), (int, float)) and r["cache_n"] > 0]
    if cached:
        notes.append("%d run(s) report cached prompt tokens (cache_n > 0); their prompt tok/s covers only the tokens read." % len(cached))
    return notes


def _squash(obj):
    """Long strings (a chat template in /props) become '<N chars>' so the one-line summary stays readable."""
    if isinstance(obj, dict):
        return {k: _squash(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_squash(v) for v in obj[:20]]
    if isinstance(obj, str) and len(obj) > 80:
        return "<%d chars>" % len(obj)
    return obj


def _short_json(obj, limit: int = 600) -> str:
    s = json.dumps(_squash(obj), sort_keys=True, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + "...(%d chars)" % len(s)


# ----------------------------------------------------------------------------------------------------------------------
# compare

def load_result(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise BenchError("cannot read %s: %s" % (path, exc))
    if isinstance(data, dict) and data.get("mode") == "concurrency" and isinstance(data.get("scenarios"), dict):
        return data
    if not isinstance(data, dict) or not isinstance(data.get("cases"), dict):
        raise BenchError("%s is not a bench result (no 'cases' or concurrency 'scenarios')" % path)
    return data


def pct_diff(a, b):
    if a is None or b is None or a == 0:
        return None
    return (b - a) / abs(a) * 100.0


def compare_results(a: dict, b: dict, name_a: str, name_b: str, same_threshold: float = 2.0) -> str:
    out = []
    for tag, d, name in (("A", a, name_a), ("B", b, name_b)):
        args = d.get("args") or {}
        out.append("%s = %s  (model %s, %s, runs %s, max_tokens %s, %s)"
                   % (tag, name, d.get("model") or "?", d.get("url") or "?", args.get("runs", "?"), args.get("max_tokens", "?"),
                      d.get("created") or "?"))
    out.append("diff = (B - A) / A. 'better' follows the metric: higher tok/s, lower TTFT and wall time.")
    cases = [c for c in a["cases"] if c in b["cases"]]
    if not cases:
        raise BenchError("the two files have no case in common (A: %s, B: %s)" % (", ".join(a["cases"]) or "-", ", ".join(b["cases"]) or "-"))
    skipped = sorted(set(a["cases"]) ^ set(b["cases"]))
    w = max(12, min(22, max(len(name_a), len(name_b))))
    for case in cases:
        ra, rb = a["cases"][case].get("runs") or [], b["cases"][case].get("runs") or []
        sa, sb = summarize(ra), summarize(rb)
        out.append("")
        out.append("== %s (median of %d run(s) in A, %d in B) ==" % (case, len(ra), len(rb)))
        out.append("%-24s %*s %*s %10s  %s" % ("metric", w, name_a[:w], w, name_b[:w], "diff", ""))
        for key, label, f, direction in METRICS:
            if key not in sa and key not in sb:
                continue
            if key == "cache_n" and not sa.get(key, {}).get("max") and not sb.get(key, {}).get("max"):
                continue                                        # nothing was cached on either side: no row
            va = sa.get(key, {}).get("median")
            vb = sb.get(key, {}).get("median")
            d = pct_diff(va, vb)
            verdict = ""
            if d is not None and direction:
                if abs(d) < same_threshold:
                    verdict = "same"
                else:
                    better_b = (d > 0) == (direction > 0)
                    verdict = "B better" if better_b else "A better"
            out.append("%-24s %*s %*s %10s  %s" % (label, w, fmt(va, f), w, fmt(vb, f), ("%+.1f%%" % d) if d is not None else "n/a", verdict))
        pa, pb = sa.get("prompt_tokens", {}).get("median"), sb.get("prompt_tokens", {}).get("median")
        if pa and pb and abs(pb - pa) / pa > 0.05:
            out.append("note: the two servers counted different prompt sizes (different tokenizers or templates); prompt tok/s and TTFT are not for the same work.")
        ca, cb = sa.get("completion_tokens", {}).get("median"), sb.get("completion_tokens", {}).get("median")
        if ca and cb and abs(cb - ca) / ca > 0.2:
            out.append("note: completion lengths differ by more than 20%; decode tok/s may not be comparable.")
        for tag, d, runs in (("A", a, ra), ("B", b, rb)):
            mt = (d.get("args") or {}).get("max_tokens")
            for n in (case_notes(runs, mt) if mt else []):
                out.append("note (%s): %s" % (tag, n))
    if skipped:
        out.append("")
        out.append("not compared (only in one file): " + ", ".join(skipped))
    return "\n".join(out)


def result_name(d: dict, path: str) -> str:
    return str(d.get("label") or os.path.splitext(os.path.basename(path))[0])


# ----------------------------------------------------------------------------------------------------------------------
# concurrency mode: N clients (threads) at once
#
# python bench.py --concurrency 1,2,3,4 --scenario shared-prefix --prefix-tokens 20000 --runs 3
#
# Scenarios (every client is a thread with its own connection; all are released together by one threading.Barrier):
#   independent    N different prompts (a different task, or a different document, per client; the run id is the first line, so
#                  nothing can be reused).
#   shared-prefix  N prompts that share one long system message (~--prefix-tokens: a header and a document, standing in for a
#                  coding agent's system prompt + tool definitions) and differ in the user message.  The prefix text is
#                  identical for every client, level and round of ONE bench run, so the server can reuse it across requests; the
#                  per-request run id is the first line of the USER message, after the prefix, so tails never match.  A per-run
#                  session id is the first line of the prefix, so a second bench run against the same server starts cold.  One
#                  priming request (not counted, its cold TTFT is reported) is sent first; --no-prime skips it, and then the
#                  first round of the first level pays the cold prefill.
#   agent          shared-prefix, then --turns turns per client: each next turn appends the assistant's answer and a new user
#                  message, so the history grows and the server's prompt cache should reuse all of the earlier turns.
#   join           N-1 clients decode a long answer (--decode-tokens); --join-delay seconds after the barrier one more client
#                  sends a cold prompt of ~--prefix-tokens.  Measures what the long prompt read does to the others: their longest
#                  gap between tokens, their tok/s while it is read, and the joiner's TTFT.  Level 1 is the joiner alone.

SCENARIOS = ("independent", "shared-prefix", "agent", "join")
CONC_VERSION = 1
DEFAULT_PREFIX_TOKENS = 20000
DEFAULT_TURNS = 3
DEFAULT_DECODE_TOKENS = 2048
DEFAULT_JOIN_DELAY_S = 3.0
CONC_MAX_CLIENTS = 64
PREFIX_SEED, JOIN_SEED = 4321, 777
BARRIER_TIMEOUT_S = 120.0
PRIME_MAX_TOKENS = 8
STATUS_POLL_S = 1.0

_TASKS = [
    "Explain in great detail how a mixture-of-experts language model routes tokens to experts, with a worked example.",
    "Write a long tutorial on designing a write-ahead log for a key-value store, with code sketches in Python.",
    "Describe step by step how a TCP connection is established, kept alive and closed, and what goes wrong on lossy links.",
    "Write a detailed comparison of B-trees and LSM-trees for a database, with workloads where each one wins.",
    "Explain how a modern compiler turns source code into machine code, phase by phase, with small examples.",
    "Write a long essay on the history of public-key cryptography and the problems each new scheme solved.",
    "Describe in detail how garbage collectors work in managed runtimes, comparing tracing and reference counting.",
    "Write a thorough guide to planning a zero-downtime migration of a large database, with a checklist per phase.",
]
_FOLLOWUPS = [
    "Good. Now go deeper on the part you found hardest and add a concrete example.",
    "Thanks. List the three most likely mistakes a beginner would make here and how to avoid each.",
    "Now summarize everything so far in a short checklist.",
    "Continue with what you left out, and keep the same level of detail.",
]
PREFIX_HEADER = ("You are a coding agent working in a large repository. The tool definitions, project notes and style guide follow; "
                 "read them before you answer.")


def task_text(i: int) -> str:
    return _TASKS[i % len(_TASKS)]


def followup_text(i: int, turn: int) -> str:
    """The user message of turn `turn` (2, 3 ...) of client `i`."""
    return "%s (client %d, turn %d)" % (_FOLLOWUPS[(i + turn) % len(_FOLLOWUPS)], i, turn)


def new_nonce() -> str:
    return uuid.uuid4().hex


def shared_prefix_text(prefix_tokens: int, session_id: str) -> str:
    """The system message shared by every client of a bench run: `Session: <id>` (so another bench run starts cold), a header
    and a document of ~prefix_tokens tokens."""
    return "Session: %s\n%s\n\n%s" % (session_id, PREFIX_HEADER, build_document(prefix_tokens, PREFIX_SEED))


def user_tail(i: int, nonce: str) -> str:
    """The part of a shared-prefix / agent request that differs between requests: the run id first, then the client's task."""
    return "Run id: %s\n\n%s" % (nonce, task_text(i))


def independent_messages(i: int, nonce: str, case: str = "short") -> list:
    """Client i's own prompt: the run id first, then a task that differs per client, or (prompt-<N>k) a document of N thousand
    tokens that differs per client."""
    target = case_target_tokens(case)
    if target is None:
        body = task_text(i)
    else:
        body = build_document(target, DOC_SEED + 1 + i) + "\n\n" + SUMMARIZE
    return [{"role": "user", "content": "Run id: %s\n\n%s" % (nonce, body)}]


def shared_prefix_messages(prefix: str, i: int, nonce: str) -> list:
    return [{"role": "system", "content": prefix}, {"role": "user", "content": user_tail(i, nonce)}]


def join_messages(prefix_tokens: int, nonce: str) -> list:
    """The joiner: a cold prompt (run id first) of ~prefix_tokens tokens, a document nobody else sends."""
    return [{"role": "user", "content": "Run id: %s\n\n%s\n\n%s" % (nonce, build_document(prefix_tokens, JOIN_SEED), SUMMARIZE)}]


class ClientPlan:
    """What one client thread does: wait `delay_s` after the barrier, send `messages`, then (agent) for every entry of
    `followups` append the answer and that user message and send again."""

    def __init__(self, index: int, role: str, messages: list, max_tokens: int, delay_s: float = 0.0, followups=()):
        self.index, self.role, self.messages = index, role, messages
        self.max_tokens, self.delay_s, self.followups = max_tokens, delay_s, list(followups)


def build_plans(scenario: str, n: int, *, prefix: str = "", case: str = "short", prefix_tokens: int = DEFAULT_PREFIX_TOKENS,
                max_tokens: int = 512, turns: int = DEFAULT_TURNS, decode_tokens: int = DEFAULT_DECODE_TOKENS,
                join_delay_s: float = DEFAULT_JOIN_DELAY_S, stagger_s: float = 0.0, nonce=new_nonce) -> list:
    """The ClientPlans of one round of `n` clients.  `nonce()` makes a fresh run id per request."""
    if scenario == "independent":
        return [ClientPlan(i, "client", independent_messages(i, nonce(), case), max_tokens, i * stagger_s) for i in range(n)]
    if scenario == "shared-prefix":
        return [ClientPlan(i, "client", shared_prefix_messages(prefix, i, nonce()), max_tokens, i * stagger_s) for i in range(n)]
    if scenario == "agent":
        return [ClientPlan(i, "client", shared_prefix_messages(prefix, i, nonce()), max_tokens, i * stagger_s,
                           [followup_text(i, k) for k in range(2, turns + 1)]) for i in range(n)]
    if scenario == "join":
        plans = [ClientPlan(i, "decoder", [{"role": "user", "content": "Run id: %s\n\n%s" % (nonce(), task_text(i))}],
                            decode_tokens, i * stagger_s) for i in range(n - 1)]
        plans.append(ClientPlan(n - 1, "joiner", join_messages(prefix_tokens, nonce()), max_tokens,
                                join_delay_s if n > 1 else 0.0))
        return plans
    raise BenchError("unknown scenario %r (use %s)" % (scenario, ", ".join(SCENARIOS)))


# ---- percentiles and small helpers

def percentile(values, p: float):
    """Nearest-rank percentile (p in 0..100) of the numbers in `values`; None when there are none."""
    vals = sorted(v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not vals:
        return None
    k = max(1, int(math.ceil(p / 100.0 * len(vals))))
    return vals[min(k, len(vals)) - 1]


def _median(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def max_gap(times: list):
    """The longest wait between two consecutive text deltas, or None for fewer than two."""
    return max((b - a for a, b in zip(times, times[1:])), default=None)


def parse_levels(text: str) -> list:
    out = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit() or not 1 <= int(tok) <= CONC_MAX_CLIENTS:
            raise BenchError("--concurrency %r: use whole numbers from 1 to %d, e.g. 1,2,3,4" % (text, CONC_MAX_CLIENTS))
        if int(tok) not in out:
            out.append(int(tok))
    if not out:
        raise BenchError("--concurrency is empty")
    return out


def parse_scenarios(text: str) -> list:
    out = [s.strip() for s in (text or "independent").split(",") if s.strip()]
    for s in out:
        if s not in SCENARIOS:
            raise BenchError("unknown scenario %r (use %s, or a comma list)" % (s, ", ".join(SCENARIOS)))
    return list(dict.fromkeys(out)) or ["independent"]


# ---- status snapshots (best effort)

_STATUS_KEEP = re.compile(r"slot|serving|requested|in_flight|queue|busy|running|active|batch|util|used_mib|concurren|vram", re.I)


def flatten_numbers(obj, prefix: str = "", depth: int = 0, out=None, limit: int = 40) -> dict:
    """The small numeric fields of a JSON status document, as {"a.b.c": number}; lists contribute their length as `a.b.len`.
    Only keys that look like slot / queue / load counters are kept."""
    out = {} if out is None else out
    if depth > 4 or len(out) >= limit:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_numbers(v, "%s.%s" % (prefix, k) if prefix else str(k), depth + 1, out, limit)
    elif isinstance(obj, list):
        if _STATUS_KEEP.search(prefix):
            out[prefix + ".len"] = len(obj)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and _STATUS_KEEP.search(prefix):
        out[prefix] = obj
    return out


class StatusSampler(threading.Thread):
    """Polls GET /v1/status and GET /metrics while a round runs; failures are ignored (an endpoint that fails twice in a row is
    left alone).  `peak` keeps the largest value seen per field, `last` the latest sample."""

    def __init__(self, client: Client, interval: float = STATUS_POLL_S):
        super().__init__(daemon=True)
        self.client, self.interval = client, interval
        self.stop_event = threading.Event()
        self.samples = 0
        self.peak, self.last = {}, {}
        self._fails = {"status": 0, "metrics": 0}

    def sample(self) -> None:
        got = False
        for name, path, root in (("status", "/status", False), ("metrics", "/metrics", True)):
            if self._fails[name] >= 2:
                continue
            try:
                doc = self.client.get_json(path, root=root, timeout=5)
            except Exception:                                   # noqa: BLE001
                self._fails[name] += 1
                continue
            self._fails[name] = 0
            for k, v in flatten_numbers(doc, name).items():
                self.last[k] = v
                self.peak[k] = max(v, self.peak.get(k, v))
            got = True
        if got:
            self.samples += 1

    def run(self) -> None:
        if self.stop_event.wait(min(0.3, self.interval)):
            return
        while True:
            self.sample()
            if self.stop_event.wait(self.interval):
                return

    def stop(self) -> dict:
        self.stop_event.set()
        self.join(timeout=10)
        return {"samples": self.samples, "peak": dict(self.peak), "last": dict(self.last)}


# ---- one round: the threads

def _client_worker(plan: ClientPlan, make_client, model: str, barrier, release: dict, out: list) -> None:
    """One client thread.  Appends one record per request to `out` (a failed request gets an `error` record)."""
    messages = list(plan.messages)
    try:
        client = make_client()
        barrier.wait(BARRIER_TIMEOUT_S)
    except Exception as exc:                                    # noqa: BLE001 (a broken barrier: the round reports it)
        barrier.abort()                                         # the other clients fail now, not after the barrier timeout
        out.append({"client": plan.index, "role": plan.role, "turn": 1, "error": "%s: %s" % (type(exc).__name__, exc)})
        return
    if plan.delay_s > 0:
        time.sleep(plan.delay_s)
    for turn in range(1, 2 + len(plan.followups)):
        rec = {"client": plan.index, "role": plan.role, "turn": turn, "messages": len(messages)}
        t_send = time.perf_counter()
        try:
            stream = client.stream_chat(chat_payload_messages(model, messages, plan.max_tokens))
            if stream["first_s"] is None:
                raise BenchError("the server streamed no text (finish_reason=%s)" % stream.get("finish_reason"))
        except Exception as exc:                                # noqa: BLE001
            rec.update(start_s=t_send - release["t"], error=str(exc) or type(exc).__name__)
            out.append(rec)
            return
        m = run_metrics(stream)
        t0 = stream["t0"] - release["t"]
        times = [t0 + x for x in stream["times"]]
        comp = m["completion_tokens"] if m["completion_tokens"] is not None else stream["deltas"]
        rec.update(start_s=t0, first_token_s=t0 + stream["first_s"], last_token_s=t0 + stream["last_s"],
                   end_s=t0 + stream["wall_s"], ttft_s=m["ttft_s"], wall_s=m["wall_s"], decode_tps=m["decode_tps"],
                   decode_tps_source=m["decode_tps_source"], client_decode_tps=m["client_decode_tps"],
                   prompt_tokens=m["prompt_tokens"], completion_tokens=comp, cache_n=m["cache_n"],
                   deltas=stream["deltas"], max_gap_s=max_gap(times), finish_reason=m["finish_reason"], _times=times)
        out.append(rec)
        if turn <= len(plan.followups):
            messages = messages + [{"role": "assistant", "content": stream["text"] or "(no answer)"},
                                   {"role": "user", "content": plan.followups[turn - 1]}]


def run_round(plans: list, make_client, model: str, sampler_client=None) -> dict:
    """Run the plans' clients as threads released together by a barrier.  Returns {records, errors, wall_s, status}; request
    times are seconds after the release.  `wall_s` is the release to the last request's end."""
    barrier_info = {"t": None}

    def mark():
        barrier_info["t"] = time.perf_counter()

    barrier = threading.Barrier(len(plans), action=mark)
    outs = [[] for _ in plans]
    threads = [threading.Thread(target=_client_worker, args=(p, make_client, model, barrier, barrier_info, outs[i]), daemon=True)
               for i, p in enumerate(plans)]
    sampler = StatusSampler(sampler_client) if sampler_client is not None else None
    for th in threads:
        th.start()
    if sampler:
        sampler.start()
    try:
        for th in threads:
            while th.is_alive():
                th.join(0.5)                                    # short joins keep Ctrl-C working on Windows
    finally:
        status = sampler.stop() if sampler else None
    records = [r for out in outs for r in out]
    ok = [r for r in records if not r.get("error")]
    errors = [{"client": r["client"], "turn": r.get("turn"), "error": r["error"]} for r in records if r.get("error")]
    wall = max((r["end_s"] for r in ok), default=None)
    return {"records": ok, "errors": errors, "wall_s": wall, "status": status}


# ---- the join scenario's metrics

def join_metrics(records: list):
    """For one round of the join scenario: what the joiner's prompt read did to the decoding streams.

    The window runs from the moment the joiner sent its request to its first token.  Per decoder that overlapped the window:
    `window_gap_s` the longest gap between its tokens that touches the window, `window_tps` its tokens inside the window per
    second, `pre_tps` its rate from its first token to the join.  None without a joiner or without decoders."""
    joiner = next((r for r in records if r["role"] == "joiner"), None)
    decoders = [r for r in records if r["role"] == "decoder"]
    if joiner is None or not decoders:
        return None
    w0, w1 = joiner["start_s"], joiner["first_token_s"]
    out = {"window_s": w1 - w0, "joiner_ttft_s": joiner["ttft_s"], "decoders": [], "finished_early": 0}
    for d in decoders:
        times = d["_times"]
        if d["last_token_s"] <= w0:
            out["finished_early"] += 1                          # it was done before the joiner asked
            continue
        per_delta = (d["completion_tokens"] / d["deltas"]) if d.get("completion_tokens") and d.get("deltas") else 1.0
        inside = [x for x in times if w0 < x <= w1]
        gaps = [b - a for a, b in zip(times, times[1:]) if b > w0 and a < w1]
        if d["last_token_s"] < w1:
            out["finished_early"] += 1
        before = [x for x in times if x <= w0]
        pre = (len(before) - 1) * per_delta / (w0 - before[0]) if len(before) > 1 and w0 > before[0] else None
        out["decoders"].append({"client": d["client"], "window_gap_s": max(gaps) if gaps else None,
                                "window_tps": len(inside) * per_delta / (w1 - w0) if w1 > w0 else None, "pre_tps": pre,
                                "max_gap_s": d["max_gap_s"]})
    return out


# ---- a level: the rounds of one (scenario, clients) pair

def round_summary(rnd: dict, scenario: str) -> dict:
    ok = rnd["records"]
    tokens = sum(r["completion_tokens"] for r in ok if isinstance(r.get("completion_tokens"), (int, float)))
    wall = rnd["wall_s"]
    out = {"wall_s": wall, "completion_tokens": tokens, "agg_tps": (tokens / wall) if wall else None,
           "requests": len(ok), "errors": len(rnd["errors"])}
    if scenario == "join":
        out["join"] = join_metrics(ok)
    return out


def summarize_level(scenario: str, n: int, rounds: list) -> dict:
    """The numbers of one level over its rounds.  `agg_tps` = completion tokens of the round / wall time of the round (release
    to the last request's end), the median over rounds; the per-stream numbers and the percentiles pool every request of every
    round.  For `join` the per-stream and gap numbers are the decoders', the TTFT and cache numbers the joiner's."""
    rs = [round_summary(r, scenario) for r in rounds]
    recs = [r for rnd in rounds for r in rnd["records"]]
    streams = [r for r in recs if r["role"] != "joiner"]         # whose decode speed and gaps count
    firsts = [r for r in recs if r["role"] != "decoder"]         # whose TTFT and cache reuse count
    aggs = [r["agg_tps"] for r in rs if r["agg_tps"] is not None]
    tps = [r["decode_tps"] for r in streams if isinstance(r.get("decode_tps"), (int, float))]
    gaps = [r["max_gap_s"] for r in streams if r.get("max_gap_s") is not None]
    reporting = [r for r in firsts if isinstance(r.get("cache_n"), (int, float))]
    cached = sum(r["cache_n"] for r in reporting)
    prompt = sum(r["prompt_tokens"] for r in reporting if isinstance(r.get("prompt_tokens"), (int, float)))
    s = {
        "agg_tps": _median(aggs), "agg_tps_min": min(aggs) if aggs else None, "agg_tps_max": max(aggs) if aggs else None,
        "stream_tps_median": _median(tps), "stream_tps_min": min(tps) if tps else None,
        "stream_tps_source": "server" if streams and all(r.get("decode_tps_source") == "server" for r in streams) else "client",
        "ttft_p50": percentile([r["ttft_s"] for r in firsts], 50), "ttft_p95": percentile([r["ttft_s"] for r in firsts], 95),
        "max_gap_s": max(gaps) if gaps else None,
        "cached_mean": (cached / len(reporting)) if reporting else None,
        "cached_pct": (cached / prompt) if reporting and prompt else None,
        "errors": sum(r["errors"] for r in rs), "requests": sum(r["requests"] for r in rs),
        "wall_s": _median([r["wall_s"] for r in rs]),
    }
    if scenario == "agent":
        turns = {}
        for k in sorted({r["turn"] for r in recs}):
            tr = [r for r in recs if r["turn"] == k]
            rep = [r for r in tr if isinstance(r.get("cache_n"), (int, float))]
            pt = sum(r["prompt_tokens"] for r in rep if isinstance(r.get("prompt_tokens"), (int, float)))
            turns[str(k)] = {"requests": len(tr), "ttft_p50": percentile([r["ttft_s"] for r in tr], 50),
                             "ttft_p95": percentile([r["ttft_s"] for r in tr], 95),
                             "prompt_tokens": _median([r.get("prompt_tokens") for r in tr]),
                             "cached_pct": (sum(r["cache_n"] for r in rep) / pt) if rep and pt else None}
        s["by_turn"] = turns
    if scenario == "join":
        joins = [r["join"] for r in rs if r.get("join")]
        decs = [d for j in joins for d in j["decoders"]]
        s.update({
            "join_stall_gap_s": _median([max((d["window_gap_s"] for d in j["decoders"] if d["window_gap_s"] is not None),
                                             default=None) for j in joins]),
            "join_stall_gap_max_s": max((d["window_gap_s"] for d in decs if d["window_gap_s"] is not None), default=None),
            "join_stream_tps": _median([d["window_tps"] for d in decs]),
            "pre_join_stream_tps": _median([d["pre_tps"] for d in decs]),
            "join_ttft_s": _median([r["ttft_s"] for r in firsts]),
            "join_window_s": _median([j["window_s"] for j in joins]),
            "join_finished_early": sum(j["finished_early"] for j in joins),
        })
    return s


def public_records(rounds: list) -> list:
    """The rounds for the JSON file: every request's numbers, without the per-token times."""
    out = []
    for rnd in rounds:
        out.append({"wall_s": rnd["wall_s"], "errors": rnd["errors"], "status": rnd["status"],
                    "summary": round_summary(rnd, rnd.get("scenario", "")),
                    "requests": [{k: v for k, v in r.items() if k != "_times"} for r in rnd["records"]]})
    return out


# ---- tables

def _f(v, f="%.1f"):
    return fmt(v, f)


def render_level(scenario: str, n: int, s: dict, rounds: list, args_info: dict) -> str:
    extra = ""
    if scenario in ("shared-prefix", "agent", "join"):
        extra = ", prefix ~%d tokens" % args_info["prefix_tokens"]
    lines = ["", "== %s, %d client(s): %d round(s), max_tokens %d%s ==" % (scenario, n, len(rounds), args_info["max_tokens"], extra),
             "%-34s %12s   %s" % ("metric", "value", "range / note")]
    lines.append("%-34s %12s   %s - %s" % ("aggregate tok/s (all streams)", _f(s["agg_tps"]), _f(s["agg_tps_min"]), _f(s["agg_tps_max"])))
    lines.append("%-34s %12s   min %s  (%s)" % ("per-stream decode tok/s (median)", _f(s["stream_tps_median"]),
                                                _f(s["stream_tps_min"]), s["stream_tps_source"] + " timings"))
    lines.append("%-34s %12s   p95 %s" % ("TTFT p50 (s)", _f(s["ttft_p50"], "%.2f"), _f(s["ttft_p95"], "%.2f")))
    lines.append("%-34s %12s" % ("max inter-token gap (s)", _f(s["max_gap_s"], "%.2f")))
    lines.append("%-34s %12s   %s of the prompt tokens" % ("cached prompt tokens (mean)", _f(s["cached_mean"], "%.0f"),
                                                          fmt(s["cached_pct"], "pct")))
    lines.append("%-34s %12d   of %d request(s)" % ("errors", s["errors"], s["requests"] + s["errors"]))
    for k, t in (s.get("by_turn") or {}).items():
        lines.append("  turn %s: %d request(s), prompt ~%s tok, TTFT p50 %s s, cached %s"
                     % (k, t["requests"], _f(t["prompt_tokens"], "%.0f"), _f(t["ttft_p50"], "%.2f"), fmt(t["cached_pct"], "pct")))
    if scenario == "join" and "join_ttft_s" in s:
        if s["join_stream_tps"] is None and s["join_stall_gap_s"] is None:
            lines.append("join: no decoding stream overlapped the joiner's prompt read" + (" (level 1 is the joiner alone: its TTFT is the "
                                                                                              "baseline)" if n == 1 else ""))
        lines.append("%-34s %12s   longest, any round %s" % ("join: decoders' longest gap (s)", _f(s["join_stall_gap_s"], "%.2f"),
                                                            _f(s["join_stall_gap_max_s"], "%.2f")))
        lines.append("%-34s %12s   before the join %s" % ("join: decoder tok/s while it reads", _f(s["join_stream_tps"]),
                                                          _f(s["pre_join_stream_tps"])))
        lines.append("%-34s %12s   window %s s" % ("join: joiner TTFT (s)", _f(s["join_ttft_s"], "%.2f"), _f(s["join_window_s"], "%.2f")))
        if s["join_finished_early"]:
            lines.append("note: %d decoder(s) finished before the joiner's first token; raise --decode-tokens." % s["join_finished_early"])
    peak = {}
    for rnd in rounds:
        for k, v in ((rnd.get("status") or {}).get("peak") or {}).items():
            peak[k] = max(v, peak.get(k, v))
    if peak:
        lines.append("server status (peak while running): " + ", ".join("%s=%g" % (k, v) for k, v in sorted(peak.items())[:8]))
    for rnd in rounds:
        for e in rnd["errors"][:3]:
            lines.append("error: client %s turn %s: %s" % (e["client"], e["turn"], e["error"]))
    return "\n".join(lines)


def render_summary(scenario: str, levels: dict) -> str:
    """One row per level: the table to read across client counts."""
    base = (levels.get("1") or {}).get("summary", {}).get("agg_tps")
    lines = ["", "== summary: %s ==" % scenario,
             "%7s %10s %8s %12s %11s %9s %9s %9s %8s" % ("clients", "agg tok/s", "vs 1", "stream med", "stream min", "TTFT p50",
                                                         "TTFT p95", "max gap", "cached")]
    for key, lv in levels.items():
        s = lv["summary"]
        ratio = ("%.2fx" % (s["agg_tps"] / base)) if base and s["agg_tps"] and scenario != "join" else "-"
        lines.append("%7s %10s %8s %12s %11s %9s %9s %9s %8s"
                     % (key, _f(s["agg_tps"]), ratio, _f(s["stream_tps_median"]), _f(s["stream_tps_min"]),
                        _f(s["ttft_p50"], "%.2f"), _f(s["ttft_p95"], "%.2f"), _f(s["max_gap_s"], "%.2f"), fmt(s["cached_pct"], "pct")))
    if scenario == "join":
        lines.append("")
        lines.append("%7s %14s %17s %17s %12s" % ("clients", "decoders' gap", "decoder tok/s", "decoder tok/s", "joiner TTFT"))
        lines.append("%7s %14s %17s %17s %12s" % ("", "(s, median)", "before the join", "while it reads", "(s)"))
        for key, lv in levels.items():
            s = lv["summary"]
            lines.append("%7s %14s %17s %17s %12s" % (key, _f(s.get("join_stall_gap_s"), "%.2f"), _f(s.get("pre_join_stream_tps")),
                                                      _f(s.get("join_stream_tps")), _f(s.get("join_ttft_s"), "%.2f")))
    return "\n".join(lines)


# ---- the run

def prime_prefix(client: Client, model: str, prefix: str) -> dict:
    """One request with the shared prefix, not counted: reads it once so the measured levels see a warm prefix.  Its TTFT is
    the cost of a cold prefix."""
    stream = client.stream_chat(chat_payload_messages(model, shared_prefix_messages(prefix, 0, new_nonce()), PRIME_MAX_TOKENS))
    m = run_metrics(stream)
    return {"ttft_s": m["ttft_s"], "prompt_tokens": m["prompt_tokens"], "cache_n": m["cache_n"], "prompt_tps": m["prompt_tps"]}


def run_concurrency(args, client: Client, model: str, result: dict, save) -> None:
    levels = parse_levels(args.concurrency)
    scenarios = parse_scenarios(args.scenario)
    prefix_tokens = args.prefix_tokens if args.prefix_tokens is not None else DEFAULT_PREFIX_TOKENS
    if prefix_tokens < 1:
        raise BenchError("--prefix-tokens must be at least 1")
    if args.turns < 1 or args.decode_tokens < 1 or args.join_delay < 0 or args.stagger_ms < 0:
        raise BenchError("--turns and --decode-tokens must be at least 1; --join-delay and --stagger-ms cannot be negative")
    info = {"prefix_tokens": prefix_tokens, "max_tokens": args.max_tokens}
    session = new_nonce()[:12]
    prefix = shared_prefix_text(prefix_tokens, session) if any(s in ("shared-prefix", "agent") for s in scenarios) else ""
    result["args"].update({"concurrency": levels, "scenario": scenarios, "prefix_tokens": prefix_tokens, "turns": args.turns,
                           "decode_tokens": args.decode_tokens, "join_delay_s": args.join_delay, "stagger_ms": args.stagger_ms,
                           "prompt": args.prompt, "prime": not args.no_prime})
    result["mode"] = "concurrency"
    result["concurrency_version"] = CONC_VERSION
    result["scenarios"] = {}
    result.pop("cases", None)

    def make_client():
        return Client(args.url, args.api_key, args.timeout)

    say("warm-up (short, max_tokens %d), not counted ..." % WARMUP_MAX_TOKENS)
    w = run_request(client, model, "short", WARMUP_MAX_TOKENS)
    say("  warm-up: %s tokens in %.1f s" % (w.get("completion_tokens"), w.get("wall_s") or 0))
    primed = False
    for sc in scenarios:
        entry = result["scenarios"][sc] = {"levels": {}}
        if sc in ("shared-prefix", "agent") and not args.no_prime and not primed:
            say("priming the shared prefix (~%d tokens, cold), not counted ..." % prefix_tokens)
            result["prime"] = prime_prefix(client, model, prefix)
            primed = True
            say("  cold prefix: prompt %s tok, TTFT %s s" % (fmt(result["prime"]["prompt_tokens"], "%.0f"),
                                                             fmt(result["prime"]["ttft_s"], "%.2f")))
        for n in levels:
            rounds = []
            for i in range(1, args.runs + 1):
                plans = build_plans(sc, n, prefix=prefix, case=args.prompt, prefix_tokens=prefix_tokens, max_tokens=args.max_tokens,
                                    turns=args.turns, decode_tokens=args.decode_tokens, join_delay_s=args.join_delay,
                                    stagger_s=args.stagger_ms / 1000.0)
                rnd = run_round(plans, make_client, model, client)
                rnd["scenario"] = sc
                rounds.append(rnd)
                rs = round_summary(rnd, sc)
                say("  %s x%d round %d/%d: %d tokens in %s s = %s tok/s%s"
                    % (sc, n, i, args.runs, rs["completion_tokens"], _f(rs["wall_s"], "%.1f"), _f(rs["agg_tps"]),
                       "" if not rs["errors"] else ", %d ERROR(S): %s" % (rs["errors"], rnd["errors"][0]["error"][:200])))
                if not rnd["records"]:
                    entry["levels"][str(n)] = {"clients": n, "rounds": public_records(rounds), "summary": summarize_level(sc, n, rounds)}
                    save(True)
                    raise BenchError("every request failed at %s x%d: %s" % (sc, n, rnd["errors"][0]["error"] if rnd["errors"] else "?"))
            summary = summarize_level(sc, n, rounds)
            entry["levels"][str(n)] = {"clients": n, "summary": summary, "rounds": public_records(rounds),
                                       "status": {"peak": {k: v for rnd in rounds for k, v in ((rnd["status"] or {}).get("peak") or {}).items()}}}
            say(render_level(sc, n, summary, rounds, info))
            save(True)
        say(render_summary(sc, entry["levels"]))


def conc_summary_of(entry: dict) -> dict:
    return {k: lv.get("summary") or {} for k, lv in (entry.get("levels") or {}).items()}


# key, label, format, direction (+1 higher is better, -1 lower is better)
CONC_METRICS = [
    ("agg_tps", "aggregate tok/s", "%.1f", 1),
    ("stream_tps_median", "per-stream tok/s (median)", "%.1f", 1),
    ("stream_tps_min", "per-stream tok/s (min)", "%.1f", 1),
    ("ttft_p50", "TTFT p50 (s)", "%.2f", -1),
    ("ttft_p95", "TTFT p95 (s)", "%.2f", -1),
    ("max_gap_s", "max inter-token gap (s)", "%.2f", -1),
    ("cached_pct", "cached prompt share", "pct", 1),
    ("join_stall_gap_s", "join: decoders' longest gap (s)", "%.2f", -1),
    ("join_stream_tps", "join: decoder tok/s while reading", "%.1f", 1),
    ("join_ttft_s", "join: joiner TTFT (s)", "%.2f", -1),
    ("errors", "errors", "%.0f", -1),
]


def verdict_of(d, direction: int, same_threshold: float) -> str:
    if d is None or not direction:
        return ""
    if abs(d) < same_threshold:
        return "same"
    return "B better" if (d > 0) == (direction > 0) else "A better"


def compare_concurrency(a: dict, b: dict, name_a: str, name_b: str, same_threshold: float = 2.0) -> str:
    out = []
    for tag, d, name in (("A", a, name_a), ("B", b, name_b)):
        args = d.get("args") or {}
        out.append("%s = %s  (model %s, %s, runs %s, max_tokens %s, prefix %s, %s)"
                   % (tag, name, d.get("model") or "?", d.get("url") or "?", args.get("runs", "?"), args.get("max_tokens", "?"),
                      args.get("prefix_tokens", "?"), d.get("created") or "?"))
    out.append("concurrency results. diff = (B - A) / A. 'better' follows the metric: higher tok/s, lower TTFT and gap.")
    sa, sb = a.get("scenarios") or {}, b.get("scenarios") or {}
    common = [s for s in sa if s in sb]
    if not common:
        raise BenchError("the two files have no scenario in common (A: %s, B: %s)" % (", ".join(sa) or "-", ", ".join(sb) or "-"))
    w = max(12, min(22, max(len(name_a), len(name_b))))
    skipped = ["%s (%s)" % (s, "A" if s in sa else "B") for s in sorted(set(sa) ^ set(sb))]
    for sc in common:
        la, lb = conc_summary_of(sa[sc]), conc_summary_of(sb[sc])
        levels = [k for k in la if k in lb]
        skipped += ["%s x%s (%s only)" % (sc, k, "A" if k in la else "B") for k in sorted(set(la) ^ set(lb), key=int)]
        for k in levels:
            x, y = la[k], lb[k]
            out.append("")
            out.append("== %s, %s client(s) ==" % (sc, k))
            out.append("%-34s %*s %*s %10s  %s" % ("metric", w, name_a[:w], w, name_b[:w], "diff", ""))
            for key, label, f, direction in CONC_METRICS:
                va, vb = x.get(key), y.get(key)
                if va is None and vb is None:
                    continue
                if key == "cached_pct" and not va and not vb:
                    continue
                if key == "errors" and not va and not vb:
                    continue
                d = pct_diff(va, vb)
                out.append("%-34s %*s %*s %10s  %s" % (label, w, fmt(va, f), w, fmt(vb, f), ("%+.1f%%" % d) if d is not None else "n/a",
                                                       verdict_of(d, direction, same_threshold)))
    if skipped:
        out.append("")
        out.append("not compared (only in one file): " + ", ".join(skipped))
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------------
# main

def parse_cases(text: str) -> list:
    cases = [c.strip() for c in (text or "").split(",") if c.strip()]
    if not cases:
        raise BenchError("--cases is empty")
    for c in cases:
        case_target_tokens(c)
    return cases


def write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def say(*parts) -> None:
    print(*parts)
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bench", description="A/B speed benchmark for an OpenAI-compatible LLM server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="concurrency mode (several clients at once; without --concurrency the single-stream benchmark runs):\n"
               "  python bench.py --concurrency 1,2,3,4 --scenario shared-prefix --prefix-tokens 20000 --runs 3 --out a.json\n"
               "  python bench.py --concurrency 1,2,3 --scenario join --label with-batch-mtp --out b.json\n"
               "  python bench.py --compare a.json b.json          (single-stream and concurrency files both work)\n"
               "Per level: aggregate tok/s = completion tokens of the round / (release of the clients to the last request's\n"
               "end), median over --runs rounds; per-stream tok/s from the server's timings when it sends them; TTFT p50/p95\n"
               "(nearest rank) and cached prompt tokens over all requests; the longest gap between two tokens of a stream.\n"
               "GET /v1/status and /metrics are polled while a round runs (failures ignored): their small numeric fields are\n"
               "saved with the result.")
    ap.add_argument("--url", default=DEFAULT_URL, help="base URL, with or without /v1 (default %(default)s)")
    ap.add_argument("--api-key", default=os.environ.get("LLMSERVE_API_KEY") or os.environ.get("STRATA_API_KEY") or "",
                    help="default: env LLMSERVE_API_KEY or STRATA_API_KEY")
    ap.add_argument("--model", default="", help="model id (default: the first id of GET /v1/models)")
    ap.add_argument("--runs", type=int, default=3, help="measured runs per case (default %(default)s)")
    ap.add_argument("--cases", default=DEFAULT_CASES, help="comma list of short, prompt-4k, prompt-16k, prompt-<N>k (default %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=512, help="max_tokens per request (default %(default)s)")
    ap.add_argument("--label", default="", help="free text saved with the results")
    ap.add_argument("--out", default="", metavar="FILE.json", help="write the raw results here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="print two saved results side by side and exit")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the server per request (default %(default)s)")
    g = ap.add_argument_group("concurrency mode", "--concurrency switches it on; --runs is the number of rounds per level and "
                              "--max-tokens the answer length of each request.")
    g.add_argument("--concurrency", default=None, metavar="1,2,3,4",
                   help="comma list of client counts; each level runs --runs rounds with that many simultaneous clients (threads "
                        "released together by a barrier)")
    g.add_argument("--scenario", default=None, metavar="NAME[,NAME]",
                   help="independent (default): every client a different prompt, run id first, nothing reusable | "
                        "shared-prefix: every client the same long system message (~--prefix-tokens: header + document, like a "
                        "system prompt with tool definitions) and its own user message; the run id is the first line of the user "
                        "message, AFTER the prefix, so the prefix is reusable across clients and rounds and the tails are not; a "
                        "per-run session id at the very top of the prefix makes it cold for a second bench run | "
                        "agent: shared-prefix, then --turns turns per client, each appending the answer and a new user message | "
                        "join: N-1 clients decode --decode-tokens tokens, --join-delay seconds later one more sends a cold prompt "
                        "of ~--prefix-tokens (reports the decoders' longest gap and tok/s while it is read, and its TTFT). A comma "
                        "list runs several scenarios")
    g.add_argument("--prefix-tokens", type=int, default=None,
                   help="size of the shared prefix (shared-prefix, agent) or the joiner's prompt (join) in tokens, ~4 characters "
                        "each (default %d)" % DEFAULT_PREFIX_TOKENS)
    g.add_argument("--prompt", default="short", metavar="CASE",
                   help="independent: the prompt each client sends, short (a different task per client) or prompt-<N>k (a "
                        "different N-thousand-token document per client) (default %(default)s)")
    g.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="agent: turns per client (default %(default)s)")
    g.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS,
                   help="join: max_tokens of the decoding clients; they must still run when the joiner's prompt has been read "
                        "(default %(default)s)")
    g.add_argument("--join-delay", type=float, default=DEFAULT_JOIN_DELAY_S,
                   help="join: seconds after the release before the joiner sends (default %(default)s)")
    g.add_argument("--stagger-ms", type=float, default=0.0, help="client i sends i x this many ms after the release (default %(default)s)")
    g.add_argument("--no-prime", action="store_true",
                   help="shared-prefix / agent: skip the uncounted request that reads the shared prefix first (then the first round "
                        "of the first level pays the cold prefill)")
    return ap


def do_compare(path_a: str, path_b: str) -> int:
    a, b = load_result(path_a), load_result(path_b)
    na, nb = result_name(a, path_a), result_name(b, path_b)
    if na == nb:
        na, nb = na + " (A)", nb + " (B)"
    ca, cb = a.get("mode") == "concurrency", b.get("mode") == "concurrency"
    if ca != cb:
        raise BenchError("cannot compare a concurrency result with a single-stream one (%s is the concurrency one)"
                         % (path_a if ca else path_b))
    say(compare_concurrency(a, b, na, nb) if ca else compare_results(a, b, na, nb))
    return 0


def do_bench(args) -> int:
    if args.runs < 1:
        raise BenchError("--runs must be at least 1")
    if args.max_tokens < 1:
        raise BenchError("--max-tokens must be at least 1")
    if not args.concurrency and (args.scenario is not None or args.prefix_tokens is not None):
        raise BenchError("--scenario and --prefix-tokens belong to the concurrency mode: add --concurrency 1,2,3")
    cases = parse_cases(args.cases)
    client = Client(args.url, args.api_key, args.timeout)
    say("server: %s" % client.base)

    models = None
    try:
        models = client.get_json("/models")
    except BenchError as exc:
        if not args.model:
            raise
        say("GET /v1/models failed (ignored, --model given): %s" % exc)
    ids = [m.get("id") for m in (models or {}).get("data", []) if isinstance(m, dict) and m.get("id")]
    model = args.model or (ids[0] if ids else "")
    if not model:
        raise BenchError("GET /v1/models listed no model; pass --model")
    say("model: %s%s" % (model, "" if not ids else "   (models: %s)" % ", ".join(ids[:5])))

    props = None
    for path in ("/props", "/health"):
        try:
            props = client.get_json(path, root=True, timeout=10)
            say("GET %s: %s" % (path, _short_json(props)))
            break
        except BenchError:
            continue

    result = {
        "version": RESULT_VERSION, "label": args.label, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "url": client.base, "model": model, "props": props,
        "args": {"runs": args.runs, "cases": cases, "max_tokens": args.max_tokens},
        "python": sys.version.split()[0], "cases": {},
    }

    def save(incomplete: bool) -> None:
        if args.out:
            result["incomplete"] = incomplete
            write_json(args.out, result)

    if args.concurrency:
        try:
            run_concurrency(args, client, model, result, save)
        except BenchError:
            save(True)
            if args.out:
                say("partial results saved to %s" % args.out)
            raise
        except KeyboardInterrupt:
            save(True)
            raise
        if args.label:
            say("\nlabel: %s" % args.label)
        save(False)
        if args.out:
            say("results written to %s" % args.out)
        return 0

    try:
        say("warm-up (short, max_tokens %d), not counted ..." % WARMUP_MAX_TOKENS)
        w = run_request(client, model, "short", WARMUP_MAX_TOKENS)
        say("  warm-up: %s tokens in %.1f s" % (w.get("completion_tokens"), w.get("wall_s") or 0))
        for case in cases:
            runs = []
            result["cases"][case] = {"runs": runs}
            for i in range(1, args.runs + 1):
                r = run_request(client, model, case, args.max_tokens)
                runs.append(r)
                say("  %s run %d/%d: prompt %s tok (%s tok/s), decode %s tok (%s tok/s), TTFT %s s"
                    % (case, i, args.runs, fmt(r["prompt_tokens"], "%.0f"), fmt(r["prompt_tps"], "%.1f"),
                       fmt(r["completion_tokens"], "%.0f"), fmt(r["decode_tps"], "%.1f"), fmt(r["ttft_s"], "%.2f")))
    except BenchError:
        save(True)
        if args.out and any(c["runs"] for c in result["cases"].values()):
            say("partial results saved to %s" % args.out)
        raise
    except KeyboardInterrupt:
        save(True)
        raise

    for case in cases:
        say(render_case(case, result["cases"][case]["runs"], args.max_tokens))
    if args.label:
        say("\nlabel: %s" % args.label)
    save(False)
    if args.out:
        say("results written to %s" % args.out)
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):                     # a Windows console may not be UTF-8
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        if args.compare:
            return do_compare(args.compare[0], args.compare[1])
        return do_bench(args)
    except BenchError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
