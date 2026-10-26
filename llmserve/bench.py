#!/usr/bin/env python3
"""A/B speed benchmark for an OpenAI-compatible chat server (LLMServe, original Strata, llama.cpp ...).

Standalone and stdlib-only (Python 3.9+): it imports nothing from this repository, so the same file runs on Windows
(`py bench.py`) and inside the container (`python -m llmserve.bench`).

    python -m llmserve.bench --label llmserve --out /data/bench-llmserve.json     # measure one server
    py bench.py --url http://127.0.0.1:8095/v1 --label strata --out bench-strata.json
    python bench.py --compare bench-llmserve.json bench-strata.json               # side by side, % difference

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
import os
import random
import re
import socket
import statistics
import sys
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


def build_document(target_tokens: int) -> str:
    """A deterministic document of about `target_tokens` tokens (1 token ~ 4 characters): prose and code-like lines,
    varied by a seeded generator so it is not one repeated sentence. The same text on every machine and every call."""
    rng = random.Random(DOC_SEED)
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

    Returns {ttft_s, wall_s, first_s, last_s, usage, timings, finish_reason, content_chars, deltas}. Times are seconds since
    `t0`. The first/last token times are those of the first and last delta with text (content or reasoning)."""
    first = last = None
    usage = timings = finish = None
    chars = deltas = 0
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
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {"ttft_s": first, "wall_s": clock() - t0, "first_s": first, "last_s": last, "usage": usage, "timings": timings,
            "finish_reason": finish, "content_chars": chars, "deltas": deltas}


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


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
        "cache_n": timings.get("cache_n"),
        "draft_n": draft_n, "draft_n_accepted": draft_ok,
        "draft_accept": (draft_ok / draft_n) if draft_n and draft_ok is not None else None,
        "finish_reason": stream.get("finish_reason"),
        "timings": stream.get("timings"), "usage": stream.get("usage"),
    }


def chat_payload(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    if not isinstance(data, dict) or not isinstance(data.get("cases"), dict):
        raise BenchError("%s is not a bench result (no 'cases')" % path)
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
    ap = argparse.ArgumentParser(prog="bench", description="A/B speed benchmark for an OpenAI-compatible LLM server.")
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
    return ap


def do_compare(path_a: str, path_b: str) -> int:
    a, b = load_result(path_a), load_result(path_b)
    na, nb = result_name(a, path_a), result_name(b, path_b)
    if na == nb:
        na, nb = na + " (A)", nb + " (B)"
    say(compare_results(a, b, na, nb))
    return 0


def do_bench(args) -> int:
    if args.runs < 1:
        raise BenchError("--runs must be at least 1")
    if args.max_tokens < 1:
        raise BenchError("--max-tokens must be at least 1")
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
