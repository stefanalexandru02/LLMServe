"""Tests for llmserve.bench against a tiny local fake SSE server (no GPU, no model).

    python -m pytest llmserve/test_bench.py -q
"""
from __future__ import annotations

import http.server
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmserve import bench as B          # noqa: E402


class Fake(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests = []            # parsed POST bodies, in order
        self.auth = []
        self.with_timings = True
        self.completion = None        # completion_tokens to report (default: max_tokens)
        self.error_status = None
        self.draft = True
        self.lock = threading.Lock()
        self.texts = []               # the prompt text of every earlier request (for cache_n)
        self.cache_style = None       # None | "timings" (timings.cache_n) | "usage" (usage.prompt_tokens_details.cached_tokens)
        self.script = None            # callable(body) -> {pieces, piece_delay, prefill}: per-request behaviour
        self.prefilling = 0           # requests in their prompt read: the others' tokens wait (a batch that does not mix)
        self.status = None            # JSON for GET /v1/status
        self.metrics = None           # JSON for GET /metrics
        self.fail_from = None         # index of the first request that gets a 500

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.server_address[1]


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": "fake-model-q2"}, {"id": "other"}]})
        elif self.path == "/props":
            self._json({"n_ctx": 65536, "model": "fake"})
        elif self.path == "/v1/status" and self.server.status is not None:
            self._json(self.server.status)
        elif self.path == "/metrics" and self.server.metrics is not None:
            self._json(self.server.metrics)
        else:
            self._json({"error": "nope"}, 404)

    def do_POST(self):
        srv = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with srv.lock:
            srv.requests.append(body)
            srv.auth.append(self.headers.get("Authorization"))
            idx = len(srv.requests) - 1               # 0 is the warm-up
            text = "\n".join(m["content"] for m in body["messages"])
            cache = max([len(os.path.commonprefix([text, old])) for old in srv.texts] or [0]) // 4
            srv.texts.append(text)
        if srv.error_status or (srv.fail_from is not None and idx >= srv.fail_from):
            self._json({"error": {"message": "boom from fake"}}, srv.error_status or 500)
            return
        script = srv.script(body) if srv.script else {}
        pieces = script.get("pieces") or ["alpha ", "beta ", "gamma"]
        delay = script.get("piece_delay", 0.03)
        prefill = script.get("prefill", 0.05)
        n = srv.completion or (len(pieces) if script else body["max_tokens"])
        ptoks = max(1, len(text) // 4)
        cache = min(cache, ptoks) if srv.cache_style else 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        def wait_for_prefill():
            while srv.prefilling:
                time.sleep(0.005)

        base = {"id": "x", "object": "chat.completion.chunk", "model": body["model"]}
        send(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
        self.wfile.write(b": keep-alive comment\n\n")
        if script.get("prefill_blocks"):
            with srv.lock:
                srv.prefilling += 1
            time.sleep(prefill)
            with srv.lock:
                srv.prefilling -= 1
        else:
            time.sleep(prefill)
        if not script:
            send(dict(base, choices=[{"index": 0, "delta": {"reasoning_content": "hmm"}, "finish_reason": None}]))
        for piece in pieces:
            time.sleep(delay)
            if not script.get("prefill_blocks"):
                wait_for_prefill()
            send(dict(base, choices=[{"index": 0, "delta": {"content": piece}, "finish_reason": None}]))
        send(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "length"}]))
        usage = {"prompt_tokens": ptoks, "completion_tokens": n, "total_tokens": ptoks + n}
        if srv.cache_style == "usage":
            usage["prompt_tokens_details"] = {"cached_tokens": cache}
        final = dict(base, choices=[], usage=usage)
        if srv.with_timings:
            t = {"cache_n": cache if srv.cache_style == "timings" else 0, "prompt_n": ptoks - cache,
                 "prompt_per_second": 1000.0 + 100 * idx if idx else 1.0,
                 "predicted_n": n, "predicted_per_second": 50.0 + 10 * idx if idx else 2.0}
            if srv.draft:
                t.update(draft_n=100, draft_n_accepted=60 + idx)
            final["timings"] = t
        send(final)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture()
def fake():
    srv = Fake()
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()
    srv.server_close()


# ---------------------------------------------------------------------------------------------------------------------

def test_normalize_url():
    assert B.normalize_url("http://h:1/v1") == "http://h:1/v1"
    assert B.normalize_url("http://h:1/v1/") == "http://h:1/v1"
    assert B.normalize_url("http://h:1") == "http://h:1/v1"
    assert B.normalize_url("http://h:1/") == "http://h:1/v1"
    assert B.normalize_url("h:1") == "http://h:1/v1"


def test_documents_are_deterministic_and_sized():
    d1, d2 = B.build_document(4000), B.build_document(4000)
    assert d1 == d2
    assert 16000 <= len(d1) < 17000
    assert 64000 <= len(B.build_document(16000)) < 65000
    blocks = d1.split("\n\n")
    assert len(blocks) > 40 and len(set(blocks)) == len(blocks)   # varied, not one repeated block
    p1, p2 = B.make_prompt("prompt-4k"), B.make_prompt("prompt-4k")
    l1, rest1 = p1.split("\n", 1)
    l2, rest2 = p2.split("\n", 1)
    assert l1.startswith("Run id: ") and l1 != l2            # the nonce is the very first line and differs per run
    assert rest1 == rest2                                    # everything else is identical
    assert p1.endswith("Summarize the document above in detail.")
    assert B.make_prompt("short").startswith("Run id: ")
    with pytest.raises(B.BenchError):
        B.parse_cases("short,bogus")


def test_stream_parsing(fake):
    client = B.Client(fake.url, "", 30)                      # no /v1 given
    payload = B.chat_payload("m", "hello", 20)
    s = client.stream_chat(payload)
    assert s["usage"]["completion_tokens"] == 20
    assert s["timings"]["predicted_n"] == 20
    assert s["finish_reason"] == "length"
    assert s["content_chars"] == len("hmm") + len("alpha beta gamma")
    assert s["ttft_s"] >= 0.04                               # the empty role chunk does not count as the first token
    assert s["last_s"] > s["first_s"]
    sent = fake.requests[0]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["temperature"] == 0 and sent["seed"] == 1
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["max_tokens"] == 20


def test_run_metrics_prefers_server_timings_and_falls_back():
    stream = {"ttft_s": 0.5, "wall_s": 3.0, "first_s": 0.5, "last_s": 2.5, "deltas": 4, "finish_reason": "stop",
              "usage": {"prompt_tokens": 1000, "completion_tokens": 101},
              "timings": {"prompt_per_second": 2000.0, "predicted_per_second": 70.0, "predicted_n": 101, "cache_n": 0,
                          "draft_n": 50, "draft_n_accepted": 40}}
    m = B.run_metrics(stream)
    assert m["prompt_tps"] == 2000.0 and m["prompt_tps_source"] == "server"
    assert m["decode_tps"] == 70.0 and m["decode_tps_source"] == "server"
    assert m["client_decode_tps"] == pytest.approx(100 / 2.0)
    assert m["draft_accept"] == pytest.approx(0.8)
    stream["timings"] = None
    m = B.run_metrics(stream)
    assert m["decode_tps"] == pytest.approx(50.0) and m["decode_tps_source"] == "client"
    assert m["prompt_tps"] == pytest.approx(2000.0) and m["prompt_tps_source"] == "client"
    assert m["draft_accept"] is None


def test_full_run_excludes_warmup_and_reports_medians(fake, tmp_path, capsys):
    out = tmp_path / "res.json"
    rc = B.main(["--url", fake.url + "/v1/", "--runs", "3", "--cases", "short,prompt-4k", "--max-tokens", "40",
                 "--label", "fakeA", "--api-key", "sekret", "--out", str(out)])
    assert rc == 0
    # 1 warm-up + 3 runs of 2 cases
    assert len(fake.requests) == 7
    assert fake.requests[0]["max_tokens"] == 64
    assert all(r["max_tokens"] == 40 for r in fake.requests[1:])
    assert set(fake.auth) == {"Bearer sekret"}
    assert all(r["model"] == "fake-model-q2" for r in fake.requests)         # first id of /v1/models
    nonces = [r["messages"][0]["content"].split("\n", 1)[0] for r in fake.requests]
    assert all(n.startswith("Run id: ") for n in nonces) and len(set(nonces)) == 7

    data = json.loads(out.read_text())
    assert data["label"] == "fakeA" and data["model"] == "fake-model-q2" and data["incomplete"] is False
    assert list(data["cases"]) == ["short", "prompt-4k"]
    short = data["cases"]["short"]["runs"]
    assert len(short) == 3
    # requests 1..3 -> prompt_per_second 1100/1200/1300; the warm-up (1.0) is not in the results
    assert [r["prompt_tps"] for r in short] == [1100.0, 1200.0, 1300.0]
    assert [r["decode_tps"] for r in short] == [60.0, 70.0, 80.0]
    assert short[0]["draft_accept"] == pytest.approx(0.61)
    assert short[0]["completion_tokens"] == 40 and short[0]["prompt_tokens"] > 0
    assert short[0]["ttft_s"] > 0 and short[0]["client_decode_tps"] > 0
    big = data["cases"]["prompt-4k"]["runs"]
    assert big[0]["prompt_tokens"] > 3000
    s = B.summarize(short)
    assert s["prompt_tps"]["median"] == 1200.0 and s["prompt_tps"]["min"] == 1100.0 and s["prompt_tps"]["max"] == 1300.0
    assert s["draft_accept"]["median"] == pytest.approx(0.62)

    text = capsys.readouterr().out
    assert "model: fake-model-q2" in text and "GET /props" in text and "n_ctx" in text
    assert "== short:" in text and "== prompt-4k:" in text
    assert "1200.0" in text and "1100.0 - 1300.0" in text and "62.0%" in text
    assert "far below max_tokens" not in text


def test_early_stop_note_and_no_timings_fallback(fake, tmp_path, capsys):
    fake.completion = 5
    fake.with_timings = False
    out = tmp_path / "r.json"
    assert B.main(["--url", fake.url, "--runs", "2", "--cases", "short", "--max-tokens", "100", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "far below max_tokens 100" in text
    runs = json.loads(out.read_text())["cases"]["short"]["runs"]
    assert all(r["decode_tps_source"] == "client" and r["decode_tps"] > 0 for r in runs)
    assert all(r["draft_accept"] is None for r in runs)


def test_http_error_prints_status_and_body(fake, capsys):
    fake.error_status = 503
    rc = B.main(["--url", fake.url, "--runs", "1", "--cases", "short"])
    err = capsys.readouterr().err
    assert rc != 0 and "HTTP 503" in err and "boom from fake" in err


def test_connection_refused_hint(capsys):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = B.main(["--url", "http://127.0.0.1:%d/v1" % port, "--runs", "1"])
    err = capsys.readouterr().err
    assert rc != 0 and "is the server running?" in err and "host.docker.internal" in err


def _result(label, scale, comp=512, prompt_tokens=4000):
    runs = [{"prompt_tps": 1000.0 * scale * k, "decode_tps": 50.0 * scale * k, "client_decode_tps": 48.0 * scale,
             "ttft_s": 4.0 / scale, "wall_s": 14.0 / scale, "prompt_tokens": prompt_tokens, "completion_tokens": comp,
             "cache_n": 0, "draft_accept": 0.6 * scale if scale <= 1 else None}
            for k in (0.9, 1.0, 1.1)]
    return {"label": label, "model": "m", "url": "http://x/v1", "created": "2026-01-01 00:00:00",
            "args": {"runs": 3, "max_tokens": 512}, "cases": {"prompt-4k": {"runs": runs}, "short": {"runs": runs[:1]}}}


def test_compare_output(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_result("llmserve", 1.0)))
    b.write_text(json.dumps(_result("strata", 1.25)))
    assert B.main(["--compare", str(a), str(b)]) == 0
    text = capsys.readouterr().out
    assert "A = llmserve" in text and "B = strata" in text
    assert "== prompt-4k" in text and "== short" in text
    row = next(l for l in text.splitlines() if l.startswith("prompt tok/s"))
    assert "1000.0" in row and "1250.0" in row and "+25.0%" in row and "B better" in row
    ttft = next(l for l in text.splitlines() if l.startswith("TTFT"))
    assert "4.00" in ttft and "3.20" in ttft and "-20.0%" in ttft and "B better" in ttft       # lower TTFT is better
    acc = next(l for l in text.splitlines() if l.startswith("draft acceptance"))
    assert "60.0%" in acc and "-" in acc
    ptoks = next(l for l in text.splitlines() if l.startswith("prompt tokens"))
    assert "+0.0%" in ptoks and "better" not in ptoks and "same" not in ptoks


def test_compare_warns_on_different_sizes_and_missing_case(tmp_path, capsys):
    ra = _result("a", 1.0)
    rb = _result("b", 1.0, prompt_tokens=5000)
    del rb["cases"]["short"]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(ra))
    b.write_text(json.dumps(rb))
    assert B.main(["--compare", str(a), str(b)]) == 0
    text = capsys.readouterr().out
    assert "different prompt sizes" in text and "not compared (only in one file): short" in text
    assert "A better" not in text and "B better" not in text


def test_compare_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert B.main(["--compare", str(bad), str(bad)]) == 2
    assert "cannot read" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------------------
# concurrency mode

def _stream(first=0.1, last=1.1, comp=101, deltas=101, text="ok", cache=None, t0=0.0, times=None, decode=None):
    """A parsed stream as parse_stream returns it, for tests that need no HTTP."""
    times = times if times is not None else [first + (last - first) * k / max(1, deltas - 1) for k in range(deltas)]
    timings = {"cache_n": cache} if cache is not None else None
    if decode:
        timings = dict(timings or {}, predicted_per_second=decode)
    return {"ttft_s": first, "wall_s": last + 0.05, "first_s": first, "last_s": last, "usage": {"prompt_tokens": 1000, "completion_tokens": comp},
            "timings": timings, "finish_reason": "stop", "content_chars": 10, "deltas": deltas, "t0": t0, "times": times, "text": text}


def _rec(client=0, role="client", turn=1, start=0.0, ttft=0.5, end=2.0, comp=100, tps=50.0, cache=None, prompt=1000, gap=0.1,
         times=None, first=None, last=None):
    return {"client": client, "role": role, "turn": turn, "start_s": start, "first_token_s": first if first is not None else start + ttft,
            "last_token_s": last if last is not None else end, "end_s": end, "ttft_s": ttft, "wall_s": end - start, "decode_tps": tps,
            "decode_tps_source": "server", "client_decode_tps": tps, "prompt_tokens": prompt, "completion_tokens": comp, "cache_n": cache,
            "deltas": comp, "max_gap_s": gap, "_times": times or []}


def test_percentile_is_nearest_rank():
    assert B.percentile([], 50) is None and B.percentile([None], 50) is None
    assert B.percentile([7], 95) == 7
    ten = list(range(1, 11))
    assert B.percentile(ten, 50) == 5 and B.percentile(ten, 95) == 10 and B.percentile(ten, 10) == 1 and B.percentile(ten, 100) == 10
    twenty = list(range(1, 21))
    assert B.percentile(twenty, 95) == 19 and B.percentile(twenty[::-1], 50) == 10      # order does not matter
    assert B.percentile([3, 1, 2], 50) == 2
    assert B.max_gap([0.0, 0.1, 0.5, 0.6]) == pytest.approx(0.4) and B.max_gap([1.0]) is None


def test_levels_and_scenarios_are_validated():
    assert B.parse_levels("1, 2,3,2,4") == [1, 2, 3, 4]
    for bad in ("", "0", "a", "1,-2", "65"):
        with pytest.raises(B.BenchError):
            B.parse_levels(bad)
    assert B.parse_scenarios("join,agent,join") == ["join", "agent"] and B.parse_scenarios(None) == ["independent"]
    with pytest.raises(B.BenchError):
        B.parse_scenarios("shared")


def test_barrier_releases_all_clients_together():
    n = 4
    sent = []

    class Stub:
        def stream_chat(self, payload):
            sent.append(time.perf_counter())
            return _stream(t0=time.perf_counter())

    made = []

    def make_client():
        time.sleep(0.06 * len(made))          # the clients are ready at very different times
        made.append(time.perf_counter())
        return Stub()

    plans = B.build_plans("independent", n)
    rnd = B.run_round(plans, make_client, "m")
    assert len(rnd["records"]) == n and not rnd["errors"]
    assert min(sent) >= max(made) - 0.005                  # nobody sent before the last client was ready
    assert max(sent) - min(sent) < 0.05                    # and they left together
    assert all(r["start_s"] >= 0 for r in rnd["records"])


def test_barrier_stagger_and_a_failing_client_do_not_hang():
    class Stub:
        def stream_chat(self, payload):
            return _stream(t0=time.perf_counter())

    plans = B.build_plans("independent", 3, stagger_s=0.1)
    rnd = B.run_round(plans, lambda: Stub(), "m")
    starts = sorted(r["start_s"] for r in rnd["records"])
    assert starts[1] - starts[0] >= 0.08 and starts[2] - starts[0] >= 0.18

    calls = []

    def make_client():
        calls.append(1)
        if len(calls) == 2:
            raise B.BenchError("cannot connect")
        return Stub()

    t0 = time.perf_counter()
    rnd = B.run_round(B.build_plans("independent", 3), make_client, "m")
    assert time.perf_counter() - t0 < 10                   # the barrier was aborted, not waited out
    assert rnd["errors"] and any("cannot connect" in e["error"] for e in rnd["errors"])


def test_aggregate_math_and_level_summary():
    # round 1: 100 + 50 tokens, the last request ends at t=3 -> 50 tok/s.  round 2: 300 tokens in 2 s -> 150 tok/s
    r1 = {"records": [_rec(0, comp=100, end=2.0, tps=60.0, ttft=0.2, cache=900, gap=0.3),
                      _rec(1, comp=50, end=3.0, tps=40.0, ttft=1.0, cache=0, gap=0.9)], "errors": [], "wall_s": 3.0, "status": None}
    r2 = {"records": [_rec(0, comp=150, end=2.0, tps=80.0, ttft=0.4, cache=900, gap=0.2),
                      _rec(1, comp=150, end=1.5, tps=70.0, ttft=0.6, cache=900, gap=0.1)], "errors": [], "wall_s": 2.0, "status": None}
    s = B.summarize_level("independent", 2, [r1, r2])
    assert s["agg_tps"] == pytest.approx(100.0)            # median of 50 and 150
    assert (s["agg_tps_min"], s["agg_tps_max"]) == (pytest.approx(50.0), pytest.approx(150.0))
    assert s["stream_tps_median"] == pytest.approx(65.0) and s["stream_tps_min"] == 40.0 and s["stream_tps_source"] == "server"
    assert s["ttft_p50"] == 0.4 and s["ttft_p95"] == 1.0     # nearest rank over the four requests
    assert s["max_gap_s"] == 0.9
    assert s["cached_mean"] == pytest.approx(675.0) and s["cached_pct"] == pytest.approx(2700 / 4000)
    assert s["errors"] == 0 and s["requests"] == 4
    r1["errors"] = [{"client": 1, "turn": 1, "error": "x"}]
    assert B.summarize_level("independent", 2, [r1, r2])["errors"] == 1


def test_scenario_independent_prompts_all_differ():
    plans = B.build_plans("independent", 4)
    firsts = [p.messages[0]["content"] for p in plans]
    assert all(c.startswith("Run id: ") for c in firsts)
    bodies = [c.split("\n", 1)[1] for c in firsts]
    assert len(set(bodies)) == 4 and len(set(c.split("\n", 1)[0] for c in firsts)) == 4      # different run ids and different tasks
    long = B.build_plans("independent", 2, case="prompt-4k")
    docs = [p.messages[0]["content"] for p in long]
    assert docs[0].split("\n", 1)[1] != docs[1].split("\n", 1)[1] and all(len(d) > 15000 for d in docs)
    assert all(p.followups == [] and p.role == "client" for p in plans)


def test_scenario_shared_prefix_is_identical_across_clients_and_nonce_comes_after_it():
    prefix = B.shared_prefix_text(500, "sess1")
    assert prefix.startswith("Session: sess1\n") and 1900 < len(prefix) < 2400
    assert B.shared_prefix_text(500, "sess1") == prefix and B.shared_prefix_text(500, "sess2") != prefix
    rounds = [B.build_plans("shared-prefix", 3, prefix=prefix) for _ in range(2)]
    systems = {p.messages[0]["content"] for plans in rounds for p in plans}
    assert systems == {prefix}                              # the same prefix for every client and every round
    users = [p.messages[1]["content"] for plans in rounds for p in plans]
    assert len(set(users)) == 6                             # but no two tails are alike
    assert all(u.startswith("Run id: ") and "Session:" not in u for u in users)
    assert all(p.messages[0]["role"] == "system" and p.messages[1]["role"] == "user" for p in rounds[0])
    assert "Run id:" not in prefix                           # the nonce is not in the shared part


def test_scenario_agent_messages_grow_by_two_per_turn():
    plans = B.build_plans("agent", 2, prefix="PREFIX", turns=3)
    assert [len(p.followups) for p in plans] == [2, 2]
    assert plans[0].followups != plans[1].followups and plans[0].followups[0] != plans[0].followups[1]
    assert B.build_plans("agent", 1, prefix="P", turns=1)[0].followups == []

    # drive one client against a stub and look at the messages it sends each turn
    seen = []

    class Stub:
        def stream_chat(self, payload):
            seen.append(payload["messages"])
            return _stream(text="answer %d" % len(seen), t0=time.perf_counter())

    rnd = B.run_round(plans[:1], lambda: Stub(), "m")
    assert [len(m) for m in seen] == [2, 4, 6]
    assert [m["role"] for m in seen[2]] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert seen[2][2]["content"] == "answer 1" and seen[2][4]["content"] == "answer 2"
    assert seen[2][:4] == seen[1] + [] or seen[2][:4] == seen[1][:4]       # earlier turns are an unchanged prefix of the next
    assert seen[1][:2] == seen[0] and seen[2][:4] == seen[1]
    assert seen[1][3]["content"] == plans[0].followups[0] and seen[2][5]["content"] == plans[0].followups[1]
    assert [r["turn"] for r in rnd["records"]] == [1, 2, 3] and [r["messages"] for r in rnd["records"]] == [2, 4, 6]


def test_scenario_join_plans():
    plans = B.build_plans("join", 4, prefix_tokens=500, decode_tokens=300, max_tokens=64, join_delay_s=2.5)
    assert [p.role for p in plans] == ["decoder", "decoder", "decoder", "joiner"]
    assert [p.max_tokens for p in plans] == [300, 300, 300, 64] and [p.delay_s for p in plans] == [0, 0, 0, 2.5]
    joiner = plans[-1].messages[0]["content"]
    assert joiner.startswith("Run id: ") and 1900 < len(joiner) < 2600                   # a cold prompt of ~500 tokens
    assert len({p.messages[0]["content"] for p in plans}) == 4
    solo = B.build_plans("join", 1, prefix_tokens=100)
    assert [p.role for p in solo] == ["joiner"] and solo[0].delay_s == 0.0


def test_join_metrics_gap_window_and_rates():
    # a decoder emits a token every 0.1 s, except nothing between t=2.0 and t=3.0 (the joiner's prompt is read from 2.0 to 3.2)
    times = [round(0.1 * k, 1) for k in range(1, 21)] + [round(3.0 + 0.1 * k, 1) for k in range(0, 11)]
    dec = _rec(0, "decoder", start=0.0, first=0.1, last=times[-1], end=times[-1], comp=len(times), times=times, gap=1.0)
    dec["deltas"] = len(times)
    joiner = _rec(1, "joiner", start=2.0, first=3.2, ttft=1.2, end=3.6, comp=10)
    m = B.join_metrics([dec, joiner])
    d = m["decoders"][0]
    assert m["window_s"] == pytest.approx(1.2) and m["joiner_ttft_s"] == 1.2 and m["finished_early"] == 0
    assert d["window_gap_s"] == pytest.approx(1.0, abs=1e-6)                 # 2.0 -> 3.0
    assert d["pre_tps"] == pytest.approx(10.0, rel=0.1)                       # one token per 0.1 s before the join
    assert d["window_tps"] == pytest.approx(3 / 1.2, rel=0.01)                # tokens at 3.0, 3.1, 3.2 inside (2.0, 3.2]
    early = _rec(2, "decoder", start=0.0, first=0.1, last=1.5, end=1.5, comp=5, times=[0.1, 0.5, 1.0, 1.5])
    m2 = B.join_metrics([early, joiner])
    assert m2["decoders"] == [] and m2["finished_early"] == 1
    assert B.join_metrics([joiner]) is None and B.join_metrics([dec]) is None

    s = B.summarize_level("join", 2, [{"records": [dec, joiner], "errors": [], "wall_s": 3.6, "status": None, "scenario": "join"}])
    assert s["join_stall_gap_s"] == pytest.approx(1.0) and s["join_ttft_s"] == 1.2 and s["ttft_p50"] == 1.2    # TTFT is the joiner's
    assert s["max_gap_s"] == 1.0 and s["pre_join_stream_tps"] == pytest.approx(10.0, rel=0.1)


def test_cached_tokens_from_timings_or_usage_details():
    assert B.cached_tokens({"timings": {"cache_n": 12}, "usage": {"prompt_tokens_details": {"cached_tokens": 99}}}) == 99
    assert B.cached_tokens({"timings": {"cache_n": 12}, "usage": {}}) == 12 and B.cached_tokens({"timings": {"cache_n": 0}}) == 0
    assert B.cached_tokens({"timings": None, "usage": {"prompt_tokens_details": {"cached_tokens": 99}}}) == 99
    assert B.cached_tokens({"timings": {}, "usage": {}}) is None
    s = _stream(cache=None)
    s["usage"] = {"prompt_tokens": 50, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 40}}
    m = B.run_metrics(s)
    assert m["cache_n"] == 40


def test_flatten_numbers_keeps_small_counters_only():
    doc = {"concurrency": {"serving": 3, "requested": 4}, "model": "x", "uptime_s": 12345, "ok": True,
           "activity": {"in_flight": 2, "requests": 99}, "live": {"slots": [{"a": 1}, {"a": 2}, {"a": 3}]},
           "machine": {"gpu": {"util_pct": 80, "name": "g", "used_mib": 21000, "temp_c": 60}}}
    flat = B.flatten_numbers(doc, "status")
    assert flat["status.concurrency.serving"] == 3 and flat["status.activity.in_flight"] == 2
    assert flat["status.live.slots.len"] == 3 and flat["status.machine.gpu.util_pct"] == 80
    assert "status.uptime_s" not in flat and "status.ok" not in flat and "status.activity.requests" not in flat


def _bench(fake, tmp_path, *argv, out="r.json"):
    path = tmp_path / out
    rc = B.main(["--url", fake.url, *argv, "--out", str(path)])
    return rc, (json.loads(path.read_text()) if path.exists() else None)


def test_shared_prefix_run_end_to_end(fake, tmp_path, capsys):
    fake.cache_style = "timings"
    rc, data = _bench(fake, tmp_path, "--concurrency", "1,3", "--scenario", "shared-prefix", "--prefix-tokens", "400", "--runs", "2",
                      "--max-tokens", "20", "--label", "fakeA")
    assert rc == 0
    # warm-up + prime + (1 + 3) clients x 2 rounds
    assert len(fake.requests) == 2 + 8
    shared = [r for r in fake.requests[2:]]
    assert len({r["messages"][0]["content"] for r in shared}) == 1 and shared[0]["messages"][0]["role"] == "system"
    assert fake.requests[1]["messages"][0]["content"] == shared[0]["messages"][0]["content"]       # the prime used the same prefix
    assert len({r["messages"][1]["content"] for r in shared}) == 8
    assert all(r["max_tokens"] == 20 for r in shared) and all(r["stream_options"] == {"include_usage": True} for r in shared)

    assert data["mode"] == "concurrency" and data["label"] == "fakeA" and data["incomplete"] is False
    assert data["args"]["concurrency"] == [1, 3] and data["args"]["scenario"] == ["shared-prefix"] and data["args"]["prefix_tokens"] == 400
    assert data["prime"]["prompt_tokens"] > 300
    lv = data["scenarios"]["shared-prefix"]["levels"]
    assert list(lv) == ["1", "3"] and len(lv["3"]["rounds"]) == 2 and len(lv["3"]["rounds"][0]["requests"]) == 3
    assert "_times" not in lv["3"]["rounds"][0]["requests"][0]
    s1, s3 = lv["1"]["summary"], lv["3"]["summary"]
    assert s3["requests"] == 6 and s3["errors"] == 0 and s3["agg_tps"] > 0 and s3["ttft_p50"] > 0 and s3["ttft_p95"] >= s3["ttft_p50"]
    assert s3["cached_pct"] > 0.8 and s3["cached_mean"] > 300         # the primed prefix is reused by every client
    assert s3["max_gap_s"] >= 0.02 and s3["stream_tps_median"] > 0
    # 20 tokens per request: aggregate = tokens / wall of the round, so three clients at once beat one
    assert s3["agg_tps"] > s1["agg_tps"] * 1.5

    text = capsys.readouterr().out
    assert "== shared-prefix, 3 client(s): 2 round(s)" in text and "== summary: shared-prefix ==" in text
    assert "aggregate tok/s" in text and "TTFT p50" in text and "priming the shared prefix" in text
    assert "clients  agg tok/s" in text.replace("clients ", "clients  ", 1) or "agg tok/s" in text


def test_no_prime_makes_the_first_request_cold(fake, tmp_path):
    fake.cache_style = "usage"                                # the other way a server reports reuse
    rc, data = _bench(fake, tmp_path, "--concurrency", "1", "--scenario", "shared-prefix", "--prefix-tokens", "300", "--runs", "2",
                      "--max-tokens", "10", "--no-prime")
    assert rc == 0 and len(fake.requests) == 3 and "prime" not in data
    reqs = data["scenarios"]["shared-prefix"]["levels"]["1"]["rounds"]
    first, second = reqs[0]["requests"][0], reqs[1]["requests"][0]
    assert first["cache_n"] < 20 and second["cache_n"] > 0.7 * second["prompt_tokens"]


def test_agent_run_end_to_end(fake, tmp_path):
    fake.cache_style = "timings"
    rc, data = _bench(fake, tmp_path, "--concurrency", "1,2", "--scenario", "agent", "--prefix-tokens", "300", "--turns", "3",
                      "--runs", "1", "--max-tokens", "10", "--no-prime")
    assert rc == 0 and len(fake.requests) == 1 + 3 + 2 * 3
    solo = [r for r in fake.requests[1:4]]
    assert [len(r["messages"]) for r in solo] == [2, 4, 6]
    assert sorted(len(r["messages"]) for r in fake.requests[4:]) == [2, 2, 4, 4, 6, 6]
    assert solo[2]["messages"][2] == {"role": "assistant", "content": "alpha beta gamma"}
    s = data["scenarios"]["agent"]["levels"]["1"]["summary"]
    assert s["requests"] == 3 and set(s["by_turn"]) == {"1", "2", "3"}
    t1, t2, t3 = (s["by_turn"][k] for k in "123")
    assert t1["cached_pct"] < 0.2 and t2["cached_pct"] > 0.8 and t3["cached_pct"] > 0.8        # turn 1 cold, the follow-ups reuse it
    assert t3["prompt_tokens"] > t2["prompt_tokens"] > t1["prompt_tokens"]
    assert data["scenarios"]["agent"]["levels"]["2"]["summary"]["requests"] == 6


def test_join_run_measures_the_decode_stall(fake, tmp_path, capsys):
    def script(body):
        content = body["messages"][0]["content"]
        if len(content) > 2000:                                # the joiner: a long prompt read that stops the others
            return {"pieces": ["j "] * 3, "piece_delay": 0.01, "prefill": 0.6, "prefill_blocks": True}
        return {"pieces": ["w "] * 120, "piece_delay": 0.02, "prefill": 0.0}

    fake.script = script
    rc, data = _bench(fake, tmp_path, "--concurrency", "1,3", "--scenario", "join", "--prefix-tokens", "600", "--decode-tokens", "120",
                      "--join-delay", "0.5", "--runs", "1", "--max-tokens", "3")
    assert rc == 0
    lv = data["scenarios"]["join"]["levels"]
    solo, s = lv["1"]["summary"], lv["3"]["summary"]
    assert solo["join_ttft_s"] >= 0.55 and solo.get("join_stall_gap_s") is None
    assert s["join_ttft_s"] >= 0.55 and s["join_window_s"] >= 0.55
    assert s["join_stall_gap_s"] >= 0.45 and s["max_gap_s"] >= 0.45                       # the decoders stood still while it was read
    assert s["pre_join_stream_tps"] > 20 and s["join_stream_tps"] < 0.5 * s["pre_join_stream_tps"]
    assert s["ttft_p50"] == s["join_ttft_s"]
    text = capsys.readouterr().out
    assert "decoders' longest gap" in text and "joiner TTFT" in text and "no decoding stream overlapped" in text


def test_status_is_sampled_while_running_and_failures_are_ignored(fake, tmp_path):
    fake.script = lambda body: {"pieces": ["w "] * 25, "piece_delay": 0.03, "prefill": 0.0}
    fake.status = {"concurrency": {"serving": 3, "requested": 3}, "model": "x", "uptime_s": 5}
    fake.metrics = {"live": {"slots": [{"a": 1}, {"a": 2}, {"a": 3}]}}
    rc, data = _bench(fake, tmp_path, "--concurrency", "2", "--runs", "1", "--max-tokens", "25")
    assert rc == 0
    lvl = data["scenarios"]["independent"]["levels"]["2"]
    peak = lvl["status"]["peak"]
    assert peak["status.concurrency.serving"] == 3 and peak["metrics.live.slots.len"] == 3
    assert lvl["rounds"][0]["status"]["samples"] >= 1
    fake.status = fake.metrics = None                          # both endpoints answer 404: nothing breaks, nothing is saved
    rc, data = _bench(fake, tmp_path, "--concurrency", "2", "--runs", "1", "--max-tokens", "25", out="r2.json")
    assert rc == 0 and data["scenarios"]["independent"]["levels"]["2"]["status"]["peak"] == {}


def test_every_request_failing_stops_with_a_message_and_saves_partial(fake, tmp_path, capsys):
    fake.fail_from = 1                                         # the warm-up works, the measured requests get a 500
    rc, data = _bench(fake, tmp_path, "--concurrency", "2", "--runs", "1")
    assert rc == 2 and "every request failed" in capsys.readouterr().err
    assert data["incomplete"] is True


def test_scenario_without_concurrency_is_refused(fake, capsys):
    assert B.main(["--url", fake.url, "--scenario", "join"]) == 2
    assert "--concurrency" in capsys.readouterr().err
    assert B.main(["--url", fake.url, "--concurrency", "0"]) == 2


def test_single_stream_mode_is_unchanged_without_the_new_flags(fake, tmp_path):
    rc, data = _bench(fake, tmp_path, "--runs", "1", "--cases", "short", "--max-tokens", "20")
    assert rc == 0 and "mode" not in data and list(data["cases"]) == ["short"] and "scenarios" not in data


def _conc_result(label, scale, scenarios=("shared-prefix",), levels=(1, 2)):
    out = {}
    for sc in scenarios:
        lv = {}
        for n in levels:
            lv[str(n)] = {"clients": n, "summary": {
                "agg_tps": 100.0 * scale * n ** 0.5, "stream_tps_median": 90.0 * scale, "stream_tps_min": 80.0 * scale,
                "ttft_p50": 2.0 / scale, "ttft_p95": 4.0 / scale, "max_gap_s": 0.5 / scale, "cached_pct": 0.9, "errors": 0,
                "join_stall_gap_s": (1.5 / scale) if sc == "join" else None}}
        out[sc] = {"levels": lv}
    return {"mode": "concurrency", "label": label, "model": "m", "url": "http://x/v1", "created": "2026-01-01 00:00:00",
            "args": {"runs": 3, "max_tokens": 512, "prefix_tokens": 20000}, "scenarios": out}


def test_compare_concurrency_results_side_by_side(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_conc_result("armA", 1.0, ("shared-prefix", "join"))))
    b.write_text(json.dumps(_conc_result("armB", 1.25, ("shared-prefix", "join", "agent"), levels=(1, 2, 3))))
    assert B.main(["--compare", str(a), str(b)]) == 0
    text = capsys.readouterr().out
    assert "A = armA" in text and "B = armB" in text
    for head in ("== shared-prefix, 1 client(s) ==", "== shared-prefix, 2 client(s) ==", "== join, 2 client(s) =="):
        assert head in text
    assert "== shared-prefix, 3 client(s)" not in text and "not compared (only in one file)" in text and "agent (B)" in text
    sec = text.split("== shared-prefix, 2 client(s) ==")[1].split("==")[0]
    agg = next(l for l in sec.splitlines() if l.startswith("aggregate tok/s"))
    assert "141.4" in agg and "176.8" in agg and "+25.0%" in agg and "B better" in agg
    ttft = next(l for l in sec.splitlines() if l.startswith("TTFT p50"))
    assert "2.00" in ttft and "1.60" in ttft and "-20.0%" in ttft and "B better" in ttft
    gap = next(l for l in text.split("== join, 2 client(s) ==")[1].splitlines() if l.startswith("join: decoders"))
    assert "1.50" in gap and "1.20" in gap and "B better" in gap
    cached = next(l for l in sec.splitlines() if l.startswith("cached prompt share"))
    assert "90.0%" in cached and "same" in cached


def test_compare_refuses_to_mix_modes_and_runs_on_real_files(fake, tmp_path, capsys):
    fake.cache_style = "timings"
    for label in ("one", "two"):
        assert B.main(["--url", fake.url, "--concurrency", "1,2", "--scenario", "independent", "--runs", "1", "--max-tokens", "10",
                       "--label", label, "--out", str(tmp_path / (label + ".json"))]) == 0
    capsys.readouterr()
    assert B.main(["--compare", str(tmp_path / "one.json"), str(tmp_path / "two.json")]) == 0
    text = capsys.readouterr().out
    assert "A = one" in text and "B = two" in text and "== independent, 2 client(s) ==" in text and "aggregate tok/s" in text
    single = tmp_path / "single.json"
    single.write_text(json.dumps(_result("s", 1.0)))
    assert B.main(["--compare", str(single), str(tmp_path / "one.json")]) == 2
    assert "concurrency" in capsys.readouterr().err
