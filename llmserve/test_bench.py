"""Tests for llmserve.bench against a tiny local fake SSE server (no GPU, no model).

    python -m pytest llmserve/test_bench.py -q
"""
from __future__ import annotations

import http.server
import json
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
        else:
            self._json({"error": "nope"}, 404)

    def do_POST(self):
        srv = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        srv.requests.append(body)
        srv.auth.append(self.headers.get("Authorization"))
        if srv.error_status:
            self._json({"error": {"message": "boom from fake"}}, srv.error_status)
            return
        idx = len(srv.requests) - 1               # 0 is the warm-up
        n = srv.completion or body["max_tokens"]
        prompt = body["messages"][0]["content"]
        ptoks = max(1, len(prompt) // 4)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        base = {"id": "x", "object": "chat.completion.chunk", "model": body["model"]}
        send(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
        self.wfile.write(b": keep-alive comment\n\n")
        time.sleep(0.05)
        send(dict(base, choices=[{"index": 0, "delta": {"reasoning_content": "hmm"}, "finish_reason": None}]))
        for piece in ("alpha ", "beta ", "gamma"):
            time.sleep(0.03)
            send(dict(base, choices=[{"index": 0, "delta": {"content": piece}, "finish_reason": None}]))
        send(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "length"}]))
        final = dict(base, choices=[], usage={"prompt_tokens": ptoks, "completion_tokens": n, "total_tokens": ptoks + n})
        if srv.with_timings:
            t = {"cache_n": 0, "prompt_n": ptoks, "prompt_per_second": 1000.0 + 100 * idx if idx else 1.0,
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
