"""serve/test_phases.py - the server's own per-request phases: template render, tokenizer, queue wait and the time from the
request's arrival to the first token handed to the client, in the response's `timings` (server_*), in the history rows and in
one log line per request.  Against the mock engine with a clock (no GPU, no pack)."""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, phases_line, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class SlowClocked(MockEngine):
    """The mock engine with the engine's clock, whose first token comes `first_s` after the request is sent."""
    first_s = 0.15

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        time.sleep(self.first_s)
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 20.0 * n,
                         "finish": "stop", "reused": 0, "hits": 9, "lookups": 10}


class Phases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = SlowClocked(tok, "</think>\n\nanswer", max_context=8192)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body, raw=False):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        return data if raw else json.loads(data)

    def chat(self, text="hello there", **extra):
        return self.post("/v1/chat/completions",
                         {"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": text}], **extra})

    def test_the_response_timings_carry_the_server_phases_and_keep_the_old_keys(self):
        b = self.chat()
        t = b["timings"]
        for old in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
                    "predicted_per_second"):
            self.assertIn(old, t)
        self.assertEqual(t["prompt_ms"], 40.0)
        for key in ("server_render_ms", "server_encode_ms", "server_queue_ms", "server_ttft_ms"):
            self.assertIsInstance(t[key], float, key)
            self.assertGreaterEqual(t[key], 0.0)
        self.assertEqual(t["server_encode_tokens"], b["usage"]["prompt_tokens"])
        self.assertGreater(t["server_prompt_chars"], len("hello there"))
        self.assertGreaterEqual(t["server_encode_reused_chars"], 0)        # the template head other prompts share
        # the first token comes 150 ms after the request was handed to the engine
        self.assertGreaterEqual(t["server_ttft_ms"], 150.0)
        self.assertGreaterEqual(t["server_ttft_ms"], t["server_render_ms"] + t["server_encode_ms"] + t["server_queue_ms"])
        self.assertLess(t["server_ttft_ms"], 5000.0)

    def test_the_stream_ends_with_the_same_fields(self):
        raw = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 20, "stream": True,
                                                 "messages": [{"role": "user", "content": "hi"}]}, raw=True)
        chunks = [json.loads(ln[6:]) for ln in raw.decode().splitlines() if ln.startswith("data: {")]
        t = chunks[-1]["timings"]
        self.assertIn("server_ttft_ms", t)
        self.assertIn("server_encode_ms", t)
        self.assertIn("predicted_per_second", t)

    def test_a_conversation_that_goes_on_reuses_its_encoded_start(self):
        base = [{"role": "user", "content": "first question " * 30}]
        first = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 5, "messages": base})["timings"]
        more = base + [{"role": "assistant", "content": "an answer"}, {"role": "user", "content": "and then?"}]
        second = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 5, "messages": more})["timings"]
        self.assertLess(first["server_encode_reused_chars"], 100)
        self.assertGreater(second["server_encode_reused_chars"], 100)
        self.assertGreater(second["server_encode_tokens"], first["server_encode_tokens"])

    def test_the_wait_for_the_engines_turn_is_the_queue_time(self):
        out = []
        slow = threading.Thread(target=lambda: out.append(self.chat("one")), daemon=True)
        slow.start()
        time.sleep(0.05)                                   # the first holds the turn for ~150 ms
        second = self.chat("two")["timings"]
        slow.join(10)
        self.assertGreaterEqual(second["server_queue_ms"], 60.0)
        self.assertLessEqual(out[0]["timings"]["server_queue_ms"], 60.0)

    def test_one_compact_log_line_per_request(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.chat("log me")
        lines = [ln for ln in buf.getvalue().splitlines() if "server phases:" in ln]
        self.assertEqual(len(lines), 1, buf.getvalue())
        self.assertRegex(lines[0], r"^\[strata\] request [0-9a-f]{12} server phases: render \d+ ms, encode \d+ ms "
                                   r"\(\d+ tokens from \d+ chars(, \d+ chars reused|, [\d,]+ tok/s)?\), queue \d+ ms, engine prompt 40 ms, "
                                   r"first token \d+ ms after the request arrived \(other \d+ ms\)$")

    def test_the_history_rows_carry_them_too(self):
        self.chat("history")
        row = list(self.svc.history)[-1]
        self.assertIn("server_ttft_ms", row)
        self.assertIn("server_encode_tokens", row)
        self.assertEqual(row["prompt_ms"], 40.0)

    def test_the_anthropic_route_logs_them_as_well(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.post("/v1/messages", {"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(len([ln for ln in buf.getvalue().splitlines() if "server phases:" in ln]), 1)

    def test_a_direct_run_without_a_request_notes_nothing_and_does_not_fail(self):
        svc = Service(SlowClocked(ByteTokenizer(), "ok", max_context=4096), ByteTokenizer(),
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.engine.first_s = 0.0
        ids, thinking, max_new = svc.prepare([{"role": "user", "content": "x"}], None, {}, 5)
        done = [x for k, x in svc.run(ids, thinking, None, max_new, {}, threading.Event()) if k == "done"][0]
        self.assertNotIn("server_render_ms", done["timings"])         # phase_begin was never called: nothing to report
        self.assertIn("server_ttft_ms", done["timings"])              # ... but the run's own clock gives the first token

    def test_phases_line_pieces(self):
        self.assertIsNone(phases_line("abc", None, None))
        line = phases_line("abc", {"t0": 0.0, "render_ms": 12.4, "encode_ms": 340.0, "encode_tokens": 17000,
                                   "prompt_chars": 52000, "encode_reused_chars": 40000, "queue_ms": 0.0,
                                   "ttft_ms": 4401.0}, 3812.0)
        self.assertEqual(line, "[strata] request abc server phases: render 12 ms, encode 340 ms (17000 tokens from 52000 "
                               "chars, 40000 chars reused), queue 0 ms, engine prompt 3812 ms, first token "
                               "4401 ms after the request arrived (other 237 ms)")
        self.assertNotIn("first token", phases_line("abc", {"render_ms": 1.0}, None))


if __name__ == "__main__":
    unittest.main()
