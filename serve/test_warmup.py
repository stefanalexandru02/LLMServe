"""serve/test_warmup.py - the start-up warm-up (serve/warmup.py): its settings, its prompts, the requests it sends through
Service (against the mock engine and against a fake engine process speaking the real line protocol), that a failing step
never stops it, and that the HTTP port opens only after it (a real `python -m serve.server` against a fake engine)."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import warmup  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Recording(MockEngine):
    """The mock engine that notes every request it is asked for: (prompt ids, max_new, sampling)."""

    def __init__(self, *a, fail=None, **kw):
        super().__init__(*a, **kw)
        self.calls, self.fail = [], fail or {}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = len(self.calls)
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        if n in self.fail:
            raise self.fail[n]
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def service(engine=None, **kw):
    tok = ByteTokenizer()
    engine = engine or Recording(tok, "OK", max_context=kw.pop("max_context", 40000))
    return Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja")), engine


class Settings(unittest.TestCase):
    def test_defaults_are_full_and_four(self):
        self.assertEqual(warmup.settings({}, {}), ("full", 4))

    def test_the_config_sets_both(self):
        self.assertEqual(warmup.settings({"warmup": "short", "warmup_parks": 2}, {}), ("short", 2))
        self.assertEqual(warmup.settings({"warmup": "off"}, {}), ("off", 4))

    def test_the_environment_overrides_the_config(self):
        env = {"LLMSERVE_WARMUP": "OFF", "LLMSERVE_WARMUP_PARKS": "0"}
        self.assertEqual(warmup.settings({"warmup": "full", "warmup_parks": 4}, env), ("off", 0))
        self.assertEqual(warmup.settings({"warmup": "short"}, {"LLMSERVE_WARMUP": ""}), ("short", 4))   # empty = not set

    def test_booleans_and_synonyms(self):
        self.assertEqual(warmup.parse_mode(True), "full")
        self.assertEqual(warmup.parse_mode(False), "off")
        self.assertEqual(warmup.parse_mode("on"), "full")
        self.assertEqual(warmup.parse_mode("no"), "off")

    def test_a_wrong_value_names_the_setting(self):
        with self.assertRaisesRegex(ValueError, "LLMSERVE_WARMUP"):
            warmup.settings({}, {"LLMSERVE_WARMUP": "fast"})
        with self.assertRaisesRegex(ValueError, 'config "warmup_parks"'):
            warmup.settings({"warmup_parks": -1}, {})
        with self.assertRaises(ValueError):
            warmup.settings({"warmup_parks": True}, {})
        with self.assertRaises(ValueError):
            warmup.settings({"warmup_parks": 65}, {})


class Text(unittest.TestCase):
    def test_the_text_is_deterministic_realistic_and_about_the_size_asked(self):
        tok = ByteTokenizer()
        a, b = warmup.text_of_tokens(tok, 6000, seed=3), warmup.text_of_tokens(tok, 6000, seed=3)
        self.assertEqual(a, b)
        self.assertNotEqual(a, warmup.text_of_tokens(tok, 6000, seed=4))
        self.assertTrue(0.85 * 6000 <= len(tok.encode(a)) <= 1.25 * 6000, len(tok.encode(a)))
        for marker in ("def ", "static int64_t", "export async function", "impl ", "func (", "SELECT ", '"name": "read_file"',
                       "Traceback", "--- a/src", "git diff"):
            self.assertIn(marker, a)
        self.assertNotIn("<|", a)                       # no control token's text, no <tool_call> tag: it is plain text
        self.assertNotIn("<tool_call>", a)

    def test_blocks_do_not_repeat(self):
        gen = warmup.blocks(1)
        seen = [next(gen) for _ in range(200)]
        self.assertGreater(len(set(seen)), 190)


class Run(unittest.TestCase):
    def logged(self, svc, mode="full", parks=4, **kw):
        lines = []
        out = warmup.run(svc, mode, parks, log=lines.append, **kw)
        return out, lines

    def test_full_sends_the_requests_in_order(self):
        svc, eng = service()
        out, lines = self.logged(svc, "full", 4)
        # prefill, decode, 4 parks, short
        self.assertEqual([k for k, _, _ in out["steps"]], ["prefill", "decode"] + ["park"] * 4 + ["short"])
        self.assertTrue(all(ok for _, ok, _ in out["steps"]))
        self.assertEqual(len(eng.calls), 7)
        sizes = [len(c[0]) for c in eng.calls]
        self.assertTrue(14000 <= sizes[0] <= 19000, sizes)           # ~16K tokens
        self.assertEqual(eng.calls[0][1], 1)                         # one new token
        self.assertTrue(900 <= sizes[1] <= 1500, sizes)              # ~1K
        self.assertEqual(eng.calls[1][1], 768)
        for i in range(2, 6):
            self.assertLess(sizes[i], 300)
        self.assertTrue(80 <= sizes[6] <= 250, sizes)                # ~120
        self.assertEqual(eng.calls[6][1], 16)

    def test_checkpoints_are_off_exactly_for_the_one_shot_requests(self):
        svc, eng = service()
        self.logged(svc, "full", 3)
        flags = [c[2].get("strata_checkpoint") for c in eng.calls]
        self.assertEqual(flags, [False, False, None, None, None, False])      # prefill, decode, 3 parks, short
        for ids, _, sampling in eng.calls:
            self.assertEqual(sampling["temperature"], 0)                       # greedy
            self.assertEqual(sampling["presence_penalty"], 0)                  # not the model card's 1.5
        # the last step is the one-shot: it parks conversation 3 when the engine starts it and keeps nothing itself
        self.assertIs(eng.calls[-1][2]["strata_checkpoint"], False)

    def test_thinking_is_off_and_every_prompt_is_one_user_message(self):
        svc, eng = service()
        self.logged(svc, "full", 2)
        tok = ByteTokenizer()
        for ids, _, _ in eng.calls:
            text = tok.decode(ids)
            self.assertEqual(text.count("<|im_start|>"), 2, text[:200])       # user turn + assistant header: no system turn
            self.assertTrue(text.startswith("<|im_start|>user\n"), text[:80])
            self.assertIn("<think>\n\n</think>", text[-40:])                  # thinking off: the template's empty block

    def test_park_conversations_all_differ_from_the_first_tokens(self):
        svc, eng = service()
        self.logged(svc, "short", 4)
        parks = [c[0] for c in eng.calls[1:5]]
        heads = {tuple(p[:60]) for p in parks}
        self.assertEqual(len(heads), 4)
        self.assertEqual(len({tuple(p) for p in parks}), 4)

    def test_short_mode_leaves_out_the_long_requests(self):
        svc, eng = service()
        out, _ = self.logged(svc, "short", 2)
        self.assertEqual([k for k, _, _ in out["steps"]], ["prefill", "park", "park", "short"])
        self.assertTrue(1500 <= len(eng.calls[0][0]) <= 3200, len(eng.calls[0][0]))
        self.assertNotIn(768, [c[1] for c in eng.calls])

    def test_zero_parks_and_off(self):
        svc, eng = service()
        out, _ = self.logged(svc, "full", 0)
        self.assertEqual([k for k, _, _ in out["steps"]], ["prefill", "decode", "short"])
        svc, eng = service()
        out, lines = self.logged(svc, "off", 4)
        self.assertEqual((eng.calls, lines), ([], []))

    def test_parks_are_capped_at_the_conversation_slots(self):
        svc, eng = service()
        eng.spawn = ("strata", ["--conversation-cache-slots", "2"])
        out, _ = self.logged(svc, "full", 9)
        self.assertEqual([k for k, _, _ in out["steps"]].count("park"), 2)

    def test_no_parks_when_the_engine_has_no_conversation_cache(self):
        svc, eng = service()
        eng.info = {"conversation_cache_mib": 0}
        out, lines = self.logged(svc, "full", 4)
        self.assertEqual([k for k, _, _ in out["steps"]], ["prefill", "decode", "short"])
        self.assertTrue(any("conversation cache off" in x for x in lines), lines)

    def test_each_step_is_logged_with_its_duration(self):
        svc, eng = service()
        _, lines = self.logged(svc, "full", 1)
        steps = [ln for ln in lines if ln.startswith("[strata] warm-up ") and "/4 " in ln]
        self.assertEqual(len(steps), 4, lines)
        for ln in steps:
            self.assertRegex(ln, r": \d+\.\d s \(prompt [\d,]+ tokens")
        self.assertRegex(lines[-1], r"warm-up done in \d+\.\d s")
        self.assertIn("the port opens when it is done", lines[0])

    def test_a_failing_step_is_logged_and_the_rest_runs(self):
        svc, eng = service(Recording(ByteTokenizer(), "OK", max_context=40000, fail={1: ValueError("boom")}))
        out, lines = self.logged(svc, "full", 1)
        self.assertEqual([ok for _, ok, _ in out["steps"]], [True, False, True, True])
        self.assertTrue(any("decode" in ln and "failed (ValueError: boom); continuing" in ln for ln in lines), lines)
        self.assertEqual(len(eng.calls), 4)
        self.assertIn("warm-up done", lines[-1])

    def test_a_dead_engine_ends_the_warm_up_but_not_the_server(self):
        svc, eng = service(Recording(ByteTokenizer(), "OK", max_context=40000, fail={0: EngineDied("gone")}))
        out, lines = self.logged(svc, "full", 2)
        self.assertTrue(out["aborted"])
        self.assertEqual(len(eng.calls), 1)
        self.assertTrue(any("the engine stopped" in ln for ln in lines), lines)

    def test_a_small_context_shrinks_or_skips(self):
        svc, eng = service(Recording(ByteTokenizer(), "OK", max_context=4096))
        out, lines = self.logged(svc, "full", 1)
        sizes = [len(c[0]) for c in eng.calls]
        self.assertLessEqual(sizes[0], 4096 - 8 - 1024 + 400)                  # shrunk to fit with room to answer
        svc, eng = service(Recording(ByteTokenizer(), "OK", max_context=1200))
        out, lines = self.logged(svc, "full", 0)
        self.assertEqual([ok for _, ok, _ in out["steps"]], [None, None, True], lines)   # prefill, decode skipped
        self.assertTrue(any("skipped" in ln for ln in lines))

    def test_a_cancel_before_the_start_stops_it(self):
        svc, eng = service()
        cancel = threading.Event()
        cancel.set()
        out, _ = self.logged(svc, "full", 2, cancel=cancel)
        self.assertTrue(out["aborted"])
        self.assertEqual(eng.calls, [])

    def test_the_warm_up_leaves_no_trace_in_the_metrics_or_the_prompt_cache(self):
        svc, eng = service()
        before = (list(svc.history), dict(svc.totals), svc.last_timings)
        self.logged(svc, "full", 2)
        self.assertEqual((list(svc.history), dict(svc.totals), svc.last_timings), before)
        self.assertEqual(svc.totals["requests"], 0)
        self.assertEqual(svc.prompts.entries, [])
        self.assertEqual(svc.latencies.snapshot()["e2e"][2], 0)            # (counts, sum, n)


# ---- the real line protocol: a fake engine process that records what it is sent
FAKE_ENGINE = r'''
import sys, time
log = open(sys.argv[sys.argv.index('--record') + 1], 'a')
delay = float(sys.argv[sys.argv.index('--delay') + 1]) if '--delay' in sys.argv else 0.0
print('READY 40000 stop', flush=True)
for line in sys.stdin:
    log.write(line.split(' ', 3)[0] + ' ' + ' '.join(line.split(' ', 3)[1:3]) + '\n'); log.flush()
    if line.startswith('QUIT'):
        break
    if line.startswith(('GEN', 'BGEN')):
        time.sleep(delay)
        print('T 79\nT 107\nDONE 1 1 0 2 stop', flush=True)
'''


def fake_engine(tmp: Path) -> str:
    py = tmp / "fake_engine.py"
    py.write_text("#!" + sys.executable + "\n" + FAKE_ENGINE)
    py.chmod(0o700)
    return str(py)


@unittest.skipIf(os.name == "nt", "the fake engine is a script with a shebang")
class WireProtocol(unittest.TestCase):
    def test_the_one_shot_requests_send_ckpt_0_and_the_parks_do_not(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Path(d) / "lines.txt"
            eng = StrataEngine(fake_engine(Path(d)), ["--record", str(rec)])
            self.addCleanup(eng.close)
            svc = Service(eng, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
            out = warmup.run(svc, "full", 3, log=lambda *_: None)
            self.assertTrue(all(ok for _, ok, _ in out["steps"]), out)
            gens = [ln.split() for ln in rec.read_text().splitlines() if ln.startswith("GEN")]
            self.assertEqual(len(gens), 6)
            keys = [g[2:] for g in gens]                      # "GEN <max_new> <keys...>": the keys are the 3rd field on
            joined = [ln for ln in rec.read_text().splitlines() if ln.startswith("GEN")]
            self.assertEqual(["ckpt=0" in ln for ln in joined], [True, True, False, False, False, True], joined)


# ---- the whole server: the port stays closed until the warm-up is over
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tiny_tokenizer_dir(tmp: Path) -> Path:
    """A byte-level tokenizer pack (256 byte tokens, specials, one merge): enough for the server to start."""
    sys.path.insert(0, str(ROOT / "tools"))
    import strata_tokenizer as ST
    d = tmp / "tokenizer"
    d.mkdir()
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["th"]      # "th": the one merge's result
    types = [1] * 257
    for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
        tokens.append(t)
        types.append(3)
    for t in ("<think>", "</think>"):
        tokens.append(t)
        types.append(4)
    (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
    (d / "merges.txt").write_text("t h", encoding="utf-8")        # one merge: the reader wants at least a line
    (d / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    (d / "tokenizer.json").write_text(json.dumps({"pre": "qwen35", "special_ids": {}}), encoding="utf-8")
    return d


@unittest.skipIf(os.name == "nt", "signals and a shebang script")
class PortOpensAfterTheWarmUp(unittest.TestCase):
    def start(self, tmp: Path, extra_cfg: dict, env_extra: dict | None = None, delay: float = 0.4):
        port = free_port()
        rec = tmp / "lines.txt"
        cfg = {"exe": fake_engine(tmp), "args": ["--record", str(rec), "--delay", str(delay)], "cwd": str(tmp),
               "tokenizer": str(tiny_tokenizer_dir(tmp)), "log": str(tmp / "engine.log"), **extra_cfg}
        cfg_path = tmp / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        env = {**os.environ, "PYTHONPATH": str(ROOT), **(env_extra or {})}
        for k in ("LLMSERVE_WARMUP", "LLMSERVE_WARMUP_PARKS", "STRATA_API_KEY"):
            if k not in (env_extra or {}):
                env.pop(k, None)
        proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(cfg_path),
                                 "--port", str(port), "--shutdown-timeout", "2"], cwd=str(ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(lambda: (proc.kill(), proc.wait(10), proc.stdout.close()))
        return proc, port, rec

    @staticmethod
    def health(port):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, ConnectionError):
            return None

    @staticmethod
    def stop(proc):
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
        return out

    def test_health_answers_only_after_the_warm_up_and_it_warmed(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proc, port, rec = self.start(tmp, {"warmup": "short", "warmup_parks": 2}, delay=0.3)
            seen_closed = False
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                h = self.health(port)
                if h is not None:
                    break
                seen_closed = True
                self.assertIsNone(proc.poll(), "the server exited")
                time.sleep(0.05)
            self.assertIsNotNone(h, "the server never answered /health")
            self.assertTrue(seen_closed, "the port was open before the warm-up could run")
            self.assertEqual(h["status"], "ok")
            n_before = len([ln for ln in rec.read_text().splitlines() if ln.startswith("GEN")])
            self.assertEqual(n_before, 4, "the warm-up (prefill, 2 parks, short) must be over when /health answers")
            out = self.stop(proc)
            self.assertIn("warm-up (short): 4 requests", out)
            self.assertIn("warm-up done in", out)
            self.assertLess(out.index("warm-up done in"), out.index("ready: http://"))

    def test_off_in_the_config_and_the_environment_override(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proc, port, rec = self.start(tmp, {"warmup": "off"}, delay=0.0)
            for _ in range(600):
                if self.health(port):
                    break
                time.sleep(0.1)
            self.assertFalse(rec.exists() and "GEN" in rec.read_text())
            self.assertNotIn("warm-up", self.stop(proc))
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proc, port, rec = self.start(tmp, {"warmup": "off"}, {"LLMSERVE_WARMUP": "short", "LLMSERVE_WARMUP_PARKS": "0"},
                                         delay=0.0)
            for _ in range(600):
                if self.health(port):
                    break
                time.sleep(0.1)
            self.assertEqual(len([ln for ln in rec.read_text().splitlines() if ln.startswith("GEN")]), 2)   # prefill + short
            self.assertIn("warm-up (short): 2 requests", self.stop(proc))

    def test_a_wrong_setting_stops_the_start_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proc, port, rec = self.start(tmp, {"warmup": "quick"})
            out, _ = proc.communicate(timeout=30)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('config "warmup"', out)
            self.assertFalse(rec.exists())                      # the engine was never started

    def test_sigterm_during_the_warm_up_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proc, port, rec = self.start(tmp, {"warmup": "full"}, delay=1.0)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not (rec.exists() and "GEN" in rec.read_text()):
                time.sleep(0.05)
            self.assertIsNone(self.health(port))                # still warming up
            out = self.stop(proc)
            self.assertIn("stopped", out)
            self.assertNotIn("ready: http://", out)


if __name__ == "__main__":
    unittest.main()
