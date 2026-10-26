"""Tests for tools/tokenizer_bench.py: its corpus and timing logic against a mock tokenizer and a fake clock (no model), the
whole run against a tiny byte-level BPE built here, and - when `tokenizers` is installed - exact id equality of the Python
tokenizer and the Hugging Face one on that tiny vocabulary.

    python -m pytest llmserve/test_tokenizer_bench.py -q
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer               # noqa: E402,F401  (imported at collection: test_prepare's fake app folders hold an empty one)
import tokenizer_bench as TB          # noqa: E402


# ---------------------------------------------------------------------------------------------------- a mock tokenizer
class MockTok:
    """One token per whitespace-separated word; a 'piece cache' that makes the second encode of a word free."""

    def __init__(self):
        self._piece_ids = {}
        self.encoded = []

    def encode(self, text, parse_special=False, plain=()):
        self.encoded.append(len(text))
        out = []
        for w in text.split():
            out.append(self._piece_ids.setdefault(w, len(self._piece_ids)))
        return out

    def decode(self, ids):
        return " ".join("w%d" % i for i in ids)


class Clock:
    """A fake perf_counter: every call advances by the next step of a script, so time_it() measures exactly it."""

    def __init__(self, steps):
        self.steps, self.now = list(steps), 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls % 2 == 0 and self.steps:        # time_it reads the clock twice: start, then end
            self.now += self.steps.pop(0)
        return self.now


# ---------------------------------------------------------------------------------------------------- corpus logic
def tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.cpp").write_text("int b;\nint bb;\n")
    (tmp_path / "src" / "a.py").write_text("print('a')\n")
    (tmp_path / "src" / "notes.bin").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "src" / "skip.dat").write_text("not a source suffix")
    (tmp_path / "src" / "empty.py").write_text("   \n")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "x.py").write_text("cached = 1\n")
    (tmp_path / "serve").mkdir()
    (tmp_path / "serve" / "s.py").write_text("x = 1\ny = 2\nz = 3\n")
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe broken utf-8")
    return tmp_path


def test_read_corpus_sorted_filtered_and_relative(tmp_path):
    base = tree(tmp_path)
    got = TB.read_corpus(["src", "serve", "bad.py"], 10_000, base=base)
    assert [n for n, _ in got] == ["src/a.py", "src/b.cpp", "serve/s.py"]       # sorted, skip.dat/.bin/empty/__pycache__/bad out
    assert got[0][1] == "print('a')\n"
    assert TB.read_corpus(["src", "src"], 10_000, base=base) == TB.read_corpus(["src"], 10_000, base=base)   # no duplicates
    assert [n for n, _ in TB.read_corpus([str(base / "serve" / "s.py")], 100, base=base)] == ["serve/s.py"]  # a file, absolute


def test_read_corpus_stops_at_the_limit_and_cuts_at_a_line_end(tmp_path):
    base = tree(tmp_path)
    got = TB.read_corpus(["src", "serve"], 11 + 8, base=base)                  # a.py is 11 chars; b.cpp has 2 lines of 7+8
    assert [n for n, _ in got] == ["src/a.py", "src/b.cpp"]
    assert got[1][1] == "int b;\n"                                             # cut after the first line, not mid-line
    assert sum(len(t) for _, t in got) <= 19
    assert TB.read_corpus(["src"], 0, base=base) == []


def test_split_halves():
    files = [("a", "x" * 10), ("b", "y" * 10), ("c", "z" * 10), ("d", "w" * 10)]
    first, second = TB.split_halves(files)
    assert [n for n, _ in first] == ["a", "b"] and [n for n, _ in second] == ["c", "d"]
    f2, s2 = TB.split_halves([("only", "line one\nline two\nline three\n")])
    assert f2[0][1] + s2[0][1] == "line one\nline two\nline three\n" and f2[0][1].endswith("\n")
    f3, s3 = TB.split_halves([("a", "1"), ("b", "2")])
    assert [n for n, _ in f3] == ["a"] and [n for n, _ in s3] == ["b"]


def test_repetitive_text_has_the_size_asked():
    t = TB.repetitive_text(1000)
    assert len(t) == 1000 and t.startswith("The quick brown fox")
    assert len(TB.repetitive_text(10)) == 10


# ---------------------------------------------------------------------------------------------------- timing logic
def test_rate_and_projection():
    assert TB.rate(1000, 2.0) == 500.0 and TB.rate(1000, 0) is None
    p = TB.project(1000.0)
    assert p == {32768: 32.768, 131072: 131.072, 262144: 262.144}
    assert TB.project(None) == {32768: None, 131072: None, 262144: None}
    assert TB.fmt_proj(TB.project(10_000.0)) == "32K: 3.3 s  128K: 13.1 s  256K: 26.2 s"
    assert TB.fmt_proj(TB.project(1_000_000.0)) == "32K: 0.03 s  128K: 0.13 s  256K: 0.26 s"


def test_measure_python_cold_warm_and_warm_new_with_a_fake_clock():
    toks = []

    def make():
        t = MockTok()
        toks.append(t)
        return t
    clock = Clock([2.0, 0.5, 1.0])                       # cold 2 s, warm 0.5 s, warm-new 1 s
    text = "alpha beta gamma delta " * 100              # 400 words
    new = "alpha epsilon zeta " * 100                    # 300 words
    m = TB.measure_python("t", make, [text], [new], clock=clock)
    assert len(toks) == 1                                # one fresh tokenizer for the whole measurement
    assert (m["tokens"], m["cold_s"], m["warm_s"]) == (400, 2.0, 0.5)
    assert (m["cold_tps"], m["warm_tps"]) == (200.0, 800.0)
    assert (m["new_tokens"], m["new_tps"]) == (300, 300.0)
    assert m["stable"] is True and m["pieces_cached"] == 6 and m["chars"] == len(text)
    lines = "\n".join(TB.python_lines(m))
    assert "cold" in lines and "200 tokens/s" in lines and "warm-new" in lines and "32K: 163.8 s" in lines


def test_first_difference_and_compare_ids():
    assert TB.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert TB.first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert TB.first_difference([1, 2], [1, 2, 3]) == 2            # one is the start of the other
    c = TB.compare_ids([[1, 2, 3], [4, 5], [7]], [[1, 2, 3], [4, 6], [7]], ["a", "b", "c"], decode=lambda ids: "-".join(map(str, ids)))
    assert (c["texts"], c["equal"], c["n_different"]) == (3, 2, 1)
    d = c["different"][0]
    assert d["name"] == "b" and d["index"] == 1 and d["py"] == [4, 5] and d["hf"] == [4, 6]
    assert d["py_text"] == "4-5" and d["hf_text"] == "4-6"
    many = TB.compare_ids([[1]] * 20, [[2]] * 20, [str(i) for i in range(20)], limit=5)
    assert many["n_different"] == 20 and len(many["different"]) == 5      # all counted, a few shown
    assert TB.compare_ids([[1]], [[1]], ["x"])["n_different"] == 0


def test_measure_incremental_reuses_the_start_of_the_conversation():
    pytest.importorskip("regex")
    import strata_tokenizer as ST
    tok_parts = tiny_parts()
    files = [("f%d.py" % i, ("def f%d(x):\n    return x + %d\n" % (i, i)) * 20) for i in range(6)]
    m = TB.measure_incremental(lambda: ST.Tokenizer(tok_parts["tokens"], tok_parts["merges"], tok_parts["types"], "qwen35", None), files)
    assert m["reused_chars"] > 0.7 * sum(len(t) for _, t in files[:-1])         # the first five messages are not encoded again
    assert m["tokens"] > m["first_tokens"] and m["added_chars"] > 0
    assert TB.measure_incremental(lambda: MockTok(), files[:2]) is None


# ---------------------------------------------------------------------------------------------------- a tiny real BPE
def tiny_parts(corpus: str = "", merges: int = 120) -> dict:
    """Byte-level BPE trained here on `corpus` (or a default text), in the layout the server reads: 256 byte tokens, the
    merged tokens, the Qwen specials.  Enough to run the real Tokenizer and the whole tool without a model."""
    import regex
    import strata_tokenizer as ST
    corpus = corpus or (("def parse_header(buf, offset=0):\n    return buf[offset:offset + 4]  # the header\n"
                         "int main() { return 0; }\n// the quick brown fox jumps over the lazy dog\n") * 30)
    pat = regex.compile(ST.QWEN35_PATTERN)
    words = collections.Counter("".join(ST.BYTE_TO_UNICODE[b] for b in w.encode()) for w in pat.findall(corpus))
    seqs = {w: list(w) for w in words}
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    rules = []
    for _ in range(merges):
        pairs = collections.Counter()
        for w, n in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        (a, b), _n = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        rules.append("%s %s" % (a, b))
        tokens.append(a + b)
        for w in seqs:
            s, out, i = seqs[w], [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
    types = [1] * len(tokens)
    for t, ty in (("<|im_start|>", 3), ("<|im_end|>", 3), ("<|endoftext|>", 3), ("<think>", 4), ("</think>", 4)):
        tokens.append(t)
        types.append(ty)
    return {"tokens": tokens, "merges": rules, "types": types, "pre": "qwen35", "special_ids": None}


def write_pack(directory: Path, parts: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(parts["tokens"])}), encoding="utf-8")
    (directory / "merges.txt").write_text("\n".join(parts["merges"]), encoding="utf-8")
    (directory / "token_type.json").write_text(json.dumps(parts["types"]), encoding="utf-8")
    (directory / "tokenizer.json").write_text(json.dumps({"pre": parts["pre"], "special_ids": {}}), encoding="utf-8")
    return directory


def args_for(tok_dir, corpus, **kw):
    ns = dict(tokenizer=str(tok_dir), corpus=[str(c) for c in corpus], max_chars=200_000, quick=False, no_hf=True, json="")
    ns.update(kw)
    return argparse.Namespace(**ns)


def test_load_parts_reads_the_pack_layout(tmp_path):
    parts = tiny_parts()
    d = write_pack(tmp_path / "tokenizer", parts)
    got = TB.load_parts(d)
    assert got["tokens"] == parts["tokens"] and got["merges"] == parts["merges"] and got["pre"] == "qwen35"
    tok = TB.make_tokenizer(got)
    ids = tok.encode("def parse_header(buf):\n", parse_special=True)
    assert tok.decode(ids) == "def parse_header(buf):\n"
    assert TB.make_tokenizer(got)._piece_ids == {} and tok._piece_ids          # a fresh tokenizer has an empty cache
    with pytest.raises(SystemExit):
        TB.load_parts(tmp_path / "missing")


def test_whole_run_without_hf_and_the_report(tmp_path, capsys):
    pytest.importorskip("regex")
    d = write_pack(tmp_path / "tokenizer", tiny_parts())
    src = tmp_path / "code"
    src.mkdir()
    for i in range(8):
        (src / ("m%d.py" % i)).write_text(("def f%d(x):\n    return x * %d  # <|im_start|> in a comment\n" % (i, i)) * 40)
    out = tmp_path / "numbers.json"
    rc = TB.main(["--tokenizer", str(d), "--corpus", str(src), "--no-hf", "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "(a) repetitive text" in text and "(b) source code of this repository" in text
    assert "cold" in text and "warm" in text and "warm-new" in text
    assert "32K:" in text and "128K:" in text and "256K:" in text
    assert "PromptEncoder, a conversation of" in text
    assert "skipped (--no-hf)" in text
    data = json.loads(out.read_text())
    b = data["python"][1]
    assert b["tokens"] > 1000 and b["cold_tps"] > 0 and b["warm_tps"] > b["cold_tps"] * 0.5
    assert data["hf"] is None


def test_hf_comparison_is_reported_when_it_is_missing(tmp_path, monkeypatch):
    pytest.importorskip("regex")
    d = write_pack(tmp_path / "tokenizer", tiny_parts())
    src = tmp_path / "code"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n" * 200)
    monkeypatch.setattr(TB, "have_hf", lambda: False)
    rep = TB.run(args_for(d, [src], no_hf=False), log=lambda *_: None)
    assert rep["hf"] is None and "pip install tokenizers" in rep["hf_note"]
    assert "pip install tokenizers" in TB.render(rep)


def test_run_with_a_mock_tokenizer_and_a_fake_clock(tmp_path):
    src = tmp_path / "code"
    src.mkdir()
    for i in range(4):
        (src / ("f%d.py" % i)).write_text("word " * 100 + "\n")
    parts = {"tokens": [], "merges": [], "types": [], "pre": "qwen35", "special_ids": None}
    rep = TB.run(args_for("unused", [src]), clock=Clock([1.0] * 20), make_tok=MockTok, parts=parts, log=lambda *_: None)
    a, b = rep["python"]
    assert a["name"].startswith("(a)") and b["name"].startswith("(b)")
    assert a["cold_tps"] == a["tokens"] / 1.0 and b["warm_tps"] == b["tokens"] / 1.0


# ---------------------------------------------------------------------------------------------------- Hugging Face
def test_hugging_face_gives_the_same_ids_as_the_python_tokenizer(tmp_path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("regex")
    import strata_tokenizer as ST
    corpus_files = [p for p in (ROOT / "serve").glob("*.py")][:12] + [ROOT / "README.md"]
    text = "".join(p.read_text(encoding="utf-8") for p in corpus_files)
    parts = tiny_parts(text[:200_000], merges=400)
    py = ST.Tokenizer(parts["tokens"], parts["merges"], parts["types"], "qwen35", None)
    hf = TB.build_hf(parts)
    sample = [p.read_text(encoding="utf-8") for p in corpus_files]
    sample += ["", " ", "\n\n\n", "naïve café — 日本語のテキスト 😀 é x́y", "tabs\t\tand  spaces   \n  \n x",
               "<|im_start|>user\nhello<|im_end|>\n<think>\n\n</think>\n", "a<|im_start|>b", "it's we're I'll they'd 123456 7.89"]
    for t in sample:
        assert hf.encode(t, add_special_tokens=False).ids == py.encode(t, parse_special=True), t[:60]
    m = TB.measure_hf(hf, sample[:5])
    assert m["tokens"] > 0 and m["batch_equal"] and m["cold_tps"] > 0


def test_whole_run_with_hf_reports_speed_and_identity(tmp_path, capsys):
    pytest.importorskip("tokenizers")
    pytest.importorskip("regex")
    d = write_pack(tmp_path / "tokenizer", tiny_parts())
    src = tmp_path / "code"
    src.mkdir()
    for i in range(6):
        (src / ("m%d.py" % i)).write_text(("def f%d(x):\n    return x * %d  # the header\n" % (i, i)) * 40)
    rc = TB.main(["--tokenizer", str(d), "--corpus", str(src)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "Hugging Face tokenizers " in text and "encode_batch" in text and "speed-up over Python" in text
    assert "ids: IDENTICAL on all 7 text(s)" in text


def test_default_tokenizer_dir_follows_the_env_then_the_run_config(tmp_path, monkeypatch):
    monkeypatch.delenv("LLMSERVE_TOKENIZER_DIR", raising=False)
    monkeypatch.setenv("LLMSERVE_DATA_DIR", str(tmp_path))
    assert TB.default_tokenizer_dir() == TB.DEFAULT_TOKENIZER                       # no cfg.json
    (tmp_path / "cfg.json").write_text(json.dumps({"tokenizer": "/data/packs/iq3_s/tokenizer"}))
    assert TB.default_tokenizer_dir() == "/data/packs/iq3_s/tokenizer"
    (tmp_path / "cfg.json").write_text("{broken")
    assert TB.default_tokenizer_dir() == TB.DEFAULT_TOKENIZER
    monkeypatch.setenv("LLMSERVE_TOKENIZER_DIR", "/elsewhere")
    assert TB.default_tokenizer_dir() == "/elsewhere"
