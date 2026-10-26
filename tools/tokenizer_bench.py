#!/usr/bin/env python3
"""tools/tokenizer_bench.py - how fast is the server's tokenizer, and would Hugging Face `tokenizers` (Rust) be worth it?

The server tokenizes every prompt in pure Python (tools/strata_tokenizer.py: the `regex` module for the pre-tokenizer,
a dictionary of already-merged pieces, a BPE merge loop for the rest).  At 100K+ tokens that is seconds the user waits
before the engine sees the first token.  This tool measures it on the model's own tokenizer:

    python tools/tokenizer_bench.py                                   # /data/packs/q2_0/tokenizer, the repo's src/ serve/ ...
    python tools/tokenizer_bench.py --tokenizer /data/packs/q2_0/tokenizer --corpus src serve --max-chars 3000000
    docker compose exec llmserve python tools/tokenizer_bench.py --quick

For each text (a: repetitive text, b: the source files of this repository, a realistic code corpus) it prints tokens/s
  cold       a fresh Tokenizer with an empty piece cache (the first prompt after a start; a new codebase)
  warm       the same Tokenizer encoding the same text again (every piece is in the cache: the upper bound)
  warm-new   (corpus b) the Tokenizer warmed by the first half of the corpus, encoding the second half: what an agent that
             reads new files meets once the common pieces (keywords, indents) are cached
and the encode time those rates give for prompts of 32K, 128K and 256K tokens.  It also times PromptEncoder's incremental
path (a long conversation that grows by one message: the common agent turn).

If the `tokenizers` package is importable (pip install tokenizers) it builds a tokenizers.Tokenizer from the same
vocabulary and merges, with the same pre-tokenizer regex (QWEN35_PATTERN, split "isolated", then byte-level without its own
regex) and the same special tokens, and reports its speed (one thread and encode_batch over all cores) AND whether the ids
are exactly the Python tokenizer's, file by file where they are not.  The server is NOT switched to it by this tool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_TOKENIZER = "/data/packs/q2_0/tokenizer"
DEFAULT_CORPUS = ("src", "serve", "include", "tools")           # relative to the repository root
SUFFIXES = (".cpp", ".cu", ".cuh", ".hpp", ".h", ".c", ".py", ".md", ".cmake", ".txt", ".sh", ".json", ".jinja", ".yml")
SKIP_DIRS = {"__pycache__", ".git", "third_party", "build", "node_modules", "fixtures"}
TARGETS = (32 * 1024, 128 * 1024, 256 * 1024)                  # prompt sizes to project, in tokens
REPETITIVE_UNIT = ("The quick brown fox jumps over the lazy dog. def f(x): return x + 1\n"
                   "    for i in range(10):  # the same line again and again\n")
SPECIAL_TYPES = (3, 4)                                         # GGUF token types: CONTROL, USER_DEFINED


# ------------------------------------------------------------------------------------------------ loading
def load_parts(directory) -> dict:
    """The pack's tokenizer/ directory as the server reads it (serve/server.py main): tokens, merges, token types, the
    pre-tokenizer name and the special ids."""
    d = Path(directory)
    if not (d / "vocab.json").exists():
        raise SystemExit(f"tokenizer_bench: no vocab.json in {d} (pass --tokenizer DIR; in the container: "
                         f"{DEFAULT_TOKENIZER})")
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((d / "token_type.json").read_text())
    cfg = {}
    try:
        cfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {"tokens": tokens, "merges": merges, "types": types, "pre": cfg.get("pre") or "qwen35",
            "special_ids": cfg.get("special_ids") or None}


def make_tokenizer(parts: dict):
    """A fresh strata_tokenizer.Tokenizer (an empty piece cache)."""
    import strata_tokenizer as ST
    return ST.Tokenizer(parts["tokens"], parts["merges"], parts["types"], parts["pre"], parts["special_ids"])


# ------------------------------------------------------------------------------------------------ the corpora
def repetitive_text(chars: int) -> str:
    """The same two lines over and over, `chars` characters of them (a boilerplate-heavy log, a generated file)."""
    return (REPETITIVE_UNIT * (chars // len(REPETITIVE_UNIT) + 1))[:chars]


def read_corpus(roots, max_chars: int, base=ROOT, suffixes=SUFFIXES) -> list:
    """[(relative path, text)] of the source files under `roots` (files or folders, relative to `base` or absolute), in
    sorted order, until `max_chars` characters are collected.  Unreadable and non-UTF-8 files are skipped; a file that
    would pass the limit is cut at a line end."""
    files, seen = [], set()
    for root in roots:
        p = Path(root)
        p = p if p.is_absolute() else Path(base) / p
        if p.is_file():
            cands = [p]
        else:
            cands = []
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                cands.extend(Path(dirpath) / f for f in sorted(filenames) if f.endswith(tuple(suffixes)))
        for f in cands:
            if f not in seen:
                seen.add(f)
                files.append(f)
    out, total = [], 0
    for f in files:
        if total >= max_chars:
            break
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not text.strip():
            continue
        last = total + len(text) > max_chars
        if last:
            cut = text.rfind("\n", 0, max_chars - total)
            text = text[:cut + 1] if cut > 0 else text[:max_chars - total]
        try:
            rel = str(f.relative_to(base))
        except ValueError:
            rel = str(f)
        out.append((rel, text))
        total += len(text)
        if last:
            break
    return out


def split_halves(files: list) -> tuple:
    """The files cut into two lists of about equal characters (for the warm-new measurement); a single file is cut at a line end."""
    total = sum(len(t) for _, t in files)
    if len(files) == 1:
        name, text = files[0]
        cut = text.rfind("\n", 0, len(text) // 2) + 1 or len(text) // 2
        return [(name, text[:cut])], [(name, text[cut:])]
    run, k = 0, 0
    for k, (_, t) in enumerate(files):
        run += len(t)
        if run * 2 >= total:
            break
    k = min(max(k, 0), len(files) - 2) if len(files) > 1 else 0
    return files[:k + 1], files[k + 1:]


# ------------------------------------------------------------------------------------------------ timing
def time_it(fn, clock=time.perf_counter):
    t0 = clock()
    result = fn()
    return clock() - t0, result


def rate(tokens: int, seconds: float):
    return tokens / seconds if seconds and seconds > 0 else None


def project(tps, targets=TARGETS) -> dict:
    """{target tokens: seconds} at `tps` tokens per second (None when there is no rate)."""
    return {t: (t / tps if tps else None) for t in targets}


def measure_python(name: str, make_tok, texts_cold: list, texts_warm_new=None, clock=time.perf_counter) -> dict:
    """Cold / warm (and warm-new) encode rates of the Python tokenizer on `texts_cold` (a list of strings, encoded one
    after the other with parse_special=True, as a prompt is).  `make_tok()` returns a fresh tokenizer."""
    tok = make_tok()
    cold_s, ids = time_it(lambda: [tok.encode(t, parse_special=True) for t in texts_cold], clock)
    n = sum(len(x) for x in ids)
    warm_s, ids2 = time_it(lambda: [tok.encode(t, parse_special=True) for t in texts_cold], clock)
    out = {"name": name, "chars": sum(len(t) for t in texts_cold), "tokens": n, "cold_s": cold_s, "warm_s": warm_s,
           "cold_tps": rate(n, cold_s), "warm_tps": rate(n, warm_s), "stable": ids == ids2}
    if texts_warm_new:
        new_s, ids3 = time_it(lambda: [tok.encode(t, parse_special=True) for t in texts_warm_new], clock)
        m = sum(len(x) for x in ids3)
        out.update(new_tokens=m, new_s=new_s, new_tps=rate(m, new_s))
    out["pieces_cached"] = len(getattr(tok, "_piece_ids", ()))
    return out


def measure_incremental(parts_or_make, files: list, clock=time.perf_counter) -> dict | None:
    """PromptEncoder on a conversation that grows: every file one user message, encoded once; then one more message
    appended and the whole conversation encoded again (what the server does for an agent's next turn).  -> seconds of the
    first (cold) and of the second encode, the tokens and the characters taken from the first."""
    from strata_tokenizer import PromptEncoder
    if len(files) < 3:
        return None
    make_tok = parts_or_make
    tok = make_tok()
    enc = PromptEncoder(tok)

    def convo(n):
        return "".join("<|im_start|>user\n%s<|im_end|>\n" % t for _, t in files[:n])
    base_text, longer = convo(len(files) - 1), convo(len(files))
    first_s, first = time_it(lambda: enc.encode(base_text), clock)
    second_s, second = time_it(lambda: enc.encode(longer), clock)
    return {"tokens": len(second), "first_s": first_s, "second_s": second_s, "reused_chars": enc.last_reused,
            "added_chars": len(longer) - len(base_text), "first_tokens": len(first)}


# ------------------------------------------------------------------------------------------------ Hugging Face tokenizers
def have_hf() -> bool:
    try:
        import tokenizers  # noqa: F401
        return True
    except ImportError:
        return False


def build_hf(parts: dict):
    """A tokenizers.Tokenizer with the same vocabulary, merges, pre-tokenizer regex and special tokens as the Python one.
    None for a family whose pre-tokenizer is a sequence of patterns (DeepSeek's)."""
    import strata_tokenizer as ST
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers, decoders
    if str(parts["pre"]).startswith("deepseek"):
        return None
    vocab = {t: i for i, t in enumerate(parts["tokens"])}
    merges = [tuple(m.split(" ")) for m in parts["merges"] if m]
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(ST.QWEN35_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    types = parts["types"] or []
    specials = [parts["tokens"][i] for i, ty in enumerate(types) if ty in SPECIAL_TYPES]
    if specials:
        tk.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])
    return tk


def measure_hf(tk, texts: list, clock=time.perf_counter) -> dict:
    """tokenizers' rates on `texts`: encode one text after the other (one thread) and encode_batch (all cores), cold
    (first call) and warm (second).  Also returns the ids of the one-thread pass, per text."""
    def one():
        return [tk.encode(t, add_special_tokens=False).ids for t in texts]

    def batch():
        return [e.ids for e in tk.encode_batch(texts, add_special_tokens=False)]
    cold_s, ids = time_it(one, clock)
    n = sum(len(x) for x in ids)
    warm_s, _ = time_it(one, clock)
    bcold_s, bids = time_it(batch, clock)
    bwarm_s, _ = time_it(batch, clock)
    return {"tokens": n, "cold_s": cold_s, "warm_s": warm_s, "cold_tps": rate(n, cold_s), "warm_tps": rate(n, warm_s),
            "batch_cold_s": bcold_s, "batch_warm_s": bwarm_s, "batch_cold_tps": rate(n, bcold_s),
            "batch_warm_tps": rate(n, bwarm_s), "ids": ids, "batch_equal": bids == ids}


def first_difference(a: list, b: list):
    """Index of the first position where the id lists differ (the shorter length when one is a prefix), None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_ids(py_ids: list, hf_ids: list, names: list, decode=None, window: int = 4, limit: int = 8) -> dict:
    """Exact equality of two lists of per-text id lists.  -> {texts, equal, different: [{name, index, py, hf, text}...]}
    with up to `limit` examples; `decode(ids)` (optional) turns the ids around the first difference into text."""
    different, n_diff = [], 0
    for name, a, b in zip(names, py_ids, hf_ids):
        i = first_difference(a, b)
        if i is None:
            continue
        n_diff += 1
        if len(different) < limit:
            lo = max(0, i - window)
            item = {"name": name, "index": i, "py": a[lo:i + window], "hf": b[lo:i + window],
                    "len_py": len(a), "len_hf": len(b)}
            if decode is not None:
                try:
                    item["py_text"], item["hf_text"] = decode(a[lo:i + window]), decode(b[lo:i + window])
                except Exception:  # noqa: BLE001 - the report must not fail on a decode error
                    pass
            different.append(item)
    return {"texts": len(names), "equal": len(names) - n_diff, "different": different, "n_different": n_diff,
            "tokens_py": sum(len(x) for x in py_ids), "tokens_hf": sum(len(x) for x in hf_ids)}


# ------------------------------------------------------------------------------------------------ report
def fmt_rate(tps) -> str:
    return "-" if tps is None else f"{tps:,.0f}"


def fmt_secs(s) -> str:
    return "-" if s is None else f"{s:.2f} s" if s < 1 else f"{s:.1f} s"


def fmt_proj(proj: dict) -> str:
    return "  ".join(f"{t // 1024}K: {fmt_secs(s)}" for t, s in proj.items())


def python_lines(m: dict) -> list:
    lines = [f"  {m['name']}: {m['chars']:,} characters = {m['tokens']:,} tokens "
             f"({m['chars'] / max(1, m['tokens']):.2f} chars/token)"]
    for label, tps in (("cold", m["cold_tps"]), ("warm", m["warm_tps"]), ("warm-new", m.get("new_tps"))):
        if label == "warm-new" and tps is None:
            continue
        lines.append(f"    {label:<9}{fmt_rate(tps):>10} tokens/s   encode time for a prompt of  {fmt_proj(project(tps))}")
    if m.get("pieces_cached") is not None:
        lines.append(f"    ({m['pieces_cached']:,} distinct pre-tokenizer pieces cached after the run)")
    return lines


def render(report: dict) -> str:
    lines = ["tokenizer_bench: " + report["tokenizer"], ""]
    lines.append("Python tokenizer (tools/strata_tokenizer.py, what the server runs)")
    for m in report["python"]:
        lines += python_lines(m)
    inc = report.get("incremental")
    if inc:
        lines += ["", f"  PromptEncoder, a conversation of {inc['first_tokens']:,} tokens that grows by one message of "
                      f"{inc['added_chars']:,} characters:",
                  f"    first turn (cold)   {inc['first_s']:.2f} s    next turn  {inc['second_s'] * 1000:.0f} ms "
                  f"({inc['reused_chars']:,} of {inc['reused_chars'] + inc['added_chars']:,} characters taken from the first)"]
    hf = report.get("hf")
    lines.append("")
    if hf is None:
        lines.append("Hugging Face tokenizers: " + report.get("hf_note", "not compared"))
    else:
        lines.append(f"Hugging Face tokenizers {hf['version']} (same vocabulary, merges, regex, specials)")
        for m in hf["measures"]:
            lines.append(f"  {m['name']}: {m['tokens']:,} tokens")
            for label, tps in (("1 thread, cold", m["cold_tps"]), ("1 thread, warm", m["warm_tps"]),
                               ("encode_batch, cold", m["batch_cold_tps"]), ("encode_batch, warm", m["batch_warm_tps"])):
                lines.append(f"    {label:<19}{fmt_rate(tps):>10} tokens/s   {fmt_proj(project(tps))}")
            py = next((p for p in report["python"] if p["name"] == m["name"]), None)
            if py and py["cold_tps"] and m["cold_tps"]:
                lines.append(f"    speed-up over Python (cold): {m['cold_tps'] / py['cold_tps']:.1f}x with one thread"
                             + (f", {m['batch_cold_tps'] / py['cold_tps']:.1f}x with encode_batch" if m["batch_cold_tps"] else ""))
        cmp = hf["identity"]
        lines.append("")
        if cmp["n_different"] == 0:
            lines.append(f"  ids: IDENTICAL on all {cmp['texts']} text(s), {cmp['tokens_py']:,} tokens")
        else:
            lines.append(f"  ids: {cmp['n_different']} of {cmp['texts']} text(s) DIFFER "
                         f"({cmp['tokens_py']:,} Python tokens, {cmp['tokens_hf']:,} Hugging Face tokens); the first "
                         f"{len(cmp['different'])}:")
            for d in cmp["different"]:
                lines.append(f"    {d['name']}: first difference at token {d['index']} "
                             f"(Python {d['len_py']:,} tokens, HF {d['len_hf']:,})")
                lines.append(f"      python {d['py']}" + (f"  {d['py_text']!r}" if "py_text" in d else ""))
                lines.append(f"      hf     {d['hf']}" + (f"  {d['hf_text']!r}" if "hf_text" in d else ""))
        if not all(m.get("batch_equal", True) for m in hf["measures"]):
            lines.append("  note: encode_batch gave other ids than encode for some text (tokenizers bug or special tokens)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ main
def run(args, clock=time.perf_counter, make_tok=None, parts=None, log=print) -> dict:
    """The whole measurement; returns the report dict (what --json saves).  `make_tok`/`parts` let a test supply them."""
    parts = parts if parts is not None else load_parts(args.tokenizer)
    make = make_tok or (lambda: make_tokenizer(parts))
    max_chars = 250_000 if args.quick else args.max_chars
    files = read_corpus(args.corpus or DEFAULT_CORPUS, max_chars)
    if not files:
        raise SystemExit("tokenizer_bench: the corpus is empty (--corpus FILE_OR_DIR ...)")
    rep_chars = min(max_chars, 600_000)
    report = {"tokenizer": str(args.tokenizer), "python": [], "corpus_files": len(files), "targets": list(TARGETS)}
    log(f"corpus: {len(files)} files, {sum(len(t) for _, t in files):,} characters; "
        f"repetitive text: {rep_chars:,} characters; encoding (a fresh tokenizer per text) ...")
    first, second = split_halves(files)
    texts_b = [t for _, t in first]                 # per file: a prompt's cost is the same, encode_batch has texts to spread
    texts_b_new = [t for _, t in second] if second else None
    report["python"].append(measure_python("(a) repetitive text", make, [repetitive_text(rep_chars)], clock=clock))
    log("  (a) done")
    report["python"].append(measure_python("(b) source code of this repository", make, texts_b, texts_b_new, clock=clock))
    log("  (b) done")
    try:
        report["incremental"] = measure_incremental(make, files, clock)
    except Exception as e:  # noqa: BLE001 - an extra figure; the rest is still worth printing
        log(f"  incremental measurement failed: {type(e).__name__}: {e}")
        report["incremental"] = None
    if args.no_hf:
        report["hf"], report["hf_note"] = None, "skipped (--no-hf)"
    elif not have_hf():
        report["hf"], report["hf_note"] = None, "not installed here (pip install tokenizers), so no speed or identity comparison"
    else:
        import tokenizers
        tk = build_hf(parts)
        if tk is None:
            report["hf"], report["hf_note"] = None, f"not compared: the {parts['pre']} pre-tokenizer is a pattern sequence"
        else:
            log("  Hugging Face tokenizers ...")
            py_tok = make()
            names = [n for n, _ in files]
            texts = [t for _, t in files]
            measures = []
            rep_text = [repetitive_text(rep_chars)]
            m_rep = measure_hf(tk, rep_text, clock)
            m_rep["name"] = "(a) repetitive text"
            ids_rep = m_rep.pop("ids")
            measures.append(m_rep)
            m_src = measure_hf(tk, texts_b, clock)      # the same texts the Python tokenizer was timed on
            m_src["name"] = "(b) source code of this repository"
            hf_b_ids = m_src.pop("ids")
            measures.append(m_src)
            hf_ids = hf_b_ids + [tk.encode(t, add_special_tokens=False).ids for t in (texts_b_new or [])]
            py_ids = [py_tok.encode(t, parse_special=True) for t in texts]
            identity = compare_ids(py_ids + [py_tok.encode(rep_text[0], parse_special=True)], hf_ids + ids_rep,
                                   names + ["(a) repetitive text"], decode=py_tok.decode)
            report["hf"] = {"version": tokenizers.__version__, "measures": measures, "identity": identity}
    return report


def default_tokenizer_dir() -> str:
    """$LLMSERVE_TOKENIZER_DIR, else the "tokenizer" of the run config ($LLMSERVE_DATA_DIR/cfg.json, /data/cfg.json in the
    container: right for whatever quant is installed), else the Q2_0 pack's."""
    if os.environ.get("LLMSERVE_TOKENIZER_DIR"):
        return os.environ["LLMSERVE_TOKENIZER_DIR"]
    try:
        cfg = json.loads((Path(os.environ.get("LLMSERVE_DATA_DIR", "/data")) / "cfg.json").read_text(encoding="utf-8"))
        if isinstance(cfg.get("tokenizer"), str) and cfg["tokenizer"]:
            return cfg["tokenizer"]
    except (OSError, ValueError, AttributeError):
        pass
    return DEFAULT_TOKENIZER


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=default_tokenizer_dir(),
                    help="the pack's tokenizer/ directory (vocab.json, merges.txt, token_type.json, tokenizer.json); "
                         "default: $LLMSERVE_TOKENIZER_DIR, else the \"tokenizer\" of /data/cfg.json, else %s" % DEFAULT_TOKENIZER)
    ap.add_argument("--corpus", nargs="*", default=None, metavar="PATH",
                    help="files or folders for corpus (b), relative to the repository root (default: %s)" % " ".join(DEFAULT_CORPUS))
    ap.add_argument("--max-chars", type=int, default=1_500_000, help="characters of corpus (b) (default %(default)s, ~400K tokens)")
    ap.add_argument("--quick", action="store_true", help="a 250,000-character corpus: a minute at most")
    ap.add_argument("--no-hf", action="store_true", help="do not compare with Hugging Face tokenizers even if installed")
    ap.add_argument("--json", default="", metavar="FILE", help="also save the numbers here")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print()
    print(render(report))
    if args.json:
        def clean(o):
            return {k: clean(v) for k, v in o.items() if k != "ids"} if isinstance(o, dict) else \
                [clean(x) for x in o] if isinstance(o, list) else o
        Path(args.json).write_text(json.dumps(clean(report), indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nnumbers saved to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
