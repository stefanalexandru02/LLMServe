"""tools/_paths.py - where the tools find llama.cpp's gguf-py (for the MTP packer and the fixtures).

Order: $STRATA_GGUF_PY, then <repo>/third_party/llama.cpp/gguf-py (a llama.cpp checkout at the pinned commit), then the
development tree's ../../.ref/llama.cpp/gguf-py, then $STRATA_INSTALL_DIR/third_party/llama.cpp/gguf-py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]


def _install_dirs() -> list[Path]:
    """$STRATA_INSTALL_DIR, when set: a folder holding third_party/llama.cpp."""
    raw = os.environ.get("STRATA_INSTALL_DIR")
    return [Path(raw).expanduser()] if raw else []


def gguf_py() -> str:
    cands = [os.environ.get("STRATA_GGUF_PY"), ENGINE / "third_party" / "llama.cpp" / "gguf-py",
             ENGINE.parents[min(1, len(ENGINE.parents) - 1)] / ".ref" / "llama.cpp" / "gguf-py"]   # the dev tree; a shallow install (G:\Strata) has no grandparent
    cands += [d / "third_party" / "llama.cpp" / "gguf-py" for d in _install_dirs()]
    for c in cands:
        if c and (Path(c) / "gguf").is_dir():
            return str(c)
    sys.exit("llama.cpp's gguf-py was not found. Set STRATA_GGUF_PY to the gguf-py folder of a llama.cpp checkout at the "
             "pinned commit (<llama.cpp>/gguf-py)")


def add_gguf_py() -> None:
    p = gguf_py()
    if p not in sys.path:
        sys.path.insert(0, p)
