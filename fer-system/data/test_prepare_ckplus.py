"""METRIC check: CK+ prep copies disgust/fear only and skips the nested duplicate.

Run from fer-system/:

    python -m data.test_prepare_ckplus
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.prepare_ckplus import copy_minority, find_ckplus48, prepare_ckplus  # noqa: E402


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _touch_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (48, 48), color=80).save(path)


def test_find_prefers_top_level_tree() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        top = root / "CK+48"
        nested = root / "ck" / "CK+48"
        for tree in (top, nested):
            _touch_png(tree / "disgust" / "a.png")
            _touch_png(tree / "fear" / "b.png")
        found = find_ckplus48(root)
        _assert(found == top, found)


def test_copy_skips_happy_contempt_and_maps_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "CK+48"
        _touch_png(src / "disgust" / "d1.png")
        _touch_png(src / "fear" / "f1.png")
        _touch_png(src / "happy" / "h1.png")
        _touch_png(src / "anger" / "a1.png")
        _touch_png(src / "contempt" / "c1.png")
        dest = Path(tmp) / "train"
        counts = copy_minority(src, dest)
        _assert(counts == {
            "angry": 0,
            "disgust": 1,
            "fear": 1,
            "happy": 0,
            "neutral": 0,
            "sad": 0,
            "surprise": 0,
        }, counts)
        _assert((dest / "disgust" / "d1.png").is_file(), "disgust not copied")
        _assert(not (dest / "happy" / "h1.png").exists(), "happy must stay empty")
        _assert(not list((dest / "angry").iterdir()), "anger is not a minority target")
        _assert((dest / "neutral").is_dir(), "neutral folder must exist")


def test_prepare_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        src = raw / "CK+48"
        for i in range(100):
            _touch_png(src / "disgust" / f"d{i}.png")
        for i in range(50):
            _touch_png(src / "fear" / f"f{i}.png")
        out = Path(tmp) / "out"
        first = prepare_ckplus(raw, out)
        second = prepare_ckplus(raw, out)
        _assert(first == second == out, (first, second, out))
        _assert((out / "prep_meta.json").is_file(), "missing meta")


def main() -> None:
    test_find_prefers_top_level_tree()
    test_copy_skips_happy_contempt_and_maps_names()
    test_prepare_is_idempotent()
    print("METRIC ok: CK+ copies disgust/fear only; nested duplicate unused; contempt dropped")


if __name__ == "__main__":
    main()
