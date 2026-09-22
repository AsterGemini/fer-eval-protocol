"""METRIC check: KDEF prep copies disgust/fear only.

Run from fer-system/:

    python -m data.test_prepare_kdef
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.prepare_kdef import copy_minority, find_kdef_root, prepare_kdef  # noqa: E402


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(80, 80, 80)).save(path, format="JPEG")


def test_find_nested_imagefolder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nested = root / "kdef-database"
        _touch_jpg(nested / "disgust" / "d.jpg")
        _touch_jpg(nested / "fear" / "f.jpg")
        found = find_kdef_root(root)
        _assert(found == nested, found)


def test_copy_skips_majority_classes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "kdef"
        _touch_jpg(src / "disgust" / "d1.jpg")
        _touch_jpg(src / "fear" / "f1.jpg")
        _touch_jpg(src / "happy" / "h1.jpg")
        _touch_jpg(src / "angry" / "a1.jpg")
        dest = Path(tmp) / "train"
        counts = copy_minority(src, dest)
        _assert(
            counts
            == {
                "angry": 0,
                "disgust": 1,
                "fear": 1,
                "happy": 0,
                "neutral": 0,
                "sad": 0,
                "surprise": 0,
            },
            counts,
        )
        _assert((dest / "disgust" / "d1.jpg").is_file(), "disgust not copied")
        _assert(not (dest / "happy" / "h1.jpg").exists(), "happy must stay empty")
        _assert(not list((dest / "angry").iterdir()), "angry is not a minority target")
        _assert((dest / "neutral").is_dir(), "neutral folder must exist")


def test_prepare_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        src = raw / "kdef"
        for i in range(300):
            _touch_jpg(src / "disgust" / f"d{i}.jpg")
            _touch_jpg(src / "fear" / f"f{i}.jpg")
        out = Path(tmp) / "out"
        first = prepare_kdef(raw, out)
        second = prepare_kdef(raw, out)
        _assert(first == second == out, (first, second, out))
        _assert((out / "prep_meta.json").is_file(), "missing meta")


def test_default_config_excludes_failed_kdef_mix() -> None:
    from src.utils import load_config

    cfg = load_config("configs/public.yaml")
    extras = {str(e.get("name")): e for e in cfg["data"].get("extra_train_sources") or []}
    _assert("kdef" not in extras, extras.keys())


def main() -> None:
    test_find_nested_imagefolder()
    test_copy_skips_majority_classes()
    test_prepare_is_idempotent()
    test_default_config_excludes_failed_kdef_mix()
    print("METRIC ok: KDEF copies disgust/fear only; majority folders stay empty")


if __name__ == "__main__":
    main()
