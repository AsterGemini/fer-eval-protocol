"""METRIC check: MMA prep uses train only and drops FER-near-dups.

Run from fer-system/:

    python -m data.test_prepare_mma_minority
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import data.prepare_mma_minority as mod  # noqa: E402
from data.prepare_mma_minority import find_mma_train, prepare_mma_minority  # noqa: E402
from data.test_prepare_minority_boost import _pattern_image, _save_jpeg_roundtrip  # noqa: E402


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_find_prefers_mmafedb_train_and_ignores_valid_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train = root / "MMAFEDB" / "train"
        valid = root / "MMAFEDB" / "valid"
        test = root / "MMAFEDB" / "test"
        for tree in (train, valid, test):
            (tree / "disgust").mkdir(parents=True)
            (tree / "fear").mkdir(parents=True)
            Image.new("L", (48, 48), color=40).save(tree / "disgust" / "a.png")
            Image.new("L", (48, 48), color=80).save(tree / "fear" / "b.png")
        found = find_mma_train(root)
        _assert(found == train, found)


def test_find_ignores_train_nested_under_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        decoy = root / "MMAFEDB" / "test" / "train"
        real = root / "MMAFEDB" / "train"
        for tree in (decoy, real):
            (tree / "disgust").mkdir(parents=True)
            (tree / "fear").mkdir(parents=True)
            Image.new("L", (48, 48), color=40).save(tree / "disgust" / "a.png")
            Image.new("L", (48, 48), color=80).save(tree / "fear" / "b.png")
        found = find_mma_train(root)
        _assert(found == real, found)


def test_prepare_drops_near_dups_of_exclusion_set() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        train = td / "raw" / "MMAFEDB" / "train"
        (train / "disgust").mkdir(parents=True)
        (train / "fear").mkdir(parents=True)
        _pattern_image(10).save(train / "disgust" / "keep.png")
        _pattern_image(11).save(train / "disgust" / "dup.png")
        _pattern_image(12).save(train / "fear" / "keep.png")

        excl = td / "excl"
        excl.mkdir()
        _save_jpeg_roundtrip(Image.open(train / "disgust" / "dup.png"), excl / "v.png")

        resolver = lambda: {  # noqa: E731
            "fer2013_primary": {
                "dirs": [excl],
                "fingerprint": "toy",
            }
        }
        # Inject hashes by wrapping load: easier to pass resolver that
        # hashes the toy exclusion dir via the production loader.
        orig_min = dict(mod.MIN_EXPECTED)
        mod.MIN_EXPECTED.update({"disgust": 1, "fear": 0})
        try:
            # default_mma_exclusion_dirs is bypassed; resolver returns
            # dirs that load_or_build_exclusion_hashes will hash.
            out = td / "out"
            result = prepare_mma_minority(
                td / "raw", out, exclusion_resolver=resolver
            )
            meta = json.loads((Path(result) / "prep_meta.json").read_text())
            _assert(meta["counts"]["disgust"] == 1, meta["counts"])
            _assert(meta["counts"]["fear"] == 1, meta["counts"])
            _assert(
                meta["accounting"]["disgust"]["excluded_fer2013_primary"] == 1,
                meta["accounting"],
            )
            _assert((Path(result) / "train" / "happy").is_dir(), "7-class tree missing")
            again = prepare_mma_minority(
                td / "raw", out, exclusion_resolver=resolver
            )
            _assert(again == result, "prep not idempotent")
        finally:
            mod.MIN_EXPECTED.update(orig_min)


def test_default_config_excludes_failed_mma_mix() -> None:
    from src.utils import load_config

    cfg = load_config("configs/public.yaml")
    extras = {str(e.get("name")): e for e in cfg["data"].get("extra_train_sources") or []}
    _assert("mma_minority" not in extras, extras.keys())


def main() -> None:
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
