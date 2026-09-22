"""METRIC check: minority_boost near-dup filtering rules.

Run from fer-system/:

    python -m data.test_prepare_minority_boost
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import data.prepare_minority_boost as mod  # noqa: E402
from data.prepare_minority_boost import (  # noqa: E402
    HASH_BYTES,
    ahash_bytes,
    hamming_min,
    hash_tree_bytes,
    select_new_crops,
)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _dist(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


def _pattern_image(seed: int, size: int = 48) -> Image.Image:
    """Blocky 16x16 random pattern upscaled to `size` — distinctive aHash bits."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (16, 16), dtype=np.uint8) * 255
    return Image.fromarray(bits, mode="L").resize((size, size), Image.NEAREST)


def _save_jpeg_roundtrip(im: Image.Image, dest: Path) -> None:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    with Image.open(buf) as reenc:
        reenc.save(dest, format="PNG")


def test_ahash_shape_and_stability() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.png"
        _pattern_image(1).save(p)
        h1 = ahash_bytes(p)
        h2 = ahash_bytes(p)
        _assert(h1 is not None and h1.shape == (HASH_BYTES,), h1)
        _assert(np.array_equal(h1, h2), "ahash not deterministic")


def test_ahash_recompressed_duplicate_is_close() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        orig = td / "orig.png"
        _pattern_image(2).save(orig)
        reenc = td / "reenc.png"
        _save_jpeg_roundtrip(Image.open(orig), reenc)
        d = _dist(ahash_bytes(orig), ahash_bytes(reenc))
        _assert(d <= 10, f"recompressed duplicate at distance {d} > 10")


def test_ahash_different_images_are_far() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a, b = td / "a.png", td / "b.png"
        _pattern_image(3).save(a)
        _pattern_image(4).save(b)
        d = _dist(ahash_bytes(a), ahash_bytes(b))
        _assert(d > 40, f"unrelated images suspiciously close: {d}")


def test_hamming_min_chunked() -> None:
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (500, HASH_BYTES), dtype=np.uint8)
    cand = rng.integers(0, 256, (300, HASH_BYTES), dtype=np.uint8)
    cand[7] = ref[42]  # planted exact duplicate
    d = hamming_min(cand, ref)
    _assert(d.shape == (300,), d.shape)
    _assert(d[7] == 0, d[7])
    _assert(d.min() == 0 and np.median(d) > 60, (d.min(), np.median(d)))
    empty = hamming_min(cand, np.zeros((0, HASH_BYTES), np.uint8))
    _assert(empty.min() > 1000, "empty reference must mark everything far")


def test_select_new_crops_excludes_near_dups() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pack = td / "raw" / "fer2013+affectnet"
        for folder in ("disgusted", "fearful"):
            (pack / "train" / folder).mkdir(parents=True)
        for i, seed in enumerate((10, 11, 12)):
            _pattern_image(seed).save(pack / "train" / "disgusted" / f"im{i}.png")
            _pattern_image(seed + 100).save(pack / "train" / "fearful" / f"im{i}.png")
        # exclusion tree holds recompressed copies of disgusted/im0 + fearful/im1
        excl = td / "excl" / "val"
        excl.mkdir(parents=True)
        _save_jpeg_roundtrip(
            Image.open(pack / "train" / "disgusted" / "im0.png"), excl / "v0.png"
        )
        _save_jpeg_roundtrip(
            Image.open(pack / "train" / "fearful" / "im1.png"), excl / "v1.png"
        )

        ref, _ = hash_tree_bytes(excl)
        kept, acct = select_new_crops(pack, {"fer2013_primary": ref})
        _assert(len(kept["disgust"]) == 2, acct)
        _assert(len(kept["fear"]) == 2, acct)
        _assert(acct["disgust"]["excluded_fer2013_primary"] == 1, acct)
        _assert(acct["fear"]["excluded_fer2013_primary"] == 1, acct)
        _assert(acct["disgust"]["files_in"] == 3, acct)


def test_prepare_end_to_end_with_injected_resolver() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pack = td / "raw" / "fer2013+affectnet"
        (pack / "train" / "disgusted").mkdir(parents=True)
        (pack / "train" / "fearful").mkdir(parents=True)
        for i in range(3):
            _pattern_image(200 + i).save(pack / "train" / "disgusted" / f"d{i}.png")
            _pattern_image(300 + i).save(pack / "train" / "fearful" / f"f{i}.png")
        out = td / "out"

        resolver = lambda: {}  # noqa: E731 — no exclusion sources in unit test
        orig_min = dict(mod.MIN_EXPECTED)
        mod.MIN_EXPECTED.update({"disgust": 1, "fear": 1})
        try:
            result = mod.prepare_minority_boost(td / "raw", out, exclusion_resolver=resolver)
            meta = json.loads((Path(result) / "prep_meta.json").read_text())
            _assert(meta["counts"]["disgust"] == 3, meta["counts"])
            _assert(meta["counts"]["fear"] == 3, meta["counts"])
            _assert((Path(result) / "train" / "happy").is_dir(), "7-class tree missing")
            again = mod.prepare_minority_boost(td / "raw", out, exclusion_resolver=resolver)
            _assert(again == result, "prep not idempotent")
        finally:
            mod.MIN_EXPECTED.update(orig_min)


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} tests passed")
