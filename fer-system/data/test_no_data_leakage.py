"""METRIC: extra train sources must not leak frozen FER val (or MMA valid/test).

Run from fer-system/:

    python -m data.test_no_data_leakage
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import data.prepare_mma_minority as mma_mod  # noqa: E402
import data.prepare_minority_boost as boost_mod  # noqa: E402
from data.prepare_ferplus_minority import select_minority_rows  # noqa: E402
from data.prepare_mma_minority import find_mma_train, prepare_mma_minority  # noqa: E402
from data.prepare_minority_boost import (  # noqa: E402
    EXCLUSION_SOURCES,
    ahash_bytes,
    hamming_min,
    hash_tree_bytes,
    select_new_crops,
)
from data.test_prepare_minority_boost import _pattern_image, _save_jpeg_roundtrip  # noqa: E402
from src.dataset import create_dataloaders, create_extra_val_loaders  # noqa: E402
from src.utils import load_config  # noqa: E402

PIXELS = " ".join(["0"] * (48 * 48))
_FORBIDDEN_TRAIN_DIR_PARTS = frozenset({"valid", "validation", "val", "test"})
_PREPARE_TRAIN_ONLY = frozenset(
    {"ferplus_minority", "ckplus", "kdef", "minority_boost", "mma_minority"}
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _plus_row(usage: str, image_name: str, *, fear: int = 0, disgust: int = 0) -> dict:
    return {
        "usage": usage,
        "image_name": image_name,
        "neutral": "0",
        "happiness": "0",
        "surprise": "0",
        "sadness": "0",
        "anger": "0",
        "disgust": str(disgust),
        "fear": str(fear),
        "contempt": "0",
        "unknown": "1",
        "nf": "0",
    }


def test_config_fer_val_overlay_disabled() -> None:
    cfg = load_config("configs/public.yaml")
    _assert(
        cfg["data"].get("val_label_corrections") in (None, "", "null"),
        cfg["data"].get("val_label_corrections"),
    )


def test_config_extra_sources_cannot_train_on_holdout_splits() -> None:
    cfg = load_config("configs/public.yaml")
    extras = list(cfg["data"].get("extra_train_sources") or [])
    names = [str(e.get("name") or "") for e in extras]
    _assert("human_feedback" not in names, names)
    _assert("val_errors" not in names, names)

    val_dir_names = [str(e.get("name")) for e in extras if e.get("val_dir")]
    _assert(val_dir_names == ["raf_db"], val_dir_names)

    for entry in extras:
        train_dir = str(entry.get("train_dir") or "").replace("\\", "/").lower()
        parts = {p for p in train_dir.split("/") if p}
        leak = parts & _FORBIDDEN_TRAIN_DIR_PARTS
        _assert(not leak, f"{entry.get('name')} train_dir={train_dir} hits {leak}")
        if str(entry.get("prepare") or "") in _PREPARE_TRAIN_ONLY:
            _assert(not entry.get("val_dir"), entry)


def test_hash_guards_require_fer_and_raf_validation() -> None:
    fer = next(s for s in EXCLUSION_SOURCES if s["key"] == "fer2013_primary")
    _assert(fer.get("mandatory") is True, fer)
    _assert("validation" in fer["splits"], fer)
    _assert("train" in fer["splits"], fer)
    raf = next(s for s in EXCLUSION_SOURCES if s["key"] == "raf_db")
    _assert(raf.get("mandatory") is True, raf)
    _assert("validation" in raf["splits"], raf)
    src = inspect.getsource(mma_mod.default_mma_exclusion_dirs)
    _assert("fer2013_affectnet_pack" in src, src)
    _assert(mma_mod.MERGED_PACK_SLUG == "prasadsomvanshih/fer2013-affectnet-dataset-emotions", mma_mod.MERGED_PACK_SLUG)


def test_loader_trains_on_extra_train_dir_never_extra_val() -> None:
    train_src = inspect.getsource(create_dataloaders)
    extra_src = inspect.getsource(create_extra_val_loaders)
    _assert('extra["train_dir"]' in train_src, train_src)
    _assert("train_parts.append(extra_ds)" in train_src, train_src)
    _assert('extra["val_dir"]' not in train_src, "create_dataloaders must not train on extra val")
    _assert('extra["val_dir"]' in extra_src, extra_src)
    _assert("never trained on" in extra_src, extra_src)


def test_mma_find_refuses_valid_or_test_only_packs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for split in ("valid", "test", "validation"):
            tree = root / "MMAFEDB" / split
            (tree / "disgust").mkdir(parents=True)
            (tree / "fear").mkdir(parents=True)
            Image.new("L", (48, 48), color=40).save(tree / "disgust" / "a.png")
            Image.new("L", (48, 48), color=80).save(tree / "fear" / "b.png")
        try:
            found = find_mma_train(root)
        except FileNotFoundError:
            found = None
        _assert(found is None, f"must not fall back to holdout split: {found}")


def test_mma_prepare_does_not_copy_valid_or_test_images() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pack = td / "raw" / "MMAFEDB"
        train = pack / "train"
        valid = pack / "valid"
        test = pack / "test"
        for tree in (train, valid, test):
            (tree / "disgust").mkdir(parents=True)
            (tree / "fear").mkdir(parents=True)
        _pattern_image(10).save(train / "disgust" / "train_keep.png")
        _pattern_image(11).save(train / "fear" / "train_fear.png")
        _pattern_image(90).save(valid / "disgust" / "val_leak.png")
        _pattern_image(91).save(test / "disgust" / "test_leak.png")
        _pattern_image(92).save(valid / "fear" / "val_fear_leak.png")

        orig_min = dict(mma_mod.MIN_EXPECTED)
        mma_mod.MIN_EXPECTED.update({"disgust": 1, "fear": 0})
        try:
            out = Path(
                prepare_mma_minority(td / "raw", td / "out", exclusion_resolver=lambda: {})
            )
        finally:
            mma_mod.MIN_EXPECTED.update(orig_min)

        disgust_names = [p.name for p in (out / "train" / "disgust").iterdir() if p.is_file()]
        fear_names = [p.name for p in (out / "train" / "fear").iterdir() if p.is_file()]
        _assert(any("train_keep" in n for n in disgust_names), disgust_names)
        _assert(not any("val_leak" in n for n in disgust_names), disgust_names)
        _assert(not any("test_leak" in n for n in disgust_names), disgust_names)
        _assert(not any("val_fear_leak" in n for n in fear_names), fear_names)


def test_mma_prepare_drops_recompressed_fer_val_near_dup() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        train = td / "raw" / "MMAFEDB" / "train"
        (train / "disgust").mkdir(parents=True)
        (train / "fear").mkdir(parents=True)
        _pattern_image(20).save(train / "disgust" / "keep.png")
        _pattern_image(21).save(train / "disgust" / "fer_val_clone.png")
        _pattern_image(22).save(train / "fear" / "keep.png")

        fer_val = td / "fer_val"
        fer_val.mkdir()
        _save_jpeg_roundtrip(
            Image.open(train / "disgust" / "fer_val_clone.png"),
            fer_val / "official_val.png",
        )

        orig_min = dict(mma_mod.MIN_EXPECTED)
        mma_mod.MIN_EXPECTED.update({"disgust": 1, "fear": 0})
        try:
            out = Path(
                prepare_mma_minority(
                    td / "raw",
                    td / "out",
                    exclusion_resolver=lambda: {
                        "fer2013_primary": {"dirs": [fer_val], "fingerprint": "toy-val"}
                    },
                )
            )
        finally:
            mma_mod.MIN_EXPECTED.update(orig_min)

        kept = list((out / "train" / "disgust").iterdir())
        _assert(len(kept) == 1, [p.name for p in kept])
        _assert("keep.png" in kept[0].name, kept[0].name)
        val_hash = ahash_bytes(fer_val / "official_val.png")
        kept_hash = ahash_bytes(kept[0])
        dist = int(hamming_min(kept_hash[None, :], val_hash[None, :])[0])
        _assert(dist > 10, f"kept crop still near FER val: hamming={dist}")
        meta = json.loads((out / "prep_meta.json").read_text(encoding="utf-8"))
        _assert(meta["accounting"]["disgust"]["excluded_fer2013_primary"] == 1, meta)


def test_minority_boost_drops_recompressed_fer_val_near_dup() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pack = td / "raw" / "fer2013+affectnet"
        for folder in ("disgusted", "fearful"):
            (pack / "train" / folder).mkdir(parents=True)
        _pattern_image(30).save(pack / "train" / "disgusted" / "keep.png")
        _pattern_image(31).save(pack / "train" / "disgusted" / "fer_val_clone.png")
        _pattern_image(32).save(pack / "train" / "fearful" / "keep.png")

        fer_val = td / "fer_val"
        fer_val.mkdir()
        _save_jpeg_roundtrip(
            Image.open(pack / "train" / "disgusted" / "fer_val_clone.png"),
            fer_val / "official_val.png",
        )
        ref, _ = hash_tree_bytes(fer_val)
        kept, acct = select_new_crops(pack, {"fer2013_primary": ref})
        _assert(len(kept["disgust"]) == 1, acct)
        _assert(kept["disgust"][0].name == "keep.png", kept["disgust"])
        _assert(acct["disgust"]["excluded_fer2013_primary"] == 1, acct)


def test_ferplus_rejects_publictest_and_privatetest() -> None:
    fer_rows = [
        {"emotion": "2", "pixels": PIXELS, "usage": "Training"},
        {"emotion": "1", "pixels": PIXELS, "usage": "PublicTest"},
        {"emotion": "1", "pixels": PIXELS, "usage": "PrivateTest"},
    ]
    plus_rows = [
        _plus_row("Training", "fer0000000.png", fear=9),
        _plus_row("PublicTest", "fer0000001.png", disgust=9),
        _plus_row("PrivateTest", "fer0000002.png", fear=9),
    ]
    selected = select_minority_rows(fer_rows, plus_rows)
    _assert(len(selected) == 1, selected)
    _assert(selected[0]["image_name"] == "fer0000000.png", selected[0])
    _assert(selected[0]["label"] == "fear", selected[0])


def main() -> None:
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"METRIC ok: {len(tests)} leakage tests passed")


if __name__ == "__main__":
    main()
