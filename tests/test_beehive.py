import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from varroa_vision.data import beehive


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "beehive"
    layout = {
        ("frame", "train"): {"a": "1 0.5 0.5 0.4 0.6\n0 0.2 0.2 0.1 0.1\n", "b": ""},
        ("frame", "valid"): {"c": "1 0.5 0.5 0.4 0.6\n"},
        ("frame", "test"): {"d": "0 0.5 0.5 0.4 0.6\n"},
        ("bottom", "train"): {"e": "1 0.1 0.1 0.1 0.1\n1 0.3 0.3 0.1 0.1\n0 0.6 0.6 0.1 0.1\n"},
        ("bottom", "valid"): {"f": "1 0.1 0.1 0.1 0.1\n"},
        ("bottom", "test"): {"g": "1 0.1 0.1 0.1 0.1\n"},
    }
    for (subset, split), files in layout.items():
        d = raw / f"{subset}_dataset" / split
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir()
        for stem, text in files.items():
            Image.new("RGB", (64, 64)).save(d / "images" / f"{stem}.jpg")
            if stem != "b":  # b has no label file at all -> background
                (d / "labels" / f"{stem}.txt").write_text(text)
    return raw


def test_remap_label():
    text, clear, degraded = beehive.remap_label("1 0.5 0.5 0.4 0.6\n0 0.2 0.2 0.1 0.1\n\n")
    assert text == "0 0.5 0.5 0.4 0.6\n0 0.2 0.2 0.1 0.1\n"
    assert (clear, degraded) == (1, 1)
    with pytest.raises(ValueError):
        beehive.remap_label("1 0.5 0.5\n")


def test_convert_merges_subsets(raw_dir: Path, tmp_path: Path):
    out = tmp_path / "bees"
    data_yaml = beehive.convert(raw_dir, out, link=False)
    train_imgs = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert train_imgs == ["bottom__e.jpg", "frame__a.jpg", "frame__b.jpg"]
    assert (out / "labels" / "train" / "frame__b.txt").read_text() == ""
    assert (out / "labels" / "train" / "bottom__e.txt").read_text().startswith("0 0.1 0.1")
    assert set((out / "labels" / "train" / "bottom__e.txt").read_text().split("\n")[:3]) == {
        "0 0.1 0.1 0.1 0.1",
        "0 0.3 0.3 0.1 0.1",
        "0 0.6 0.6 0.1 0.1",
    }
    stats = json.loads((out / "stats.json").read_text())
    assert stats["train"] == {"images": 3, "boxes": 5, "frame_images": 2, "bottom_images": 1}
    assert stats["val"]["images"] == 2 and stats["test"]["images"] == 2
    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["names"] == {0: "bee"} and cfg["val"] == "images/val"
    with (out / "meta.csv").open() as fh:
        rows = {r["image"]: r for r in csv.DictReader(fh)}
    assert rows["bottom__e.jpg"]["n_clear"] == "1" and rows["bottom__e.jpg"]["n_degraded"] == "2"


def test_convert_single_subset(raw_dir: Path, tmp_path: Path):
    beehive.convert(raw_dir, tmp_path / "frame_only", subsets=("frame",), link=False)
    assert sorted(p.name for p in (tmp_path / "frame_only" / "images" / "train").iterdir()) == ["frame__a.jpg", "frame__b.jpg"]
