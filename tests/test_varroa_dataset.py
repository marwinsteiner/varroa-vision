import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from varroa_vision.data import varroa_dataset as vd

W, H = 160, 280

ROWS = [
    # healthy bee, no boxes -> empty label file (background image)
    ("train/videos/2017-09-20_19-24-55/a.png", 0, []),
    # label 1, one box
    ("train/videos/2017-09-20_19-24-55/b.png", 1, [(84, 143, 109, 172)]),
    # label 3 counts as infected; second box is degenerate and dropped
    ("val/videos/2017-09-25_16-03-38/c.png", 3, [(10, 20, 40, 50), (30, 30, 30, 60)]),
    # box running past the right edge is clipped
    ("test/videos/2017-10-17_1-39-36/d.png", 1, [(150, 100, 175, 130)]),
]


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    lines = []
    for rel, label, boxes in ROWS:
        p = raw / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (W, H), (120, 100, 40)).save(p)
        coords = " ".join(str(v) for b in boxes for v in b)
        lines.append(f"{rel} {label}" + (f" {coords}" if coords else ""))
    (raw / "gt.csv").write_text("\n".join(lines) + "\n")
    return raw


def _labels(out: Path, split: str, stem: str) -> list[list[float]]:
    text = (out / "labels" / split / f"{stem}.txt").read_text()
    return [list(map(float, line.split())) for line in text.splitlines()]


def test_parse_gt(raw_dir: Path):
    samples = vd.parse_gt(raw_dir / "gt.csv")
    assert [s.label_raw for s in samples] == [0, 1, 3, 1]
    assert samples[2].infected and not samples[0].infected
    assert samples[2].boxes == ((10, 20, 40, 50), (30, 30, 30, 60))
    assert samples[1].session == "2017-09-20_19-24-55"
    assert samples[1].date == "2017-09-20"


def test_parse_gt_rejects_malformed(tmp_path: Path):
    bad = tmp_path / "gt.csv"
    bad.write_text("train/videos/s/a.png 1 1 2 3\n")
    with pytest.raises(ValueError, match="expected"):
        vd.parse_gt(bad)


def test_to_yolo_line():
    assert vd.to_yolo_line((84, 143, 109, 172), W, H) == "0 0.603125 0.562500 0.156250 0.103571"


def test_convert_official(raw_dir: Path, tmp_path: Path):
    out = tmp_path / "yolo"
    data_yaml = vd.convert(raw_dir, out, "official", link=False)

    assert _labels(out, "train", "a") == []
    assert _labels(out, "train", "b") == [[0, 0.603125, 0.5625, 0.15625, 0.103571]]
    # label 3 -> one valid box kept, degenerate dropped
    assert len(_labels(out, "val", "c")) == 1
    # clipped: x2 becomes 160 -> cx=(150+160)/2/160, w=10/160
    (clipped,) = _labels(out, "test", "d")
    assert clipped[1] == pytest.approx(155 / 160, abs=1e-6)
    assert clipped[3] == pytest.approx(10 / 160, abs=1e-6)

    for split, name in [("train", "a.png"), ("train", "b.png"), ("val", "c.png"), ("test", "d.png")]:
        assert (out / "images" / split / name).exists()

    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["names"] == {0: "varroa"}
    assert cfg["train"] == "images/train" and Path(cfg["path"]).is_absolute()

    stats = json.loads((out / "stats.json").read_text())
    assert stats["train"] == {"images": 2, "infected": 1, "boxes": 1, "boxes_dropped": 0}
    assert stats["val"] == {"images": 1, "infected": 1, "boxes": 1, "boxes_dropped": 1}

    with (out / "meta.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["image"]: r["infected"] for r in rows} == {"a.png": "0", "b.png": "1", "c.png": "1", "d.png": "1"}
    assert rows[2]["label_raw"] == "3" and rows[2]["n_boxes"] == "1"


def test_convert_date_split(raw_dir: Path, tmp_path: Path):
    out = tmp_path / "yolo_date"
    vd.convert(raw_dir, out, "date", link=False)
    # 2017-09-20 -> train, 2017-09-25 -> val, 2017-10-17 -> test (same as official here)
    assert (out / "images" / "train" / "a.png").exists()
    assert (out / "images" / "val" / "c.png").exists()
    assert (out / "images" / "test" / "d.png").exists()
    stats = json.loads((out / "stats.json").read_text())
    assert stats["split_mode"] == "date"


def test_convert_dedupes_repeated_rows(raw_dir: Path, tmp_path: Path):
    with (raw_dir / "gt.csv").open("a") as fh:
        fh.write("train/videos/2017-09-20_19-24-55/b.png 1 90 140 110 170\n")
    out = tmp_path / "dedup"
    vd.convert(raw_dir, out, "official", link=False)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["train"]["images"] == 2
    assert _labels(out, "train", "b") == [[0, 0.603125, 0.5625, 0.15625, 0.103571]]  # first row kept


def test_convert_rejects_duplicate_names(raw_dir: Path, tmp_path: Path):
    dup = raw_dir / "test/videos/2017-09-01_10-54-26/a.png"
    dup.parent.mkdir(parents=True)
    Image.new("RGB", (W, H)).save(dup)
    with (raw_dir / "gt.csv").open("a") as fh:
        fh.write("test/videos/2017-09-01_10-54-26/a.png 0\n")
    with pytest.raises(ValueError, match="duplicate"):
        vd.convert(raw_dir, tmp_path / "x", "official", link=False)
