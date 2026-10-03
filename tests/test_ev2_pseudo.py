import csv
import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from varroa_vision.data import ev2_pseudo


def _ev2_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ev2_bees"
    (d / "images" / "all").mkdir(parents=True)
    rows = [
        ("v1_frame0.png", "varroa_infested/v1.MTS", "infested", 1),
        ("v1_frame1.png", "varroa_infested/v1.MTS", "infested", 1),
        ("v1_frame2.png", "varroa_infested/v1.MTS", "infested", 0),
        ("v2_frame0.png", "varroa_free/v2.MTS", "free", 0),
    ]
    for name, *_ in rows:
        Image.new("RGB", (400, 300)).save(d / "images" / "all" / name)
    pd.DataFrame(rows, columns=["image", "video", "video_class", "infected"]).to_csv(d / "meta.csv", index=False)
    return d


def test_video_split_is_deterministic_and_covers_all_buckets():
    names = [f"varroa_infested/{i}.MTS" for i in range(300)]
    splits = [ev2_pseudo.video_split(n) for n in names]
    assert splits == [ev2_pseudo.video_split(n) for n in names]
    counts = {s: splits.count(s) for s in ("train", "val", "test")}
    assert counts["train"] > counts["val"] > 0 and counts["test"] > 0


def test_build_keeps_detected_positives_and_all_negatives(tmp_path: Path, monkeypatch):
    ev2 = _ev2_dir(tmp_path)
    dets = {
        "v1_frame0.png": [(0.5, 0.5, 0.1, 0.1, 0.9)],
        "v1_frame1.png": [],  # visible mite but no detection -> dropped from train
        "v1_frame2.png": [(0.2, 0.2, 0.1, 0.1, 0.3)],  # negative with a detection -> background anyway
        "v2_frame0.png": [],
    }
    # pin the split: v1 trains, v2 is held out
    monkeypatch.setattr(ev2_pseudo, "video_split", lambda video, seed=0: "train" if "v1" in video else "test")
    out = tmp_path / "pseudo"
    data_yaml = ev2_pseudo.build(ev2, dets, out, link=False)
    assert (out / "labels" / "train" / "v1_frame0.txt").read_text() == "0 0.500000 0.500000 0.100000 0.100000\n"
    assert not (out / "images" / "train" / "v1_frame1.png").exists()
    assert (out / "labels" / "train" / "v1_frame2.txt").read_text() == ""
    assert (out / "images" / "test" / "v2_frame0.png").exists()
    assert not (out / "labels" / "test" / "v2_frame0.txt").exists()
    stats = json.loads((out / "stats.json").read_text())
    assert stats["train"]["dropped_positive"] == 1 and stats["train"]["positives"] == 1
    assert stats["train"]["unlabelled_detections_on_negatives"] == 1
    assert stats["test"]["images"] == 1
    with (out / "meta.csv").open() as fh:
        rows = {r["image"]: r for r in csv.DictReader(fh)}
    assert rows["v1_frame0.png"]["n_boxes"] == "1" and rows["v1_frame2.png"]["n_boxes"] == "0"
    assert rows["v2_frame0.png"]["n_boxes"] == "" and rows["v2_frame0.png"]["pseudo"] == "0"
    assert yaml.safe_load(data_yaml.read_text())["names"] == {0: "varroa"}


def test_heldout_positives_are_kept(tmp_path: Path, monkeypatch):
    ev2 = _ev2_dir(tmp_path)
    monkeypatch.setattr(ev2_pseudo, "video_split", lambda video, seed=0: "test")
    out = tmp_path / "pseudo"
    ev2_pseudo.build(ev2, {}, out, link=False)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["test"] == {"images": 4, "positives": 2}


def test_write_combined(tmp_path: Path):
    base = tmp_path / "base"
    (base / "images" / "train").mkdir(parents=True)
    (base / "meta.csv").write_text("image,split,infected,n_boxes\n")
    pseudo = tmp_path / "pseudo"
    (pseudo / "images" / "train").mkdir(parents=True)
    out = tmp_path / "combined"
    cfg = yaml.safe_load(ev2_pseudo.write_combined(base, pseudo, out).read_text())
    assert len(cfg["train"]) == 2 and cfg["train"][1].endswith("train")
    assert cfg["val"].endswith("val") and (out / "meta.csv").exists()
