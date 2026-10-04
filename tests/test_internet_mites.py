import csv
import json
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from varroa_vision.data import combine, internet_mites, roboflow_bees


def _project(tmp_path: Path, name: str, names: list[str], items: dict[str, tuple[tuple[int, int], str]]) -> Path:
    """items: file name -> ((w, h), label text). Roboflow layout with a data.yaml."""
    d = tmp_path / name
    for split in ("train", "valid"):
        (d / split / "images").mkdir(parents=True)
        (d / split / "labels").mkdir(parents=True)
    (d / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    for fname, ((w, h), text) in items.items():
        split = "valid" if fname.startswith("v_") else "train"
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(d / split / "images" / fname)
        (d / split / "labels" / (Path(fname).stem + ".txt")).write_text(text)
    return d


def test_source_stem_and_dedupe(tmp_path: Path):
    d = _project(tmp_path, "p", ["bees", "varroa mites"], {
        "a_jpg.rf.0123456789abcdef0123456789abcdef.jpg": ((400, 300), "1 0.5 0.5 0.05 0.05\n"),
        "a_jpg.rf.fedcba9876543210fedcba9876543210.jpg": ((400, 300), "1 0.5 0.5 0.05 0.05\n"),
        "2017-09-01_3-01-01-mp4-bee_id_2300-48975-1_png.rf.0123456789abcdef0123456789abcdef.jpg": ((160, 280), "1 0.5 0.5 0.1 0.1\n"),
    })
    assert internet_mites.source_stem("a_jpg.rf.0123456789abcdef0123456789abcdef.jpg") == "a_jpg"
    imgs = internet_mites.unique_images(d)
    assert len(imgs) == 1  # one copy of a, VarroaDataset re-upload dropped


def test_boxes_in_window_clips_and_filters():
    lines, n = internet_mites.boxes_in_window([(10, 10, 30, 30), (90, 90, 130, 130)], (0, 0, 100, 100))
    assert n == 1 and lines[0].startswith("0 0.200000 0.200000 0.200000 0.200000")


def test_windows_for_image_kinds():
    rng = random.Random(0)
    wins = internet_mites.windows_for_image(1000, 800, mites=[(500, 400, 520, 420)], bees=[], rng=rng)
    kinds = [k for k, _ in wins]
    assert kinds.count("mite") == 1 and 1 <= kinds.count("negative") <= 2
    _, (x1, y1, x2, y2) = wins[0]
    assert 160 <= x2 - x1 <= 200 and x1 <= 510 <= x2 and y1 <= 410 <= y2
    wins_b = internet_mites.windows_for_image(1000, 800, mites=[(500, 400, 520, 420)], bees=[(450, 350, 600, 500)], rng=rng)
    kinds_b = [k for k, _ in wins_b]
    assert kinds_b[0] == "bee" and "mite" not in kinds_b


def test_build_internet_mites(tmp_path: Path):
    p1 = _project(tmp_path, "src1", ["varroa_mite"], {
        "img1.jpg": ((800, 600), "0 0.5 0.5 0.03 0.04\n0 0.2 0.2 0.03 0.04\n"),
        "img2.jpg": ((640, 640), ""),
    })
    p2 = _project(tmp_path, "src2", ["bees", "varroa mites"], {
        "img3.jpg": ((900, 700), "0 0.5 0.5 0.3 0.3\n1 0.52 0.5 0.03 0.03\n"),
    })
    out = tmp_path / "internet"
    data_yaml = internet_mites.build({"src1": p1, "src2": p2}, out, test_fraction=0.0, val_fraction=0.0)
    with (out / "meta.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    kinds = {r["kind"] for r in rows}
    assert "mite" in kinds and "bee" in kinds
    assert sum(int(r["infected"]) for r in rows if r["kind"] == "bee") == 1
    assert all(r["infected"] == "0" for r in rows if r["kind"] == "negative")
    assert all(r["split"] == "train" for r in rows)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["train"]["boxes"] >= 3
    assert yaml.safe_load(data_yaml.read_text())["names"] == {0: "varroa"}


def test_roboflow_bees_and_combine(tmp_path: Path):
    p = _project(tmp_path, "hofer", ["bee", "mite", "pollen", "queen", "varroa"], {
        "comb1.jpg": ((640, 640), "0 0.5 0.5 0.1 0.1\n3 0.3 0.3 0.1 0.1\n2 0.2 0.2 0.05 0.05\n"),
        "board1.jpg": ((640, 640), "4 0.5 0.5 0.02 0.02\n"),
    })
    out = tmp_path / "bees"
    data_yaml = roboflow_bees.build(p, out, test_fraction=0.0, val_fraction=0.0, link=False)
    labels = list((out / "labels" / "train").iterdir())
    assert [l.name for l in labels] == ["comb1.txt"]
    assert labels[0].read_text().count("\n") == 2  # bee + queen, pollen dropped
    combined = combine.write_combined(out, [out], tmp_path / "comb", names={0: "bee"})
    cfg = yaml.safe_load(combined.read_text())
    assert len(cfg["train"]) == 2 and cfg["val"].endswith("val") and (tmp_path / "comb" / "meta.csv").exists()
    assert yaml.safe_load(data_yaml.read_text())["names"] == {0: "bee"}
