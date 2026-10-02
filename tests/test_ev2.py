import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from varroa_vision.data import ev2

ROWS = [
    {"video": "varroa_infested/1_00953.MTS", "id": "frame_4", "varroa_visible": "yes", "coord_1": [702, 708], "coord_2": [1044, 969]},
    {"video": "varroa_infested/1_00953.MTS", "id": "frame_6", "varroa_visible": "no", "coord_1": [701, 634], "coord_2": [1126, 971]},
    {"video": "varroa_free/10_01026.MTS", "id": "frame_0", "varroa_visible": "no", "coord_1": [10, 20], "coord_2": [400, 300]},
]


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "ev2"
    (raw / "dataset_free").mkdir(parents=True)
    (raw / "dataset_infested").mkdir()
    with (raw / "labels.txt").open("w") as fh:
        for i, r in enumerate(ROWS):
            fh.write(json.dumps(r) + "\n")
            if i == 1:
                fh.write("\n")  # blank separator lines exist in the real file
    Image.new("RGB", (342, 261)).save(raw / "dataset_infested" / "1_00953.MTS_frame4.png")
    Image.new("RGB", (425, 337)).save(raw / "dataset_free" / "1_00953.MTS_frame6.png")
    Image.new("RGB", (390, 280)).save(raw / "dataset_free" / "10_01026.MTS_frame0.png")
    return raw


def test_parse_labels(raw_dir: Path):
    crops = ev2.parse_labels(raw_dir / "labels.txt")
    assert len(crops) == 3
    assert crops[0].file_name == "1_00953.MTS_frame4.png" and crops[0].visible
    assert crops[0].folder == "dataset_infested" and crops[1].folder == "dataset_free"
    assert crops[0].video_class == "infested" and crops[2].video_class == "free"
    assert crops[0].box == (702, 708, 1044, 969)


def test_parse_labels_rejects_bad_id(tmp_path: Path):
    p = tmp_path / "labels.txt"
    p.write_text(json.dumps({"video": "v/x.MTS", "id": "f4", "varroa_visible": "yes", "coord_1": [0, 0], "coord_2": [1, 1]}) + "\n")
    with pytest.raises(ValueError, match="unexpected id"):
        ev2.parse_labels(p)


def test_convert(raw_dir: Path, tmp_path: Path):
    out = tmp_path / "ev2_bees"
    data_yaml = ev2.convert(raw_dir, out, link=False)
    assert sorted(p.name for p in (out / "images" / "all").iterdir()) == [
        "10_01026.MTS_frame0.png",
        "1_00953.MTS_frame4.png",
        "1_00953.MTS_frame6.png",
    ]
    with (out / "meta.csv").open() as fh:
        rows = {r["image"]: r for r in csv.DictReader(fh)}
    assert rows["1_00953.MTS_frame4.png"]["infected"] == "1"
    assert rows["1_00953.MTS_frame6.png"]["infected"] == "0"
    assert rows["1_00953.MTS_frame6.png"]["video_class"] == "infested"
    assert rows["10_01026.MTS_frame0.png"]["video_class"] == "free"
    assert (rows["1_00953.MTS_frame4.png"]["width"], rows["1_00953.MTS_frame4.png"]["height"]) == ("342", "261")
    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["val"] == "images/all" and "train" not in cfg
