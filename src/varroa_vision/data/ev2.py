"""Prepare the EV2 dataset (Zenodo 13771384) as a bee-level evaluation set.

EV2 is not a frame dataset: every PNG is already a crop of one bee (roughly 300 to 700 px
on a side) cut from a 1920 x 1080 camcorder frame, and the box in ``labels.txt`` gives
the crop's position in that source frame. There are no mite boxes. What EV2 does give is
a per-crop flag ``varroa_visible`` on 5,170 crops from 32 videos, recorded with a
different camera, lighting and viewpoint than VarroaDataset. That makes it a clean
cross-domain test of the stage 2 detector at bee level: a crop counts as positive if the
detector fires on it.

Input layout after unzipping ``dataset.zip``::

    <raw>/labels.txt                    JSON lines: video, id, varroa_visible, coord_1, coord_2
    <raw>/dataset_free/<video>_frame<n>.png      varroa_visible == "no"
    <raw>/dataset_infested/<video>_frame<n>.png  varroa_visible == "yes"

Output layout::

    <out>/images/all/<file>.png     hard-linked (or copied)
    <out>/meta.csv                  image, split, video, video_class, frame, infected, width, height
    <out>/data.yaml                 val -> images/all (for predict); no labels, so no box metrics

Note the folder names: ``dataset_free`` means "no mite visible in this crop", and 699 of
its 1,987 crops come from videos of infested bees where the mite is hidden in that frame.
``video_class`` keeps that distinction.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger
from PIL import Image
from tqdm import tqdm

from varroa_vision.data.varroa_dataset import CLASS_NAMES, _place


@dataclass(frozen=True)
class Crop:
    video: str  # e.g. "varroa_infested/1_00953.MTS"
    frame: int
    visible: bool
    box: tuple[int, int, int, int]  # position in the source frame, not in the crop

    @property
    def video_class(self) -> str:
        return self.video.split("/")[0].removeprefix("varroa_")

    @property
    def file_name(self) -> str:
        return f"{self.video.split('/')[1]}_frame{self.frame}.png"

    @property
    def folder(self) -> str:
        return "dataset_infested" if self.visible else "dataset_free"


def parse_labels(labels_txt: Path) -> list[Crop]:
    crops = []
    with labels_txt.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            m = re.fullmatch(r"frame_(\d+)", r["id"])
            if m is None:
                raise ValueError(f"{labels_txt}:{lineno}: unexpected id {r['id']!r}")
            if r["varroa_visible"] not in ("yes", "no"):
                raise ValueError(f"{labels_txt}:{lineno}: unexpected varroa_visible {r['varroa_visible']!r}")
            box = (*r["coord_1"], *r["coord_2"])
            crops.append(Crop(r["video"], int(m.group(1)), r["varroa_visible"] == "yes", box))
    return crops


def ensure_extracted(raw_dir: Path) -> None:
    if (raw_dir / "labels.txt").exists():
        return
    archive = raw_dir / "dataset.zip"
    if not archive.exists():
        raise FileNotFoundError(f"neither {raw_dir / 'labels.txt'} nor {archive} exists")
    logger.info("extracting {}", archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)


def convert(raw_dir: Path, out_dir: Path, link: bool = True) -> Path:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    ensure_extracted(raw_dir)
    crops = parse_labels(raw_dir / "labels.txt")
    names = Counter(c.file_name for c in crops)
    dupes = [n for n, k in names.items() if k > 1]
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate crop names: {dupes[:3]}")

    img_out = out_dir / "images" / "all"
    img_out.mkdir(parents=True, exist_ok=True)
    rows = []
    stats: Counter = Counter()
    for c in tqdm(crops, desc="ev2", unit="img"):
        src = raw_dir / c.folder / c.file_name
        with Image.open(src) as im:
            width, height = im.size
        _place(src, img_out / c.file_name, link)
        rows.append(
            {
                "image": c.file_name,
                "split": "all",
                "video": c.video,
                "video_class": c.video_class,
                "frame": c.frame,
                "infected": int(c.visible),
                "width": width,
                "height": height,
                "src_x1": c.box[0], "src_y1": c.box[1], "src_x2": c.box[2], "src_y2": c.box[3],
            }
        )
        stats[(c.video_class, c.visible)] += 1

    with (out_dir / "meta.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"path": str(out_dir.resolve()), "val": "images/all", "names": CLASS_NAMES},
            sort_keys=False,
        )
    )
    for (vc, vis), n in sorted(stats.items()):
        logger.info("{:9s} visible={!s:5s} {}", vc, vis, n)
    return data_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ev2"))
    parser.add_argument("--out", type=Path, default=Path("datasets/ev2_bees"))
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args(argv)
    logger.info("wrote {}", convert(args.raw, args.out, link=not args.copy))


if __name__ == "__main__":
    main()
