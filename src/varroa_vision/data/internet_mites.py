"""Build a mite-detection dataset from Roboflow Universe exports of internet photos.

These projects mix sources (comb photos, mites on pupae, macro shots, sticky boards)
and ship augmented copies of each image. This module:

1. keeps one copy per source image (Roboflow names copies ``<stem>.rf.<hash>``), and
   drops re-uploaded VarroaDataset crops (``bee_id`` in the name);
2. maps every mite-like class to ``0 = varroa`` and ignores the rest;
3. turns each image into stage 2 training crops at a bee-like scale:
   - a crop around every bee box (with margin) where the project has bee boxes,
   - otherwise a window of 6 to 10 times the mite box around each mite, jittered,
   - plus up to two random windows per image that contain no mite, as backgrounds;
4. splits by source image with a deterministic hash (default 80/20 train/test) and
   writes ``data.yaml``, ``meta.csv`` (source, stem, window kind, infected, n_boxes) and
   ``stats.json``.

    python -m varroa_vision.data.internet_mites --projects beproj_internet bolo_beehive sammy_var_sahi hofer_bees
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import cv2
import yaml
from loguru import logger

from varroa_vision.data.varroa_dataset import CLASS_NAMES

MITE_CLASSES = {"varroa", "varroa mites", "varroa_mite", "mite", "varroa-mite", "varroa mite"}
BEE_CLASSES = {"bee", "bees", "honey bee", "honeybee", "queen"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def source_stem(file_name: str) -> str:
    return RF_SUFFIX.sub("", Path(file_name).stem)


def split_for(stem: str, test_fraction: float, seed: int) -> str:
    h = hashlib.sha1(f"{seed}:{stem}".encode()).digest()
    return "test" if int.from_bytes(h[:8], "big") / 2**64 < test_fraction else "train"


def read_labels(path: Path, names: list[str]) -> list[tuple[str, float, float, float, float]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((names[int(p[0])], *map(float, p[1:5])))
    return out


def yolo_to_px(box, W: int, H: int) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H


def unique_images(project_dir: Path) -> list[tuple[Path, Path, list[str]]]:
    """(image, label, names) for one copy per source stem across all splits."""
    names = yaml.safe_load((project_dir / "data.yaml").read_text())["names"]
    seen, out = set(), []
    for split in ("train", "valid", "test"):
        img_dir = project_dir / split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = source_stem(img.name)
            if stem in seen or "bee_id" in img.name:
                continue
            seen.add(stem)
            out.append((img, project_dir / split / "labels" / f"{img.stem}.txt", names))
    return out


def boxes_in_window(boxes, win, min_inside: float = 0.5):
    """Boxes (x1,y1,x2,y2 px) clipped to ``win`` and converted to YOLO lines; a box is
    kept if at least ``min_inside`` of its area lies inside the window."""
    wx1, wy1, wx2, wy2 = win
    ww, wh = wx2 - wx1, wy2 - wy1
    lines, kept = [], 0
    for x1, y1, x2, y2 in boxes:
        ix1, iy1, ix2, iy2 = max(x1, wx1), max(y1, wy1), min(x2, wx2), min(y2, wy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) < min_inside * (x2 - x1) * (y2 - y1):
            continue
        cx, cy = ((ix1 + ix2) / 2 - wx1) / ww, ((iy1 + iy2) / 2 - wy1) / wh
        lines.append(f"0 {cx:.6f} {cy:.6f} {(ix2 - ix1) / ww:.6f} {(iy2 - iy1) / wh:.6f}")
        kept += 1
    return lines, kept


def clamp_window(cx, cy, side, W, H):
    side = min(side, W, H)
    x1 = int(min(max(0, cx - side / 2), W - side))
    y1 = int(min(max(0, cy - side / 2), H - side))
    return x1, y1, int(x1 + side), int(y1 + side)


def windows_for_image(W, H, mites, bees, rng: random.Random, margin: float = 0.15, k_range=(6.0, 10.0), n_neg: int = 2, min_side: int = 160, max_mite_windows: int = 10):
    """Yield (kind, window) for one image: bee crops, mite windows, negative windows.
    Mite windows are capped per image so sticky boards with dozens of mites do not
    swamp the comb photos."""
    wins = []
    for x1, y1, x2, y2 in bees:
        mx, my = margin * (x2 - x1), margin * (y2 - y1)
        wins.append(("bee", (int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(W, x2 + mx)), int(min(H, y2 + my)))))
    if not bees:
        for x1, y1, x2, y2 in rng.sample(mites, min(len(mites), max_mite_windows)):
            side = max(min_side, rng.uniform(*k_range) * max(x2 - x1, y2 - y1))
            cx = (x1 + x2) / 2 + rng.uniform(-0.25, 0.25) * side
            cy = (y1 + y2) / 2 + rng.uniform(-0.25, 0.25) * side
            wins.append(("mite", clamp_window(cx, cy, side, W, H)))
    sides = [w[2] - w[0] for _, w in wins] or [min(W, H)]
    for _ in range(n_neg * 4):
        if sum(1 for k, _ in wins if k == "negative") >= n_neg:
            break
        side = rng.choice(sides)
        win = clamp_window(rng.uniform(0, W), rng.uniform(0, H), side, W, H)
        if not any(max(x1, win[0]) < min(x2, win[2]) and max(y1, win[1]) < min(y2, win[3]) for x1, y1, x2, y2 in mites):
            wins.append(("negative", win))
    return wins


def build(project_dirs: dict[str, Path], out_dir: Path, test_fraction: float = 0.2, seed: int = 0, val_fraction: float = 0.1) -> Path:
    """``val_fraction`` of the non-test source images become a val split (hashed from the
    train portion, so the test split is unchanged by the choice of val_fraction)."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    for s in ("train", "val", "test"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
    rows, stats = [], {"train": Counter(), "val": Counter(), "test": Counter()}
    for source, pdir in project_dirs.items():
        for img_path, lbl_path, names in unique_images(Path(pdir)):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            labels = read_labels(lbl_path, names)
            mites = [yolo_to_px(b[1:], W, H) for b in labels if b[0].lower() in MITE_CLASSES]
            bees = [yolo_to_px(b[1:], W, H) for b in labels if b[0].lower() in BEE_CLASSES]
            if not mites and not bees:
                continue
            stem = source_stem(img_path.name)
            split = split_for(f"{source}/{stem}", test_fraction, seed)
            if split == "train" and val_fraction > 0 and split_for(f"val:{source}/{stem}", val_fraction / (1 - test_fraction), seed) == "test":
                split = "val"
            for j, (kind, win) in enumerate(windows_for_image(W, H, mites, bees, rng)):
                x1, y1, x2, y2 = win
                if x2 - x1 < 32 or y2 - y1 < 32:
                    continue
                lines, n = boxes_in_window(mites, win)
                if kind == "negative" and n:
                    continue
                name = f"{source}__{stem}__{kind}{j}"
                cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out_dir / "labels" / split / f"{name}.txt").write_text("".join(l + "\n" for l in lines))
                st = stats[split]
                st["images"] += 1
                st[f"{kind}_windows"] += 1
                st["boxes"] += n
                st["infected"] += int(n > 0)
                rows.append({"image": f"{name}.jpg", "split": split, "source": source, "stem": stem, "kind": kind, "infected": int(n > 0), "n_boxes": n, "width": x2 - x1, "height": y2 - y1})
    with (out_dir / "meta.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "stats.json").write_text(json.dumps({s: dict(c) for s, c in stats.items()}, indent=2))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASS_NAMES}, sort_keys=False))
    for s, c in stats.items():
        logger.info("{:5s} {}", s, dict(c))
    return data_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projects", nargs="+", default=["beproj_internet", "bolo_beehive", "sammy_var_sahi", "hofer_bees"], help="keys under --raw")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/roboflow"))
    parser.add_argument("--out", type=Path, default=Path("datasets/internet_mites"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logger.info("wrote {}", build({k: args.raw / k for k in args.projects}, args.out, args.test_fraction, args.seed))


if __name__ == "__main__":
    main()
