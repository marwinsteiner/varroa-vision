"""Run both stages over a folder of frames and render annotated frames, an MP4 and a summary.

    python -m varroa_vision.visualize --images datasets/beehive_bees/images/test \
        --bee-weights runs/bee/<run>/weights/best.pt --mite-weights runs/mite/<run>/weights/best.pt \
        --conf 0.30 --out runs/viz/beehive_test

Per frame: green boxes are bees (stage 1), red boxes are mites at or above ``--conf``
(stage 2, run on each bee crop and mapped back to frame coordinates), orange boxes are
weaker mite detections between ``--weak-conf`` and ``--conf`` shown for context only.
The overlay carries per-frame and cumulative counts. Frames are processed in sorted
order; tracking is not used here because BEEHIVE frames are not one continuous video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from varroa_vision.pipeline.video import crop_with_margin

GREEN, RED, ORANGE, WHITE, BLACK = (80, 200, 80), (40, 40, 230), (30, 150, 255), (255, 255, 255), (0, 0, 0)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def crop_box_to_frame(box_xyxy_in_crop, crop_origin_xy) -> tuple[int, int, int, int]:
    """Map a box from crop pixel coordinates back to frame coordinates."""
    ox, oy = crop_origin_xy
    x1, y1, x2, y2 = box_xyxy_in_crop
    return int(round(x1 + ox)), int(round(y1 + oy)), int(round(x2 + ox)), int(round(y2 + oy))


def crop_origin(img_shape, xyxy, margin: float) -> tuple[int, int]:
    """Top-left corner of ``crop_with_margin`` for the same inputs."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return max(0, int(x1 - margin * (x2 - x1))), max(0, int(y1 - margin * (y2 - y1)))


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, max(0, y - th - 6)), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE if color != ORANGE else BLACK, 1, cv2.LINE_AA)


def annotate_frame(img, bees, mites, counts: dict) -> np.ndarray:
    """``bees``: [(xyxy, conf)], ``mites``: [(xyxy, conf, strong)] in frame coordinates."""
    out = img.copy()
    for (x1, y1, x2, y2), c in bees:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
        _label(out, f"bee {c:.2f}", int(x1), int(y1), GREEN)
    for (x1, y1, x2, y2), c, strong in mites:
        color = RED if strong else ORANGE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _label(out, f"mite {c:.2f}", x1, y1, color)
    lines = [
        f"frame: bees {counts['bees']}  mites {counts['mites']}",
        f"total: bees {counts['bees_total']}  bees with mites {counts['infested_total']}  mites {counts['mites_total']}",
        f"mites per 100 bee sightings: {counts['rate']:.1f}",
    ]
    y = 22
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (6, y - th - 6), (12 + tw, y + 6), BLACK, -1)
        cv2.putText(out, line, (9, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
        y += th + 14
    return out


def run(images: list[Path] | Path, bee_weights: Path | None, mite_weights: Path, out: Path, conf: float, weak_conf: float, bee_conf: float, imgsz_bee: int, imgsz_mite: int, margin: float, device, fps: int, max_frames: int | None, save_frames: str = "all", video: bool = True) -> dict:
    """``bee_weights=None`` skips stage 1 and treats every image as one bee crop (for
    datasets that are already single-bee crops). ``save_frames`` is ``all``,
    ``detections`` (only frames with at least a weak detection) or ``none``."""
    from ultralytics import YOLO

    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    bee_model = YOLO(str(bee_weights)) if bee_weights else None
    mite_model = YOLO(str(mite_weights))
    dirs = [Path(images)] if isinstance(images, (str, Path)) else [Path(d) for d in images]
    paths = sorted(p for d in dirs for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_frames:
        paths = paths[:max_frames]
    totals = {"bees_total": 0, "infested_total": 0, "mites_total": 0}
    per_frame = []
    writer = None
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if bee_model is None:
            h, w = img.shape[:2]
            bees = [((0.0, 0.0, float(w), float(h)), 1.0)]
            use_margin = 0.0
        else:
            r = bee_model.predict(img, imgsz=imgsz_bee, conf=bee_conf, device=device, verbose=False)[0]
            bees = [(tuple(b), float(c)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())] if r.boxes is not None else []
            use_margin = margin
        mites = []
        infested = 0
        if bees:
            crops = [crop_with_margin(img, xyxy, use_margin) for xyxy, _ in bees]
            origins = [crop_origin(img.shape, xyxy, use_margin) for xyxy, _ in bees]
            results = mite_model.predict(crops, imgsz=imgsz_mite, conf=weak_conf, device=device, verbose=False)
            for mr, origin in zip(results, origins):
                strong_here = 0
                if mr.boxes is not None:
                    for b, c in zip(mr.boxes.xyxy.tolist(), mr.boxes.conf.tolist()):
                        strong = c >= conf
                        strong_here += strong
                        mites.append((crop_box_to_frame(b, origin), float(c), strong))
                infested += strong_here > 0
        n_strong = sum(1 for m in mites if m[2])
        totals["bees_total"] += len(bees)
        totals["infested_total"] += infested
        totals["mites_total"] += n_strong
        counts = {"bees": len(bees), "mites": n_strong, **totals, "rate": 100 * totals["mites_total"] / max(totals["bees_total"], 1)}
        want_frame = save_frames == "all" or (save_frames == "detections" and mites)
        if want_frame or video:
            frame = annotate_frame(img, [] if bee_model is None else bees, mites, counts)
        if want_frame:
            cv2.imwrite(str(out / "frames" / f"{i:04d}_{p.stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        per_frame.append({"index": i, "image": p.name, "dir": p.parent.name, "bees": len(bees), "infested": infested, "mites": n_strong, "weak_mites": len(mites) - n_strong, "mite_confs": [round(m[1], 3) for m in mites]})
    if writer is not None:
        writer.release()
    summary = {"frames": len(per_frame), "conf": conf, "weak_conf": weak_conf, **totals, "rate_per100": 100 * totals["mites_total"] / max(totals["bees_total"], 1), "per_frame": per_frame}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("{} frames: bee sightings {}, bees with mites {}, mites {} -> {:.1f} per 100 sightings", len(per_frame), totals["bees_total"], totals["infested_total"], totals["mites_total"], summary["rate_per100"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="one or more folders of frames")
    parser.add_argument("--bee-weights", type=Path, default=None, help="stage 1 weights; omit for datasets of single-bee crops")
    parser.add_argument("--mite-weights", type=Path, required=True)
    parser.add_argument("--save-frames", choices=("all", "detections", "none"), default="all")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--weak-conf", type=float, default=0.10)
    parser.add_argument("--bee-conf", type=float, default=0.30)
    parser.add_argument("--imgsz-bee", type=int, default=640)
    parser.add_argument("--imgsz-mite", type=int, default=320)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    run(args.images, args.bee_weights, args.mite_weights, args.out, args.conf, args.weak_conf, args.bee_conf, args.imgsz_bee, args.imgsz_mite, args.margin, args.device, args.fps, args.max_frames, save_frames=args.save_frames, video=not args.no_video)


if __name__ == "__main__":
    main()
