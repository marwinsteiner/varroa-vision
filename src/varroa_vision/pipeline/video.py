"""Scan a video: track bees (stage 1), detect mites on each bee crop (stage 2), aggregate.

    python -m varroa_vision.pipeline.video --video comb.mp4 \
        --bee-weights runs/bee/.../best.pt --mite-weights runs/mite/.../best.pt --threshold 3

The two models are duck-typed: anything with ultralytics' ``track`` / ``predict``
interface works, which is how the tests run without weights. The stage 1 bee detector
does not exist yet (see docs/roadmap.md), so this module is exercised only by its tests
until then.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from loguru import logger

from varroa_vision.pipeline.aggregate import (
    AggregationRule,
    FrameObs,
    ScanEstimate,
    aggregate,
    decide,
    group_by_track,
)


def crop_with_margin(img: np.ndarray, xyxy, margin: float = 0.1) -> np.ndarray:
    """Crop ``xyxy`` from ``img`` (H x W x C) with a relative margin, clipped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    xa, ya = max(0, int(x1 - mx)), max(0, int(y1 - my))
    xb, yb = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    return img[ya:yb, xa:xb]


def iter_tracked_bees(bee_model, source, imgsz: int = 640, conf: float = 0.3, tracker: str = "bytetrack.yaml", device=None, vid_stride: int = 1) -> Iterator[tuple[int, np.ndarray, list[tuple[int, tuple[float, float, float, float]]]]]:
    """Yield (frame_index, frame, [(track_id, xyxy), ...]) for every processed frame."""
    stream = bee_model.track(source=source, stream=True, persist=True, imgsz=imgsz, conf=conf, tracker=tracker, device=device, vid_stride=vid_stride, verbose=False)
    for frame_idx, r in enumerate(stream):
        boxes = r.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            yield frame_idx, r.orig_img, []
            continue
        ids = boxes.id.int().tolist()
        xyxy = boxes.xyxy.tolist()
        yield frame_idx, r.orig_img, [(int(t), tuple(b)) for t, b in zip(ids, xyxy)]


def scan(bee_model, mite_model, source, rule: AggregationRule = AggregationRule(), imgsz_bee: int = 640, imgsz_mite: int = 320, bee_conf: float = 0.3, margin: float = 0.1, device=None, vid_stride: int = 1, mite_batch: int = 32) -> ScanEstimate:
    """Run both stages over ``source`` and return the aggregated estimate."""
    records: list[tuple[int, FrameObs]] = []
    min_conf = min(rule.conf, rule.conf_high) * 0.5  # keep weak detections for the sweep
    for frame_idx, frame, bees in iter_tracked_bees(bee_model, source, imgsz_bee, bee_conf, device=device, vid_stride=vid_stride):
        if not bees:
            continue
        crops = [crop_with_margin(frame, xyxy, margin) for _, xyxy in bees]
        confs_per_crop: list[tuple[float, ...]] = []
        for start in range(0, len(crops), mite_batch):
            batch = crops[start : start + mite_batch]
            for r in mite_model.predict(batch, imgsz=imgsz_mite, conf=min_conf, device=device, verbose=False):
                c = r.boxes.conf if r.boxes is not None else None
                confs_per_crop.append(tuple(float(v) for v in (c.tolist() if c is not None else [])))
        for (tid, (x1, y1, x2, y2)), confs in zip(bees, confs_per_crop):
            records.append((tid, FrameObs(frame_idx, min(x2 - x1, y2 - y1), confs)))
    estimate = aggregate(group_by_track(records), rule)
    logger.info("bees {} infested {} mites {} rate {:.2f}/100 CI [{:.2f}, {:.2f}]", estimate.bees, estimate.infested_bees, estimate.mites, estimate.rate_per100, *estimate.ci95_per100)
    return estimate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--bee-weights", type=Path, required=True)
    parser.add_argument("--mite-weights", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=3.0, help="action threshold, mites per 100 bees")
    parser.add_argument("--conf", type=float, default=AggregationRule.conf, help="mite confidence (chosen on val)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="write the estimate as JSON")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    rule = dataclasses.replace(AggregationRule(), conf=args.conf)
    est = scan(YOLO(str(args.bee_weights)), YOLO(str(args.mite_weights)), args.video, rule, device=args.device, vid_stride=args.vid_stride)
    verdict = decide(est, args.threshold)
    payload = {**{k: v for k, v in dataclasses.asdict(est).items() if k != "verdicts"}, "decision": verdict, "threshold_per100": args.threshold}
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps({**payload, "tracks": [dataclasses.asdict(v) for v in est.verdicts]}, indent=2))


if __name__ == "__main__":
    main()
