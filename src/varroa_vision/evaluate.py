"""Evaluate a mite detector at box level and at the level the app cares about: bees.

Box level: ultralytics ``val`` on a split (mAP50, mAP50-95, precision, recall).

Bee level: each image is one bee. A bee is predicted infested if it has at least one
detection at or above a confidence threshold. Against the ``infected`` column of
``meta.csv`` this gives sensitivity, specificity, precision and F1 per threshold, plus
the count statistics that drive the app's output: predicted vs true number of mites and
the resulting mites-per-100-bees rate.

Pick the operating threshold on the val split, report on test:

    python -m varroa_vision.evaluate --weights runs/mite/x/weights/best.pt --split val
    python -m varroa_vision.evaluate --weights runs/mite/x/weights/best.pt --split test --conf 0.35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

DEFAULT_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def predict_split(model, data_yaml: Path, split: str, imgsz: int, device, batch: int, min_conf: float) -> pd.DataFrame:
    """One row per detection (image, conf) plus a row with conf=NaN for images without any."""
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    img_dir = Path(cfg["path"]) / cfg[split]
    rows = []
    for r in model.predict(source=str(img_dir), imgsz=imgsz, device=device, batch=batch, conf=min_conf, stream=True, verbose=False):
        name = Path(r.path).name
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.empty(0)
        if confs.size == 0:
            rows.append({"image": name, "conf": np.nan})
        for c in confs:
            rows.append({"image": name, "conf": float(c)})
    return pd.DataFrame(rows)


def bee_level_table(dets: pd.DataFrame, meta: pd.DataFrame, grid=DEFAULT_GRID) -> pd.DataFrame:
    """Per-threshold confusion and count statistics. ``meta`` needs image, infected, n_boxes."""
    meta = meta.set_index("image")
    missing = set(dets["image"]) - set(meta.index)
    if missing:
        raise KeyError(f"{len(missing)} predicted images not in meta.csv, e.g. {sorted(missing)[:3]}")
    n_bees = len(meta)
    true_pos_bees = meta["infected"].sum()
    true_mites = meta["n_boxes"].sum()
    rows = []
    for t in grid:
        hit = dets[dets["conf"] >= t]
        n_det = hit.groupby("image").size().reindex(meta.index, fill_value=0)
        pred_inf = n_det > 0
        tp = int((pred_inf & (meta["infected"] == 1)).sum())
        fp = int((pred_inf & (meta["infected"] == 0)).sum())
        fn = int((~pred_inf & (meta["infected"] == 1)).sum())
        tn = int((~pred_inf & (meta["infected"] == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-9)
        pred_mites = int(n_det.sum())
        rows.append(
            {
                "conf": float(t),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
                "bees": n_bees,
                "infested_true": int(true_pos_bees),
                "infested_pred": int(pred_inf.sum()),
                "mites_true": int(true_mites),
                "mites_pred": pred_mites,
                "rate_true_per100": 100 * true_mites / n_bees,
                "rate_pred_per100": 100 * pred_mites / n_bees,
                "count_mae": float((n_det - meta["n_boxes"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(weights: Path, data_yaml: Path, split: str, imgsz: int, device, batch: int, conf: float | None, out_dir: Path) -> dict:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    box = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, batch=batch, plots=False, verbose=False, project=str(out_dir), name="box", exist_ok=True)
    box_metrics = {
        "mAP50": float(box.box.map50),
        "mAP50-95": float(box.box.map),
        "precision": float(box.box.mp),
        "recall": float(box.box.mr),
    }
    logger.info("box level ({}): {}", split, box_metrics)

    meta = pd.read_csv(Path(data_yaml).parent / "meta.csv")
    meta = meta[meta["split"] == split]
    dets = predict_split(model, data_yaml, split, imgsz, device, batch, min_conf=0.01)
    dets.to_csv(out_dir / "detections.csv", index=False)
    table = bee_level_table(dets, meta)
    table.to_csv(out_dir / "bee_level.csv", index=False)

    best = table.iloc[table["f1"].idxmax()]
    chosen = table.iloc[(table["conf"] - conf).abs().idxmin()] if conf is not None else best
    summary = {
        "weights": str(weights),
        "split": split,
        "imgsz": imgsz,
        "box": box_metrics,
        "bee_level_best_f1": best.to_dict(),
        "bee_level_at_conf": chosen.to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("bee level, best F1 at conf {:.2f}: sens {:.3f} spec {:.3f} prec {:.3f} F1 {:.3f}", best["conf"], best["sensitivity"], best["specificity"], best["precision"], best["f1"])
    logger.info("rate per 100 bees at conf {:.2f}: true {:.2f} pred {:.2f}", chosen["conf"], chosen["rate_true_per100"], chosen["rate_pred_per100"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/varroa_mite/data.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--conf", type=float, default=None, help="operating threshold to report; default best F1")
    parser.add_argument("--out", type=Path, default=None, help="default runs/eval/<run>/<split>")
    args = parser.parse_args(argv)
    run = args.weights.resolve().parent.parent.name
    out = args.out or Path("runs/eval") / run / args.split
    evaluate(args.weights, args.data, args.split, args.imgsz, args.device, args.batch, args.conf, out)


if __name__ == "__main__":
    main()
