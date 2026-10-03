"""Fine-tune a YOLO detector from a config file.

Usage:
    python -m varroa_vision.train --config configs/mite_yolo11n.yaml
    python -m varroa_vision.train --config configs/mite_yolo11n.yaml --device cpu --epochs 1 --fraction 0.05

Any ``key=value`` pair after the known flags is forwarded to ultralytics, so
``--set batch=32 lr0=0.005`` works for anything not exposed as a flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from loguru import logger


def load_config(path: Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _parse_set(items: list[str] | None) -> dict:
    out: dict = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        out[key] = yaml.safe_load(raw)
    return out


def train(cfg: dict, resume_from: Path | None = None) -> dict:
    """Run ultralytics training and return a summary dict with weights and metrics.

    With ``resume_from`` (a ``last.pt``), training continues from that checkpoint with the
    arguments stored in it; ``cfg`` is ignored except for logging.
    """
    from ultralytics import YOLO

    cfg = dict(cfg)
    model_name = cfg.pop("model")
    if "project" in cfg:
        # ultralytics nests a relative project dir under its own runs dir; keep ours.
        cfg["project"] = str(Path(cfg["project"]).resolve())
    if resume_from is not None:
        # ultralytics restores every argument from the checkpoint; device, batch and
        # imgsz are the ones it allows overriding, which is what moving CPU -> GPU needs.
        keep = {k: cfg[k] for k in ("device", "batch", "workers") if cfg.get(k) is not None}
        logger.info("resuming from {} with {}", resume_from, keep)
        results = YOLO(str(resume_from)).train(resume=True, **keep)
    else:
        logger.info("training {} with {}", model_name, json.dumps(cfg, default=str))
        results = YOLO(model_name).train(**cfg)
    save_dir = Path(results.save_dir)
    summary = {
        "weights": str(save_dir / "weights" / "best.pt"),
        "save_dir": str(save_dir),
        "metrics": {k: float(v) for k, v in results.results_dict.items()},
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("best weights: {}", summary["weights"])
    logger.info("final val metrics: {}", summary["metrics"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", help="override data.yaml path")
    parser.add_argument("--device", help="e.g. 0, cpu, 0,1")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--name", help="run name under the config's project dir")
    parser.add_argument("--fraction", type=float, help="fraction of the train set, for smoke tests")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="extra ultralytics args")
    parser.add_argument("--resume", type=Path, metavar="LAST_PT", help="continue an interrupted run")
    args = parser.parse_args(argv)

    overrides = {
        "data": args.data,
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "name": args.name,
        "fraction": args.fraction,
        "workers": args.workers,
    }
    overrides.update(_parse_set(args.set))
    train(load_config(args.config, overrides), resume_from=args.resume)


if __name__ == "__main__":
    main()
