"""Export trained weights for the Android app and for ONNX-based tooling.

    python -m varroa_vision.export --weights runs/mite/x/weights/best.pt --formats saved_model onnx --conf 0.30
    python -m varroa_vision.export --weights ... --formats tflite --quantize 8 --data datasets/varroa_mite/data.yaml

Two routes to a ``.tflite``:

- ``--formats saved_model`` goes ONNX -> onnx2tf -> TensorFlow SavedModel and writes
  ``best_float32.tflite`` and ``best_float16.tflite`` next to it. Works in the training
  environment (``uv sync --extra export-tflite``). This is the route used so far.
- ``--formats tflite`` is ultralytics 8.4's LiteRT path through litert-torch, which pins
  torch < 2.14 and therefore needs the separate environment from
  ``scripts/make_export_env.sh``. It is the route for INT8 (``--quantize 8 --data ...``).

``quantize`` selects precision where the format allows it: unset means float32, ``8``
INT8 (calibration data required), ``w8a16``/``w8a32`` INT8 weights with wider
activations. Parity of an exported model is checked with ``varroa_vision.evaluate`` by
passing the exported file as ``--weights``.

Every export also writes ``model_card.json`` next to the weights: input size, classes,
the operating thresholds chosen on validation, and the per-track aggregation defaults,
so the app reads its runtime constants from one file instead of hard-coding them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from varroa_vision import __version__
from varroa_vision.pipeline.aggregate import AggregationRule


def model_card(weights: Path, outputs: list[str], imgsz: int, conf: float, weak_conf: float, quantize: str | None, stage: str, names: dict[int, str] | None) -> dict:
    """Runtime constants for the app, written beside the exported files."""
    rule = AggregationRule(conf=conf) if stage == "mite" else None
    return {
        "varroa_vision": __version__,
        "exported": datetime.now(UTC).date().isoformat(),
        "stage": stage,
        "source_weights": str(weights),
        "run": weights.resolve().parent.parent.name,
        "outputs": outputs,
        "imgsz": imgsz,
        "quantize": quantize or "float32",
        "batch": 1,
        "classes": names or {},
        "conf": conf,
        "weak_conf": weak_conf,
        "iou": 0.5,
        "aggregation": dataclasses.asdict(rule) if rule else None,
        "note": "conf is the rate-matched threshold chosen on the validation split; see docs/results.md",
    }


def export(weights: Path, formats: list[str], imgsz: int, quantize: str | None, data: Path | None, nms: bool, conf: float = 0.30, weak_conf: float = 0.10, stage: str = "mite") -> list[str]:
    from ultralytics import YOLO

    if quantize in {"8", "w8a16", "w8a32"} and data is None:
        raise SystemExit(f"--quantize {quantize} needs --data for calibration")
    model = YOLO(str(weights))
    outputs = []
    for fmt in formats:
        kwargs: dict = {"format": fmt, "imgsz": imgsz, "nms": nms}
        if quantize:
            kwargs["quantize"] = int(quantize) if quantize.isdigit() else quantize
        if data is not None:
            kwargs["data"] = str(data)
        logger.info("export {} -> {} {}", weights, fmt, kwargs)
        outputs.append(str(model.export(**kwargs)))
    names = {int(k): v for k, v in (getattr(model, "names", None) or {}).items()}
    card_path = Path(weights).parent / "model_card.json"
    card_path.write_text(json.dumps(model_card(Path(weights), outputs, imgsz, conf, weak_conf, quantize, stage, names), indent=2))
    for o in [*outputs, str(card_path)]:
        logger.info("wrote {}", o)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["tflite", "onnx"])
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--quantize", default=None, help="8, w8a16, w8a32, or 16 where the format allows it; default float32")
    parser.add_argument("--data", type=Path, default=None, help="data.yaml for INT8 calibration")
    parser.add_argument("--nms", action="store_true", help="bake NMS into the graph (simpler app code)")
    parser.add_argument("--conf", type=float, default=0.30, help="operating threshold to record in model_card.json")
    parser.add_argument("--weak-conf", type=float, default=0.10)
    parser.add_argument("--stage", choices=("mite", "bee"), default="mite")
    args = parser.parse_args(argv)
    export(args.weights, args.formats, args.imgsz, args.quantize, args.data, args.nms, args.conf, args.weak_conf, args.stage)


if __name__ == "__main__":
    main()
