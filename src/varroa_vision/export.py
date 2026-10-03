"""Export trained weights for the Android app and for ONNX-based tooling.

    python -m varroa_vision.export --weights runs/mite/x/weights/best.pt --formats tflite onnx
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
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def export(weights: Path, formats: list[str], imgsz: int, quantize: str | None, data: Path | None, nms: bool) -> list[str]:
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
    for o in outputs:
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
    args = parser.parse_args(argv)
    export(args.weights, args.formats, args.imgsz, args.quantize, args.data, args.nms)


if __name__ == "__main__":
    main()
