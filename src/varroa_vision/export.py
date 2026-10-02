"""Export trained weights for the Android app and for ONNX-based tooling.

    python -m varroa_vision.export --weights runs/mite/x/weights/best.pt --formats tflite onnx
    python -m varroa_vision.export --weights ... --formats tflite --int8 --data datasets/varroa_mite/data.yaml

TFLite float16 is the default mobile target (runs on the Android GPU delegate). INT8
needs a calibration set, which is why ``--data`` is required with ``--int8``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def export(weights: Path, formats: list[str], imgsz: int, half: bool, int8: bool, data: Path | None, nms: bool) -> list[str]:
    from ultralytics import YOLO

    if int8 and data is None:
        raise SystemExit("--int8 needs --data for calibration")
    model = YOLO(str(weights))
    outputs = []
    for fmt in formats:
        kwargs = {"format": fmt, "imgsz": imgsz, "nms": nms}
        if fmt in {"tflite", "onnx", "ncnn", "engine"}:
            kwargs["half"] = half and not int8
        if fmt == "tflite" and int8:
            kwargs.update(int8=True, data=str(data))
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
    parser.add_argument("--no-half", action="store_true", help="export float32 instead of float16")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--nms", action="store_true", help="bake NMS into the graph (simpler app code)")
    args = parser.parse_args(argv)
    export(args.weights, args.formats, args.imgsz, not args.no_half, args.int8, args.data, args.nms)


if __name__ == "__main__":
    main()
