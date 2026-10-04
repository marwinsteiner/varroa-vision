"""Write a data.yaml whose train set is the union of several YOLO datasets.

    python -m varroa_vision.data.combine --base datasets/varroa_mite \
        --extra datasets/ev2_pseudo datasets/internet_mites --out datasets/mite_all

``val`` and ``test`` (and ``meta.csv``, used for bee-level evaluation) come from
``--base`` unless ``--val-from`` names another dataset, so the in-domain numbers of the
base stay comparable across runs while the extra domains only add training images.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from loguru import logger


def _split_dir(dataset: Path, key: str) -> str:
    cfg = yaml.safe_load((Path(dataset) / "data.yaml").read_text())
    return str(Path(dataset).resolve() / cfg.get(key, cfg["val"]))


def write_combined(base: Path, extras: list[Path], out: Path, val_from: Path | None = None, names: dict | None = None, val_extras: list[Path] | None = None) -> Path:
    """``val_extras`` adds other datasets' val splits to validation (a list in
    data.yaml), so model selection sees every training domain, not only the base."""
    base, out = Path(base).resolve(), Path(out)
    val_src = Path(val_from).resolve() if val_from else base
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = yaml.safe_load((base / "data.yaml").read_text())
    val = _split_dir(val_src, "val")
    cfg = {
        "path": str(out.resolve()),
        "train": [_split_dir(base, "train")] + [_split_dir(e, "train") for e in extras],
        "val": [val] + [_split_dir(v, "val") for v in val_extras] if val_extras else val,
        "test": _split_dir(val_src, "test"),
        "names": names or base_cfg["names"],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy(val_src / "meta.csv", out / "meta.csv")
    logger.info("train dirs: {}", cfg["train"])
    return out / "data.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-from", type=Path, default=None)
    parser.add_argument("--val-extra", type=Path, nargs="*", default=None, help="datasets whose val splits join validation")
    args = parser.parse_args(argv)
    logger.info("wrote {}", write_combined(args.base, args.extra, args.out, args.val_from, val_extras=args.val_extra))


if __name__ == "__main__":
    main()
