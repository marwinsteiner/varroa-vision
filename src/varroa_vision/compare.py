"""Evaluate several stage 2 runs the same way and print one comparison table.

    python -m varroa_vision.compare runs/mite/a runs/mite/b --device 0
    python -m varroa_vision.compare runs/mite/* --ev2 datasets/ev2_bees/data.yaml

For each run: box mAP50 on val and test, the rate-matched threshold chosen on val, and at
that threshold the test sensitivity, specificity and predicted vs true mites per 100
bees. With ``--ev2`` also the EV2 bee-level sensitivity and specificity at the same
threshold. Existing ``runs/eval/<run>/...`` results are reused unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

COLUMNS = ["run", "best_ep", "val_mAP50", "test_mAP50", "conf", "test_sens", "test_spec", "test_rate_pred", "test_rate_true", "ev2_sens", "ev2_spec"]


def _summary(run: Path, data_yaml: Path, split: str, device, conf: float | None, bee_only: bool, force: bool) -> dict:
    out = Path("runs/eval") / run.name / f"{data_yaml.resolve().parent.name}_{split}"
    path = out / "summary.json"
    if path.exists() and not force:
        s = json.loads(path.read_text())
        stale = split != "all" and "bee_level_best_rate" not in s  # written by an older evaluate
        if not stale and (conf is None or abs(s["bee_level_at_conf"]["conf"] - conf) < 1e-9):
            return s
    from varroa_vision.evaluate import evaluate

    return evaluate(run / "weights" / "best.pt", data_yaml, split, 320, device, 64, conf, out, bee_only=bee_only)


def best_epoch(run: Path) -> int | None:
    results = run / "results.csv"
    if not results.exists():
        return None
    import pandas as pd

    df = pd.read_csv(results)
    df.columns = [c.strip() for c in df.columns]
    fit = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    return int(df["epoch"].iloc[int(fit.idxmax())])


def compare_run(run: Path, data_yaml: Path, ev2_yaml: Path | None, device, force: bool) -> dict:
    val = _summary(run, data_yaml, "val", device, None, False, force)
    conf = float(val.get("bee_level_best_rate", val["bee_level_best_f1"])["conf"])
    test = _summary(run, data_yaml, "test", device, conf, False, force)
    t = test["bee_level_at_conf"]
    row = {
        "run": run.name,
        "best_ep": best_epoch(run),
        "val_mAP50": val["box"]["mAP50"],
        "test_mAP50": test["box"]["mAP50"],
        "conf": conf,
        "test_sens": t["sensitivity"],
        "test_spec": t["specificity"],
        "test_rate_pred": t["rate_pred_per100"],
        "test_rate_true": t["rate_true_per100"],
        "ev2_sens": None,
        "ev2_spec": None,
    }
    if ev2_yaml is not None:
        e = _summary(run, ev2_yaml, "all", device, conf, True, force)["bee_level_at_conf"]
        row["ev2_sens"], row["ev2_spec"] = e["sensitivity"], e["specificity"]
    return row


def format_table(rows: list[dict]) -> str:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}" if v < 10 else f"{v:.1f}"
        return str(v)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in COLUMNS) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("datasets/varroa_mite/data.yaml"))
    parser.add_argument("--ev2", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="also write the table (markdown) and rows (json)")
    args = parser.parse_args(argv)
    rows = []
    for run in args.runs:
        if not (run / "weights" / "best.pt").exists():
            logger.warning("skipping {}: no best.pt", run)
            continue
        rows.append(compare_run(run, args.data, args.ev2, args.device, args.force))
    table = format_table(rows)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
        args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
