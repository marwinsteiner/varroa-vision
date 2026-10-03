from pathlib import Path

from varroa_vision.compare import best_epoch, format_table


def test_format_table_handles_missing_and_scales():
    rows = [{"run": "a", "best_ep": 12, "val_mAP50": 0.8194, "test_mAP50": 0.8009, "conf": 0.25, "test_sens": 0.83, "test_spec": 0.968, "test_rate_pred": 32.07, "test_rate_true": 31.34}]
    table = format_table(rows)
    lines = table.splitlines()
    assert lines[0].startswith("| run | best_ep | val_mAP50")
    assert "| a | 12 | 0.819 | 0.801 | 0.250 | 0.830 | 0.968 | 32.1 | 31.3 | - | - |" in lines[2]


def test_best_epoch_uses_fitness(tmp_path: Path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.10\n2,0.9,0.12\n3,0.6,0.30\n"
    )
    assert best_epoch(run) == 3  # 0.1*0.6+0.9*0.30 beats 0.1*0.9+0.9*0.12
    assert best_epoch(tmp_path / "missing") is None
