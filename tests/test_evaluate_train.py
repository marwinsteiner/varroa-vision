from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from varroa_vision import train
from varroa_vision.evaluate import bee_level_table


def test_load_config_overrides(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: yolo11n.pt\nepochs: 60\nimgsz: 320\n")
    out = train.load_config(cfg, {"epochs": 1, "device": None, "fraction": 0.05})
    assert out == {"model": "yolo11n.pt", "epochs": 1, "imgsz": 320, "fraction": 0.05}


def test_parse_set_types():
    assert train._parse_set(["batch=32", "lr0=0.005", "cos_lr=false", "name=x"]) == {
        "batch": 32,
        "lr0": 0.005,
        "cos_lr": False,
        "name": "x",
    }
    with pytest.raises(SystemExit):
        train._parse_set(["novalue"])


def test_bee_level_table():
    # four bees: a healthy (no dets), b infected 1 mite (det 0.9), c infected 2 mites
    # (dets 0.6 and 0.2), d healthy with a false positive at 0.3
    meta = pd.DataFrame(
        {
            "image": ["a", "b", "c", "d"],
            "infected": [0, 1, 1, 0],
            "n_boxes": [0, 1, 2, 0],
        }
    )
    dets = pd.DataFrame(
        {
            "image": ["a", "b", "c", "c", "d"],
            "conf": [np.nan, 0.9, 0.6, 0.2, 0.3],
        }
    )
    table = bee_level_table(dets, meta, grid=np.array([0.1, 0.5]))

    low = table.iloc[0]
    assert (low["tp"], low["fp"], low["fn"], low["tn"]) == (2, 1, 0, 1)
    assert low["sensitivity"] == 1.0 and low["specificity"] == 0.5
    assert low["mites_true"] == 3 and low["mites_pred"] == 4
    assert low["rate_true_per100"] == pytest.approx(75.0)
    assert low["rate_pred_per100"] == pytest.approx(100.0)

    high = table.iloc[1]
    assert (high["tp"], high["fp"], high["fn"], high["tn"]) == (2, 0, 0, 2)
    assert high["mites_pred"] == 2  # c's second mite at 0.2 is lost
    assert high["count_mae"] == pytest.approx(0.25)
    assert high["f1"] == 1.0


def test_bee_level_table_without_counts():
    meta = pd.DataFrame({"image": ["a", "b"], "infected": [0, 1]})
    dets = pd.DataFrame({"image": ["a", "b"], "conf": [np.nan, 0.7]})
    table = bee_level_table(dets, meta, grid=np.array([0.5]))
    row = table.iloc[0]
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 0, 0, 1)
    assert row["mites_pred"] == 1 and row["rate_pred_per100"] == pytest.approx(50.0)
    assert "mites_true" not in table.columns and "count_mae" not in table.columns


def test_bee_level_table_unknown_image():
    meta = pd.DataFrame({"image": ["a"], "infected": [0], "n_boxes": [0]})
    dets = pd.DataFrame({"image": ["zzz"], "conf": [0.5]})
    with pytest.raises(KeyError, match="not in meta"):
        bee_level_table(dets, meta)
