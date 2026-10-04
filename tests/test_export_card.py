from pathlib import Path

from varroa_vision.export import model_card


def test_model_card_contents():
    card = model_card(Path("runs/mite/yolo11n_320_all/weights/best.pt"), ["a.tflite"], 320, 0.3, 0.1, "8", "mite", {0: "varroa"})
    assert card["run"] == "yolo11n_320_all" and card["imgsz"] == 320 and card["quantize"] == "8"
    assert card["classes"] == {0: "varroa"} and card["conf"] == 0.3
    assert card["aggregation"]["conf"] == 0.3 and card["aggregation"]["min_hits"] == 2
    bee = model_card(Path("runs/bee/x/weights/best.pt"), [], 640, 0.3, 0.1, None, "bee", {0: "bee"})
    assert bee["aggregation"] is None and bee["quantize"] == "float32"
