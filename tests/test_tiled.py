from types import SimpleNamespace

import numpy as np
import torch

from varroa_vision.pipeline.tiled import merge_boxes, predict_tiled, tile_grid


def test_tile_grid_covers_image_with_overlap():
    tiles = list(tile_grid(1500, 1000, 640, 0.2))
    assert all(x2 - x1 == 640 and y2 - y1 == 640 for x1, y1, x2, y2 in tiles)
    assert max(x2 for _, _, x2, _ in tiles) == 1500 and max(y2 for _, _, _, y2 in tiles) == 1000
    assert tiles[0] == (0, 0, 640, 640) and tiles[1][0] == 512  # step = 640 * 0.8
    small = list(tile_grid(500, 400, 640, 0.2))
    assert small == [(0, 0, 500, 400)]


def test_merge_boxes_keeps_highest_and_drops_duplicates():
    kept = merge_boxes([((0, 0, 100, 100), 0.6), ((5, 5, 105, 105), 0.9), ((300, 300, 350, 350), 0.4)])
    assert [c for _, c in kept] == [0.9, 0.4]
    assert kept[0][0] == (5.0, 5.0, 105.0, 105.0)
    assert merge_boxes([]) == []


class FakeDetector:
    """Returns one box at the tile centre of every tile, so duplicates arise only where
    tiles overlap on the same object (they don't here) and counts equal tile count."""

    def predict(self, crops, **kw):
        out = []
        for crop in crops:
            h, w = crop.shape[:2]
            out.append(SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[w / 2 - 10, h / 2 - 10, w / 2 + 10, h / 2 + 10]]), conf=torch.tensor([0.8]))))
        return out


def test_predict_tiled_maps_boxes_back_to_image_coordinates():
    img = np.zeros((1000, 1500, 3), dtype=np.uint8)
    boxes = predict_tiled(FakeDetector(), img, tile=640, overlap=0.2)
    tiles = list(tile_grid(1500, 1000, 640, 0.2))
    assert len(boxes) == len(tiles)
    (x1, y1, x2, y2), c = boxes[0]
    assert (x1, y1, x2, y2) == (310.0, 310.0, 330.0, 330.0) and abs(c - 0.8) < 1e-6
    one = predict_tiled(FakeDetector(), np.zeros((400, 500, 3), dtype=np.uint8), tile=640)
    assert len(one) == 1 and one[0][0] == (240.0, 190.0, 260.0, 210.0)
