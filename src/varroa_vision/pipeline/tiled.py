"""Tiled (sliced) inference for large stills: run a detector on overlapping tiles and
merge the boxes with NMS.

A 12 to 48 MP phone photo of a whole comb resized to 640 px turns a bee into 20 px and a
mite into nothing. Slicing the photo into overlapping tiles at native resolution keeps
the objects at the scale the detector was trained on. This is the SAHI idea without the
dependency; it is only for still photos, the video path tracks on resized frames.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

Box = tuple[float, float, float, float]


def tile_grid(width: int, height: int, tile: int, overlap: float) -> Iterator[tuple[int, int, int, int]]:
    """Yield (x1, y1, x2, y2) tiles covering the image, overlapping by ``overlap`` of a tile.
    Edge tiles are shifted inward so every tile is ``tile`` wide when the image allows."""
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, width - tile + 1), step))
    ys = list(range(0, max(1, height - tile + 1), step))
    if xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    for y in ys:
        for x in xs:
            yield x, y, min(x + tile, width), min(y + tile, height)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def merge_boxes(boxes: list[tuple[Box, float]], iou_thresh: float = 0.5) -> list[tuple[Box, float]]:
    """Greedy NMS over (xyxy, conf) pairs from all tiles; highest confidence wins."""
    if not boxes:
        return []
    arr = np.array([b for b, _ in boxes], dtype=float)
    conf = np.array([c for _, c in boxes], dtype=float)
    order = np.argsort(-conf)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_matrix(arr[i : i + 1], arr[order[1:]])[0]
        order = order[1:][ious < iou_thresh]
    return [(tuple(arr[i].tolist()), float(conf[i])) for i in keep]


def predict_tiled(model, img: np.ndarray, tile: int = 640, overlap: float = 0.2, conf: float = 0.3, iou_thresh: float = 0.5, device=None, imgsz: int | None = None) -> list[tuple[Box, float]]:
    """Run ``model.predict`` on overlapping tiles of ``img`` and return merged
    (xyxy, conf) boxes in image coordinates. Falls back to one pass when the image is
    not larger than a tile."""
    h, w = img.shape[:2]
    imgsz = imgsz or tile
    if w <= tile and h <= tile:
        tiles = [(0, 0, w, h)]
    else:
        tiles = list(tile_grid(w, h, tile, overlap))
    crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
    found: list[tuple[Box, float]] = []
    for (x1, y1, _, _), r in zip(tiles, model.predict(crops, imgsz=imgsz, conf=conf, device=device, verbose=False)):
        if r.boxes is None:
            continue
        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            found.append(((b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1), float(c)))
    return merge_boxes(found, iou_thresh)
