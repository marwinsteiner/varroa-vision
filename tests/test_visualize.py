import numpy as np

from varroa_vision.pipeline.video import crop_with_margin
from varroa_vision.visualize import annotate_frame, crop_box_to_frame, crop_origin


def test_crop_origin_matches_crop_with_margin():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[150, 210] = (1, 2, 3)  # a marker pixel inside the bee box
    xyxy = (200, 140, 300, 240)
    crop = crop_with_margin(img, xyxy, margin=0.1)
    ox, oy = crop_origin(img.shape, xyxy, margin=0.1)
    assert (ox, oy) == (190, 130)
    assert tuple(crop[150 - oy, 210 - ox]) == (1, 2, 3)
    assert crop_box_to_frame((20.0, 20.0, 30.0, 30.0), (ox, oy)) == (210, 150, 220, 160)


def test_annotate_frame_draws_without_error():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    out = annotate_frame(
        img,
        bees=[((10, 10, 100, 100), 0.9)],
        mites=[((40, 40, 50, 50), 0.6, True), ((60, 60, 70, 70), 0.15, False)],
        counts={"bees": 1, "mites": 1, "bees_total": 5, "infested_total": 1, "mites_total": 1, "rate": 20.0},
    )
    assert out.shape == img.shape and out.any()
