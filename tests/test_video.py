"""Exercise the video driver with fake models that mimic the ultralytics interface."""

from types import SimpleNamespace

import numpy as np
import torch

from varroa_vision.pipeline.aggregate import AggregationRule
from varroa_vision.pipeline.video import crop_with_margin, scan


class FakeBeeModel:
    """Two bees: track 1 visible in all frames, track 2 only in the first frame."""

    def __init__(self, n_frames=4):
        self.n_frames = n_frames

    def track(self, source, **kw):
        for i in range(self.n_frames):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            if i == 0:
                xyxy = torch.tensor([[100.0, 100.0, 220.0, 200.0], [300.0, 300.0, 340.0, 330.0]])
                ids = torch.tensor([1.0, 2.0])
            else:
                xyxy = torch.tensor([[100.0 + i, 100.0, 220.0 + i, 200.0]])
                ids = torch.tensor([1.0])
            yield SimpleNamespace(orig_img=img, boxes=SimpleNamespace(id=ids, xyxy=xyxy))


class FakeMiteModel:
    """Fires with conf 0.6 on every crop wider than 100 px, never otherwise."""

    def __init__(self):
        self.calls = []

    def predict(self, batch, **kw):
        self.calls.append(len(batch))
        out = []
        for crop in batch:
            conf = torch.tensor([0.6]) if crop.shape[1] > 100 else torch.zeros(0)
            out.append(SimpleNamespace(boxes=SimpleNamespace(conf=conf)))
        return out


def test_crop_with_margin_clips_to_image():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    c = crop_with_margin(img, (0, 0, 50, 40), margin=0.5)
    assert c.shape == (60, 75, 3)  # margin pushes past the top-left edge, clipped at 0
    c2 = crop_with_margin(img, (180, 90, 200, 100), margin=0.5)
    assert c2.shape == (15, 30, 3)  # grows up/left into the image, clipped at the bottom-right


def test_scan_counts_bees_once_and_pools_mites():
    mite = FakeMiteModel()
    rule = AggregationRule(conf=0.35, conf_high=0.8, min_hits=2, min_frames=3, min_bee_px=60)
    est = scan(FakeBeeModel(n_frames=4), mite, "fake.mp4", rule)
    # track 1: 4 frames, 100 px tall box, mite in every frame -> one infested bee
    # track 2: 1 frame, 30 px box -> not counted
    assert est.bees == 1 and est.infested_bees == 1 and est.mites == 1
    assert sum(mite.calls) == 5  # one crop per bee per frame
    by_id = {v.track_id: v for v in est.verdicts}
    assert by_id[1].counted and by_id[1].hits == 4
    assert not by_id[2].counted
