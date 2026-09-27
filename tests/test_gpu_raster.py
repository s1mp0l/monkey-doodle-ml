"""Батчевый torch-растеризатор (обучение) = эталон doodle.preprocess.rasterize."""
import json
from pathlib import Path

import numpy as np
import torch

from doodle import preprocess as pp
from doodle.gpu_raster import AugParams, StrokeBank, augment, make_aug_table, rasterize_batch

VEC = Path(__file__).parent / "vectors"


def _bank(drawings, device):
    pts, sid, start, length = [], [], [], []
    off = 0
    for d in drawings:
        n = 0
        for k, s in enumerate(d):
            for x, y in zip(s[0], s[1]):
                pts.append((x, y))
                sid.append(k)
                n += 1
        start.append(off)
        length.append(n)
        off += n
    return StrokeBank(np.array(pts, np.uint8).reshape(-1, 2), np.array(sid, np.int16),
                      np.array(start), np.array(length), device)


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_matches_reference_on_vectors():
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(VEC.glob("*.json"))]
    docs = [d for d in docs if d["simplified"] and d["params"]["size"] == 64]
    for dev in _devices():
        bank = _bank([d["simplified"] for d in docs], dev)
        pts, sid, valid = bank.gather(torch.arange(len(docs), device=dev))
        x = rasterize_batch(pts, sid, valid, 64, 2.0, 2.0).cpu().numpy()[:, 0]
        for d, img in zip(docs, x):
            np.testing.assert_allclose(img, np.load(VEC / f"{d['name']}.npy"), atol=1e-4, err_msg=d["name"])


def test_augment_runs_and_keeps_ink():
    rng = np.random.default_rng(1)
    drawings = [pp.normalize([[(rng.random(8) * 200).tolist(), (rng.random(8) * 200).tolist()] for _ in range(k + 1)])
                for k in range(16)]
    bank = _bank(drawings, "cpu")
    pts, sid, valid = bank.gather(torch.arange(16))
    labels = torch.tensor([i % 2 for i in range(16)])
    table = make_aug_table([AugParams(), AugParams.heavy()], "cpu")
    q, v, lw, sh = augment(pts, sid, valid, labels, table, torch.Generator().manual_seed(0))
    x = rasterize_batch(q, sid, v, 64, 2.0, lw, sh)
    assert x.shape == (16, 1, 64, 64)
    assert torch.isfinite(x).all()
    assert (x.flatten(1).amax(1) > 0.5).all()      # у каждого рисунка остались чернила
