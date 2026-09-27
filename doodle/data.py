"""Загрузка data/prepared/<profile>/ и сбалансированная по классам выборка."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .gpu_raster import AugParams, StrokeBank


class PreparedData:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        with open(self.root / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.classes = self.meta["classes"]
        self.pre = self.meta["preprocess"]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def arrays(self, split: str) -> dict[str, np.ndarray]:
        r = self.root
        return {k: np.load(r / f"{split}_{k}.npy", mmap_mode="r") for k in ("pts", "sid", "start", "len", "label", "src")}

    def images(self, split: str) -> np.ndarray:
        return np.load(self.root / f"{split}_images.npy", mmap_mode="r")

    def bank(self, split: str, device) -> tuple[StrokeBank, torch.Tensor]:
        a = self.arrays(split)
        bank = StrokeBank(np.asarray(a["pts"]), np.asarray(a["sid"]), np.asarray(a["start"]), np.asarray(a["len"]), device)
        labels = torch.as_tensor(np.asarray(a["label"]).astype(np.int64), device=device)
        return bank, labels

    def aug_params(self) -> list[AugParams]:
        return [AugParams.heavy() if c.get("augment") == "heavy" else AugParams() for c in self.classes]


def class_weights(counts: np.ndarray, sampling_weight: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Доля класса в эпохе: sampling_weight · min(1, n / median)^alpha, нормировано.
    Большие классы уравниваются; маленькие (rude) подтягиваются, но не до равной доли —
    иначе горстка рисунков повторялась бы тысячи раз за эпоху."""
    counts = counts.astype(np.float64)
    med = np.median(counts[counts > 0]) if (counts > 0).any() else 1.0
    w = sampling_weight * np.minimum(1.0, counts / med) ** alpha
    w[counts == 0] = 0
    return w / w.sum()


class BalancedSampler:
    """Индексы эпохи: число примеров класса ~ multinomial(weights); внутри класса — без повторов,
    пока хватает, дальше по кругу (oversampling)."""

    def __init__(self, labels: np.ndarray, num_classes: int, weights: np.ndarray, epoch_size: int, seed: int = 0):
        self.by_class = [np.flatnonzero(labels == c) for c in range(num_classes)]
        self.weights = weights
        self.epoch_size = epoch_size
        self.rng = np.random.default_rng(seed)

    def epoch(self) -> np.ndarray:
        k = self.rng.multinomial(self.epoch_size, self.weights)
        parts = []
        for c, n in enumerate(k):
            pool = self.by_class[c]
            if n == 0 or len(pool) == 0:
                continue
            reps = int(np.ceil(n / len(pool)))
            idx = np.concatenate([self.rng.permutation(pool) for _ in range(reps)])[:n]
            parts.append(idx)
        out = np.concatenate(parts)
        self.rng.shuffle(out)
        return out

    def expected_repeats(self) -> np.ndarray:
        """Сколько раз за эпоху в среднем показывается один пример каждого класса."""
        n = np.array([len(p) for p in self.by_class], dtype=np.float64)
        return np.where(n > 0, self.weights * self.epoch_size / np.maximum(n, 1), 0)
