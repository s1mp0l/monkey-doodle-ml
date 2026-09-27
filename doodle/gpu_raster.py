"""Батчевая растеризация и аугментации на GPU (torch) — та же математика, что doodle/preprocess.rasterize.

Без аугментаций результат совпадает с эталоном с точностью float32 (tests/test_gpu_raster.py).
Данные целиком лежат на GPU: точки всех рисунков (uint8 → float), номера штрихов, начало/длина.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

SCALE_MAX = 255.0


@dataclass
class AugParams:
    """Сила аугментаций. heavy — для маленьких своих классов (rude)."""
    rotate_deg: float = 12.0
    scale: tuple[float, float] = (0.85, 1.15)     # анизотропный масштаб по осям до «подгонки» к 255
    shear: float = 0.15
    flip_p: float = 0.5                           # зеркало по горизонтали
    jitter: float = 1.2                           # σ шума точек, единицы сетки 255
    final_scale: tuple[float, float] = (0.8, 1.0) # уменьшение после подгонки (поля разной ширины)
    shift: float = 8.0                            # сдвиг, единицы сетки 255
    line_width: tuple[float, float] = (1.6, 2.8)  # толщина линии, пиксели выхода
    drop_p: float = 0.3                           # вероятность выкинуть штрихи у длинного рисунка
    drop_stroke_p: float = 0.15                   # доля выкидываемых штрихов
    drop_min_strokes: int = 4

    @staticmethod
    def heavy() -> "AugParams":
        return AugParams(rotate_deg=35.0, scale=(0.7, 1.3), shear=0.3, jitter=2.0, final_scale=(0.6, 1.0),
                         shift=14.0, line_width=(1.4, 3.2), drop_p=0.4, drop_stroke_p=0.2, drop_min_strokes=3)


class StrokeBank:
    """Точки одного split на устройстве + сборка паддингованного батча."""

    def __init__(self, pts, sid, start, length, device):
        self.pts = torch.from_numpy(np.array(pts, dtype=np.float32)).to(device)       # (P, 2)
        self.sid = torch.from_numpy(np.array(sid, dtype=np.int32)).to(device)         # (P,)
        self.start = torch.from_numpy(np.array(start, dtype=np.int64)).to(device)     # (M,)
        self.len = torch.from_numpy(np.array(length, dtype=np.int64)).to(device)      # (M,)
        self.device = device

    def gather(self, idx: torch.Tensor, max_points: int = 1024):
        """→ pts (B, L, 2), sid (B, L), valid (B, L). Рисунки длиннее max_points обрезаются (редкость)."""
        lens = self.len[idx].clamp(max=max_points)
        L = int(lens.max().item()) if len(idx) else 1
        ar = torch.arange(L, device=self.device)
        valid = ar[None, :] < lens[:, None]
        pos = (self.start[idx][:, None] + ar[None, :]).clamp(max=len(self.pts) - 1)
        pts = self.pts[pos]
        sid = torch.where(valid, self.sid[pos], torch.full_like(self.sid[pos], -1))
        return pts, sid, valid


def _bbox(pts, valid):
    big = 1e9
    mn = torch.where(valid[..., None], pts, torch.full_like(pts, big)).amin(dim=1)
    mx = torch.where(valid[..., None], pts, torch.full_like(pts, -big)).amax(dim=1)
    return mn, mx


def make_aug_table(params: list[AugParams], device) -> dict[str, torch.Tensor]:
    """Параметры аугментаций по классам → тензоры (C,) для выборки по меткам батча."""
    tab: dict[str, list[float]] = {}
    for p in params:
        for k, v in vars(p).items():
            if isinstance(v, tuple):
                tab.setdefault(k + "_lo", []).append(float(v[0]))
                tab.setdefault(k + "_hi", []).append(float(v[1]))
            else:
                tab.setdefault(k, []).append(float(v))
    return {k: torch.tensor(v, device=device) for k, v in tab.items()}


def augment(pts, sid, valid, labels, table: dict[str, torch.Tensor], gen: torch.Generator | None = None):
    """Аугментации на штрихах (до растеризации), параметры — по классу каждого элемента.
    Возвращает (pts, valid, line_width (B,), center_shift (B, 2)) для rasterize_batch."""
    B, L, _ = pts.shape
    dev = pts.device

    def per(k):
        return table[k][labels]

    def rnd(*shape):
        return torch.rand(*shape, device=dev, generator=gen)

    def uni(k):
        lo, hi = per(k + "_lo"), per(k + "_hi")
        return lo + (hi - lo) * rnd(B)

    # 1) выкидывание штрихов у длинных рисунков (хотя бы один штрих остаётся)
    n_str = sid.amax(dim=1) + 1                                            # (B,), у невалидных sid = -1
    S = max(int(n_str.max().item()) if B else 1, 1)
    apply = (rnd(B) < per("drop_p")) & (n_str >= per("drop_min_strokes"))
    kill = (rnd(B, S) < per("drop_stroke_p")[:, None]) & apply[:, None]
    kill &= torch.arange(S, device=dev)[None, :] < n_str[:, None]
    kill &= ~(kill.sum(1) >= n_str)[:, None]
    killed_pt = torch.gather(kill, 1, sid.clamp(min=0).long()) & valid
    valid = valid & ~killed_pt

    # 2) аффинное преобразование вокруг центра рамки: M = R · Shear · Scale · Flip
    mn, mx = _bbox(pts, valid)
    q = pts - ((mn + mx) / 2)[:, None, :]
    ang = (rnd(B) * 2 - 1) * per("rotate_deg") * math.pi / 180
    sx, sy = uni("scale"), uni("scale")
    sh = (rnd(B) * 2 - 1) * per("shear")
    flip = torch.where(rnd(B) < per("flip_p"), -1.0, 1.0)
    ca, sa = torch.cos(ang), torch.sin(ang)
    m00, m01 = ca * sx * flip, (ca * sh - sa) * sy
    m10, m11 = sa * sx * flip, (sa * sh + ca) * sy
    x, y = q[..., 0], q[..., 1]
    q = torch.stack([m00[:, None] * x + m01[:, None] * y, m10[:, None] * x + m11[:, None] * y], dim=-1)

    # 3) дрожание точек
    q = q + torch.randn(q.shape, device=dev, generator=gen) * per("jitter")[:, None, None]

    # 4) подгонка как в normalize(): большая сторона рамки = 255·final_scale; сдвиг — отдельно
    mn, mx = _bbox(q, valid)
    ext = (mx - mn).amax(dim=1).clamp(min=1e-6)
    fs = uni("final_scale")
    q = (q - mn[:, None, :]) * (SCALE_MAX / ext * fs)[:, None, None]
    # сдвиг не больше свободного места (чтобы рисунок не обрезался краем) + 3 единицы
    max_shift = torch.minimum(per("shift"), (1 - fs) * SCALE_MAX / 2 + 3)
    shift = (rnd(B, 2) * 2 - 1) * max_shift[:, None]
    return q, valid, uni("line_width"), shift


def rasterize_batch(pts, sid, valid, size: int = 64, pad: float = 2.0, line_width=2.0,
                    center_shift: torch.Tensor | None = None, seg_chunk: int = 16):
    """Эталонная растеризация батчем. pts — нормализованные координаты (сетка 255, Y вниз).
    line_width — число или тензор (B,). center_shift (B, 2) — сдвиг рисунка от центра, единицы сетки.
    → (B, 1, size, size) float32."""
    B, L, _ = pts.shape
    dev = pts.device
    if B == 0:
        return torch.zeros((0, 1, size, size), device=dev)
    mn, mx = _bbox(pts, valid)
    c = (mn + mx) / 2
    if center_shift is not None:
        c = c - center_shift
    f = (size - 2.0 * pad) / SCALE_MAX
    P = size / 2.0 + (pts - c[:, None, :]) * f                                   # (B, L, 2) пиксели
    # отрезок i: (P[i], P[i+1]) если следующий в том же штрихе, иначе вырожденный (P[i], P[i])
    nxt = torch.cat([P[:, 1:], P[:, -1:]], dim=1)
    nsid = torch.cat([sid[:, 1:], torch.full_like(sid[:, :1], -2)], dim=1)
    nval = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1)
    same = valid & nval & (nsid == sid)
    Bp = torch.where(same[..., None], nxt, P)
    A = P
    ys, xs = torch.meshgrid(torch.arange(size, device=dev), torch.arange(size, device=dev), indexing="ij")
    px = (xs.reshape(-1).float() + 0.5)
    py = (ys.reshape(-1).float() + 0.5)
    lens = valid.sum(1)
    order = torch.argsort(lens, descending=True)
    lens_sorted = lens[order]
    best = torch.full((B, size * size), float("inf"), device=dev)
    for s0 in range(0, L, seg_chunk):
        nrows = int((lens_sorted > s0).sum().item())
        if nrows == 0:
            break
        rows = order[:nrows]
        a = A[rows, s0:s0 + seg_chunk]                        # (b, k, 2)
        b = Bp[rows, s0:s0 + seg_chunk]
        v = valid[rows, s0:s0 + seg_chunk]
        abx = (b[..., 0] - a[..., 0])[..., None]
        aby = (b[..., 1] - a[..., 1])[..., None]
        # l2 == 0 (точка): числитель тоже 0 → t = 0, как в эталоне
        inv_l2 = 1.0 / (abx * abx + aby * aby).clamp(min=1e-12)
        apx = px[None, None, :] - a[..., 0][..., None]
        apy = py[None, None, :] - a[..., 1][..., None]
        t = ((apx * abx + apy * aby) * inv_l2).clamp_(0.0, 1.0)
        dx = apx - t * abx
        dy = apy - t * aby
        d = dx * dx + dy * dy                                 # квадрат расстояния; sqrt — после min
        d = d + torch.where(v, 0.0, float("inf"))[..., None]
        best[rows] = torch.minimum(best[rows], d.amin(dim=1))
    if not torch.is_tensor(line_width):
        line_width = torch.full((B,), float(line_width), device=dev)
    r = (line_width / 2.0)[:, None]
    img = (r + 0.5 - torch.sqrt(best)).clamp(0.0, 1.0)
    return img.view(B, 1, size, size)
