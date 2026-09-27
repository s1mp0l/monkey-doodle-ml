"""Каноническая предобработка рисунка — ЕДИНСТВЕННАЯ для обучения, редактора и порта на C#.

Спецификация для порта: docs/PREPROCESS.md. Эталонные векторы: tests/vectors/.

Конвейер (все функции чистые, без глобального состояния):

    raw strokes  ──normalize()──►  int strokes 0..255  ──rasterize()──►  float32 [N, N], 0 = фон, 1 = чернила

Формат штрихов (как в Quick, Draw!): список штрихов, штрих = [xs, ys] или [xs, ys, ts];
ось Y направлена ВНИЗ (экранные координаты). Эталон считает в float64; порт может считать
во float32 — расхождение интенсивности пикселей укладывается в 1e-3.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

EPSILON = 2.0        # RDP, в единицах нормализованной сетки
SCALE_MAX = 255.0    # после нормализации большая сторона рамки = 255
SIZE = 64            # N
PAD = 2.0            # поля, пиксели выхода
LINE_WIDTH = 2.0     # толщина линии, пиксели выхода

Strokes = Sequence[Sequence[Sequence[float]]]


# ───────────────────────────────────────────────────────────── вспомогательное

def to_point_arrays(strokes: Strokes) -> list[np.ndarray]:
    """[[xs, ys, (ts)], ...] → список массивов (n, 2) float64. Пустые штрихи и нечисла выкидываются."""
    out = []
    for s in strokes:
        if len(s) < 2:
            continue
        xs = np.asarray(s[0], dtype=np.float64)
        ys = np.asarray(s[1], dtype=np.float64)
        n = min(len(xs), len(ys))
        if n == 0:
            continue
        p = np.stack([xs[:n], ys[:n]], axis=1)
        p = p[np.isfinite(p).all(axis=1)]
        if len(p):
            out.append(p)
    return out


def dedupe(p: np.ndarray) -> np.ndarray:
    """Удалить точки, в точности совпадающие с предыдущей."""
    if len(p) < 2:
        return p
    keep = np.ones(len(p), dtype=bool)
    keep[1:] = np.any(p[1:] != p[:-1], axis=1)
    return p[keep]


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Расстояние от точек p (n, 2) до ОТРЕЗКА ab (с отсечением проекции, не до прямой)."""
    ab = b - a
    l2 = float(ab[0] * ab[0] + ab[1] * ab[1])
    ap = p - a
    if l2 == 0.0:
        return np.sqrt(ap[:, 0] ** 2 + ap[:, 1] ** 2)
    t = (ap[:, 0] * ab[0] + ap[:, 1] * ab[1]) / l2
    t = np.clip(t, 0.0, 1.0)
    dx = ap[:, 0] - t * ab[0]
    dy = ap[:, 1] - t * ab[1]
    return np.sqrt(dx * dx + dy * dy)


def rdp(p: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Ramer–Douglas–Peucker. Итеративно (стек), расстояние — до отрезка, выбирается ПЕРВЫЙ максимум,
    точка сохраняется при d > epsilon (строго). Первая и последняя точки всегда сохраняются."""
    n = len(p)
    if n < 3:
        return p.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d = point_segment_distance(p[i + 1:j], p[i], p[j])
        k = int(np.argmax(d))          # argmax возвращает первый максимум
        if d[k] > epsilon:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return p[keep]


def resample(p: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Шаг Google «resample with 1 px spacing»: вставить точки на каждом отрезке через ≤ spacing.

    В каноническом конвейере НЕ вызывается: расстояние до хорды выпукло вдоль отрезка, поэтому максимум
    RDP всегда достигается в исходной вершине, и ресемплинг перед RDP не меняет результат
    (проверяется в tests/test_preprocess.py::test_resample_is_noop_for_rdp)."""
    if len(p) < 2:
        return p.copy()
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, math.ceil(L / spacing))
        t = (np.arange(1, n + 1, dtype=np.float64) / n)[:, None]
        out.append(a + (b - a) * t)
    return np.concatenate(out, axis=0)


def round_half_up(v: np.ndarray) -> np.ndarray:
    """Округление floor(v + 0.5) — однозначно портируется (Math.Floor(v + 0.5))."""
    return np.floor(v + 0.5)


# ───────────────────────────────────────────────────────────── 1. нормализация

def normalize(strokes: Strokes, epsilon: float = EPSILON) -> list[list[list[int]]]:
    """Нормализация «как Quick Draw simplified».

    1. выкинуть пустые штрихи; 2. рамка по ВСЕМ точкам: min → (0, 0);
    3. равномерный масштаб s = 255 / max(w, h) (если max(w, h) == 0 → s = 1);
    4. по каждому штриху: убрать дубли подряд → RDP(epsilon) → floor(v + 0.5) → убрать дубли подряд.
    Возвращает [[xs], [ys]] целых 0..255 на штрих (штрих из одной точки допустим).
    """
    pts = to_point_arrays(strokes)
    if not pts:
        return []
    allp = np.concatenate(pts, axis=0)
    mn = allp.min(axis=0)
    ext = float((allp.max(axis=0) - mn).max())
    s = SCALE_MAX / ext if ext > 0 else 1.0
    out = []
    for p in pts:
        q = (p - mn) * s
        q = dedupe(q)
        q = rdp(q, epsilon)
        q = dedupe(round_half_up(q))
        out.append([q[:, 0].astype(int).tolist(), q[:, 1].astype(int).tolist()])
    return out


# ───────────────────────────────────────────────────────────── 2. растеризация

def to_pixel_segments(strokes: Strokes, size: int = SIZE, pad: float = PAD) -> np.ndarray:
    """Нормализованные штрихи → отрезки (S, 4) [ax, ay, bx, by] в пикселях выхода.

    Центрирование: центр рамки ((min+max)/2 по всем точкам) → центр изображения (size/2, size/2);
    масштаб фиксированный f = (size − 2·pad) / 255 (НЕ по фактической рамке).
    Штрих из n ≥ 2 точек даёт n−1 отрезков, из одной точки — вырожденный отрезок (a = b)."""
    pts = to_point_arrays(strokes)
    if not pts:
        return np.zeros((0, 4), dtype=np.float64)
    allp = np.concatenate(pts, axis=0)
    c = (allp.min(axis=0) + allp.max(axis=0)) / 2.0
    f = (size - 2.0 * pad) / SCALE_MAX
    segs = []
    for p in pts:
        q = size / 2.0 + (p - c) * f
        if len(q) == 1:
            segs.append(np.concatenate([q, q], axis=1))
        else:
            segs.append(np.concatenate([q[:-1], q[1:]], axis=1))
    return np.concatenate(segs, axis=0)


def rasterize(strokes: Strokes, size: int = SIZE, pad: float = PAD, line_width: float = LINE_WIDTH) -> np.ndarray:
    """Растеризация «расстояние до ближайшего отрезка».

    Центр пикселя (col, row) = (col + 0.5, row + 0.5); d = min по отрезкам расстояния до отрезка;
    I = clamp(line_width/2 + 0.5 − d, 0, 1). Строка 0 — верх (ось Y вниз). float32 [size, size]."""
    segs = to_pixel_segments(strokes, size, pad)
    img = np.zeros((size, size), dtype=np.float32)
    if len(segs) == 0:
        return img
    r = line_width / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    px = (xs.reshape(-1) + 0.5).astype(np.float64)
    py = (ys.reshape(-1) + 0.5).astype(np.float64)
    best = np.full(px.shape, np.inf)
    for s0 in range(0, len(segs), 256):              # порциями — ограничение памяти
        sg = segs[s0:s0 + 256]
        ax, ay, bx, by = (sg[:, i][:, None] for i in range(4))
        abx, aby = bx - ax, by - ay
        l2 = abx * abx + aby * aby
        apx, apy = px[None, :] - ax, py[None, :] - ay
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(l2 > 0, (apx * abx + apy * aby) / np.where(l2 > 0, l2, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        dx, dy = apx - t * abx, apy - t * aby
        best = np.minimum(best, np.sqrt(dx * dx + dy * dy).min(axis=0))
    img[:] = np.clip(r + 0.5 - best, 0.0, 1.0).reshape(size, size)
    return img


# ───────────────────────────────────────────────────────────── целиком

def preprocess(raw_strokes: Strokes, size: int = SIZE, pad: float = PAD, line_width: float = LINE_WIDTH,
               epsilon: float = EPSILON) -> np.ndarray:
    """Сырые штрихи (любые единицы, Y вниз) → float32 [size, size]. Вход модели — [1, 1, size, size]."""
    return rasterize(normalize(raw_strokes, epsilon), size, pad, line_width)


def from_texture_strokes(strokes_px: Iterable[Iterable[Sequence[float]]], texture_height: float) -> list[list[list[float]]]:
    """Штрихи игры (списки точек (x, y) в пикселях Texture2D, Y ВВЕРХ, как в DrawingSurface) →
    штрихи Quick Draw (Y вниз): x' = x, y' = texture_height − y."""
    out = []
    for s in strokes_px:
        pts = list(s)
        if not pts:
            continue
        out.append([[float(p[0]) for p in pts], [float(texture_height - p[1]) for p in pts]])
    return out


def to_png_bytes(img: np.ndarray, invert: bool = True, scale: int = 1) -> bytes:
    """float [H, W] 0..1 → PNG (8-bit grayscale) без сторонних библиотек. invert: тёмные чернила на белом."""
    import struct
    import zlib

    a = np.clip(img, 0.0, 1.0)
    if invert:
        a = 1.0 - a
    g = (a * 255.0 + 0.5).astype(np.uint8)
    if scale > 1:
        g = np.repeat(np.repeat(g, scale, axis=0), scale, axis=1)
    h, w = g.shape
    raw = b"".join(b"\x00" + g[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
