"""Процедурные простые подтипы rude: свастика (и «SS»), попа, грудь — в сыром формате штрихов Quick Draw.

Геометрия строится в локальных единицах (размах ~1), затем — те же «рука» и «мышь», что у фаллоса
(synth._humanize, synth._mouse_sample): дрожание, перелёт/недолёт на стыках, профиль скорости, паузы.
Калибровки по людям нет (людей этих подтипов мало) — только низкоуровневые распределения (шаг времени,
скорость, паузы, размер) из rude_calibration.json, если она есть. Всё детерминировано от seed.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from . import preprocess as pp
from . import synth
from .synth import CANVAS, _from_quantiles, _humanize, _mouse_sample

# подтип в записи (как в data/custom/rude_subtypes.json) → генератор
SHAPES = ("swastika", "butt", "breasts")
SUBTYPE = {"swastika": "swastika", "butt": "butt", "breasts": "body"}


def _seg(*pts):
    return np.array(pts, dtype=float)


# ───────────────────────────────────────────── свастика

def _swastika(rng) -> list[np.ndarray]:
    if rng.random() < 0.08:
        return _ss(rng)
    th0 = math.radians(rng.choice([0.0, 45.0], p=[0.6, 0.4]) + rng.normal(0, 6) if rng.random() < 0.85
                       else rng.uniform(0, 90))
    chir = 1 if rng.random() < 0.75 else -1                  # обе стороны; «правильная» — чаще
    arm = 0.5 * np.clip(rng.normal(1, 0.12, 4), 0.7, 1.3)
    hook = arm * rng.uniform(0.6, 1.2) * np.clip(rng.normal(1, 0.15, 4), 0.6, 1.4)
    c = rng.normal(0, 0.02, 2)
    ends, tips = [], []
    for k in range(4):
        a = th0 + k * math.pi / 2 + rng.normal(0, 0.06)
        d = np.array([math.cos(a), math.sin(a)])
        e = c + arm[k] * d
        h = a + chir * (math.pi / 2 + rng.normal(0, 0.12))
        ends.append(e)
        tips.append(e + hook[k] * np.array([math.cos(h), math.sin(h)]))
    scheme = rng.choice(["2Z", "4L", "cross+hooks", "Z+2L", "one"], p=[0.4, 0.2, 0.25, 0.1, 0.05])
    if scheme == "2Z":
        out = [_seg(tips[k], ends[k], c, ends[k + 2], tips[k + 2]) for k in (0, 1)]
    elif scheme == "4L":
        out = [_seg(c, ends[k], tips[k]) for k in range(4)]
    elif scheme == "cross+hooks":
        out = [_seg(ends[0], ends[2]), _seg(ends[1], ends[3])] + [_seg(ends[k], tips[k]) for k in range(4)]
    elif scheme == "Z+2L":
        out = [_seg(tips[0], ends[0], c, ends[2], tips[2]), _seg(c, ends[1], tips[1]), _seg(c, ends[3], tips[3])]
    else:
        out = [_seg(tips[0], ends[0], c, ends[1], tips[1], ends[1], c, ends[2], tips[2], ends[2], c, ends[3], tips[3])]
    if rng.random() < 0.25:                                   # порядок штрихов у людей разный
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def _ss(rng) -> list[np.ndarray]:
    """«SS» — две молнии."""
    out, gap = [], rng.uniform(0.45, 0.7)
    for k in range(2):
        x0 = k * gap
        w, h = rng.uniform(0.25, 0.4), rng.uniform(0.9, 1.1)
        out.append(_seg([x0 + w, h / 2], [x0, 0.02 * rng.normal()], [x0 + w, 0.0], [x0, -h / 2]))
    return out


# ───────────────────────────────────────────── попа и грудь

def _blob(c, r, rng, n=48):
    """Неровный круг: радиус чуть «плывёт», начало и охват случайны (недоведённый или перехлёст)."""
    a0 = rng.uniform(0, 2 * math.pi)
    sweep = 2 * math.pi * rng.uniform(0.92, 1.12)
    a = a0 + np.linspace(0, sweep, n) * rng.choice([1, -1])
    rr = r * (1 + 0.08 * np.sin(a * rng.integers(1, 3) + rng.random() * 6))
    sq = rng.uniform(0.85, 1.15)
    return np.stack([c[0] + rr * np.cos(a), c[1] + rr * sq * np.sin(a)], 1)


def _dot(c, r, rng):
    if rng.random() < 0.5:                                    # закрашенная точка-каракуля
        a = np.linspace(0, 2 * math.pi * rng.uniform(2, 4), 40)
        rr = r * np.linspace(1, 0.2, len(a))
        return np.stack([c[0] + rr * np.cos(a), c[1] + rr * np.sin(a)], 1)
    return _blob(c, r, rng, n=16)


def _butt(rng) -> list[np.ndarray]:
    r = 0.5
    sep = r * rng.uniform(1.6, 2.05)
    rr = r * np.clip(rng.normal(1, 0.08, 2), 0.8, 1.2)
    cl, cr = np.array([-sep / 2, 0.0]), np.array([sep / 2, rng.normal(0, 0.04)])
    out = [_blob(cl, rr[0], rng), _blob(cr, rr[1], rng)]
    if rng.random() < 0.8:                                    # дырка/линия между половинами
        mid = (cl + cr) / 2 + np.array([0, rng.normal(0.15, 0.1)])
        out.append(_dot(mid, r * rng.uniform(0.06, 0.14), rng) if rng.random() < 0.7
                   else _seg(mid + [0, -0.25], mid + [0, 0.25]))
    return out


def _breasts(rng) -> list[np.ndarray]:
    r = 0.5
    sep = r * rng.uniform(2.0, 2.6)
    rr = r * np.clip(rng.normal(1, 0.08, 2), 0.8, 1.2)
    cs = [np.array([-sep / 2, 0.0]), np.array([sep / 2, rng.normal(0, 0.05)])]
    out = [_blob(c, rk, rng) for c, rk in zip(cs, rr)]
    off = rng.normal(0, 0.12, 2) * r
    for c in cs:
        out.append(_dot(c + off + [0, -0.1 * r], r * rng.uniform(0.1, 0.28), rng))
    return out


GEN = {"swastika": _swastika, "butt": _butt, "breasts": _breasts}


# ───────────────────────────────────────────── «рука» и «мышь» → запись

def render(shape: str, rng, low: dict) -> tuple[list, dict]:
    parts = GEN[shape](rng)
    rot = rng.normal(0, 12) if (shape != "swastika" and rng.random() < 0.85) else (
        rng.uniform(-180, 180) if shape != "swastika" else 0.0)
    th = math.radians(rot)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    # локальные (x вправо, y вверх) → холст (Y вниз)
    world = [np.stack([p[:, 0], -p[:, 1]], 1) @ R.T for p in parts]
    allp = np.concatenate(world)
    mn, mx = allp.min(0), allp.max(0)
    u = rng.random()
    size = rng.uniform(0.05, 0.14) if u < 0.1 else rng.uniform(0.75, 0.97) if u < 0.18 else \
        min(max(_from_quantiles(rng, low["size_q"], 0.05), 0.04), 0.97)
    scale = size * CANVAS / max(float((mx - mn).max()), 1e-6)
    span = (mx - mn) * scale
    off = np.array([rng.uniform(4, max(5, CANVAS - 4 - span[0])), rng.uniform(4, max(5, CANVAS - 4 - span[1]))])
    P = {"wobble": float(rng.uniform(0.003, 0.02)), "join_err": float(rng.uniform(0.0, 0.04))}
    speed = _from_quantiles(rng, low["speed_q"], 0.08) * float(np.exp(rng.normal(0, 0.25)))
    strokes, t = [], 0.0
    for q in world:
        path = np.clip(_humanize((q - mn) * scale + off, P, size * CANVAS, rng), 0, CANVAS - 1)
        if rng.random() < 0.2:
            path = path[::-1]
        xs, ys, ts = _mouse_sample(path, speed * rng.uniform(0.85, 1.2), low["dt"], rng, t)
        strokes.append([[round(v, 1) for v in xs], [round(v, 1) for v in ys], [int(round(v)) for v in ts]])
        t = ts[-1] + _from_quantiles(rng, low["pause_q"], 60)
    return strokes, {"rotation_deg": round(float(rot), 1), "size": round(float(size), 3), **P}


def generate(counts: dict[str, int], seed: int = 0, calibration: dict | None = None) -> list[dict]:
    """counts: {"swastika": n, "butt": n, "breasts": n} → записи rude (source: synthetic, subtype)."""
    low = (calibration or {}).get("low") or synth.DEFAULT_LOW
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    out = []
    for si, shape in enumerate(SHAPES):
        for i in range(int(counts.get(shape, 0))):
            r = np.random.default_rng([seed, 1000 + si, i])
            d, info = render(shape, r, low)
            out.append({
                "key_id": f"syn-{shape[:3]}-{seed:04x}-{i:06x}",
                "word": "rude",
                "author": "synth",
                "timestamp": now,
                "prompt": shape,
                "device": "mouse",
                "canvas": {"w": int(CANVAS), "h": int(CANVAS), "format": "square"},
                "source": "synthetic",
                "subtype": SUBTYPE[shape],
                "generator": {"name": "synth_shapes", "version": 1, "seed": seed, "index": i, "shape": shape,
                              **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}},
                "drawing": d,
                "simplified": pp.normalize(d),
            })
    return out
