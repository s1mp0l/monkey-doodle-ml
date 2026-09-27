"""Офлайн-аугментация рисунков людей и составные рисунки (композиции) — в сыром формате штрихов.

Все преобразования работают с сырыми штрихами на холсте 800×800 (Y вниз) и заново «проводят мышью»
изменённые линии (доли секунд между точками, паузы между штрихами — из калибровки), поэтому плотность
точек, упрощение RDP и дрожание соответствуют новому размеру рисунка.

Преобразования (каждое записывается в поле `transforms` записи):
    fill:<hatch|scribble|spiral|cross>   заливка замкнутой области отдельными штрихами, с лёгким вылетом
    size:<tiny|small|medium|huge>        реальный размер на холсте (сторона рамки 20–50 / 50–120 / 120–350 / 350–760 px)
    affine:rot=..,shear=..               поворот и сдвиг
    jitter                               плавное дрожание линий
    dropout / order / reverse            выкинуть штрих (у длинных рисунков), переставить, развернуть
    text:spacing=..,slant=..,wave=..     для надписей: межбуквенный интервал, наклон строки, волна базовой линии
"""
from __future__ import annotations

import math

import numpy as np

from . import preprocess as pp
from .synth import DEFAULT_LOW, _from_quantiles, _mouse_sample, _smooth_noise

CANVAS = 800.0
SIZE_BUCKETS = {"tiny": (20, 50), "small": (50, 120), "medium": (120, 350), "huge": (350, 760)}
FILLS = ("hatch", "scribble", "spiral", "cross")
NO_FILL = ("word", "hate", "swastika")          # линейные символы и надписи не заливают


def _paths(drawing):
    return [np.stack([np.asarray(s[0], float), np.asarray(s[1], float)], 1) for s in drawing if len(s[0])]


def _bbox(paths):
    a = np.concatenate(paths)
    return a.min(0), a.max(0)


def _retime(paths, rng, low, speed_mul=1.0):
    """Пути (px) → сырые штрихи [[xs],[ys],[ts]] с «мышиным» временем."""
    out, t = [], 0.0
    speed = _from_quantiles(rng, low["speed_q"], 0.08) * speed_mul
    for p in paths:
        p = np.clip(p, 0, CANVAS - 1)
        if len(p) == 1 or float(np.hypot(*(p[-1] - p[0]))) == 0 and len(p) < 3:
            out.append([[round(float(p[0, 0]), 1)], [round(float(p[0, 1]), 1)], [int(round(t))]])
            t += 40
        else:
            xs, ys, ts = _mouse_sample(p, speed * rng.uniform(0.85, 1.2), low["dt"], rng, t)
            out.append([[round(v, 1) for v in xs], [round(v, 1) for v in ys], [int(round(v)) for v in ts]])
            t = ts[-1]
        t += _from_quantiles(rng, low["pause_q"], 60)
    return out


# ───────────────────────────────────────────── геометрия

def fit_size(paths, rng, bucket: str):
    lo, hi = SIZE_BUCKETS[bucket]
    mn, mx = _bbox(paths)
    ext = float((mx - mn).max()) or 1.0
    target = rng.uniform(lo, hi)
    s = target / ext
    span = (mx - mn) * s
    off = np.array([rng.uniform(4, max(5, CANVAS - 4 - span[0])), rng.uniform(4, max(5, CANVAS - 4 - span[1]))])
    return [(p - mn) * s + off for p in paths]


def affine(paths, rng, rot_deg, shear):
    mn, mx = _bbox(paths)
    c = (mn + mx) / 2
    th = math.radians(rot_deg)
    M = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) @ np.array([[1, shear], [0, 1]])
    return [(p - c) @ M.T + c for p in paths]


def jitter(paths, rng, amp_frac=0.006):
    mn, mx = _bbox(paths)
    amp = amp_frac * float((mx - mn).max() or 1)
    out = []
    for p in paths:
        if len(p) < 3:
            out.append(p)
            continue
        n = len(p)
        out.append(p + np.stack([_smooth_noise(rng, n, amp), _smooth_noise(rng, n, amp)], 1))
    return out


def text_layout(paths, rng, spacing, slant, wave):
    """Надпись: штрихи раздвигаются/сдвигаются по X пропорционально позиции (интервал), строка наклоняется,
    базовая линия идёт волной. Штрихи одной буквы двигаются вместе (группировка по пересечению по X)."""
    mn, mx = _bbox(paths)
    w = float(mx[0] - mn[0]) or 1.0
    h = float(mx[1] - mn[1]) or 1.0
    cx = [float(p[:, 0].mean()) for p in paths]
    out = []
    for p, x in zip(paths, cx):
        u = (x - mn[0]) / w                                   # 0..1 — позиция в строке
        dx = (spacing - 1.0) * u * w
        dy = slant * u * w + wave * h * math.sin(2 * math.pi * u * rng.uniform(0.8, 1.6))
        out.append(p + np.array([dx, dy]))
    return out


# ───────────────────────────────────────────── заливка

def _closed_region(paths):
    """Самый крупный почти замкнутый штрих (конец рядом с началом) → многоугольник, иначе None."""
    best, area = None, 0.0
    for p in paths:
        if len(p) < 6:
            continue
        ext = float((p.max(0) - p.min(0)).max())
        if ext <= 0 or np.hypot(*(p[-1] - p[0])) > 0.25 * ext:
            continue
        x, y = p[:, 0], p[:, 1]
        a = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if a > area:
            best, area = p, a
    return best


def _inside(poly, pts):
    x, y = pts[:, 0], pts[:, 1]
    px, py = poly[:, 0], poly[:, 1]
    inside = np.zeros(len(pts), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        inside ^= cond
        j = i
    return inside


def fill_strokes(poly, rng, kind: str, overflow: float = 0.06):
    """Штрихи заливки внутри многоугольника, как мышью: зигзаг, туда-сюда каракули, спираль, крест-накрест."""
    c = poly.mean(0)
    grow = poly + (poly - c) * overflow                      # лёгкий вылет за контур
    mn, mx = grow.min(0), grow.max(0)
    ext = float((mx - mn).max())
    step = ext * rng.uniform(0.05, 0.1)
    strokes = []

    def hatch(angle):
        d = np.array([math.cos(angle), math.sin(angle)])
        nrm = np.array([-d[1], d[0]])
        span = ext * 1.5
        offs = np.arange(-span / 2, span / 2, step)
        zig = []
        for k, o in enumerate(offs):
            line = c + nrm * o + np.outer(np.linspace(-span / 2, span / 2, 60), d)
            m = _inside(grow, line)
            if m.sum() < 2:
                continue
            seg = line[m]
            seg = seg if k % 2 == 0 else seg[::-1]
            zig.append(seg[[0, -1]])
        return np.concatenate(zig) if zig else None

    if kind in ("hatch", "cross"):
        a0 = rng.uniform(0, math.pi)
        z = hatch(a0)
        if z is not None:
            strokes.append(z)
        if kind == "cross":
            z = hatch(a0 + rng.uniform(1.2, 1.9))
            if z is not None:
                strokes.append(z)
    elif kind == "scribble":
        pts = [c]
        for _ in range(int(rng.integers(25, 60))):
            for _try in range(8):
                q = pts[-1] + rng.normal(0, ext * 0.25, 2)
                if _inside(grow, q[None])[0]:
                    pts.append(q)
                    break
        if len(pts) > 3:
            strokes.append(np.array(pts))
    elif kind == "spiral":
        turns = rng.uniform(3, 6)
        t = np.linspace(0, 1, int(80 * turns))
        ang = 2 * math.pi * turns * t
        r = (1 - t)[:, None] * (grow - c).max(0) * 0.9
        sp = c + np.stack([np.cos(ang), np.sin(ang)], 1) * r
        m = _inside(grow, sp)
        if m.sum() > 5:
            strokes.append(sp[m])
    return strokes


# ───────────────────────────────────────────── вариант аугментации

def augment_one(drawing, rng, subtype: str, low: dict | None = None, spec: dict | None = None):
    """Один аугментированный вариант. spec: явный набор {"fill": .., "size": .., "rot": .., ...};
    без spec — случайный набор. → (сырые штрихи, список преобразований)."""
    low = low or DEFAULT_LOW
    spec = dict(spec or {})
    paths = _paths(drawing)
    tr = []
    if not paths:
        return drawing, tr
    is_text = subtype in ("word",)
    if spec.get("text", is_text and rng.random() < 0.8):
        sp, sl, wv = rng.uniform(0.8, 1.35), rng.uniform(-0.12, 0.12), rng.uniform(0, 0.12)
        paths = text_layout(paths, rng, sp, sl, wv)
        tr.append(f"text:spacing={sp:.2f},slant={sl:.2f},wave={wv:.2f}")
    if len(paths) >= 4 and spec.get("dropout", rng.random() < 0.2):
        k = int(rng.integers(len(paths)))
        paths = paths[:k] + paths[k + 1:]
        tr.append("dropout")
    rot = spec.get("rot", float(rng.normal(0, 12) if rng.random() < 0.7 else rng.uniform(-60, 60)))
    shear = spec.get("shear", float(rng.uniform(-0.2, 0.2)))
    if is_text:
        rot = float(np.clip(rot, -20, 20))
    paths = affine(paths, rng, rot, shear)
    tr.append(f"affine:rot={rot:.0f},shear={shear:.2f}")
    if spec.get("jitter", rng.random() < 0.6):
        paths = jitter(paths, rng)
        tr.append("jitter")
    fill = spec.get("fill", rng.choice(FILLS) if rng.random() < 0.35 else None)
    if fill and subtype not in NO_FILL:
        poly = _closed_region(paths)
        if poly is not None:
            fs = fill_strokes(poly, rng, fill)
            if fs:
                pos = int(rng.integers(0, len(paths) + 1)) if rng.random() < 0.3 else len(paths)
                paths = paths[:pos] + fs + paths[pos:]
                tr.append(f"fill:{fill}")
    if spec.get("order", rng.random() < 0.2) and len(paths) > 1:
        perm = rng.permutation(len(paths))
        paths = [paths[i] for i in perm]
        tr.append("order")
    if spec.get("reverse", rng.random() < 0.25):
        paths = [p[::-1] if rng.random() < 0.5 else p for p in paths]
        tr.append("reverse")
    size = spec.get("size", str(rng.choice(list(SIZE_BUCKETS), p=[0.12, 0.28, 0.4, 0.2])))
    paths = fit_size(paths, rng, size)
    tr.append(f"size:{size}")
    return _retime(paths, rng, low), tr


def augment_records(parents: list[dict], subtypes: dict, per_parent: int, seed: int, low: dict | None = None) -> list[dict]:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    out = []
    for pi, par in enumerate(parents):
        st = subtypes.get(par["key_id"], "unknown")
        for k in range(per_parent):
            rng = np.random.default_rng([seed, pi, k])
            d, tr = augment_one(par["drawing"], rng, st, low)
            if not pp.normalize(d):
                continue
            out.append({
                "key_id": f"aug-{par['key_id']}-{seed}-{k}",
                "word": par.get("word", "rude"), "author": par.get("author", "аноним"), "timestamp": now,
                "prompt": par.get("prompt", ""), "device": "mouse",
                "canvas": {"w": int(CANVAS), "h": int(CANVAS), "format": "square"},
                "source": "augmented", "parent": par["key_id"], "subtype": st, "transforms": tr,
                "augment": {"seed": seed, "index": k},
                "drawing": d, "simplified": pp.normalize(d),
            })
    return out


# ───────────────────────────────────────────── композиции

def qd_to_paths(drawing):
    """Quick Draw simplified (0..255, без времени) → пути."""
    return [np.stack([np.asarray(s[0], float), np.asarray(s[1], float)], 1) for s in drawing if len(s[0])]


def _overlap(a, b):
    """Доля площади меньшего прямоугольника, закрытая пересечением."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return w * h / max(1e-9, min(sa, sb))


def _primary_frac(placed, primary=0):
    """Отношение большей стороны рамки главного объекта к большей стороне рамки всей композиции."""
    allp = np.concatenate([p for part in placed for p in part])
    B = float((allp.max(0) - allp.min(0)).max()) or 1.0
    pp0 = np.concatenate(placed[primary])
    P = float((pp0.max(0) - pp0.min(0)).max())
    return P / B


def compose(parts: list[tuple[list, float]], rng, low: dict | None = None, min_frac: float = 0.0, primary: int = 0):
    """parts: [(пути, относительный масштаб)] → один рисунок на холсте: случайные места и перекрытия.
    Масштаб 1.0 — «обычный» объект (сторона ~ 160–320 px); порядок рисования частей — случайный.
    min_frac: главный объект (parts[primary]) должен занимать ≥ этой доли большей стороны всей композиции
    (проверяется по РЕАЛЬНОМУ размещению; при недоборе прочие объекты стягиваются к главному)."""
    low = low or DEFAULT_LOW
    placed = []
    base = rng.uniform(140, 260)
    # объекты кучкуются в области ~1.9–2.7 базового размера (иначе после нормализации всё мелкое)
    region = min(CANVAS - 8, base * rng.uniform(1.9, 2.7))
    r0 = np.array([rng.uniform(4, CANVAS - 4 - region), rng.uniform(4, CANVAS - 4 - region)])
    allow = rng.random() < 0.3                     # иногда объекты заметно перекрываются (торчит из фигуры и т. п.)
    boxes = []
    for paths, rel in parts:
        mn, mx = _bbox(paths)
        ext = float((mx - mn).max()) or 1.0
        s = base * rel * rng.uniform(0.85, 1.15) / ext
        span = (mx - mn) * s
        best, best_ov = None, 9.0
        for _ in range(30):
            off = r0 + np.array([rng.uniform(0, max(1, region - span[0])), rng.uniform(0, max(1, region - span[1]))])
            box = np.concatenate([off, off + span])
            ov = max([_overlap(box, b) for b in boxes] or [0.0])
            if ov < best_ov:
                best, best_ov = off, ov
            if ov <= (0.45 if allow else 0.12):
                break
        boxes.append(np.concatenate([best, best + span]))
        placed.append([(p - mn) * s + best for p in paths])
    # добор минимальной относительной величины главного объекта: стягиваем прочие к его центру
    if min_frac > 0 and len(placed) > 1:
        pc = np.concatenate(placed[primary]).mean(0)
        for _ in range(12):
            if _primary_frac(placed, primary) >= min_frac:
                break
            placed = [part if i == primary else [pc + (p - pc) * 0.85 for p in part] for i, part in enumerate(placed)]
    order = rng.permutation(len(placed))
    paths = [p for i in order for p in placed[i]]
    return _retime(jitter(paths, rng, 0.003), rng, low)
