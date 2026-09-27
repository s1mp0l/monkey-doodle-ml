"""Процедурный генератор грубых рисунков фаллоса (класс rude) в сыром формате штрихов Quick Draw.

Рисунок собирается из частей в локальной системе (ствол вдоль +u, длина 1, поперёк — v):
ствол (U-контур / две линии / замкнутый / «одной линией» вместе с яйцами), головка (круглая, заострённая,
плоская; поперечная линия-«венчик», щель), яйца (0/1/2 круга, дуги или один двойной «лепесток»),
допы (волоски, капли, брызги, линии движения). Затем — «рука»: поворот/масштаб/место на холсте 800×800,
плавное дрожание, перелёт/недолёт на стыках, повторные штрихи, порядок и направление штрихов, и «мышь»:
профиль скорости minimum-jerk, шаг времени и паузы между штрихами — из калибровки по рисункам людей.

Всё детерминировано от seed. Калибровка (calibrate()) снимает с рисунков людей распределения
низкого уровня (шаг времени, скорость, паузы, размер) и маргинали признаков для «раскачки» (raking)
пула кандидатов: число штрихов, доля самого длинного штриха, соотношение сторон рамки, угол оси,
число точек после упрощения, длительность.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import preprocess as pp

CANVAS = 800.0

# Значения по умолчанию (до калибровки) — сняты с 84 рисунков фаллоса мышью в веб-сборщике.
DEFAULT_LOW = {
    "dt": [2] * 80 + [4] * 7 + [3] * 5 + [1] * 5 + [6] * 2 + [8],           # мс между событиями мыши
    "speed_q": [0.25, 0.525, 1.0, 1.485, 2.326],                           # px/мс, квантили 5/25/50/75/95
    "pause_q": [194.5, 230.0, 277.0, 377.0, 650.5],                        # мс между штрихами
    "size_q": [0.141, 0.225, 0.363, 0.546, 0.697],                         # большая сторона рамки / 800
}


def _from_quantiles(rng, q, lo_clip=None):
    """Случайное значение по 5 квантилям (5/25/50/75/95) — кусочно-линейная обратная CDF."""
    ps = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    qv = np.array(q, dtype=float)
    u = rng.random()
    if u < 0.05:
        v = qv[0] - (qv[1] - qv[0]) * (0.05 - u) / 0.2
    elif u > 0.95:
        v = qv[4] + (qv[4] - qv[3]) * (u - 0.95) / 0.2
    else:
        v = float(np.interp(u, ps, qv))
    return max(v, lo_clip) if lo_clip is not None else v


# ───────────────────────────────────────────── геометрия (локальные единицы, ствол длины 1)

def _bezier(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _arc(c, rx, ry, a0, a1, n):
    a = np.linspace(a0, a1, n)
    return np.stack([c[0] + rx * np.cos(a), c[1] + ry * np.sin(a)], 1)


def _polyline_len(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


@dataclass
class Part:
    name: str
    pts: np.ndarray            # (n, 2) в локальных координатах (v, u)


def _shaft_sides(P, rng):
    """Левая и правая стороны ствола (снизу вверх) + центры для головки."""
    W, taper, curv = P["width"], P["taper"], P["curvature"]
    n = 40
    # центральная линия — квадратичная Безье с боковым изгибом
    c0, c1, c2 = np.array([0.0, 0.0]), np.array([curv, 0.5]), np.array([curv * 0.6, 1.0])
    center = _bezier(c0, c1, c2, n)
    tang = np.gradient(center, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    half = (W / 2) * (1 - taper * np.linspace(0, 1, n))[:, None]
    left = center - norm * half
    right = center + norm * half
    return left, right, center[-1], tang[-1], norm[-1], half[-1, 0]


def _cap(P, tip_c, tang, norm, half, left_end, right_end):
    """Головка: кривая от конца левой стороны к концу правой."""
    h = P["cap_height"]
    style = P["cap_style"]
    n = 24
    if style == "pointed":
        apex = tip_c + tang * h * 1.4
        a = _bezier(left_end, left_end + tang * h * 0.9, apex, n // 2)
        b = _bezier(apex, right_end + tang * h * 0.9, right_end, n // 2)
        return np.concatenate([a, b[1:]])
    if style == "flat":
        k1 = left_end + tang * h * 0.6
        k2 = right_end + tang * h * 0.6
        return np.concatenate([_bezier(left_end, left_end + tang * h * 0.5, k1, 6),
                               np.linspace(k1, k2, 10)[1:], _bezier(k2, right_end + tang * h * 0.5, right_end, 6)[1:]])
    # round: полуэллипс
    ang = np.linspace(math.pi, 0, n)
    loc = np.stack([np.cos(ang) * half, np.sin(ang) * h], 1)
    pts = tip_c + loc[:, :1] * norm + loc[:, 1:] * tang
    pts[0], pts[-1] = left_end, right_end
    return pts


def _balls(P, left0, right0, rng):
    """Яйца у основания. → список частей."""
    n, r, style = P["balls"], P["ball_r"], P["ball_style"]
    if n == 0:
        return []
    spread, drop = P["ball_spread"], P["ball_drop"]
    # яйца — по бокам ниже основания ствола, почти касаются друг друга
    cv = max(r * spread, abs(left0[0]) * 0.6)
    cl = np.array([-cv, -r * drop])
    cr = np.array([cv, -r * drop])
    rx = r * P["ball_squash"]
    parts = []
    if style == "lobe" and n == 2:
        # один штрих: от левой стороны ствола вокруг левого, между, вокруг правого — к правой стороне
        d = math.pi / 180
        a = _arc(cl, rx, r, 60 * d, 330 * d, 34)          # левое: сверху у ствола → наружу → вниз → к середине
        b = _arc(cr, rx, r, 210 * d, 480 * d, 34)         # правое: от середины снизу → наружу → вверх к стволу
        parts.append(Part("balls_lobe", np.concatenate([a, b])))
        return parts
    centers = [cl, cr] if n == 2 else [cl if P["single_left"] else cr]
    for i, c in enumerate(centers):
        if style == "arc":
            # внешняя часть круга (со стороны от ствола)
            d = math.pi / 180
            pts = _arc(c, rx, r, 60 * d, 330 * d, 32) if c is cl else _arc(c, rx, r, 210 * d, 480 * d, 32)
        else:
            a0 = rng.uniform(0, 2 * math.pi)
            over = rng.uniform(0.0, 0.35)                   # перелёт при замыкании круга
            pts = _arc(c, rx, r, a0, a0 + 2 * math.pi + over, 40)
            if rng.random() < 0.5:
                pts = pts[::-1]
        parts.append(Part(f"ball{i}", pts))
    return parts


def _ball_centers(P, left0):
    r = P["ball_r"]
    cv = max(r * P["ball_spread"], abs(left0[0]) * 0.6)
    cs = [np.array([-cv, -r * P["ball_drop"]]), np.array([cv, -r * P["ball_drop"]])]
    return cs if P["balls"] == 2 else [cs[0] if P["single_left"] else cs[1]]


def _trim_inside(side, centers, P):
    r, rx = P["ball_r"], P["ball_r"] * P["ball_squash"]
    k = 0
    while k < len(side) - 3:
        v, u = side[k]
        if any(((v - c[0]) / rx) ** 2 + ((u - c[1]) / r) ** 2 < 1.0 for c in centers):
            k += 1
        else:
            break
    return side[k:]


def build_parts(P, rng) -> list[Part]:
    left, right, tip_c, tang, norm, half = _shaft_sides(P, rng)
    cap = _cap(P, tip_c, tang, norm, half, left[-1], right[-1])
    parts: list[Part] = []
    ss = P["shaft_style"]
    balls = _balls(P, left[0], right[0], rng)
    full_left, full_right = left, right
    if balls:
        # стороны ствола начинаются там, где кончается контур яйца (люди не ведут линию сквозь круг)
        centers = _ball_centers(P, left[0])
        left, right = _trim_inside(left, centers, P), _trim_inside(right, centers, P)
    if ss == "two_lines":
        parts.append(Part("shaft_left", left))
        if P["cap_attached"]:
            parts.append(Part("shaft_right_cap", np.concatenate([right, cap[::-1][1:]])))
        else:
            parts.append(Part("shaft_right", right))
            parts.append(Part("cap", cap))
    else:
        u = np.concatenate([left, cap[1:], right[::-1][1:]])
        if ss == "closed" and not balls:
            u = np.concatenate([u, np.linspace(u[-1], u[0], 8)[1:]])
        if ss == "one_line" and len(balls) >= 2 and not balls[0].name.endswith("lobe"):
            b0, b1 = balls[0].pts, balls[1].pts
            u = np.concatenate([b0, left, cap[1:], right[::-1][1:], b1])
            balls = []
        parts.append(Part("shaft", u))
    if P["ridge"]:
        k = int(len(full_left) * (1 - P["ridge_off"]))
        k = min(max(k, 1), len(full_left) - 1)
        a, b = full_left[k], full_right[k]
        mid = (a + b) / 2 + tang * P["ridge_bend"] * half
        parts.append(Part("ridge", _bezier(a, mid, b, 12)))
    if P["slit"]:
        tip = cap[len(cap) // 2]
        parts.append(Part("slit", np.stack([tip, tip - tang * P["cap_height"] * 0.5])))
    ball_first = P["order"] == "balls_first"
    parts = (balls + parts) if ball_first else (parts + balls)

    # допы
    for k in range(P["hair"]):
        base = (left[0] if k % 2 == 0 else right[0]) + np.array([rng.normal(0, 0.1), rng.uniform(-0.25, 0.05)])
        z = [base]
        for _ in range(rng.integers(3, 7)):
            z.append(z[-1] + rng.normal(0, 0.035, 2))
        parts.append(Part("hair", np.array(z)))
    for _ in range(P["drops"]):
        c = tip_c + tang * rng.uniform(0.3, 0.8) + norm * rng.normal(0, 0.15)
        r = rng.uniform(0.02, 0.05)
        parts.append(Part("drop", _arc(c, r, r * 1.3, 0, 2 * math.pi + 0.2, 14)))
    for _ in range(P["splash"]):
        a = rng.normal(0, 0.8)
        d = tang * math.cos(a) + norm * math.sin(a)
        s0 = tip_c + tang * P["cap_height"] + d * rng.uniform(0.08, 0.2)
        parts.append(Part("splash", np.stack([s0, s0 + d * rng.uniform(0.08, 0.2)])))
    for k in range(P["motion"]):
        side = norm * (half + 0.12 + 0.07 * k) * (1 if k % 2 == 0 else -1)
        a = tip_c * 0.3 + side
        parts.append(Part("motion", np.stack([a, a + tang * rng.uniform(0.25, 0.45)])))
    return parts


# ───────────────────────────────────────────── параметры

STYLES = ("clean", "one_line", "sketchy", "minimal", "tiny", "huge")


def sample_params(rng, style: str | None = None, weights: dict | None = None) -> dict:
    if style is None:
        w = weights or {"clean": 0.5, "one_line": 0.08, "sketchy": 0.12, "minimal": 0.12, "tiny": 0.1, "huge": 0.08}
        names = list(w)
        p = np.array([w[k] for k in names], float)
        style = names[int(rng.choice(len(names), p=p / p.sum()))]
    P = {
        "style": style,
        "width": float(rng.uniform(0.28, 0.6)),
        "taper": float(rng.uniform(-0.1, 0.25)),
        "curvature": float(rng.normal(0, 0.1) if rng.random() < 0.8 else rng.normal(0, 0.25)),
        "cap_style": str(rng.choice(["round", "pointed", "flat"], p=[0.45, 0.4, 0.15])),
        "cap_height": 0.0,
        "cap_attached": bool(rng.random() < 0.5),
        "ridge": bool(rng.random() < 0.7),
        "ridge_off": float(rng.uniform(0.08, 0.28)),
        "ridge_bend": float(rng.normal(0.25, 0.3)),
        "slit": bool(rng.random() < 0.15),
        "balls": int(rng.choice([0, 1, 2], p=[0.05, 0.07, 0.88])),
        "ball_r": 0.0,
        "ball_style": str(rng.choice(["circle", "arc", "lobe"], p=[0.7, 0.15, 0.15])),
        "single_left": bool(rng.random() < 0.5),
        "ball_spread": float(rng.uniform(0.75, 1.1)),
        "ball_drop": float(rng.uniform(0.35, 0.95)),
        "ball_squash": float(rng.uniform(0.9, 1.5)),
        "shaft_style": str(rng.choice(["U", "two_lines", "closed", "one_line"], p=[0.62, 0.22, 0.1, 0.06])),
        "order": str(rng.choice(["shaft_first", "balls_first"], p=[0.8, 0.2])),
        "hair": int(rng.integers(2, 6)) if rng.random() < 0.08 else 0,
        "drops": int(rng.integers(1, 4)) if rng.random() < 0.07 else 0,
        "splash": int(rng.integers(3, 7)) if rng.random() < 0.05 else 0,
        "motion": int(rng.integers(2, 4)) if rng.random() < 0.04 else 0,
        "wobble": float(rng.uniform(0.003, 0.018)),
        "join_err": float(rng.uniform(0.0, 0.035)),
        "reverse_p": float(rng.uniform(0.0, 0.4)),
        "overstroke_p": 0.0,
        "rotation_deg": _sample_rotation(rng),
        "size": 0.0,
        "speed_mul": float(np.exp(rng.normal(0, 0.25))),
    }
    P["cap_height"] = float(P["width"] * rng.uniform(0.35, 0.95))
    P["ball_r"] = float(P["width"] * rng.uniform(0.42, 0.78))
    if style == "one_line":
        P["shaft_style"] = "one_line"
        P["balls"] = 2
        P["ball_style"] = "circle"
        P["ridge"] = bool(rng.random() < 0.3)
    elif style == "sketchy":
        P["overstroke_p"] = float(rng.uniform(0.3, 0.8))
        P["wobble"] *= 1.6
        P["join_err"] *= 1.5
    elif style == "minimal":
        P["ridge"] = False
        P["slit"] = False
        P["ball_style"] = "lobe" if rng.random() < 0.6 else P["ball_style"]
        P["shaft_style"] = "U"
        P["hair"] = P["drops"] = P["splash"] = P["motion"] = 0
    return P


def _sample_rotation(rng) -> float:
    """Угол оси от «кончиком вверх», градусы, + по часовой (вправо). Люди чаще рисуют вверх и вверх-вправо."""
    u = rng.random()
    if u < 0.38:
        return float(rng.normal(0, 12))
    if u < 0.75:
        return float(rng.uniform(20, 70))
    if u < 0.85:
        return float(rng.uniform(70, 100))
    if u < 0.93:
        return float(rng.uniform(-100, -20))
    return float(rng.uniform(-180, 180))


# ───────────────────────────────────────────── «рука» и «мышь»

def _smooth_noise(rng, n, amp, waves=4):
    s = np.linspace(0, 1, n)
    out = np.zeros(n)
    for _ in range(waves):
        f = rng.uniform(0.5, 5.0)
        out += np.sin(2 * math.pi * (f * s + rng.random())) / waves
    return out * amp


def _humanize(pts, P, scale, rng):
    """Дрожание по нормали, перелёт/недолёт на концах."""
    if len(pts) < 2:
        return pts
    # сгустить до ~2 px, чтобы шум был гладким
    L = _polyline_len(pts)
    n = max(2, int(L / 2.0))
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    if s[-1] <= 0:
        return pts
    t = np.linspace(0, s[-1], n)
    q = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)
    tang = np.gradient(q, axis=0)
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(nrm, 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    q = q + normal * _smooth_noise(rng, n, P["wobble"] * scale)[:, None]
    q = q + rng.normal(0, 0.25, q.shape)                            # мелкий тремор
    je = P["join_err"] * scale
    for end, sgn in ((0, -1), (-1, 1)):
        d = rng.normal(0, je)
        tv = tang[end] * sgn
        if d > 0:                                                    # перелёт
            ext = q[end] + tv * np.linspace(0, d, max(2, int(d / 2)))[:, None]
            q = np.concatenate([ext[::-1][:-1], q]) if end == 0 else np.concatenate([q, ext[1:]])
        elif d < 0:                                                  # недолёт
            k = min(len(q) - 2, int(-d / 2.0))
            if k > 0:
                q = q[k:] if end == 0 else q[:-k]
    return q


def _mouse_sample(path, speed, dt_pool, rng, t0):
    """Путь → точки мыши: minimum-jerk вдоль длины, шаг времени из dt_pool, запись точки ≥ 1 px (как холст)."""
    L = _polyline_len(path)
    T = max(40.0, L / max(speed, 0.05))
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    xs, ys, ts = [path[0, 0]], [path[0, 1]], [t0]
    t = 0.0
    while t < T:
        t += float(dt_pool[rng.integers(len(dt_pool))])
        tau = min(1.0, t / T)
        pos = L * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
        x, y = np.interp(pos, s, path[:, 0]), np.interp(pos, s, path[:, 1])
        if (x - xs[-1]) ** 2 + (y - ys[-1]) ** 2 >= 1.0:
            xs.append(x)
            ys.append(y)
            ts.append(t0 + t)
    return xs, ys, ts


def render(P: dict, rng, low: dict) -> list[list[list[float]]]:
    """Параметры → сырой рисунок [[xs], [ys], [ts]] на холсте 800×800 (Y вниз)."""
    parts = build_parts(P, rng)
    # локальные (v, u) → холст: u вверх = -y; поворот по часовой на rotation_deg
    th = math.radians(P["rotation_deg"])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    world = []
    for part in parts:
        p = part.pts.copy()
        q = np.stack([p[:, 0], -p[:, 1]], 1) @ R.T
        world.append((part.name, q))
    allp = np.concatenate([w for _, w in world])
    mn, mx = allp.min(0), allp.max(0)
    ext = float((mx - mn).max())
    size = P["size"]
    scale = size * CANVAS / max(ext, 1e-6)
    span = (mx - mn) * scale
    off = np.array([rng.uniform(4, max(5, CANVAS - 4 - span[0])), rng.uniform(4, max(5, CANVAS - 4 - span[1]))])
    strokes = []
    t = 0.0
    speed = _from_quantiles(rng, low["speed_q"], 0.08) * P["speed_mul"] * (0.6 if P["style"] == "tiny" else 1.0)
    for name, q in world:
        path = (q - mn) * scale + off
        path = _humanize(path, P, size * CANVAS, rng)
        if rng.random() < P["reverse_p"] and not name.startswith("shaft"):
            path = path[::-1]
        reps = 2 if rng.random() < P["overstroke_p"] else 1
        for k in range(reps):
            pth = path if k == 0 else _humanize(path + rng.normal(0, 0.012 * size * CANVAS, 2), P, size * CANVAS, rng)
            pth = np.clip(pth, 0, CANVAS - 1)
            xs, ys, ts = _mouse_sample(pth, speed * rng.uniform(0.85, 1.2), low["dt"], rng, t)
            strokes.append([[round(v, 1) for v in xs], [round(v, 1) for v in ys], [int(round(v)) for v in ts]])
            t = ts[-1] + _from_quantiles(rng, low["pause_q"], 60)
    return strokes


def sample_size(rng, P, low):
    s = _from_quantiles(rng, low["size_q"], 0.05)
    if P["style"] == "tiny":
        s = rng.uniform(0.05, 0.14)
    elif P["style"] == "huge":
        s = rng.uniform(0.75, 0.97)
    return float(min(max(s, 0.04), 0.97))


# ───────────────────────────────────────────── признаки (для калибровки и отчёта)

FEATURES = ("strokes", "longest_share", "aspect", "axis_deg", "simp_points", "duration_s")
BINS = {
    "strokes": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.5, 99],
    "longest_share": [0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.01],
    "aspect": [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
    "axis_deg": [0, 30, 60, 90, 120, 150, 180.1],
    "simp_points": [0, 30, 40, 50, 60, 80, 120, 9999],
    "duration_s": [0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 9999],
}


def features(drawing) -> dict:
    X = np.concatenate([np.asarray(s[0], float) for s in drawing])
    Y = np.concatenate([np.asarray(s[1], float) for s in drawing])
    w, h = np.ptp(X), np.ptp(Y)
    lens = [_polyline_len(np.stack([s[0], s[1]], 1).astype(float)) for s in drawing]
    c = np.stack([X - X.mean(), Y - Y.mean()])
    ev, evec = np.linalg.eigh(c @ c.T)
    ax = evec[:, 1]
    ang = math.degrees(math.atan2(ax[0], -ax[1])) % 180          # 0 = вертикаль, 90 = горизонталь
    simp = pp.normalize(drawing)
    return {
        "strokes": len(drawing),
        "longest_share": max(lens) / max(sum(lens), 1e-9),
        "aspect": min(w, h) / max(w, h, 1e-9),
        "axis_deg": ang,
        "simp_points": sum(len(s[0]) for s in simp),
        "duration_s": (drawing[-1][2][-1] - drawing[0][2][0]) / 1000.0,
    }


def hist(values, bins):
    h, _ = np.histogram(np.clip(values, bins[0], bins[-1] - 1e-9), bins=bins)
    return (h + 0.5) / (h.sum() + 0.5 * len(h))                 # сглаживание — нет пустых корзин


def calibrate(drawings: list) -> dict:
    """Снять распределения с рисунков людей → словарь калибровки (JSON-сериализуемый)."""
    dts, speeds, pauses, sizes = [], [], [], []
    feats = [features(d) for d in drawings]
    for d in drawings:
        X = np.concatenate([s[0] for s in d])
        Y = np.concatenate([s[1] for s in d])
        sizes.append(max(np.ptp(X), np.ptp(Y)) / CANVAS)
        for s in d:
            x, y, t = (np.asarray(v, float) for v in s)
            dd, dt = np.hypot(np.diff(x), np.diff(y)), np.diff(t)
            dts += [int(v) for v in dt if 0 < v <= 40]
            m = dt > 0
            if m.any() and (t[-1] - t[0]) > 0:
                speeds.append(float(dd.sum() / (t[-1] - t[0])))          # средняя скорость штриха
        pauses += [b[2][0] - a[2][-1] for a, b in zip(d[:-1], d[1:]) if b[2][0] - a[2][-1] >= 0]
    q = lambda a: np.percentile(a, [5, 25, 50, 75, 95]).round(4).tolist() if len(a) else None  # noqa: E731
    return {
        "n": len(drawings),
        "low": {
            "dt": [int(v) for v in (dts if len(dts) <= 5000 else np.random.default_rng(0).choice(dts, 5000))],
            "speed_q": q(speeds) or DEFAULT_LOW["speed_q"],
            "pause_q": q(pauses) or DEFAULT_LOW["pause_q"],
            "size_q": q(sizes) or DEFAULT_LOW["size_q"],
        },
        "marginals": {k: hist([f[k] for f in feats], BINS[k]).round(5).tolist() for k in FEATURES},
        "summary": {k: q([f[k] for f in feats]) for k in FEATURES},
    }


def rake(feats: list[dict], marginals: dict, iters: int = 8) -> np.ndarray:
    """Iterative proportional fitting: веса кандидатов, чтобы маргинали признаков совпали с людьми."""
    n = len(feats)
    w = np.ones(n)
    idx = {k: np.clip(np.digitize([f[k] for f in feats], BINS[k]) - 1, 0, len(BINS[k]) - 2) for k in marginals}
    for _ in range(iters):
        for k, target in marginals.items():
            cur = np.bincount(idx[k], weights=w, minlength=len(target))
            cur = cur / cur.sum()
            ratio = np.where(cur > 0, np.asarray(target) / np.maximum(cur, 1e-12), 0)
            w = w * ratio[idx[k]]
            w = np.clip(w / w.mean(), 0.02, 25)                 # не давать одному кандидату доминировать
    return w / w.sum()


# ───────────────────────────────────────────── генерация записей

def generate(n: int, seed: int = 0, calibration: dict | None = None, pool_factor: float = 5.0) -> list[dict]:
    """n записей rude (source: synthetic). С калибровкой — пул pool_factor·n и выборка по весам raking."""
    rng = np.random.default_rng(seed)
    low = (calibration or {}).get("low") or DEFAULT_LOW
    m = int(n * pool_factor) if calibration else n
    cands = []
    for i in range(m):
        r = np.random.default_rng([seed, i])
        P = sample_params(r)
        P["size"] = sample_size(r, P, low)
        d = render(P, r, low)
        cands.append((i, P, d))
    if calibration:
        feats = [features(d) for _, _, d in cands]
        w = rake(feats, calibration["marginals"])
        # взвешенная выборка без возвращения (Efraimidis–Spirakis)
        keys = np.log(rng.random(m)) / np.maximum(w * m, 1e-12)        # лог-ключи: без потери точности
        chosen = np.sort(np.argsort(-keys)[:n])
        cands = [cands[k] for k in chosen]
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    out = []
    for i, P, d in cands:
        out.append({
            "key_id": f"syn-{seed:04x}-{i:06x}",
            "word": "rude",
            "author": "synth",
            "timestamp": now,
            "prompt": P["style"],
            "device": "mouse",
            "canvas": {"w": int(CANVAS), "h": int(CANVAS), "format": "square"},
            "source": "synthetic",
            "generator": {"name": "synth_rude", "version": 1, "seed": seed, "index": i,
                          "calibrated": bool(calibration), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in P.items()}},
            "drawing": d,
            "simplified": pp.normalize(d),
        })
    return out
