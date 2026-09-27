"""Сгенерировать эталонные векторы предобработки tests/vectors/ (для pytest и порта на C#).

Каждый случай — три файла:
    <name>.json  {name, about, params, input (сырые штрихи Quick Draw, Y вниз), [texture], simplified}
    <name>.npy   float32 [N, N] — ожидаемый вход модели (без осей batch/channel)
    <name>.png   то же картинкой (тёмные чернила на белом, ×4) — для глаз

Перегенерировать ТОЛЬКО при осознанном изменении спецификации (docs/PREPROCESS.md):
    python scripts/make_test_vectors.py
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from _common import RAW_QD, ROOT
from doodle import preprocess as pp

OUT = ROOT / "tests" / "vectors"


def cases():
    rng = np.random.default_rng(12345)
    c = []
    c.append(("empty", "нет штрихов → нулевое изображение", [], None))
    c.append(("dot", "один штрих из одной точки (тычок)", [[[10], [20]]], None))
    c.append(("line_horizontal", "горизонтальная линия: высота рамки 0, центр по Y",
              [[[0, 50, 100], [7, 7, 7]]], None))
    c.append(("line_vertical_float", "вертикальная линия, дробные и отрицательные координаты",
              [[[-3.25, -3.25], [-40.5, 12.75]]], None))
    c.append(("square_closed", "замкнутый квадрат одним штрихом (первая точка = последняя)",
              [[[0, 100, 100, 0, 0], [0, 0, 100, 100, 0]]], None))
    c.append(("duplicates", "повторы точек, пустой штрих и штрих из совпадающих точек",
              [[[5, 5, 5, 30, 30, 60], [5, 5, 5, 30, 30, 5]], [[], []], [[40, 40, 40], [70, 70, 70]]], None))
    t = np.linspace(0, 2 * math.pi, 400)
    circle = [[(120 + 80 * np.cos(t) + rng.normal(0, 0.4, t.size)).round(2).tolist(),
               (90 + 80 * np.sin(t) + rng.normal(0, 0.4, t.size)).round(2).tolist()]]
    c.append(("circle_dense_noisy", "плотная шумная окружность (400 точек) — проверка RDP", circle, None))
    c.append(("tiny", "крошечный рисунок 3×2 px — растягивается до 255",
              [[[100, 103, 101], [50, 51, 52]]], None))
    house = [[[0, 0, 120, 120, 0], [200, 80, 80, 200, 200]], [[0, 60, 120], [80, 0, 80]],
             [[45, 45, 75, 75], [200, 140, 140, 200]]]
    c.append(("house_multistroke", "домик из трёх штрихов", house, None))

    # игра: точки DrawingSurface в пикселях Texture2D листа 480×706, ось Y ВВЕРХ
    arc = [(140 + 200 * s, 380 + 150 * math.sin(math.pi * s) - 60 * s) for s in np.linspace(0, 1, 90)]
    arc2 = [(140 + 200 * s, 380 + 110 * math.sin(math.pi * s) - 60 * s) for s in np.linspace(0, 1, 70)]
    tex = {"width": 480, "height": 706, "strokes": [[[round(x, 2), round(y, 2)] for x, y in arc],
                                                    [[round(x, 2), round(y, 2)] for x, y in arc2]]}
    c.append(("game_texture_banana", "банан мышью в игре: координаты Texture2D (Y вверх) → from_texture_strokes",
              pp.from_texture_strokes(tex["strokes"], tex["height"]), tex))

    # реальный рисунок Quick Draw (simplified) — если скачан
    qd = RAW_QD / "banana.ndjson"
    if qd.exists():
        with open(qd, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        c.append(("quickdraw_banana", f"Quick Draw simplified banana key_id={rec['key_id']}", rec["drawing"], None))
    return c


def main():
    ap = argparse.ArgumentParser(description="Перегенерировать tests/vectors/ по текущей эталонной предобработке.")
    ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = [("", dict(size=64, pad=2.0, line_width=2.0)),
                ("_n32", dict(size=32, pad=1.0, line_width=1.5))]
    n = 0
    for name, about, strokes, tex in cases():
        simp = pp.normalize(strokes, pp.EPSILON)
        for suffix, prm in variants:
            if suffix and name not in ("house_multistroke", "circle_dense_noisy", "dot"):
                continue
            img = pp.rasterize(simp, **prm)
            doc = {"name": name + suffix, "about": about,
                   "params": {"epsilon": pp.EPSILON, **prm},
                   "input": strokes, "simplified": simp}
            if tex:
                doc["texture"] = tex
            (OUT / f"{name}{suffix}.json").write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
            np.save(OUT / f"{name}{suffix}.npy", img.astype(np.float32))
            (OUT / f"{name}{suffix}.png").write_bytes(pp.to_png_bytes(img, invert=True, scale=4))
            n += 1
    print(f"{n} векторов → {OUT}")


if __name__ == "__main__":
    main()
