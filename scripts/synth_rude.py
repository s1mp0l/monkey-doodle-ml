"""Синтетические рисунки rude (фаллос) — генерация, калибровка по людям, контактный лист.

    # калибровка по своим рисункам (только подтип phallus, если есть разметка data/custom/rude_subtypes.json)
    python scripts/synth_rude.py --calibrate data/custom/rude.ndjson
    # генерация 5000 (с калибровкой, если она есть) → data/synthetic/rude.ndjson + reports/synth_rude_sheet.png
    python scripts/synth_rude.py -n 5000 --seed 1
    python scripts/synth_rude.py -n 200 --no-calibration --out data/synthetic/test.ndjson

Калибровка пишется в data/synthetic/rude_calibration.json (там только статистики, не рисунки).
Контактный лист: слева 100 синтетических, справа до 100 рисунков людей — через каноническую
предобработку 64×64 (ровно то, что видит модель). Рядом — reports/synth_rude_stats.md со сравнением
распределений (число штрихов, точки, длительность, рамка, доля самого длинного штриха).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import DATA, ROOT
from doodle import preprocess as pp
from doodle import synth
from doodle.ndjson_io import iter_records

SYN_DIR = DATA / "synthetic"
CAL_PATH = SYN_DIR / "rude_calibration.json"
REPORTS = ROOT / "reports"
SUBTYPES = DATA / "custom" / "rude_subtypes.json"


def human_drawings(path: Path, subtype: str | None) -> list[dict]:
    recs = [r for r in iter_records(path) if r.get("drawing") and r.get("source") != "synthetic"]
    if subtype and SUBTYPES.exists():
        sub = json.loads(SUBTYPES.read_text(encoding="utf-8"))["subtypes"]
        sel = [r for r in recs if sub.get(r.get("key_id")) == subtype]
        print(f"разметка подтипов: {len(sel)} из {len(recs)} — «{subtype}»")
        return sel
    return recs


def contact_sheet(syn: list[dict], hum: list[dict], path: Path, n: int = 100, seed: int = 0):
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(seed)
    n_hum_total = len(hum)
    syn = [syn[i] for i in rng.choice(len(syn), min(n, len(syn)), replace=False)]
    hum = [hum[i] for i in rng.choice(len(hum), min(n, len(hum)), replace=False)] if hum else []
    cell, cols, gap, top = 68, 10, 30, 30
    rows = 10
    W = cols * cell * 2 + gap
    im = Image.new("L", (W, top + rows * cell), 255)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    d.text((6, 6), f"синтетика ({len(syn)} случайных)", fill=0, font=font)
    d.text((cols * cell + gap + 6, 6), f"люди ({n_hum_total} всего, показано {len(hum)})", fill=0, font=font)
    for block, recs in ((0, syn), (1, hum)):
        x0 = block * (cols * cell + gap)
        for k, r in enumerate(recs[: rows * cols]):
            rr, cc = divmod(k, cols)
            a = (255 - pp.preprocess(r["drawing"]) * 255 + 0.5).astype(np.uint8)
            im.paste(Image.fromarray(a), (x0 + cc * cell + 2, top + rr * cell + 2))
            d.rectangle([x0 + cc * cell + 1, top + rr * cell + 1, x0 + cc * cell + 66, top + rr * cell + 66], outline=210)
    d.line([(cols * cell + gap // 2, 0), (cols * cell + gap // 2, im.height)], fill=120, width=2)
    im = im.resize((im.width * 3 // 2, im.height * 3 // 2), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def stats_table(syn: list[dict], hum: list[dict]) -> str:
    fs = [synth.features(r["drawing"]) for r in syn]
    fh = [synth.features(r["drawing"]) for r in hum]

    def q(v):
        return "{:.2f} / {:.2f} / {:.2f}".format(*np.percentile(v, [25, 50, 75])) if v else "—"

    pts = lambda rs: [sum(len(s[0]) for s in r["drawing"]) for r in rs]  # noqa: E731
    lines = ["| признак (квартили 25/50/75) | синтетика | люди |", "|---|---|---|"]
    for k in synth.FEATURES:
        lines.append(f"| {k} | {q([f[k] for f in fs])} | {q([f[k] for f in fh])} |")
    lines.append(f"| raw points | {q(pts(syn))} | {q(pts(hum))} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Генератор синтетических рисунков rude + калибровка по людям + контактный лист.")
    ap.add_argument("-n", type=int, default=0, help="сколько сгенерировать (0 — не генерировать)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(SYN_DIR / "rude.ndjson"), help="куда писать (файл перезаписывается)")
    ap.add_argument("--calibrate", default=None, metavar="NDJSON", help="снять калибровку с рисунков людей")
    ap.add_argument("--subtype", default="phallus", help="подтип из data/custom/rude_subtypes.json для калибровки ('' — все)")
    ap.add_argument("--no-calibration", action="store_true", help="генерировать с настройками по умолчанию")
    ap.add_argument("--human", default=str(DATA / "custom" / "rude.ndjson"), help="рисунки людей для контактного листа")
    ap.add_argument("--sheet", default=str(REPORTS / "synth_rude_sheet.png"))
    ap.add_argument("--sheet-only", action="store_true", help="не генерировать — перерисовать лист по готовому --out")
    args = ap.parse_args()

    if args.calibrate:
        hum = human_drawings(Path(args.calibrate), args.subtype or None)
        if len(hum) < 10:
            raise SystemExit(f"для калибровки мало рисунков: {len(hum)}")
        cal = synth.calibrate([r["drawing"] for r in hum])
        cal["source"] = str(args.calibrate)
        cal["subtype"] = args.subtype or None
        SYN_DIR.mkdir(parents=True, exist_ok=True)
        CAL_PATH.write_text(json.dumps(cal, ensure_ascii=False), encoding="utf-8")
        print(f"калибровка по {cal['n']} рисункам → {CAL_PATH}")
        for k, v in cal["summary"].items():
            print(f"  {k:<14} квантили 5/25/50/75/95: {v}")

    if args.sheet_only:
        recs = list(iter_records(Path(args.out)))
        hum_all = human_drawings(Path(args.human), None)
        contact_sheet(recs, hum_all, Path(args.sheet), seed=args.seed)
        print(f"контактный лист: {args.sheet} (синтетика {len(recs)}, люди {len(hum_all)})")
        return
    if args.n <= 0:
        return
    cal = None
    if not args.no_calibration and CAL_PATH.exists():
        cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
        print(f"калибровка: {CAL_PATH} ({cal['n']} рисунков)")
    recs = synth.generate(args.n, args.seed, cal)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{len(recs)} рисунков → {out}")

    hum_all = human_drawings(Path(args.human), None) if Path(args.human).exists() else []
    hum = human_drawings(Path(args.human), args.subtype or None) if Path(args.human).exists() else []
    contact_sheet(recs, hum_all, Path(args.sheet), seed=args.seed)     # справа — все рисунки людей класса
    table = stats_table(recs, hum)                                       # статистики — против подтипа калибровки
    md = Path(args.sheet).with_name("synth_rude_stats.md")
    md.write_text(f"# Синтетика rude vs люди\n\nсинтетика: {len(recs)} (seed {args.seed}, калибровка: {bool(cal)}), "
                  f"люди: {len(hum)} (подтип «{args.subtype}» из {len(hum_all)})\n\n{table}\n", encoding="utf-8")
    print(f"контактный лист: {args.sheet}\n{table}")


if __name__ == "__main__":
    main()
