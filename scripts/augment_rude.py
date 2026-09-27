"""Офлайн-аугментация рисунков людей класса rude + составные рисунки (композиции) + листы для просмотра.

    python scripts/augment_rude.py                 # всё: аугментации, композиции, листы в reports/
    python scripts/augment_rude.py --sheets-only   # только перерисовать листы по готовым файлам

Выход (data/augmented/):
    rude.ndjson            варианты рисунков людей: source "augmented", parent, subtype, transforms
    rude_composites.ndjson рисунок rude (человек или синтетика) + 1–3 обычных рисунка Quick Draw → rude
    other_composites.ndjson 2–4 обычных рисунка Quick Draw сопоставимого размера → other
Разбиение: у варианта/композиции split родителя. Композиции с родителем из test — отдельный набор
«ctest» (для отдельной оценки), из val — не создаются; объекты Quick Draw берутся из того же split, что
у них в prepare.py (crc32 ключа). prepare.py ещё раз проверяет split родителя и выкидывает несовпадения.
Правило меток для «безобидных» композиций: всегда `other`; композиции, где один объект-задание явно
доминирует (в игре это рисунок задания с мелкими деталями), НЕ генерируются — объекты подбираются
сопоставимого размера (масштабы 0.7–1.3), чтобы метка `other` была однозначной.
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from _common import DATA, RAW_QD, ROOT
from doodle import augment as aug
from doodle import preprocess as pp
from doodle.config import load_config
from doodle.ndjson_io import custom_samples, iter_records
from prepare import human_split

OUT = DATA / "augmented"
REPORTS = ROOT / "reports"
SUBTYPES = DATA / "custom" / "rude_subtypes.json"
CAL = DATA / "synthetic" / "rude_calibration.json"

# обычные объекты для композиций: задания урока + несколько частых предметов из other
COMPOSE_QD = ["house", "tree", "sun", "car", "cloud", "flower", "apple", "banana", "fish", "star", "cat", "moon",
              "smiley face", "umbrella", "bicycle", "airplane", "snowman", "dog", "bird", "hat", "chair", "table", "door"]


def qd_split(key: str, val_pct: int, test_pct: int) -> str:
    h = zlib.crc32(f"qd:{key}".encode("utf-8")) % 100
    return "test" if h < test_pct else "val" if h < test_pct + val_pct else "train"


def write(path: Path, recs: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def record(key, word, drawing, source, **extra):
    return {"key_id": key, "word": word, "author": "composite", "timestamp": None, "prompt": "", "device": "mouse",
            "canvas": {"w": 800, "h": 800, "format": "square"}, "source": source, **extra,
            "drawing": drawing, "simplified": pp.normalize(drawing)}


def main():
    ap = argparse.ArgumentParser(description="Аугментация рисунков людей rude, композиции с Quick Draw, контактные листы.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target-per-subtype", type=int, default=600, help="примерно столько вариантов на подтип")
    ap.add_argument("--max-per-parent", type=int, default=25)
    ap.add_argument("--rude-composites", type=int, default=1500)
    ap.add_argument("--benign-composites", type=int, default=1500)
    ap.add_argument("--sheets-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    spec = cfg.by_id("rude")
    sub = json.loads(SUBTYPES.read_text(encoding="utf-8"))["subtypes"] if SUBTYPES.exists() else {}
    low = json.loads(CAL.read_text(encoding="utf-8"))["low"] if CAL.exists() else None
    human = [r for r in custom_samples(cfg.custom_dir, "rude") if r.get("source") != "synthetic"]
    hsplit = human_split([(f"custom:{r['key_id']}", r) for r in human], spec.human_split or {})
    split_of = {r["key_id"]: hsplit[f"custom:{r['key_id']}"] for r in human}

    if not args.sheets_only:
        # ── аугментация: больше вариантов для редких подтипов
        n_sub = Counter(sub.get(r["key_id"], "unknown") for r in human)
        per = {st: int(np.clip(round(args.target_per_subtype / n), 2, args.max_per_parent)) for st, n in n_sub.items()}
        recs = []
        for st, k in sorted(per.items()):
            parents = [r for r in human if sub.get(r["key_id"], "unknown") == st]
            recs += aug.augment_records(parents, sub, k, args.seed, low)
        write(OUT / "rude.ndjson", recs)
        print(f"аугментация: {len(recs)} вариантов; на родителя: {per}")

        # ── композиции
        val_pct, test_pct = int(cfg.split.get("val", 5)), int(cfg.split.get("test", 5))
        pools = {"train": [], "test": []}
        for q in COMPOSE_QD:
            p = RAW_QD / f"{q.replace(' ', '_')}.ndjson"
            if not p.exists():
                continue
            for i, r in enumerate(iter_records(p)):
                if i >= 1500:
                    break
                sp = qd_split(r["key_id"], val_pct, test_pct)
                if sp in pools:
                    pools[sp].append((q, r))
        print(f"объекты Quick Draw для композиций: train {len(pools['train'])}, test {len(pools['test'])}")
        syn = list(iter_records(DATA / "synthetic" / "rude.ndjson"))[:2000]
        rng = np.random.default_rng(args.seed)
        rude_c, benign_c = [], []
        hum_by = {"train": [r for r in human if split_of[r["key_id"]] == "train"],
                  "test": [r for r in human if split_of[r["key_id"]] == "test"]}
        for k in range(args.rude_composites):
            sp = "test" if k % 5 == 0 else "train"             # ~20 % — отдельный набор для оценки
            use_syn = sp == "train" and syn and rng.random() < 0.3
            par = syn[int(rng.integers(len(syn)))] if use_syn else hum_by[sp][int(rng.integers(len(hum_by[sp])))]
            st = "phallus" if use_syn else sub.get(par["key_id"], "unknown")
            rel = rng.uniform(0.6, 0.8) if rng.random() < 0.25 else rng.uniform(0.9, 1.4)     # иногда мельче остального
            # надпись должна остаться читаемой при 64 px — ей минимум побольше
            min_frac = 0.4 if st == "word" else (0.3 if rng.random() < 0.85 else 0.22)
            objs = [pools[sp][int(rng.integers(len(pools[sp])))] for _ in range(int(rng.integers(1, 4)))]
            parts = [(aug._paths(par["drawing"]), rel)] + [(aug.qd_to_paths(o["drawing"]), rng.uniform(0.7, 1.3)) for _, o in objs]
            d = aug.compose(parts, np.random.default_rng([args.seed, 1, k]), low, min_frac=min_frac, primary=0)
            rude_c.append(record(f"comp-r-{args.seed}-{k}", "rude", d, "composite", parent=par["key_id"], subtype=st,
                                 split_hint="ctest" if sp == "test" else "train", rude_scale=round(float(rel), 2), n_objects=len(objs),
                                 objects=[q for q, _ in objs], qd_parents=[str(o["key_id"]) for _, o in objs]))
        for k in range(args.benign_composites):
            sp = "test" if k % 5 == 0 else "train"
            objs = [pools[sp][int(rng.integers(len(pools[sp])))] for _ in range(int(rng.integers(2, 5)))]
            parts = [(aug.qd_to_paths(o["drawing"]), rng.uniform(0.7, 1.3)) for _, o in objs]
            d = aug.compose(parts, np.random.default_rng([args.seed, 2, k]), low)
            benign_c.append(record(f"comp-b-{args.seed}-{k}", "other", d, "composite", subtype="benign_composite",
                                   split_hint="ctest" if sp == "test" else "train",
                                   objects=[q for q, _ in objs], qd_parents=[str(o["key_id"]) for _, o in objs]))
        write(OUT / "rude_composites.ndjson", rude_c)
        write(OUT / "other_composites.ndjson", benign_c)
        print(f"композиции: rude {len(rude_c)}, other {len(benign_c)} (по ~20 % — набор ctest для отдельной оценки)")

    sheets(human, sub, low, args.seed)


def _tile(im, d, drawing, x, y, cell=66):
    from PIL import Image

    a = (255 - pp.preprocess(drawing) * 255 + 0.5).astype(np.uint8)
    im.paste(Image.fromarray(a), (x + 1, y + 1))
    d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=210)


def _font(size=14):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def sheets(human, sub, low, seed):
    from PIL import Image, ImageDraw

    REPORTS.mkdir(exist_ok=True)
    f = _font()
    cell = 66
    # ── аугментация: строки = родитель; столбцы: оригинал | 4 заливки | 4 размера | 4 поворота
    rng = np.random.default_rng(seed)
    by_st = {}
    for r in human:
        by_st.setdefault(sub.get(r["key_id"], "unknown"), []).append(r)
    order = ["phallus", "word", "swastika", "body", "poop", "butt", "hate", "finger"]
    rows = []
    for st in order:
        lst = by_st.get(st, [])
        for i in rng.choice(len(lst), min(2, len(lst)), replace=False) if lst else []:
            rows.append((st, lst[int(i)]))
    cols = ["оригинал"] + [f"fill:{k}" for k in aug.FILLS] + [f"size:{k}" for k in aug.SIZE_BUCKETS] + ["rot −40", "rot −15", "rot +15", "rot +40"]
    left, top = 90, 40
    im = Image.new("L", (left + len(cols) * cell, top + len(rows) * cell), 255)
    d = ImageDraw.Draw(im)
    for j, c in enumerate(cols):
        d.text((left + j * cell + 3, 22), c, fill=0, font=_font(10))
    for i, (st, r) in enumerate(rows):
        y = top + i * cell
        d.text((4, y + 24), st, fill=0, font=f)
        _tile(im, d, r["drawing"], left, y)
        for j, fl in enumerate(aug.FILLS):
            g = np.random.default_rng([seed, i, j])
            dd, _ = aug.augment_one(r["drawing"], g, st, low, {"fill": fl, "size": "medium", "rot": 0, "shear": 0,
                                                              "jitter": False, "dropout": False, "order": False, "reverse": False, "text": False})
            _tile(im, d, dd, left + (1 + j) * cell, y)
        for j, sz in enumerate(aug.SIZE_BUCKETS):
            g = np.random.default_rng([seed, i, 10 + j])
            dd, _ = aug.augment_one(r["drawing"], g, st, low, {"fill": None, "size": sz, "rot": 0, "shear": 0,
                                                              "jitter": False, "dropout": False, "order": False, "reverse": False, "text": False})
            _tile(im, d, dd, left + (5 + j) * cell, y)
        for j, rot in enumerate((-40, -15, 15, 40)):
            g = np.random.default_rng([seed, i, 20 + j])
            dd, _ = aug.augment_one(r["drawing"], g, st, low, {"fill": None, "size": "medium", "rot": rot, "shear": 0.1,
                                                              "jitter": True, "dropout": False, "order": False, "reverse": False})
            _tile(im, d, dd, left + (9 + j) * cell, y)
    d.text((4, 4), "аугментация рисунков людей: оригинал → заливки → размеры (реальный размер на холсте) → повороты",
           fill=0, font=f)
    im = im.resize((im.width * 3 // 2, im.height * 3 // 2), Image.NEAREST)
    im.save(REPORTS / "augment_rude_variants.png")

    # ── композиции
    for name, path, title in (("rude", OUT / "rude_composites.ndjson", "композиции с rude (метка rude)"),
                              ("benign", OUT / "other_composites.ndjson", "безобидные композиции Quick Draw (метка other)")):
        recs = list(iter_records(path))
        if not recs:
            continue
        pick = [recs[int(i)] for i in np.random.default_rng(seed).choice(len(recs), min(100, len(recs)), replace=False)]
        im = Image.new("L", (10 * cell, 30 + 10 * cell), 255)
        d = ImageDraw.Draw(im)
        d.text((4, 6), f"{title}: {len(recs)} всего, показано {len(pick)}", fill=0, font=f)
        for k, r in enumerate(pick):
            _tile(im, d, r["drawing"], (k % 10) * cell, 30 + (k // 10) * cell)
        im = im.resize((im.width * 3 // 2, im.height * 3 // 2), Image.NEAREST)
        im.save(REPORTS / f"composites_{name}.png")

    # ── разбивка
    recs = list(iter_records(OUT / "rude.ndjson"))
    mix = Counter()
    for r in recs:
        t = r.get("transforms", [])
        mix[("filled" if any(x.startswith("fill:") for x in t) else "outline")] += 1
        mix[next((x for x in t if x.startswith("size:")), "size:?")] += 1
    by = Counter(r.get("subtype") for r in recs)
    md = ["# Аугментация rude", "", f"вариантов: {len(recs)}", "", "по подтипам: " + ", ".join(f"{k} {v}" for k, v in by.most_common()),
          "", "заливка/контур и размеры: " + ", ".join(f"{k} {v}" for k, v in sorted(mix.items()))]
    for name, path in (("rude", OUT / "rude_composites.ndjson"), ("other", OUT / "other_composites.ndjson")):
        cr = list(iter_records(path))
        md.append(f"\nкомпозиции {name}: {len(cr)} (train {sum(r.get('split_hint') == 'train' for r in cr)}, "
                  f"ctest {sum(r.get('split_hint') == 'ctest' for r in cr)})")
    (REPORTS / "augment_rude_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("листы: reports/augment_rude_variants.png, reports/composites_rude.png, reports/composites_benign.png")


if __name__ == "__main__":
    main()
