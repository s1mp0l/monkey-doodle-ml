"""Подготовка обучающих данных: Quick Draw + свои рисунки → нормализованные штрихи + split.

Выход: data/prepared/<profile>/
    meta.json                    классы, счётчики, параметры предобработки, предупреждения
    <split>_pts.npy   uint8 (P, 2)   точки нормализованных штрихов (0..255)
    <split>_sid.npy   int16 (P,)     номер штриха точки внутри рисунка
    <split>_start.npy int64 (M,)     начало рисунка в pts;  <split>_len.npy int32 (M,) число точек
    <split>_label.npy int16 (M,)     индекс класса;         <split>_src.npy uint8 (M,) 0=Quick Draw, 1=свои
    val_images.npy / test_images.npy  uint8 (M, N, N)  эталонная растеризация (doodle.preprocess), ×255
Штрихи в train растеризуются на GPU во время обучения — с аугментациями (doodle/gpu_raster.py).
Split детерминирован: crc32(key_id) % 100 → test / val / train, поэтому новые данные не «перетасовывают» старые.

    python scripts/prepare.py --profile smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from datetime import datetime
from multiprocessing import Pool

import numpy as np

from _common import PREPARED, RAW_QD
from doodle import preprocess as pp
from doodle.config import load_config
from doodle.ndjson_io import custom_samples, iter_records

SPLITS = ("train", "val", "test")


def split_of(key: str, val_pct: int, test_pct: int) -> str:
    h = zlib.crc32(key.encode("utf-8")) % 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def _normalize_job(args):
    drawings, eps = args
    return [pp.normalize(d, eps) for d in drawings]


def _raster_job(args):
    drawings, size, pad, lw = args
    return np.stack([(pp.rasterize(d, size, pad, lw) * 255.0 + 0.5).astype(np.uint8) for d in drawings]) \
        if drawings else np.zeros((0, size, size), np.uint8)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    ap = argparse.ArgumentParser(description="Подготовить data/prepared/<profile>/ из скачанного Quick Draw и data/custom.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--out", default=None, help="каталог вывода (по умолчанию data/prepared/<profile>)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--include-flagged", action="store_true", help="брать и помеченные (flag) свои рисунки")
    args = ap.parse_args()

    cfg = load_config(args.config, args.profile)
    P = cfg.preprocess
    eps, size, pad, lw = float(P.get("epsilon", 2.0)), int(P.get("size", 64)), float(P.get("pad", 2.0)), float(P.get("line_width", 2.0))
    out = PREPARED / cfg.profile if not args.out else __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val_pct, test_pct = int(cfg.split.get("val", 5)), int(cfg.split.get("test", 5))
    warnings: list[str] = []

    def warn(msg):
        warnings.append(msg)
        print("ВНИМАНИЕ:", msg)

    # ── собрать сырьё по классам
    t0 = time.time()
    per_class: list[tuple] = []   # (spec, [(key, drawing, src)])
    for spec in cfg.classes:
        items = []
        for q in spec.quickdraw:
            path = RAW_QD / f"{q.replace(' ', '_')}.ndjson"
            if not path.exists():
                warn(f"{spec.id}: нет {path.name} — запустите scripts/download.py --profile {cfg.profile}")
                continue
            n = 0
            for r in iter_records(path):
                if n >= spec.cap:
                    break
                items.append((f"qd:{r['key_id']}", r["drawing"], 0))
                n += 1
            if n < spec.cap:
                print(f"  {spec.id}/{q}: {n} < cap {spec.cap} (скачано меньше)")
        excl = ("delete",) if args.include_flagged else ("delete", "flag")
        own = custom_samples(cfg.custom_dir, spec.id, exclude=excl)
        for r in own:
            if r.get("drawing"):
                items.append((f"custom:{r.get('key_id')}", r["drawing"], 1))
        if own:
            print(f"  {spec.id}: своих рисунков {len(own)}")
        if not items:
            warn(f"класс '{spec.id}' пуст — исключён из обучения" +
                 (f" (соберите рисунки в редакторе: data/custom/{spec.id}.ndjson)" if spec.kind == "custom" else ""))
            continue
        if spec.min_samples and len(items) < spec.min_samples:
            warn(f"класс '{spec.id}': всего {len(items)} рисунков < min_samples {spec.min_samples} — "
                 f"модель будет ненадёжна на нём (нужно больше данных)")
        per_class.append((spec, items))

    if len(per_class) < 2:
        sys.exit("Меньше двух непустых классов — обучать нечего.")

    classes = [s for s, _ in per_class]
    print(f"Классов: {len(classes)}; сбор за {time.time() - t0:.1f} с")

    # ── нормализация (параллельно)
    flat = [(ci, key, d, src) for ci, (_, items) in enumerate(per_class) for key, d, src in items]
    t0 = time.time()
    with Pool(args.workers) as pool:
        normed = []
        for res in pool.imap(_normalize_job, ((c, eps) for c in chunks([f[2] for f in flat], 2000))):
            normed.extend(res)
    print(f"Нормализовано {len(flat):,} рисунков за {time.time() - t0:.1f} с")

    # ── раскладка по split
    buckets = {s: {"pts": [], "sid": [], "len": [], "label": [], "src": [], "draw": []} for s in SPLITS}
    counts = {s: [0] * len(classes) for s in SPLITS}
    skipped = 0
    for (ci, key, _, src), nd in zip(flat, normed):
        if not nd:
            skipped += 1
            continue
        sp = split_of(key, val_pct, test_pct)
        xs = np.concatenate([np.asarray(s[0], np.int64) for s in nd])
        ys = np.concatenate([np.asarray(s[1], np.int64) for s in nd])
        sid = np.concatenate([np.full(len(s[0]), k, np.int16) for k, s in enumerate(nd)])
        reps = cfg.custom_repeat if (src == 1 and sp == "train") else 1
        b = buckets[sp]
        b["pts"].append(np.stack([xs, ys], 1).astype(np.uint8))
        b["sid"].append(sid)
        b["len"].append(len(xs))
        b["label"].append(ci)
        b["src"].append(src)
        b.setdefault("rep", []).append(reps)
        if sp != "train":
            b["draw"].append(nd)
        counts[sp][ci] += 1
    if skipped:
        print(f"  пустых после нормализации: {skipped}")

    t0 = time.time()
    for sp in SPLITS:
        b = buckets[sp]
        lens = np.asarray(b["len"], np.int32)
        starts = np.zeros(len(lens), np.int64)
        if len(lens):
            starts[1:] = np.cumsum(lens[:-1])
        rep = np.asarray(b.get("rep", []), np.int64)
        idx = np.repeat(np.arange(len(lens)), rep) if len(rep) else np.zeros(0, np.int64)
        np.save(out / f"{sp}_pts.npy", np.concatenate(b["pts"]) if b["pts"] else np.zeros((0, 2), np.uint8))
        np.save(out / f"{sp}_sid.npy", np.concatenate(b["sid"]) if b["sid"] else np.zeros(0, np.int16))
        np.save(out / f"{sp}_start.npy", starts[idx])
        np.save(out / f"{sp}_len.npy", lens[idx])
        np.save(out / f"{sp}_label.npy", np.asarray(b["label"], np.int16)[idx])
        np.save(out / f"{sp}_src.npy", np.asarray(b["src"], np.uint8)[idx])
        if sp != "train":
            with Pool(args.workers) as pool:
                parts = pool.map(_raster_job, [(c, size, pad, lw) for c in chunks(b["draw"], 1000)])
            imgs = np.concatenate(parts) if parts else np.zeros((0, size, size), np.uint8)
            np.save(out / f"{sp}_images.npy", imgs)
    print(f"Сохранено за {time.time() - t0:.1f} с")

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "profile": cfg.profile,
        "config": str(cfg.path),
        "preprocess": {"epsilon": eps, "size": size, "pad": pad, "line_width": lw, "scale_max": pp.SCALE_MAX},
        "split_pct": {"val": val_pct, "test": test_pct},
        "custom_repeat": cfg.custom_repeat,
        "classes": [{
            "id": s.id, "ru": s.ru, "kind": s.kind, "quickdraw": s.quickdraw, "task": s.task, "ru_acc": s.ru_acc,
            "sampling_weight": s.sampling_weight, "augment": s.augment, "target_precision": s.target_precision,
            "min_samples": s.min_samples,
        } for s in classes],
        "excluded_classes": [s.id for s in cfg.classes if s.id not in {c.id for c in classes}],
        "counts": {sp: dict(zip([c.id for c in classes], counts[sp])) for sp in SPLITS},
        "warnings": warnings,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    tot = {sp: sum(counts[sp]) for sp in SPLITS}
    disk = sum(p.stat().st_size for p in out.glob("*.npy"))
    print(f"Итого: train {tot['train']:,} / val {tot['val']:,} / test {tot['test']:,}; "
          f"{out} — {disk / 1e6:.0f} МБ")
    for w in warnings:
        print("ВНИМАНИЕ:", w)


if __name__ == "__main__":
    main()
