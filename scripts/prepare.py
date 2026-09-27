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

from _common import PREPARED, RAW_QD, ROOT
from doodle import preprocess as pp
from doodle.config import load_config
from doodle.ndjson_io import custom_samples, iter_records

SPLITS = ("train", "val", "test")
SRC_QD, SRC_HUMAN, SRC_SYN = 0, 1, 2


def human_split(items: list[tuple[str, dict]], cfg: dict) -> dict[str, str]:
    """Рисунки людей → {key: split}. Авторов > 1 — целые авторы в один split; иначе — блоки по `block`
    рисунков подряд по времени. Группы упорядочены по crc32 и набираются в test, затем в val до нужных
    процентов, остальное — train. Детерминировано."""
    test_pct, val_pct, block = float(cfg.get("test", 25)), float(cfg.get("val", 15)), int(cfg.get("block", 8))
    authors = {(rec or {}).get("author") or "аноним" for _, rec in items}
    groups: dict[str, list[str]] = {}
    if len(authors) > 1:
        for key, rec in items:
            groups.setdefault("a:" + ((rec or {}).get("author") or "аноним"), []).append(key)
    else:
        order = sorted(items, key=lambda it: ((it[1] or {}).get("timestamp") or "", it[0]))
        for k, (key, _) in enumerate(order):
            groups.setdefault(f"b:{order[k - k % block][0]}", []).append(key)
    names = sorted(groups, key=lambda g: zlib.crc32(g.encode("utf-8")))
    total = max(1, len(items))
    out, acc = {}, 0
    for g in names:
        frac = acc / total
        sp = "test" if frac < test_pct / 100 else "val" if frac < (test_pct + val_pct) / 100 else "train"
        for key in groups[g]:
            out[key] = sp
        acc += len(groups[g])
    return out


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
    ap.add_argument("--no-synthetic", action="store_true", help="без синтетики (вариант «только люди»)")
    ap.add_argument("--max-synthetic-share", type=float, default=None,
                    help="макс. доля синтетики в train класса (по уникальным рисункам); по умолчанию из конфига")
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
    per_class: list[tuple] = []   # (spec, [(key, drawing, src, record|None)])
    for spec in cfg.classes:
        items = []
        for q in spec.sources:
            path = RAW_QD / f"{q.replace(' ', '_')}.ndjson"
            if not path.exists():
                warn(f"{spec.id}: нет {path.name} — запустите scripts/download.py --profile {cfg.profile}")
                continue
            n, cap = 0, spec.cap_for(q)
            for r in iter_records(path):
                if n >= cap:
                    break
                items.append((f"qd:{r['key_id']}", r["drawing"], SRC_QD, None))
                n += 1
            if n < cap:
                print(f"  {spec.id}/{q}: {n} < cap {cap} (скачано меньше)")
        excl = ("delete",) if args.include_flagged else ("delete", "flag")
        own = custom_samples(cfg.custom_dir, spec.id, exclude=excl)
        for r in own:
            if r.get("drawing") and r.get("source") != "synthetic":
                items.append((f"custom:{r.get('key_id')}", r["drawing"], SRC_HUMAN, r))
        if own:
            print(f"  {spec.id}: своих рисунков {len(own)}")
        if spec.synthetic and not args.no_synthetic:
            syn_path = ROOT / spec.synthetic.get("file", f"data/synthetic/{spec.id}.ndjson")
            if syn_path.exists():
                syn = [r for r in iter_records(syn_path) if r.get("drawing")]
                items += [(f"syn:{r['key_id']}", r["drawing"], SRC_SYN, r) for r in syn]
                print(f"  {spec.id}: синтетики в файле {len(syn)} ({syn_path.name})")
        if not items:
            warn(f"класс '{spec.id}' пуст — исключён из обучения" +
                 (f" (соберите рисунки в редакторе: data/custom/{spec.id}.ndjson)" if spec.kind == "custom" else ""))
            continue
        n_real = sum(1 for it in items if it[2] != SRC_SYN)
        if spec.min_samples and n_real < spec.min_samples:
            warn(f"класс '{spec.id}': всего {n_real} настоящих рисунков < min_samples {spec.min_samples} — "
                 f"модель будет ненадёжна на нём (нужно больше данных)")
        per_class.append((spec, items))

    if len(per_class) < 2:
        sys.exit("Меньше двух непустых классов — обучать нечего.")

    classes = [s for s, _ in per_class]
    print(f"Классов: {len(classes)}; сбор за {time.time() - t0:.1f} с")

    # ── нормализация (параллельно)
    # ── split: люди отдельно (по авторам / блокам по времени), синтетика — только train, с лимитом доли
    split_override: dict[str, str] = {}
    source_counts: dict[str, dict] = {}
    for ci, (spec, items) in enumerate(per_class):
        if not spec.human_split:
            continue
        hum = [(key, rec) for key, _, src, rec in items if src == SRC_HUMAN]
        split_override.update(human_split(hum, spec.human_split))
        n_hum_train = sum(1 for key, _ in hum if split_override[key] == "train")
        syn = [it for it in items if it[2] == SRC_SYN]
        share = float((spec.synthetic or {}).get("max_share", 0.75)) if args.max_synthetic_share is None \
            else args.max_synthetic_share
        limit = int(share / max(1e-9, 1 - share) * n_hum_train) if share < 1 else len(syn)
        if len(syn) > limit:
            rng = np.random.default_rng(0)
            keep = set(rng.choice(len(syn), limit, replace=False).tolist())
            drop = {syn[k][0] for k in range(len(syn)) if k not in keep}
            per_class[ci] = (spec, [it for it in items if it[0] not in drop])
            print(f"  {spec.id}: синтетики беру {limit} из {len(syn)} (≤ {share:.0%} train-набора класса)")
            syn = [syn[k] for k in sorted(keep)]
        for key, *_ in syn:
            split_override[key] = "train"
        hs = {s: sum(1 for key, _ in hum if split_override[key] == s) for s in SPLITS}
        source_counts[spec.id] = {"human": hs, "synthetic_train": len(syn)}
        print(f"  {spec.id}: люди train/val/test = {hs['train']}/{hs['val']}/{hs['test']}, синтетика train = {len(syn)}")
        if hs["test"] == 0:
            warn(f"{spec.id}: нет рисунков людей в test — оценка на людях невозможна")

    flat = [(ci, key, d, src) for ci, (_, items) in enumerate(per_class) for key, d, src, _ in items]
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
        sp = split_override.get(key) or split_of(key, val_pct, test_pct)
        xs = np.concatenate([np.asarray(s[0], np.int64) for s in nd])
        ys = np.concatenate([np.asarray(s[1], np.int64) for s in nd])
        sid = np.concatenate([np.full(len(s[0]), k, np.int16) for k, s in enumerate(nd)])
        reps = cfg.custom_repeat if (src == SRC_HUMAN and sp == "train") else 1
        b = buckets[sp]
        b["pts"].append(np.stack([xs, ys], 1).astype(np.uint8))
        b["sid"].append(sid)
        b["len"].append(len(xs))
        b["label"].append(ci)
        b["src"].append(src)
        b.setdefault("rep", []).append(reps)
        if sp != "train":
            b["draw"].append(nd)
            b.setdefault("keys", []).append(key)
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
            (out / f"{sp}_keys.json").write_text(json.dumps(b.get("keys", [])), encoding="utf-8")
    print(f"Сохранено за {time.time() - t0:.1f} с")

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "profile": cfg.profile,
        "config": str(cfg.path),
        "preprocess": {"epsilon": eps, "size": size, "pad": pad, "line_width": lw, "scale_max": pp.SCALE_MAX},
        "split_pct": {"val": val_pct, "test": test_pct},
        "custom_repeat": cfg.custom_repeat,
        "sources": {"quickdraw": SRC_QD, "human": SRC_HUMAN, "synthetic": SRC_SYN},
        "source_loss_weights": cfg.source_loss_weights,
        "source_counts": source_counts,
        "synthetic_used": not args.no_synthetic,
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
