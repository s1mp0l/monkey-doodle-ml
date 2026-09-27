"""Импорт рисунков из веб-сборщика (webapp/, файлы .ndjson от участников) в data/custom/<класс>.ndjson.

    python scripts/import_contrib.py inbox/rude_Вася_2026-09-27_40.ndjson
    python scripts/import_contrib.py inbox/                 # все *.ndjson в папке (рекурсивно)
    python scripts/import_contrib.py inbox/ --dry-run       # только проверить

Каждая запись проверяется (схема, число штрихов/точек, границы координат, время), дубли отсекаются
по key_id и по хешу рисунка (и внутри пачки, и против уже импортированного). Принятым проставляется
source: "web", contributor, imported_at; simplified пересчитывается канонической предобработкой
(doodle/preprocess.normalize). Запись — одной атомарной дозаписью на класс (как у редактора).
После импорта рисунки видны в галерее локального редактора — там их можно пометить/удалить.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT  # noqa: F401  (sys.path)
from doodle.config import load_config
from doodle.ndjson_io import append_records, iter_records
from doodle.preprocess import normalize

MAX_STROKES = 200
MIN_POINTS = 2
MAX_POINTS = 20000
MAX_ABS_COORD = 100000.0
MIN_EXTENT = 3.0                 # рисунок не может быть одной точкой
MAX_DURATION_MS = 10 * 60 * 1000
KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
DEVICES = ("mouse", "pen", "touch")


class Reject(Exception):
    pass


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def drawing_hash(drawing) -> str:
    """Хеш координат (без времени) — ловит повторную отправку того же рисунка под новым key_id."""
    xy = [[[round(float(v), 1) for v in s[0]], [round(float(v), 1) for v in s[1]]] for s in drawing]
    return hashlib.sha1(json.dumps(xy, separators=(",", ":")).encode()).hexdigest()


def validate(rec, allowed: set[str]) -> dict:
    """Проверить запись участника → очищенная запись (без служебных полей) или Reject(причина)."""
    if not isinstance(rec, dict):
        raise Reject("не объект")
    key = rec.get("key_id")
    if not isinstance(key, str) or not KEY_RE.match(key):
        raise Reject("плохой key_id")
    word = rec.get("word")
    if word not in allowed:
        raise Reject(f"класс не разрешён: {word!r}")
    d = rec.get("drawing")
    if not isinstance(d, list) or not d:
        raise Reject("нет рисунка")
    if len(d) > MAX_STROKES:
        raise Reject("слишком много штрихов")
    total, t_prev, t_first, t_last = 0, None, None, None
    xs_all, ys_all = [], []
    clean = []
    for s in d:
        if not isinstance(s, list) or len(s) != 3:
            raise Reject("штрих не [xs, ys, ts]")
        xs, ys, ts = s
        if not all(isinstance(a, list) for a in s) or not (len(xs) == len(ys) == len(ts)) or not xs:
            raise Reject("разные длины xs/ys/ts или пустой штрих")
        if not all(_num(v) for v in xs + ys + ts):
            raise Reject("нечисловые координаты/время")
        if any(abs(v) > MAX_ABS_COORD for v in xs + ys):
            raise Reject("координаты вне допустимых границ")
        for t in ts:
            if t < 0 or (t_prev is not None and t < t_prev):
                raise Reject("время штрихов убывает или отрицательно")
            t_prev = t
        t_first = ts[0] if t_first is None else t_first
        t_last = ts[-1]
        total += len(xs)
        xs_all += xs
        ys_all += ys
        clean.append([[_c(v) for v in xs], [_c(v) for v in ys], [int(round(v)) for v in ts]])
    if total < MIN_POINTS:
        raise Reject("слишком мало точек")
    if total > MAX_POINTS:
        raise Reject("слишком много точек")
    if max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all)) < MIN_EXTENT:
        raise Reject("рисунок — точка")
    if t_last - t_first > MAX_DURATION_MS:
        raise Reject("рисовали дольше 10 минут")
    canvas = rec.get("canvas") if isinstance(rec.get("canvas"), dict) else {}
    cw, ch = canvas.get("w"), canvas.get("h")
    if _num(cw) and _num(ch) and cw > 0 and ch > 0:
        mx, my = cw * 0.5, ch * 0.5            # указатель может «уехать» за край при захвате
        if min(xs_all) < -mx or max(xs_all) > cw + mx or min(ys_all) < -my or max(ys_all) > ch + my:
            raise Reject("точки далеко за пределами холста")
    simplified = normalize(clean)
    if not simplified:
        raise Reject("пусто после упрощения")
    ts_str = rec.get("timestamp")
    if not isinstance(ts_str, str) or not _iso_ok(ts_str):
        ts_str = None
    return {
        "key_id": key,
        "word": word,
        "author": str(rec.get("author") or "").strip()[:40] or "аноним",
        "timestamp": ts_str,
        "prompt": str(rec.get("prompt") or "")[:80],
        "device": rec.get("device") if rec.get("device") in DEVICES else "unknown",
        "canvas": {k: canvas[k] for k in ("w", "h", "format") if k in canvas and (_num(canvas[k]) or isinstance(canvas[k], str))},
        "source": "web",
        "drawing": clean,
        "simplified": simplified,
    }


def _c(v):
    r = round(float(v), 1)
    return int(r) if r == int(r) else r


def _iso_ok(s: str) -> bool:
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False


def collect_files(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*.ndjson") if q.is_file())
        elif p.is_file():
            out.append(p)
        else:
            print(f"нет такого файла/папки: {p}")
    return out


def import_files(files: list[Path], custom_dir: Path, allowed: set[str], contributor: str | None = None,
                 dry_run: bool = False) -> dict:
    """→ {"files": {имя: {"accepted": n, "rejected": Counter}}, "by_class": {класс: n}}"""
    seen_keys, seen_hash = set(), set()
    for w in allowed:
        for r in iter_records(custom_dir / f"{w}.ndjson"):
            if isinstance(r.get("key_id"), str):
                seen_keys.add(r["key_id"])
            if isinstance(r.get("drawing"), list):
                try:
                    seen_hash.add(drawing_hash(r["drawing"]))
                except (TypeError, ValueError, IndexError):
                    pass
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    pending: dict[str, list[dict]] = defaultdict(list)
    report = {"files": {}, "by_class": Counter()}
    for f in files:
        acc, rej = 0, Counter()
        try:
            raw = f.read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError) as e:
            rej[f"файл не читается: {e.__class__.__name__}"] += 1
            report["files"][str(f)] = {"accepted": 0, "rejected": rej}
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = validate(json.loads(line), allowed)
            except ValueError:
                rej["битый JSON"] += 1
                continue
            except Reject as e:
                rej[str(e)] += 1
                continue
            h = drawing_hash(rec["drawing"])
            if rec["key_id"] in seen_keys:
                rej["дубль key_id"] += 1
                continue
            if h in seen_hash:
                rej["дубль рисунка"] += 1
                continue
            seen_keys.add(rec["key_id"])
            seen_hash.add(h)
            rec["timestamp"] = rec["timestamp"] or now
            rec["contributor"] = contributor or (rec["author"] if rec["author"] != "аноним" else f.stem)
            rec["imported_at"] = now
            rec["import_file"] = f.name
            # порядок полей как у редактора + служебные в конце
            rec = {k: rec[k] for k in ("key_id", "word", "author", "timestamp", "prompt", "device", "canvas", "source",
                                       "contributor", "imported_at", "import_file", "drawing", "simplified")}
            pending[rec["word"]].append(rec)
            acc += 1
        report["files"][str(f)] = {"accepted": acc, "rejected": rej}
    for word, recs in pending.items():
        if recs and not dry_run:
            append_records(custom_dir / f"{word}.ndjson", recs)
        report["by_class"][word] += len(recs)
    return report


def main():
    ap = argparse.ArgumentParser(description="Импортировать .ndjson из веб-сборщика в data/custom/ (с проверкой и дедупликацией).")
    ap.add_argument("paths", nargs="+", help="файлы .ndjson или папки")
    ap.add_argument("--classes", default="rude", help="разрешённые классы через запятую (по умолчанию rude)")
    ap.add_argument("--contributor", default=None, help="кто прислал (по умолчанию — ник из записи или имя файла)")
    ap.add_argument("--custom-dir", default=None, help="куда писать (по умолчанию data/custom)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="проверить, ничего не записывая")
    args = ap.parse_args()

    cfg = load_config(args.config)
    allowed = {c.strip() for c in args.classes.split(",") if c.strip()}
    unknown = allowed - set(cfg.class_ids)
    if unknown:
        sys.exit(f"нет таких классов в конфиге: {sorted(unknown)}")
    custom_dir = Path(args.custom_dir) if args.custom_dir else cfg.custom_dir
    files = collect_files(args.paths)
    if not files:
        sys.exit("нечего импортировать")
    rep = import_files(files, custom_dir, allowed, args.contributor, args.dry_run)
    tot_a = tot_r = 0
    for name, r in rep["files"].items():
        nrej = sum(r["rejected"].values())
        tot_a += r["accepted"]
        tot_r += nrej
        print(f"{name}: принято {r['accepted']}, отклонено {nrej}")
        for why, n in r["rejected"].most_common():
            print(f"    - {why}: {n}")
    print(("[dry-run] " if args.dry_run else "") + f"Итого: принято {tot_a}, отклонено {tot_r}; " +
          ", ".join(f"{w} +{n}" for w, n in rep["by_class"].items()) +
          ("" if args.dry_run else f" → {custom_dir}"))


if __name__ == "__main__":
    main()
