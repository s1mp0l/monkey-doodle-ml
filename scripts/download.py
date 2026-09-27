"""Скачивание Quick, Draw! (simplified ndjson) по классам из конфига.

Источник: https://storage.googleapis.com/quickdraw_dataset/full/simplified/<name>.ndjson
Берутся только рисунки с recognized: true, не больше cap на класс. Докачка: состояние в
data/raw/quickdraw/<name>.state.json (сколько байт источника прочитано, размер вывода), при
повторном запуске запрос идёт с HTTP Range с того же места. Файл читается потоком — скачивается
только нужная голова файла, а не весь класс.

Примеры:
    python scripts/download.py --profile smoke
    python scripts/download.py                     # полный набор из config/classes.yaml
    python scripts/download.py --estimate          # только размеры (HEAD), ничего не качает
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from _common import RAW_QD
from doodle.config import load_config

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
FLUSH_EVERY = 2000
_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


def url_for(name: str) -> str:
    return BASE + urllib.parse.quote(name) + ".ndjson"


def safe_name(name: str) -> str:
    return name.replace(" ", "_")


def load_state(path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(path, st: dict) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path)


def head_size(name: str) -> int | None:
    req = urllib.request.Request(url_for(name), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def download_one(name: str, cap: int, retries: int = 6) -> dict:
    RAW_QD.mkdir(parents=True, exist_ok=True)
    out_path = RAW_QD / f"{safe_name(name)}.ndjson"
    st_path = RAW_QD / f"{safe_name(name)}.state.json"
    st = load_state(st_path)
    if not out_path.exists():
        st = {}
    st.setdefault("name", name)
    st.setdefault("src_offset", 0)
    st.setdefault("out_size", 0)
    st.setdefault("kept", 0)
    st.setdefault("seen", 0)
    st.setdefault("eof", False)
    st["cap"] = cap

    if st["kept"] >= cap or st["eof"]:
        return {**st, "status": "готово"}

    # Отбросить хвост вывода, записанный после последнего сохранения состояния.
    with open(out_path, "ab") as f:
        f.truncate(st["out_size"])

    attempt = 0
    while True:
        try:
            _stream(name, cap, st, out_path, st_path)
            return {**st, "status": "скачано"}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (404, 403):
                raise SystemExit(f"[{name}] HTTP {e.code}: нет такого класса Quick Draw? {url_for(name)}")
            attempt += 1
            if attempt > retries:
                raise
            wait = min(60, 2 ** attempt)
            log(f"[{name}] сеть: {e!r}; повтор через {wait} с (докачка с байта {st['src_offset']})")
            time.sleep(wait)
            with open(out_path, "ab") as f:
                f.truncate(st["out_size"])


def _stream(name: str, cap: int, st: dict, out_path, st_path) -> None:
    req = urllib.request.Request(url_for(name))
    if st["src_offset"] > 0:
        req.add_header("Range", f"bytes={st['src_offset']}-")
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "ab") as out:
        if st["src_offset"] > 0 and resp.status != 206:
            raise ConnectionError("сервер не поддержал Range")
        total = resp.headers.get("Content-Range", "").split("/")[-1] or resp.headers.get("Content-Length")
        if total and total.isdigit() and "total_size" not in st:
            st["total_size"] = int(total) + (st["src_offset"] if resp.status != 206 else 0)
        buf = b""
        pending: list[bytes] = []
        consumed = st["src_offset"]
        last_log = time.monotonic()

        def flush():
            if pending:
                data = b"".join(pending)
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
                st["out_size"] += len(data)
                pending.clear()
            st["src_offset"] = consumed
            save_state(st_path, st)

        while st["kept"] < cap:
            chunk = resp.read(1 << 20)
            if not chunk:
                if buf.strip():
                    # последняя строка без \n
                    consumed += len(buf)
                    _take(buf, st, pending, cap)
                st["eof"] = True
                break
            buf += chunk
            lines = buf.split(b"\n")
            buf = lines.pop()
            for line in lines:
                consumed += len(line) + 1
                _take(line, st, pending, cap)
                if len(pending) >= FLUSH_EVERY:
                    flush()
                if st["kept"] >= cap:
                    break
            if time.monotonic() - last_log > 15:
                last_log = time.monotonic()
                log(f"[{name}] {st['kept']}/{cap}  ({consumed / 1e6:.1f} МБ источника)")
        flush()


def _take(line: bytes, st: dict, pending: list, cap: int) -> None:
    line = line.strip()
    if not line or st["kept"] >= cap:
        return
    st["seen"] += 1
    try:
        d = json.loads(line)
    except ValueError:
        return
    if not d.get("recognized"):
        return
    rec = {"key_id": str(d.get("key_id")), "word": d.get("word"), "countrycode": d.get("countrycode"),
           "drawing": d.get("drawing")}
    pending.append((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))
    st["kept"] += 1


def main():
    ap = argparse.ArgumentParser(description="Скачать Quick Draw simplified ndjson (только recognized, с лимитом, с докачкой).")
    ap.add_argument("--config", default=None, help="путь к classes.yaml")
    ap.add_argument("--profile", default="full", help="профиль из конфига (full, smoke, ...)")
    ap.add_argument("--cap", type=int, default=None, help="переопределить лимит на класс для ВСЕХ источников")
    ap.add_argument("--only", default=None, help="через запятую: только эти имена Quick Draw")
    ap.add_argument("--workers", type=int, default=6, help="параллельных загрузок")
    ap.add_argument("--estimate", action="store_true", help="только оценить размеры (HEAD-запросы)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.profile)
    sources = cfg.quickdraw_sources()
    if args.only:
        only = {s.strip() for s in args.only.split(",")}
        sources = [s for s in sources if s[0] in only]
    if args.cap:
        sources = [(n, args.cap) for n, _ in sources]
    print(f"Профиль {cfg.profile}: {len(sources)} источников Quick Draw, "
          f"всего до {sum(c for _, c in sources):,} рисунков")

    if args.estimate:
        estimate(sources, args.workers)
        return

    t0 = time.time()
    fails = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download_one, n, c): n for n, c in sources}
        for fu in as_completed(futs):
            n = futs[fu]
            try:
                r = fu.result()
                log(f"  {n:<16} {r['status']:>8}: {r['kept']:>6}/{r['cap']}  "
                    f"(из {r['seen']} просмотренных, {r['out_size'] / 1e6:.1f} МБ)")
            except BaseException as e:  # noqa: BLE001
                fails.append(n)
                log(f"  {n}: ОШИБКА {e}")
    size = sum(p.stat().st_size for p in RAW_QD.glob("*.ndjson"))
    print(f"Готово за {time.time() - t0:.0f} с. data/raw/quickdraw: {size / 1e6:.0f} МБ")
    if fails:
        print("Не скачались (запустите ещё раз — докачаются):", ", ".join(fails))
        sys.exit(1)


def estimate(sources, workers):
    # средний размер строки источника на «принятый» рисунок — из уже скачанного, иначе ~ 1.0 КБ
    ratios = []
    for p in RAW_QD.glob("*.state.json"):
        st = load_state(p)
        if st.get("kept"):
            ratios.append(st["src_offset"] / st["kept"])
    per = sum(ratios) / len(ratios) if ratios else 1000.0
    total_full = 0
    need = 0
    with ThreadPoolExecutor(workers) as ex:
        sizes = list(ex.map(lambda s: (s[0], s[1], head_size(s[0])), sources))
    for n, cap, sz in sizes:
        if sz is None:
            print(f"  НЕТ В QUICK DRAW: {n}")
            continue
        total_full += sz
        need += min(sz, cap * per)
    print(f"Полные файлы классов: {total_full / 1e9:.2f} ГБ; с лимитами будет прочитано ≈ {need / 1e9:.2f} ГБ "
          f"(≈ {per:.0f} байт источника на рисунок), на диске ≈ {need * 0.95 / 1e9:.2f} ГБ")


if __name__ == "__main__":
    main()
