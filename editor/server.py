"""Редактор датасета: локальный веб-сервер на стандартной библиотеке (http.server) + numpy.

Вкладка «Тест» использует onnxruntime (если установлен и есть models/latest/doodle.onnx).
Запуск — scripts/editor.py. API (JSON):
    GET  /api/config                        классы, цели, подсказки разнообразия, счётчики, модель
    GET  /api/samples?word=&author=&status=&offset=&limit=   список своих рисунков (новые сверху)
    GET  /api/thumb?word=&key=              PNG 64×64 — вход модели (каноническая предобработка)
    POST /api/samples  {word, author, prompt, device, canvas, drawing}   сохранить рисунок
    POST /api/review   {word, key_id, action: delete|flag|ok, author}    мягкое удаление / пометка
    POST /api/predict  {drawing}            top-5 текущей ONNX-модели + что видит модель
"""
from __future__ import annotations

import base64
import json
import math
import mimetypes
import secrets
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from doodle import preprocess as pp
from doodle.config import ROOT, load_config
from doodle.ndjson_io import REVIEW_FILE, append_record, read_records, review_state

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 4 * 1024 * 1024
MAX_STROKES = 300
MAX_POINTS = 30000
# явные типы: на Windows mimetypes берёт их из реестра и бывает text/plain для .js
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
          ".ico": "image/x-icon", ".json": "application/json; charset=utf-8"}


class ApiError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status
        self.msg = msg


def validate_drawing(d, need_ts: bool = True) -> list[list[list[float]]]:
    """[[xs], [ys], [ts]] на штрих; числа конечные, длины равны."""
    if not isinstance(d, list) or not d:
        raise ApiError(400, "пустой рисунок")
    if len(d) > MAX_STROKES:
        raise ApiError(400, "слишком много штрихов")
    total = 0
    out = []
    for s in d:
        if not isinstance(s, list) or len(s) < 2 or (need_ts and len(s) < 3):
            raise ApiError(400, "штрих должен быть [xs, ys, ts]")
        arrs = s[:3] if len(s) >= 3 else s[:2]
        n = len(arrs[0])
        if n == 0 or any(not isinstance(a, list) or len(a) != n for a in arrs):
            raise ApiError(400, "у штриха разные длины xs/ys/ts или он пуст")
        for a in arrs:
            for v in a:
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                    raise ApiError(400, "координаты должны быть числами")
        total += n
        out.append([list(map(float, a)) for a in arrs])
    if total > MAX_POINTS:
        raise ApiError(400, "слишком много точек")
    return out


def _compact(v: float):
    r = round(v, 1)
    return int(r) if r == int(r) else r


class Model:
    """ONNX-модель для вкладки «Тест»: ленивая загрузка, перезагрузка при смене файла."""

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.lock = threading.Lock()
        self.sess = None
        self.labels = None
        self.mtime = None
        self.error = None

    def get(self):
        onnx_path, lab_path = self.dir / "doodle.onnx", self.dir / "labels.json"
        with self.lock:
            if not onnx_path.exists() or not lab_path.exists():
                self.sess, self.error = None, f"нет модели в {self.dir} — запустите scripts/export_onnx.py"
                return None
            m = (onnx_path.stat().st_mtime, lab_path.stat().st_mtime)
            if self.sess is None or m != self.mtime:
                try:
                    import onnxruntime as ort
                    so = ort.SessionOptions()
                    so.intra_op_num_threads = 2
                    self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
                    self.labels = json.loads(lab_path.read_text(encoding="utf-8"))
                    self.mtime, self.error = m, None
                except Exception as e:  # noqa: BLE001
                    self.sess, self.error = None, f"не удалось загрузить модель: {e}"
                    return None
            return self.sess, self.labels

    def info(self):
        got = self.get()
        if not got:
            return {"available": False, "error": self.error}
        _, lab = got
        return {"available": True, "run": lab.get("run"), "classes": len(lab["classes"]), "dir": str(self.dir)}


class App:
    def __init__(self, config_path=None, custom_dir: Path | None = None, model_dir: Path | None = None):
        self.cfg = load_config(config_path)
        self.custom_dir = Path(custom_dir) if custom_dir else self.cfg.custom_dir
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        self.model = Model(Path(model_dir) if model_dir else ROOT / "models" / "latest")
        P = self.cfg.preprocess
        self.pre = {"size": int(P.get("size", 64)), "pad": float(P.get("pad", 2.0)),
                    "line_width": float(P.get("line_width", 2.0)), "epsilon": float(P.get("epsilon", 2.0))}
        self.classes = {c.id: c for c in self.cfg.classes}
        self.thumbs: OrderedDict[tuple, bytes] = OrderedDict()
        self.thumb_lock = threading.Lock()
        self._cache: dict[str, tuple] = {}
        self._cache_lock = threading.Lock()

    # ── данные
    def path_for(self, word: str) -> Path:
        if word not in self.classes:
            raise ApiError(400, f"неизвестный класс: {word}")
        return self.custom_dir / f"{word}.ndjson"

    def records(self, word: str) -> list[dict]:
        """Записи класса с кэшем по (mtime, size) файла."""
        p = self.path_for(word)
        if not p.exists():
            return []
        st = p.stat()
        key = (st.st_mtime_ns, st.st_size)
        with self._cache_lock:
            c = self._cache.get(word)
            if c and c[0] == key:
                return c[1]
        recs = read_records(p)
        with self._cache_lock:
            self._cache[word] = (key, recs)
        return recs

    def counts(self) -> dict:
        rs = review_state(self.custom_dir)
        out = {}
        for cid in self.classes:
            recs = self.records(cid)
            st = [rs.get(r.get("key_id"), "ok") for r in recs]
            out[cid] = {"active": sum(s == "ok" for s in st), "flagged": sum(s == "flag" for s in st),
                        "deleted": sum(s == "delete" for s in st)}
        return out

    def config(self) -> dict:
        ed = self.cfg.editor
        targets = ed.get("targets", {})
        return {
            "classes": [{"id": c.id, "ru": c.ru, "ru_acc": c.ru_acc, "kind": c.kind, "task": c.task,
                         "target": int(targets.get(c.id, c.target or targets.get("default", 300)))}
                        for c in self.cfg.classes],
            "default_class": ed.get("default_class", "rude"),
            "variety_prompts": ed.get("variety_prompts", []),
            "counts": self.counts(),
            "model": self.model.info(),
            "preprocess": self.pre,
        }

    def save(self, body: dict) -> dict:
        word = body.get("word")
        path = self.path_for(word)
        drawing = validate_drawing(body.get("drawing"))
        author = str(body.get("author") or "").strip()[:40] or "аноним"
        prompt = str(body.get("prompt") or "").strip()[:80]
        device = body.get("device") if body.get("device") in ("mouse", "pen", "touch") else "unknown"
        canvas = body.get("canvas") if isinstance(body.get("canvas"), dict) else {}
        simplified = pp.normalize(drawing, self.pre["epsilon"])
        if not simplified:
            raise ApiError(400, "пустой рисунок")
        rec = {
            "key_id": secrets.token_hex(8),
            "word": word,
            "author": author,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "prompt": prompt,
            "device": device,
            "canvas": {k: canvas.get(k) for k in ("w", "h", "format") if k in canvas},
            "source": "editor",
            "drawing": [[[_compact(v) for v in a] for a in s] for s in drawing],
            "simplified": simplified,
        }
        append_record(path, rec)
        return {"ok": True, "key_id": rec["key_id"], "count": self.counts()[word]}

    def review(self, body: dict) -> dict:
        word, key, action = body.get("word"), body.get("key_id"), body.get("action")
        self.path_for(word)
        if action not in ("delete", "flag", "ok"):
            raise ApiError(400, "action: delete | flag | ok")
        if not isinstance(key, str) or not any(r.get("key_id") == key for r in self.records(word)):
            raise ApiError(404, "нет такого рисунка")
        append_record(self.custom_dir / REVIEW_FILE, {
            "key_id": key, "word": word, "action": action, "author": str(body.get("author") or "")[:40],
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")})
        return {"ok": True, "status": action, "count": self.counts()[word]}

    def samples(self, q: dict) -> dict:
        word = q.get("word", self.cfg.editor.get("default_class", "rude"))
        author = q.get("author") or None
        status = q.get("status", "active")
        offset, limit = max(0, int(q.get("offset", 0))), min(200, max(1, int(q.get("limit", 60))))
        rs = review_state(self.custom_dir)
        recs = self.records(word)
        authors = sorted({r.get("author", "") for r in recs})
        items = []
        for r in reversed(recs):
            s = rs.get(r.get("key_id"), "ok")
            if status == "active" and s != "ok":
                continue
            if status == "flagged" and s != "flag":
                continue
            if status == "deleted" and s != "delete":
                continue
            if author and r.get("author") != author:
                continue
            items.append({"key_id": r.get("key_id"), "author": r.get("author"), "timestamp": r.get("timestamp"),
                          "prompt": r.get("prompt"), "device": r.get("device"), "status": s,
                          "strokes": len(r.get("drawing") or [])})
        return {"word": word, "total": len(items), "items": items[offset:offset + limit], "authors": authors}

    def thumb(self, word: str, key: str) -> bytes:
        ck = (word, key)
        with self.thumb_lock:
            if ck in self.thumbs:
                self.thumbs.move_to_end(ck)
                return self.thumbs[ck]
        rec = next((r for r in self.records(word) if r.get("key_id") == key), None)
        if rec is None:
            raise ApiError(404, "нет такого рисунка")
        img = pp.preprocess(rec["drawing"], self.pre["size"], self.pre["pad"], self.pre["line_width"], self.pre["epsilon"])
        png = pp.to_png_bytes(img, invert=True)
        with self.thumb_lock:
            self.thumbs[ck] = png
            while len(self.thumbs) > 5000:
                self.thumbs.popitem(last=False)
        return png

    def predict(self, body: dict) -> dict:
        drawing = validate_drawing(body.get("drawing"), need_ts=False)
        got = self.model.get()
        pre = self.pre
        if got:
            sess, lab = got
            pre = {**pre, **{k: lab["preprocess"][k] for k in ("size", "pad", "line_width", "epsilon")}}
        t0 = time.perf_counter()
        img = pp.preprocess(drawing, int(pre["size"]), pre["pad"], pre["line_width"], pre["epsilon"])
        t_pre = (time.perf_counter() - t0) * 1000
        preview = "data:image/png;base64," + base64.b64encode(pp.to_png_bytes(img, invert=True)).decode()
        if not got:
            return {"available": False, "error": self.model.error, "preview": preview}
        t1 = time.perf_counter()
        probs = sess.run(None, {"input": img[None, None].astype(np.float32)})[0][0]
        t_inf = (time.perf_counter() - t1) * 1000
        order = np.argsort(-probs)[:5]
        cls = lab["classes"]
        top = [{"id": cls[i]["id"], "ru": cls[i]["ru"], "p": float(probs[i])} for i in order]
        best = cls[int(order[0])]
        thr = best.get("threshold")
        accepted = best["id"] != "other" and thr is not None and probs[order[0]] >= thr
        return {"available": True, "top": top, "preview": preview, "run": lab.get("run"),
                "verdict": {"id": best["id"], "ru": best["ru"], "p": float(probs[order[0]]),
                            "threshold": thr, "accepted": bool(accepted)},
                "ms": {"preprocess": round(t_pre, 2), "inference": round(t_inf, 2)}}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DoodleEditor/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # тише: только ошибки и сохранения
            if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
                super().log_message(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                raise ApiError(413 if n > MAX_BODY else 400, "пустой или слишком большой запрос")
            try:
                obj = json.loads(self.rfile.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ApiError(400, "некорректный JSON")
            if not isinstance(obj, dict):
                raise ApiError(400, "ожидался JSON-объект")
            return obj

        def _handle(self, fn):
            try:
                fn()
            except ApiError as e:
                self._json({"ok": False, "error": e.msg}, e.status)
            except (ValueError, KeyError) as e:
                self._json({"ok": False, "error": f"некорректный запрос: {e}"}, 400)
            except BrokenPipeError:
                pass
            except Exception as e:  # noqa: BLE001
                self.log_error("ошибка: %r", e)
                self._json({"ok": False, "error": "внутренняя ошибка сервера"}, 500)

        def do_GET(self):
            self._handle(self._get)

        def do_POST(self):
            self._handle(self._post)

        def _get(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/api/config":
                return self._json(app.config())
            if u.path == "/api/samples":
                return self._json(app.samples(q))
            if u.path == "/api/thumb":
                return self._send(200, app.thumb(q.get("word", ""), q.get("key", "")), "image/png", cache=True)
            name = "index.html" if u.path in ("/", "/index.html") else u.path.removeprefix("/static/")
            f = (STATIC / name).resolve()
            if f.parent != STATIC.resolve() or not f.is_file():
                raise ApiError(404, "не найдено")
            ctype = CTYPES.get(f.suffix.lower()) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            self._send(200, f.read_bytes(), ctype)

        def _post(self):
            u = urlparse(self.path)
            if u.path == "/api/samples":
                return self._json(app.save(self._body()))
            if u.path == "/api/review":
                return self._json(app.review(self._body()))
            if u.path == "/api/predict":
                return self._json(app.predict(self._body()))
            raise ApiError(404, "не найдено")

    return Handler


def serve(app: App, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    return httpd
