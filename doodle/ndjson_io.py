"""Чтение/запись ndjson: построчный JSON, безопасное дописывание из нескольких потоков и процессов."""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_thread_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve()).lower()
    with _guard:
        lk = _thread_locks.get(key)
        if lk is None:
            lk = _thread_locks[key] = threading.Lock()
        return lk


@contextmanager
def file_lock(path: Path, timeout: float = 30.0):
    """Межпроцессная блокировка через файл <path>.lock (msvcrt на Windows, fcntl на POSIX)
    + блокировка потоков внутри процесса."""
    path = Path(path)
    tl = _thread_lock(path)
    with tl:
        lock_path = path.with_name(path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            deadline = time.monotonic() + timeout
            if os.name == "nt":
                import msvcrt
                while True:
                    try:
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"не удалось заблокировать {lock_path}")
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def dumps_line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def append_records(path: str | os.PathLike, records: list[dict]) -> None:
    """Дописать записи одной операцией write под блокировкой, затем fsync.
    Если файл оборван (нет \\n в конце после сбоя) — сначала дописывается перевод строки,
    чтобы новая запись не склеилась с «битой»."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(dumps_line(r) for r in records)
    with file_lock(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o666)
        try:
            size = os.fstat(fd).st_size
            if size > 0:
                with open(path, "rb") as f:
                    f.seek(size - 1)
                    if f.read(1) != b"\n":
                        data = b"\n" + data
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)


def append_record(path: str | os.PathLike, record: dict) -> None:
    append_records(path, [record])


def iter_records(path: str | os.PathLike) -> Iterator[dict]:
    """Все корректные записи файла; битые/оборванные строки пропускаются."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(obj, dict):
                yield obj


def read_records(path: str | os.PathLike) -> list[dict]:
    return list(iter_records(path))


# ───────────────────────────────────────────── журнал ревью (мягкое удаление / пометка)

REVIEW_FILE = "_review.ndjson"


def review_state(custom_dir: str | os.PathLike) -> dict[str, str]:
    """key_id → последнее действие ('delete' | 'flag' | 'ok'). Журнал только дописывается."""
    state: dict[str, str] = {}
    for r in iter_records(Path(custom_dir) / REVIEW_FILE):
        k, a = r.get("key_id"), r.get("action")
        if isinstance(k, str) and a in ("delete", "flag", "ok"):
            state[k] = a
    return state


def custom_samples(custom_dir: str | os.PathLike, class_id: str, exclude: tuple[str, ...] = ("delete", "flag")) -> list[dict]:
    """Свои рисунки класса без исключённых по журналу ревью (по умолчанию — удалённые и помеченные)."""
    recs = read_records(Path(custom_dir) / f"{class_id}.ndjson")
    if not exclude:
        return recs
    st = review_state(custom_dir)
    return [r for r in recs if st.get(r.get("key_id")) not in exclude]
