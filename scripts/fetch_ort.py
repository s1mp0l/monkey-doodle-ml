"""Скачать и «вшить» onnxruntime-web (закреплённая версия) в playground/vendor/ — без CDN в рантайме.

    python scripts/fetch_ort.py

Берёт npm-тарбол onnxruntime-web@ORT_VERSION, проверяет sha512 из package.json npm, кладёт минимум
для WASM-бэкенда: ort.wasm.min.js (обёртка) и ort-wasm-simd-threaded.{mjs,wasm} (ядро). Так CSP
остаётся connect-src 'self' (файлы свои), нужен лишь 'wasm-unsafe-eval' для компиляции WASM.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

from _common import ROOT

ORT_VERSION = "1.30.0"
VENDOR = ROOT / "playground" / "vendor"
# Для браузера-ESM: самодостаточный бандл .mjs (вшивает JS-загрузчик) + бинарь WASM. CDN не нужен.
WANT = [
    "package/dist/ort.wasm.bundle.min.mjs",
    "package/dist/ort-wasm-simd-threaded.mjs",
    "package/dist/ort-wasm-simd-threaded.wasm",
    "package/LICENSE",
]


def main():
    meta_url = f"https://registry.npmjs.org/onnxruntime-web/{ORT_VERSION}"
    meta = json.loads(urllib.request.urlopen(meta_url, timeout=60).read())
    dist = meta["dist"]
    tarball, integrity = dist["tarball"], dist["integrity"]
    print(f"onnxruntime-web@{ORT_VERSION}: {tarball}")
    data = urllib.request.urlopen(tarball, timeout=180).read()

    algo, b64 = integrity.split("-", 1)
    digest = base64.b64encode(getattr(hashlib, algo)(data).digest()).decode()
    if digest != b64:
        raise SystemExit(f"integrity mismatch: {algo} {digest} != {b64}")
    print(f"integrity ok ({algo})")

    VENDOR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        names = set(t.getnames())
        for member in WANT:
            if member not in names:
                if member.endswith("LICENSE"):
                    print("  LICENSE не в тарболе — пропускаю (текст лицензии MIT уже в ATTRIBUTION.md)")
                    continue
                raise SystemExit(f"нет в тарболе: {member}")
            out = VENDOR / Path(member).name
            out.write_bytes(t.extractfile(member).read())
            print(f"  {out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.2f} МБ)")
    (VENDOR / "VERSION").write_text(f"onnxruntime-web {ORT_VERSION}\n{tarball}\nintegrity {integrity}\n", encoding="utf-8")
    print(f"готово → {VENDOR}")


if __name__ == "__main__":
    main()
