"""Запуск редактора датасета (рисовать / галерея / тест модели) в браузере.

    python scripts/editor.py                 # http://127.0.0.1:8765  (только этот компьютер)
    python scripts/editor.py --lan           # 0.0.0.0 — можно рисовать с других ПК и телефонов в сети
    python scripts/editor.py --port 9000 --model-dir models/20260927-201500

Рисунки дописываются в data/custom/<класс>.ndjson, журнал удалений/пометок — data/custom/_review.ndjson.
Авторизации нет: --lan включайте только в доверенной локальной сети.
"""
from __future__ import annotations

import argparse
import socket
import webbrowser

from _common import ROOT
from editor.server import App, serve


def lan_ips() -> list[str]:
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))          # пакет не отправляется — только выбор интерфейса
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def main():
    ap = argparse.ArgumentParser(description="Редактор датасета рисунков (локальный веб-сервер).")
    ap.add_argument("--lan", action="store_true", help="слушать 0.0.0.0 (доступ из локальной сети)")
    ap.add_argument("--host", default=None, help="адрес (по умолчанию 127.0.0.1, с --lan — 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config", default=None, help="classes.yaml")
    ap.add_argument("--custom-dir", default=None, help="куда писать рисунки (по умолчанию data/custom)")
    ap.add_argument("--model-dir", default=None, help="модель для вкладки «Тест» (по умолчанию models/latest)")
    ap.add_argument("--open", action="store_true", help="открыть браузер")
    args = ap.parse_args()

    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
    model_dir = (ROOT / args.model_dir) if args.model_dir and not args.model_dir[1:2] == ":" else args.model_dir
    app = App(args.config, args.custom_dir, model_dir)
    httpd = serve(app, host, args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Редактор: {url}")
    if host == "0.0.0.0":
        for ip in lan_ips():
            print(f"  в сети:  http://{ip}:{args.port}/")
    info = app.model.info()
    print("Модель для «Теста»: " + (f"{info['run']} ({info['classes']} классов)" if info["available"] else info["error"]))
    print(f"Рисунки: {app.custom_dir}\nCtrl+C — остановить")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
