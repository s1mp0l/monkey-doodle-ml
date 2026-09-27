"""Сгенерировать config/lesson_prompts.json — список заданий урока рисования для Unity.

Источник — классы с task: true в config/classes.yaml. Каждое задание — класс Quick, Draw!:
  id        метка модели (= labels.json → classes[].id)
  quickdraw точное имя класса Quick Draw
  ru        именительный падеж («Банан») — подпись задания
  ru_acc    винительный падеж («Нарисуйте: рыбку»)

    python scripts/lesson_prompts.py            # перезаписать файл
    python scripts/lesson_prompts.py --check    # только проверить, что файл актуален (код 1, если нет)
"""
from __future__ import annotations

import argparse
import json
import sys

from _common import ROOT
from doodle.config import load_config

OUT = ROOT / "config" / "lesson_prompts.json"


def build(config_path=None) -> str:
    cfg = load_config(config_path)
    doc = {
        "about": "Задания урока рисования (классы Quick, Draw!). Генерируется scripts/lesson_prompts.py "
                 "из config/classes.yaml (task: true) — не править руками.",
        "prompts": cfg.lesson_prompts(),
    }
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Записать config/lesson_prompts.json из classes.yaml (task: true).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--check", action="store_true", help="не писать, а проверить актуальность")
    args = ap.parse_args()
    text = build(args.config)
    if args.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != text:
            print("config/lesson_prompts.json устарел — запустите scripts/lesson_prompts.py")
            sys.exit(1)
        print("config/lesson_prompts.json актуален")
        return
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"{OUT}: {len(json.loads(text)['prompts'])} заданий")


if __name__ == "__main__":
    main()
