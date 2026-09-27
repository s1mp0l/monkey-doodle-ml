"""Скопировать модель для «Угадайки» в playground/model/ (модель в git не идёт — публикуется отдельно).

    python scripts/publish_playground.py --model models/latest
    python scripts/publish_playground.py --model models/20260927-202406 --fp32

Кладёт model.onnx (по умолчанию fp16, если его паритет с fp32 сходится — из export_report.json,
иначе fp32), labels.json и config/lesson_prompts.json в playground/model/. Также обновляет
playground/core.js из webapp/core.js (единый источник упрощения/растеризации).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _common import ROOT

DST = ROOT / "playground" / "model"


def main():
    ap = argparse.ArgumentParser(description="Опубликовать модель в playground/model/ для запуска в браузере.")
    ap.add_argument("--model", default="models/latest", help="каталог экспорта (models/<ts> или models/latest)")
    ap.add_argument("--fp32", action="store_true", help="взять fp32 даже если fp16 годится")
    args = ap.parse_args()

    src = ROOT / args.model if not Path(args.model).is_absolute() else Path(args.model)
    labels = src / "labels.json"
    if not labels.exists():
        raise SystemExit(f"нет {labels} — сначала scripts/export_onnx.py")

    use_fp16 = False
    rep = src / "export_report.json"
    if not args.fp32 and (src / "doodle_fp16.onnx").exists() and rep.exists():
        r = json.loads(rep.read_text(encoding="utf-8"))
        use_fp16 = r.get("fp16_top1_agreement", 0) >= 0.999 and r.get("parity_max_abs_fp16", 1) < 5e-3
    model_file = "doodle_fp16.onnx" if use_fp16 else "doodle.onnx"

    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / model_file, DST / "model.onnx")
    shutil.copy2(labels, DST / "labels.json")
    lp = ROOT / "config" / "lesson_prompts.json"
    if lp.exists():
        shutil.copy2(lp, DST / "lesson_prompts.json")
    shutil.copy2(ROOT / "webapp" / "core.js", ROOT / "playground" / "core.js")

    size = (DST / "model.onnx").stat().st_size
    lab = json.loads(labels.read_text(encoding="utf-8"))
    print(f"опубликовано в {DST}:")
    print(f"  model.onnx ← {model_file} ({size / 1e6:.2f} МБ, {'fp16' if use_fp16 else 'fp32'})")
    print(f"  labels.json ({len(lab['classes'])} классов, прогон {lab.get('run')})")
    print(f"  lesson_prompts.json, core.js обновлён из webapp/")
    print("Локально: cd playground && python -m http.server 8080  →  http://127.0.0.1:8080")


if __name__ == "__main__":
    main()
