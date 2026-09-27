"""Общее для скриптов: путь к корню проекта в sys.path, стандартные каталоги."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Консоль Windows: не падать на кириллице при выводе в cp1251/cp866.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DATA = ROOT / "data"
RAW_QD = DATA / "raw" / "quickdraw"
PREPARED = DATA / "prepared"
RUNS = ROOT / "runs"
MODELS = ROOT / "models"
