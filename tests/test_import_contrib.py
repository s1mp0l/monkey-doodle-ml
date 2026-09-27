"""Импорт файлов веб-сборщика + паритет JS-упрощения (webapp/core.js) с эталоном."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from doodle.ndjson_io import read_records
from doodle.preprocess import normalize
from import_contrib import import_files

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "webapp" / "core.js"
NODE = shutil.which("node")


def rec(key, dx=0.0, **kw):
    r = {"key_id": key, "word": "rude", "author": "вася", "timestamp": "2026-09-27T20:00:00.000+03:00",
         "prompt": "маленький", "device": "touch", "canvas": {"w": 800, "h": 800, "format": "square"}, "source": "web",
         "drawing": [[[100 + dx, 200, 300.5], [100, 400, 120], [0, 16, 40]], [[250], [90], [300]]]}
    r.update(kw)
    return r


def write(path, records, extra_lines=()):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records) + "".join(extra_lines), encoding="utf-8")
    return path


def test_valid_import_and_fields(tmp_path):
    f = write(tmp_path / "in.ndjson", [rec("aaaaaaaa-1"), rec("bbbbbbbb-2", dx=5)])
    rep = import_files([f], tmp_path / "custom", {"rude"})
    assert rep["files"][str(f)]["accepted"] == 2
    out = read_records(tmp_path / "custom" / "rude.ndjson")
    assert [r["key_id"] for r in out] == ["aaaaaaaa-1", "bbbbbbbb-2"]
    r = out[0]
    assert r["source"] == "web" and r["contributor"] == "вася" and r["import_file"] == "in.ndjson"
    assert r["simplified"] == normalize(r["drawing"])
    assert list(r)[:8] == ["key_id", "word", "author", "timestamp", "prompt", "device", "canvas", "source"]


def test_malformed_rejected_with_reasons(tmp_path):
    bad = [
        rec("short"),                                                  # плохой key_id
        rec("cccccccc-1", word="banana"),                              # класс не разрешён
        rec("dddddddd-1", drawing=[]),                                 # нет рисунка
        rec("eeeeeeee-1", drawing=[[[1, 2], [3], [0, 1]]]),            # разные длины
        rec("ffffffff-1", drawing=[[[1, "x"], [3, 4], [0, 1]]]),       # нечисловые
        rec("gggggggg-1", drawing=[[[1, 2], [3, 4], [5, 1]]]),         # время убывает
        rec("hhhhhhhh-1", drawing=[[[5, 5.5], [5, 5], [0, 1]]]),       # точка
        rec("iiiiiiii-1", drawing=[[[0, 5000], [0, 10], [0, 1]]]),     # далеко за холстом
        rec("jjjjjjjj-1", drawing=[[[0, 50], [0, 10], [0, 11 * 60 * 1000]]]),  # 11 минут
        rec("kkkkkkkk-1", drawing=[[[0, 1e9], [0, 10], [0, 1]]], canvas={}),   # границы
    ]
    f = write(tmp_path / "bad.ndjson", bad, extra_lines=['{"key_id": "broken\n', "not json\n"])
    rep = import_files([f], tmp_path / "custom", {"rude"})
    r = rep["files"][str(f)]
    assert r["accepted"] == 0 and sum(r["rejected"].values()) == len(bad) + 2
    assert r["rejected"]["битый JSON"] == 2
    assert not (tmp_path / "custom" / "rude.ndjson").exists()


def test_duplicates_by_key_and_by_drawing(tmp_path):
    custom = tmp_path / "custom"
    f1 = write(tmp_path / "a.ndjson", [rec("aaaaaaaa-1"), rec("aaaaaaaa-1", dx=3), rec("bbbbbbbb-2")])
    rep = import_files([f1], custom, {"rude"})
    r = rep["files"][str(f1)]
    # второй — дубль key_id, третий — тот же рисунок под другим key_id
    assert r["accepted"] == 1 and r["rejected"]["дубль key_id"] == 1 and r["rejected"]["дубль рисунка"] == 1
    # повторный импорт того же файла ничего не добавляет
    rep = import_files([f1], custom, {"rude"})
    assert rep["files"][str(f1)]["accepted"] == 0
    assert len(read_records(custom / "rude.ndjson")) == 1


def test_dry_run_writes_nothing(tmp_path):
    f = write(tmp_path / "in.ndjson", [rec("aaaaaaaa-1")])
    rep = import_files([f], tmp_path / "custom", {"rude"}, dry_run=True)
    assert rep["by_class"]["rude"] == 1 and not (tmp_path / "custom" / "rude.ndjson").exists()


@pytest.mark.skipif(NODE is None, reason="нет node")
def test_js_serializer_file_imports_and_matches_reference(tmp_path):
    js = f"""
    const C = require({json.dumps(str(CORE))});
    let seed = 7; const rnd = () => {{ seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }};
    const recs = [];
    for (let k = 0; k < 200; k++) {{
      const d = [];
      for (let s = 0; s < 1 + k % 4; s++) {{
        const n = 10 + Math.floor(rnd() * 80); let x = 100 + rnd() * 600, y = 100 + rnd() * 600; const xs = [], ys = [], ts = [];
        for (let i = 0; i < n; i++) {{ x += (rnd() - .5) * 30; y += (rnd() - .5) * 30; xs.push(x); ys.push(y); ts.push(s * 2000 + i * 16); }}
        d.push([xs, ys, ts]);
      }}
      recs.push(C.makeRecord({{ word: "rude", author: "js", prompt: "сбоку", device: "mouse",
        canvas: {{ w: 800, h: 800, format: "square" }}, drawing: d }}));
    }}
    process.stdout.write(C.toNdjson(recs));
    """
    out = subprocess.run([NODE, "-e", js], capture_output=True, check=True, timeout=60).stdout.decode("utf-8")
    f = tmp_path / "rude_js_2026-09-27_200.ndjson"
    f.write_text(out, encoding="utf-8")
    web = [json.loads(line) for line in out.splitlines()]
    assert all(r["simplified"] == normalize(r["drawing"]) for r in web)      # JS-порт = эталон бит-в-бит
    rep = import_files([f], tmp_path / "custom", {"rude"})
    assert rep["files"][str(f)]["accepted"] == 200, rep["files"][str(f)]["rejected"]


@pytest.mark.skipif(NODE is None, reason="нет node")
def test_js_normalize_on_reference_vectors():
    vec = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "tests" / "vectors").glob("*.json"))]
    js = f"""
    const C = require({json.dumps(str(CORE))}); const fs = require("fs");
    const inp = JSON.parse(fs.readFileSync(0, "utf8"));
    process.stdout.write(JSON.stringify(inp.map(d => C.normalize(d))));
    """
    res = subprocess.run([NODE, "-e", js], input=json.dumps([v["input"] for v in vec]).encode(), capture_output=True,
                         check=True, timeout=60).stdout
    assert json.loads(res) == [v["simplified"] for v in vec]
