"""Playground: JS-порт растеризации = эталон, и JS+onnxruntime-web ↔ Python+onnxruntime дают тот же top-1."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "webapp" / "core.js"
ORT_MJS = ROOT / "playground" / "vendor" / "ort.wasm.bundle.min.mjs"
MODEL = ROOT / "playground" / "model" / "model.onnx"
LABELS = ROOT / "playground" / "model" / "labels.json"
NODE = shutil.which("node")
VEC = ROOT / "tests" / "vectors"


@pytest.mark.skipif(NODE is None, reason="нет node")
def test_js_rasterize_matches_reference_vectors(tmp_path):
    script = f"""
    const C = require({json.dumps(str(CORE))}), fs = require('fs');
    const dir = {json.dumps(str(VEC))}, out = {json.dumps(str(tmp_path))};
    for (const f of fs.readdirSync(dir)) {{
      if (!f.endsWith('.json')) continue;
      const d = JSON.parse(fs.readFileSync(dir + '/' + f)); const p = d.params;
      const img = C.rasterize(C.normalize(d.input, p.epsilon), p.size, p.pad, p.line_width);
      fs.writeFileSync(out + '/' + d.name + '.json', JSON.stringify(Array.from(img)));
    }}
    """
    subprocess.run([NODE, "-e", script], check=True, timeout=60)
    n = 0
    for f in VEC.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        js = np.array(json.loads((tmp_path / f"{d['name']}.json").read_text()), np.float32)
        exp = np.load(f.with_suffix(".npy")).reshape(-1)
        np.testing.assert_allclose(js, exp, atol=1e-6, err_msg=d["name"])
        n += 1
    assert n >= 10


@pytest.mark.skipif(NODE is None or not MODEL.exists(), reason="нет node или не опубликована модель (scripts/publish_playground.py)")
def test_js_ortweb_top1_matches_python(tmp_path):
    import onnxruntime as ort  # noqa: F401

    from doodle import preprocess as pp

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    pre = labels["preprocess"]
    # держим-аут: несколько QD-классов, если скачаны, иначе эталонные векторы
    drawings = []
    qd = ROOT / "data" / "raw" / "quickdraw"
    for name in ("banana", "sun", "house", "fish", "apple", "tree", "car", "star"):
        p = qd / f"{name}.ndjson"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 40:
                    break
                drawings.append(json.loads(line)["drawing"])
    if len(drawings) < 20:
        drawings = [json.loads(f.read_text(encoding="utf-8"))["input"] for f in sorted(VEC.glob("*.json"))
                    if json.loads(f.read_text(encoding="utf-8"))["input"]]
    (tmp_path / "draw.json").write_text(json.dumps(drawings), encoding="utf-8")

    # Python: preprocess + onnxruntime
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    py_top1 = []
    for d in drawings:
        img = pp.preprocess(d, pre["size"], pre["pad"], pre["line_width"], pre["epsilon"]).astype(np.float32)
        probs = sess.run(None, {"input": img[None, None]})[0][0]
        py_top1.append(int(np.argmax(probs)))

    # JS: core.preprocess + onnxruntime-web
    runner = tmp_path / "run.mjs"
    runner.write_text(f"""
    import {{ createRequire }} from 'module'; import fs from 'fs'; import {{ pathToFileURL }} from 'url';
    const ort = await import(pathToFileURL({json.dumps(str(ORT_MJS).replace(chr(92), '/'))}).href);
    const require = createRequire(import.meta.url);
    const C = require({json.dumps(str(CORE))});
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = pathToFileURL({json.dumps(str(ORT_MJS.parent).replace(chr(92), '/') + '/')}).href;
    const pre = {json.dumps(pre)};
    const drawings = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'draw.json')).replace(chr(92), '/')}));
    const buf = new Uint8Array(fs.readFileSync({json.dumps(str(MODEL)).replace(chr(92), '/')}));
    const s = await ort.InferenceSession.create(buf, {{ executionProviders: ['wasm'] }});
    const out = [];
    for (const d of drawings) {{
      const img = C.preprocess(d, pre.size, pre.pad, pre.line_width, pre.epsilon);
      const r = await s.run({{ input: new ort.Tensor('float32', img, [1, 1, pre.size, pre.size]) }});
      const p = r.probs.data; let bi = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[bi]) bi = i;
      out.push(bi);
    }}
    fs.writeFileSync({json.dumps(str(tmp_path / 'js.json')).replace(chr(92), '/')}, JSON.stringify(out));
    """, encoding="utf-8")
    subprocess.run([NODE, str(runner)], check=True, timeout=180, cwd=str(tmp_path))
    js_top1 = json.loads((tmp_path / "js.json").read_text())
    assert len(js_top1) == len(py_top1)
    agree = np.mean([a == b for a, b in zip(js_top1, py_top1)])
    assert agree >= 0.99, f"top-1 совпадение JS↔Python {agree:.3f} на {len(py_top1)} рисунках"
