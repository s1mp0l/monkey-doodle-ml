"""Экспорт в ONNX для Unity Inference Engine (+ labels.json, fp16-вариант, проверки).

    python scripts/export_onnx.py --run runs/<время>

Выход: models/<время>/ и копия в models/latest/ (её использует вкладка «Тест» редактора):
    doodle.onnx        fp32, вход input [1,1,N,N] float32 (0 = фон, 1 = чернила), выход probs [1,C] (softmax)
    doodle_fp16.onnx   веса во float16 + Cast → float32 (вход/выход и вычисления fp32) — вдвое меньше
    labels.json        классы (индекс, id, Quick Draw, русские имена, пороги), параметры предобработки
    export_report.json размеры, паритет torch ↔ onnxruntime, задержка на CPU
Опсет 15: Unity Inference Engine (com.unity.ai.inference) официально поддерживает ONNX opset 7–15.
В граф попадают только Conv, Relu, MaxPool, GlobalAveragePool, Flatten, Gemm, Softmax (+Cast в fp16);
BatchNorm влит в Conv, температура калибровки — в веса Gemm.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from _common import MODELS, ROOT
from doodle import preprocess as pp
from doodle.model import ExportWrapper, count_params, fuse_for_export, load_checkpoint

OPSET = 15
VEC = ROOT / "tests" / "vectors"


def to_fp16_weights(model: onnx.ModelProto) -> onnx.ModelProto:
    """Веса float32 → float16 в файле, перед использованием Cast обратно во float32."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    new_inits, casts = [], []
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT:
            new_inits.append(init)
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        h = numpy_helper.from_array(arr, init.name + "_fp16")
        new_inits.append(h)
        casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    return m


def ort_session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def latency_ms(sess: ort.InferenceSession, n: int, size: int, runs: int = 300) -> float:
    x = np.zeros((1, 1, size, size), np.float32)
    x[0, 0, size // 4:3 * size // 4, size // 2] = 1
    for _ in range(20):
        sess.run(None, {"input": x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {"input": x})
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description="Экспорт прогона в ONNX (opset 15) + labels.json + проверки паритета/скорости.")
    ap.add_argument("--run", required=True, help="каталог runs/<время>")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default=None, help="каталог вывода (по умолчанию models/<имя прогона>)")
    ap.add_argument("--no-latest", action="store_true", help="не обновлять models/latest")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    net, ck = load_checkpoint(run / args.ckpt, "cpu")
    classes, pre = ck["classes"], ck["preprocess"]
    size = int(pre["size"])
    cal_path = run / "eval" / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
    else:
        print("ВНИМАНИЕ: нет eval/calibration.json — сначала scripts/evaluate.py. Беру T = 1, пороги 0.6.")
        cal = {"temperature": 1.0, "thresholds": {}}
    T = float(cal["temperature"])

    out = Path(args.out) if args.out else MODELS / run.name
    out.mkdir(parents=True, exist_ok=True)
    onnx_path, fp16_path = out / "doodle.onnx", out / "doodle_fp16.onnx"

    ref_model = ExportWrapper(net, T).eval()                 # без слияния BN — эталон для паритета
    fused = fuse_for_export(net)
    with torch.no_grad():                                    # температуру вливаем в последний слой: W/T, b/T
        fused.fc.weight.mul_(1.0 / T)
        fused.fc.bias.mul_(1.0 / T)
    exp_model = ExportWrapper(fused, 1.0).eval()
    dummy = torch.zeros(1, 1, size, size)
    warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
    torch.onnx.export(exp_model, (dummy,), str(onnx_path), opset_version=OPSET, input_names=["input"],
                      output_names=["probs"], do_constant_folding=True, dynamo=False)
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    model.producer_name = "monkey-doodle-ml"
    model.doc_string = f"DoodleNet {len(classes)} классов; вход input [1,1,{size},{size}] 0..1 (1 = чернила); выход probs (softmax, T={T:.3f})"
    onnx.save(model, str(onnx_path))
    ops = sorted({n.op_type for n in model.graph.node})
    opset = [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import]
    onnx.save(to_fp16_weights(model), str(fp16_path))

    labels = {
        "model": "doodle.onnx",
        "model_fp16": "doodle_fp16.onnx",
        "run": run.name, "epoch": ck.get("epoch"), "opset": OPSET,
        "input": {"name": "input", "shape": [1, 1, size, size], "dtype": "float32", "layout": "NCHW",
                  "values": "0 = фон, 1 = чернила; строка 0 — верх", "spec": "docs/PREPROCESS.md"},
        "output": {"name": "probs", "shape": [1, len(classes)], "softmax": True, "temperature": T},
        "preprocess": {"epsilon": pre["epsilon"], "scale_max": pre.get("scale_max", 255.0), "size": size,
                       "pad": pre["pad"], "line_width": pre["line_width"]},
        "reject": {"label": "не разобрал",
                   "rule": "top-1 = other ИЛИ probs[top-1] < threshold класса → «не разобрал»"},
        "classes": [{
            "index": i, "id": c["id"], "quickdraw": c.get("quickdraw", []), "ru": c["ru"],
            "ru_acc": c.get("ru_acc"), "task": bool(c.get("task")), "kind": c.get("kind", "quickdraw"),
            "threshold": cal["thresholds"].get(c["id"], None if c["id"] == "other" else 0.6),
        } for i, c in enumerate(classes)],
    }
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ── паритет torch ↔ onnxruntime на эталонных векторах и на тестовых картинках
    xs, names = [], []
    for p in sorted(VEC.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["params"]["size"] != size or not d["simplified"]:
            continue
        xs.append(pp.preprocess(d["input"], size, pre["pad"], pre["line_width"], pre["epsilon"]))
        names.append(d["name"])
    try:
        timg = np.load(Path(ck["data"]) / "test_images.npy", mmap_mode="r")[:256].astype(np.float32) / 255.0
        xs += list(timg)
    except OSError:
        pass
    X = np.stack(xs)[:, None].astype(np.float32)
    with torch.no_grad():
        pt = ref_model(torch.from_numpy(X)).numpy()
    s32, s16 = ort_session(onnx_path), ort_session(fp16_path)
    o32 = np.concatenate([s32.run(None, {"input": X[i:i + 1]})[0] for i in range(len(X))])
    o16 = np.concatenate([s16.run(None, {"input": X[i:i + 1]})[0] for i in range(len(X))])
    d32, d16 = float(np.abs(o32 - pt).max()), float(np.abs(o16 - pt).max())
    agree16 = float((o16.argmax(1) == pt.argmax(1)).mean())

    lat1 = latency_ms(ort_session(onnx_path, 1), len(classes), size)
    latn = latency_ms(ort_session(onnx_path), len(classes), size)
    lat16 = latency_ms(ort_session(fp16_path, 1), len(classes), size)
    rep = {
        "onnx": str(onnx_path), "opset": opset, "ops": ops, "params": count_params(net),
        "size_fp32_bytes": onnx_path.stat().st_size, "size_fp16_bytes": fp16_path.stat().st_size,
        "parity_max_abs_fp32": d32, "parity_max_abs_fp16": d16, "fp16_top1_agreement": agree16,
        "parity_samples": len(X), "vector_names": names,
        "cpu_ms_1thread": lat1, "cpu_ms_default_threads": latn, "cpu_ms_fp16_1thread": lat16,
        "onnxruntime": ort.__version__, "torch": torch.__version__,
    }
    (out / "export_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = d32 < 1e-4
    print(f"ONNX opset {OPSET} ({', '.join(opset)}), операции: {', '.join(ops)}")
    print(f"  {onnx_path.name}: {rep['size_fp32_bytes'] / 1e6:.2f} МБ;  {fp16_path.name}: {rep['size_fp16_bytes'] / 1e6:.2f} МБ")
    print(f"  паритет torch↔ORT на {len(X)} входах: fp32 max|Δ| = {d32:.2e} {'OK' if ok else 'ПЛОХО'}; "
          f"fp16 max|Δ| = {d16:.2e}, совпадение top-1 {agree16:.3f}")
    print(f"  CPU (onnxruntime): {lat1:.3f} мс (1 поток), {latn:.3f} мс (все потоки), fp16 {lat16:.3f} мс")
    if not args.no_latest:
        latest = MODELS / "latest"
        latest.mkdir(parents=True, exist_ok=True)
        for f in ("doodle.onnx", "doodle_fp16.onnx", "labels.json", "export_report.json"):
            shutil.copy2(out / f, latest / f)
        print(f"  скопировано в {latest}")
    if not ok:
        raise SystemExit("Паритет fp32 не сошёлся — экспорт некорректен.")


if __name__ == "__main__":
    main()
