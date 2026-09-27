"""Оценка обученной модели на test (эталонная растеризация из prepare.py) + калибровка и пороги.

    python scripts/evaluate.py --run runs/<время>

Пишет в runs/<время>/eval/:
    report.md          точность, per-class precision/recall/F1, калибровка, рекомендованные пороги
    metrics.json       то же машиночитаемо
    confusion.png      матрица ошибок (строки — истинный класс, нормировано по строке)
    calibration.json   температура T и пороги по классам → их берёт scripts/export_onnx.py
Пороги подбираются на VAL (не на test): минимальный порог уверенности, при котором точность
«узнал класс c» ≥ target_precision класса. Ниже порога (или top-1 = other) → «не разобрал».
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _common import ROOT  # noqa: F401
from doodle.data import PreparedData
from doodle.model import load_checkpoint


def logits_for(net, images_u8: np.ndarray, dev, batch=2048) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(images_u8), batch):
            x = torch.from_numpy(np.asarray(images_u8[i:i + batch], dtype=np.float32) / 255.0).to(dev)[:, None]
            out.append(net(x).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0), np.float32)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    lg = torch.from_numpy(logits).double()
    yt = torch.from_numpy(y).long()
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def per_class(y: np.ndarray, pred: np.ndarray, C: int):
    cm = np.zeros((C, C), np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(np.float64)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return cm, prec, rec, f1


def pick_thresholds(probs: np.ndarray, y: np.ndarray, classes: list[dict], other_idx: int | None, floor: float = 0.5):
    """Порог для класса c: минимальный t ≥ floor из сетки, при котором precision(top1 = c и p ≥ t) ≥ target."""
    grid = np.round(np.arange(floor, 0.991, 0.01), 2)
    pred, conf = probs.argmax(1), probs.max(1)
    out = []
    for c, spec in enumerate(classes):
        target = float(spec.get("target_precision", 0.9))
        n_true = int((y == c).sum())
        if c == other_idx:
            out.append({"id": spec["id"], "threshold": None, "note": "класс-отказ: top-1 = other → «не разобрал»"})
            continue
        chosen = None
        for t in grid:
            m = (pred == c) & (conf >= t)
            if m.sum() == 0:
                break
            p = (y[m] == c).mean()
            if p >= target:
                chosen = (float(t), float(p), float(m[y == c].mean()) if n_true else 0.0, int(m.sum()))
                break
        if chosen is None:
            out.append({"id": spec["id"], "threshold": 0.99, "precision": None, "recall": None,
                        "note": f"точность {target:.2f} недостижима на val — нужен порог выше/больше данных"})
        else:
            t, p, r, n = chosen
            out.append({"id": spec["id"], "threshold": t, "precision": round(p, 4), "recall": round(r, 4),
                        "target_precision": target, "support_val": n_true})
    return out


def draw_confusion(cm: np.ndarray, names: list[str], path: Path):
    from PIL import Image, ImageDraw, ImageFont

    C = len(names)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cell = 34 if C <= 20 else 26
    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):        # нужен шрифт с кириллицей
        try:
            font = ImageFont.truetype(name, 12)
            break
        except OSError:
            pass
    if font is None:
        font = ImageFont.load_default()
    left, top = 110, 110
    W, H = left + C * cell + 10, top + C * cell + 10
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for i in range(C):
        for j in range(C):
            v = norm[i, j]
            shade = int(255 - 215 * v)
            col = (shade, shade, 255) if i == j else (255, shade, shade)
            x0, y0 = left + j * cell, top + i * cell
            d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=col, outline=(230, 230, 230))
            if v >= 0.01:
                txt = f"{int(round(v * 100))}"
                d.text((x0 + cell / 2, y0 + cell / 2), txt, fill="black" if v < 0.6 else "white", font=font, anchor="mm")
    for i, n in enumerate(names):
        d.text((left - 6, top + i * cell + cell / 2), n[:14], fill="black", font=font, anchor="rm")
        # подписи столбцов — повёрнутым текстом
        tw = Image.new("RGBA", (100, cell), (255, 255, 255, 0))
        ImageDraw.Draw(tw).text((4, cell / 2), n[:14], fill="black", font=font, anchor="lm")
        tw = tw.rotate(90, expand=True)
        im.paste(tw, (left + i * cell, top - 102), tw)
    d.text((6, 6), "строки: истинный класс, столбцы: предсказание, % по строке", fill=(90, 90, 90), font=font)
    im.save(path)


def rude_report(names, yt, pt, tvec, other_idx, data, out: Path) -> dict | None:
    """rude на отложенных рисунках ЛЮДЕЙ (test не участвует ни в обучении, ни в порогах)."""
    if "rude" not in names:
        return None
    r = names.index("rude")
    src = np.asarray(data.arrays("test")["src"]).astype(np.int64)
    human_id = data.meta.get("sources", {}).get("human", 1)
    pred, conf = pt.argmax(1), pt.max(1)
    acc_r = (pred == r) & (conf >= tvec[r])
    is_r = yt == r
    hum_r = is_r & (src == human_id)
    n_h = int(hum_r.sum())
    if n_h == 0:
        return {"human_test": 0}

    def prf(p_mask):
        tp = int((p_mask & hum_r).sum())
        fp = int((p_mask & ~is_r).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, n_h)
        return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-12, prec + rec), "tp": tp, "fp": fp}

    res = {"human_test": n_h, "top1": prf(pred == r), "with_threshold": prf(acc_r), "threshold": float(tvec[r])}
    # кого называют rude и как называют rude
    called_rude = []
    for c, n in enumerate(names):
        if c == r:
            continue
        m = yt == c
        k = int((m & (pred == r)).sum())
        if k:
            called_rude.append({"true": n, "count": k, "rate": k / max(1, int(m.sum()))})
    called_rude.sort(key=lambda d: -d["count"])
    rude_as = [{"pred": names[c], "count": int((hum_r & (pred == c)).sum())} for c in range(len(names)) if c != r]
    rude_as = sorted([d for d in rude_as if d["count"]], key=lambda d: -d["count"])
    res["called_rude"] = called_rude[:10]
    res["rude_called_as"] = rude_as[:10]
    # по подтипам (ручная разметка data/custom/rude_subtypes.json, если есть)
    keys_path = data.root / "test_keys.json"
    sub_path = ROOT / "data" / "custom" / "rude_subtypes.json"
    if keys_path.exists() and sub_path.exists():
        keys = json.loads(keys_path.read_text(encoding="utf-8"))
        sub = json.loads(sub_path.read_text(encoding="utf-8"))["subtypes"]
        per = {}
        for i in np.flatnonzero(hum_r):
            st = sub.get(keys[i].split(":", 1)[-1], "?")
            d = per.setdefault(st, [0, 0, 0])
            d[0] += 1
            d[1] += int(pred[i] == r)
            d[2] += int(acc_r[i])
        res["by_subtype"] = {k: {"n": v[0], "recall_top1": v[1] / v[0], "recall_thr": v[2] / v[0]} for k, v in sorted(per.items())}
    # картинка ошибок: пропущенные rude и ложные rude
    try:
        from PIL import Image
        imgs = data.images("test")
        miss = [i for i in np.flatnonzero(hum_r & ~acc_r)][:40]
        fps = [i for i in np.argsort(-pt[:, r]) if not is_r[i] and acc_r[i]][:40]
        cell = 66
        im = Image.new("L", (10 * cell, (4 + 4) * cell + 10), 255)
        for row0, lst in ((0, miss), (4, fps)):
            for k, i in enumerate(lst):
                a = 255 - np.asarray(imgs[i])
                im.paste(Image.fromarray(a.astype(np.uint8)), ((k % 10) * cell + 1, (row0 + k // 10) * cell + 1 + (10 if row0 else 0)))
        im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
        im.save(out / "rude_errors.png")
        res["errors_png"] = "rude_errors.png (верх: пропущенные rude людей; низ: ложные rude)"
    except Exception as e:  # noqa: BLE001
        res["errors_png"] = f"не нарисовано: {e}"
    return res


def main():
    ap = argparse.ArgumentParser(description="Оценить прогон: test-метрики, калибровка, пороги «узнал / не разобрал».")
    ap.add_argument("--run", required=True, help="каталог runs/<время>")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data", default=None, help="каталог prepared (по умолчанию из чекпоинта)")
    ap.add_argument("--min-threshold", type=float, default=0.5, help="нижняя граница порога уверенности")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ck = load_checkpoint(run / args.ckpt, dev)
    data = PreparedData(args.data or ck["data"])
    classes = ck["classes"]
    names = [c["id"] for c in classes]
    if [c["id"] for c in data.classes] != names:
        raise SystemExit("Классы данных и чекпоинта не совпадают — укажите --data от этого прогона.")
    C = len(names)
    other_idx = names.index("other") if "other" in names else None

    yv = np.asarray(data.arrays("val")["label"]).astype(np.int64)
    yt = np.asarray(data.arrays("test")["label"]).astype(np.int64)
    lv = logits_for(net, data.images("val"), dev)
    lt = logits_for(net, data.images("test"), dev)

    T = fit_temperature(lv, yv)
    pt_raw, pt = softmax(lt, 1.0), softmax(lt, T)
    pv = softmax(lv, T)
    pred = pt.argmax(1)
    acc = float((pred == yt).mean())
    top3 = float((np.argsort(-pt, 1)[:, :3] == yt[:, None]).any(1).mean())
    cm, prec, rec, f1 = per_class(yt, pred, C)
    thr = pick_thresholds(pv, yv, classes, other_idx, args.min_threshold)

    # итог «в игре»: принятие / отказ на test с выбранными порогами
    tvec = np.array([t["threshold"] if t["threshold"] is not None else 2.0 for t in thr])
    conf = pt.max(1)
    accepted = (conf >= tvec[pred]) & (pred != (other_idx if other_idx is not None else -1))
    is_target = yt != (other_idx if other_idx is not None else -1)
    game = {
        "target_correct_accept": float((accepted & (pred == yt) & is_target).sum() / max(1, is_target.sum())),
        "target_wrong_accept": float((accepted & (pred != yt) & is_target).sum() / max(1, is_target.sum())),
        "target_rejected": float((~accepted & is_target).sum() / max(1, is_target.sum())),
        "other_false_accept": float((accepted & ~is_target).sum() / max(1, (~is_target).sum())) if other_idx is not None else None,
    }

    out = run / "eval"
    out.mkdir(exist_ok=True)
    rude = rude_report(names, yt, pt, tvec, other_idx, data, out)
    draw_confusion(cm, names, out / "confusion.png")
    metrics = {
        "checkpoint": str(run / args.ckpt), "epoch": ck.get("epoch"), "test_size": int(len(yt)),
        "accuracy": acc, "top3": top3, "temperature": T,
        "ece_raw": ece(pt_raw, yt), "ece_calibrated": ece(pt, yt),
        "per_class": [{"id": n, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(cm[i].sum())}
                      for i, (n, p, r, f) in enumerate(zip(names, prec, rec, f1))],
        "thresholds": thr, "game_simulation": game, "rude": rude,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out / "calibration.json", "w", encoding="utf-8") as f:
        json.dump({"temperature": T, "thresholds": {t["id"]: t["threshold"] for t in thr}}, f, ensure_ascii=False, indent=2)

    # самые частые путаницы
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], names[i], names[j]) for i in range(C) for j in range(C) if off[i, j] > 0), reverse=True)[:10]

    lines = [f"# Оценка {run.name}", "",
             f"- test: {len(yt)} рисунков, эпоха {ck.get('epoch')}",
             f"- **accuracy {acc:.4f}**, top-3 {top3:.4f}",
             f"- калибровка: T = {T:.3f}; ECE {metrics['ece_raw']:.4f} → {metrics['ece_calibrated']:.4f}",
             "", "## По классам (test)", "", "| класс | precision | recall | F1 | n |", "|---|---|---|---|---|"]
    lines += [f"| {m['id']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |" for m in metrics["per_class"]]
    lines += ["", "## Пороги «узнал» (подобраны на val, вероятности после калибровки)", "",
              "| класс | порог | precision | recall | примечание |", "|---|---|---|---|---|"]
    for t in thr:
        lines.append(f"| {t['id']} | {t['threshold'] if t['threshold'] is not None else '—'} | "
                     f"{t.get('precision') if t.get('precision') is not None else '—'} | "
                     f"{t.get('recall') if t.get('recall') is not None else '—'} | {t.get('note', '')} |")
    lines += ["", "## Симуляция игры на test (с порогами)", "",
              f"- задание узнано верно: {game['target_correct_accept']:.3f}",
              f"- принято, но не тот класс: {game['target_wrong_accept']:.3f}",
              f"- «не разобрал»: {game['target_rejected']:.3f}"]
    if game["other_false_accept"] is not None:
        lines.append(f"- «другое» принято за задание (ложное срабатывание): {game['other_false_accept']:.3f}")
    if rude and rude.get("human_test"):
        t1, tw = rude["top1"], rude["with_threshold"]
        lines += ["", f"## rude на отложенных рисунках людей (test: {rude['human_test']})", "",
                  "| правило | precision | recall | F1 | TP | FP |", "|---|---|---|---|---|---|",
                  f"| top-1 | {t1['precision']:.3f} | {t1['recall']:.3f} | {t1['f1']:.3f} | {t1['tp']} | {t1['fp']} |",
                  f"| top-1 и p ≥ {rude['threshold']:.2f} | {tw['precision']:.3f} | {tw['recall']:.3f} | {tw['f1']:.3f} | {tw['tp']} | {tw['fp']} |",
                  "", "Кого модель называет rude (test): " + (", ".join(f"{d['true']} {d['count']} ({d['rate']:.1%})" for d in rude["called_rude"]) or "никого"),
                  "", "Как называет рисунки rude людей: " + (", ".join(f"{d['pred']} {d['count']}" for d in rude["rude_called_as"]) or "всегда rude")]
        if rude.get("by_subtype"):
            lines += ["", "| подтип | n | recall top-1 | recall с порогом |", "|---|---|---|---|"]
            lines += [f"| {k} | {v['n']} | {v['recall_top1']:.2f} | {v['recall_thr']:.2f} |" for k, v in rude["by_subtype"].items()]
    lines += ["", "## Частые путаницы (истинный → предсказанный: число)", ""]
    lines += [f"- {a} → {b}: {n}" for n, a, b in pairs] or ["- нет"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:6]))
    print(f"Отчёт: {out / 'report.md'}\nДальше: python scripts/export_onnx.py --run {run}")


if __name__ == "__main__":
    main()
