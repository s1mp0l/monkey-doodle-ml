"""Обучение DoodleNet на подготовленных данных (GPU-растеризация + аугментации на лету).

    python scripts/train.py --data data/prepared/smoke --epochs 6
    python scripts/train.py --data data/prepared/full           # полное обучение

Выход: runs/<YYYYmmdd-HHMMSS>/  best.pt, last.pt, metrics.csv, train_args.json, aug_samples.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from _common import ROOT, RUNS
from doodle.data import BalancedSampler, PreparedData, class_weights
from doodle.gpu_raster import augment, make_aug_table, rasterize_batch
from doodle.model import DoodleNet, count_params
from doodle.preprocess import to_png_bytes


def evaluate_split(net, bank, labels, pre, batch=1024, amp=True):
    net.eval()
    correct, loss_sum, n = 0, 0.0, len(labels)
    with torch.no_grad():
        for i in range(0, n, batch):
            idx = torch.arange(i, min(n, i + batch), device=labels.device)
            pts, sid, valid = bank.gather(idx)
            x = rasterize_batch(pts, sid, valid, pre["size"], pre["pad"], pre["line_width"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = net(x)
            y = labels[idx]
            loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
    return correct / max(1, n), loss_sum / max(1, n)


def main():
    ap = argparse.ArgumentParser(description="Обучить классификатор рисунков (CNN) на data/prepared/<profile>.")
    ap.add_argument("--data", default=str(ROOT / "data" / "prepared" / "full"), help="каталог prepare.py")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3, help="пиковый LR (OneCycle)")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--epoch-size", type=int, default=0, help="примеров за эпоху (0 = размер train)")
    ap.add_argument("--patience", type=int, default=5, help="ранняя остановка: эпох без улучшения val acc")
    ap.add_argument("--widths", default="32,64,128,256", help="ширины блоков CNN")
    ap.add_argument("--no-amp", action="store_true", help="без смешанной точности (bf16)")
    ap.add_argument("--no-aug", action="store_true", help="без аугментаций (для отладки)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-weights", default=None,
                    help="вес примера в loss по источнику, напр. synthetic=0.5,human=1 (по умолчанию — из meta.json/конфига)")
    ap.add_argument("--out", default=None, help="каталог прогона (по умолчанию runs/<время>)")
    ap.add_argument("--cpu", action="store_true", help="разрешить обучение на CPU (очень медленно)")
    args = ap.parse_args()

    if not torch.cuda.is_available() and not args.cpu:
        sys.exit("CUDA недоступна (torch.cuda.is_available() == False). Проверьте установку PyTorch с CUDA (README) или --cpu.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and dev.type == "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data = PreparedData(args.data)
    pre = data.pre
    C = data.num_classes
    names = [c["id"] for c in data.classes]
    counts = np.array([data.meta["counts"]["train"][n] for n in names])
    for w in data.meta.get("warnings", []):
        print("ВНИМАНИЕ (prepare):", w)
    for c, n in zip(data.classes, counts):
        if c.get("min_samples") and n < c["min_samples"]:
            print(f"ВНИМАНИЕ: класс '{c['id']}' — {n} примеров в train < min_samples {c['min_samples']}: "
                  f"сильный oversampling + тяжёлые аугментации, но точность на нём будет низкой.")
    for cid in data.meta.get("excluded_classes", []):
        print(f"ВНИМАНИЕ: класс '{cid}' пуст и исключён из модели.")

    bank, labels = data.bank("train", dev)
    src_ids = data.meta.get("sources", {"quickdraw": 0, "human": 1, "synthetic": 2})
    sw_cfg = {**{"quickdraw": 1.0, "human": 1.0, "synthetic": 1.0}, **data.meta.get("source_loss_weights", {})}
    if args.source_weights:
        for kv in args.source_weights.split(","):
            k, v = kv.split("=")
            sw_cfg[k.strip()] = float(v)
    w_table = torch.ones(max(src_ids.values()) + 1, device=dev)
    for k, i in src_ids.items():
        w_table[i] = float(sw_cfg.get(k, 1.0))
    train_src = torch.as_tensor(np.asarray(data.arrays("train")["src"]).astype(np.int64), device=dev)
    src_count = {k: int((train_src == i).sum()) for k, i in src_ids.items()}
    print("  источники train: " + ", ".join(f"{k}={src_count[k]} (вес {sw_cfg.get(k, 1.0)})" for k in src_ids))
    vbank, vlabels = data.bank("val", dev)
    labels_np = labels.cpu().numpy()
    sw = np.array([c.get("sampling_weight", 1.0) for c in data.classes])
    weights = class_weights(counts, sw)
    epoch_size = args.epoch_size or len(labels_np)
    sampler = BalancedSampler(labels_np, C, weights, epoch_size, args.seed)
    reps = sampler.expected_repeats()
    print(f"Классов {C}, train {len(labels_np):,}, val {len(vlabels):,}, эпоха {epoch_size:,} примеров")
    print("  повторов примера за эпоху: " + ", ".join(f"{n}={r:.2f}" for n, r in zip(names, reps)))
    table = make_aug_table(data.aug_params(), dev)

    widths = tuple(int(w) for w in args.widths.split(","))
    net = DoodleNet(C, widths).to(dev).to(memory_format=torch.channels_last)
    print(f"DoodleNet {widths}: {count_params(net):,} параметров; AMP bf16: {amp}")
    steps_per_epoch = math.ceil(epoch_size / args.batch)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15, div_factor=20, final_div_factor=200)

    run = ROOT / args.out if args.out else RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "train_args.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "data": str(data.root), "classes": names}, f, ensure_ascii=False, indent=2)
    print(f"Прогон: {run}")

    # образцы аугментаций — глазами проверить, что модель видит
    with torch.no_grad():
        idx = torch.as_tensor(sampler.epoch()[:64], device=dev)
        pts, sid, valid = bank.gather(idx)
        q, v2, lw, sh = augment(pts, sid, valid, labels[idx], table)
        x = rasterize_batch(q, sid, v2, pre["size"], pre["pad"], lw, sh).cpu().numpy()[:, 0]
        grid = np.ones((8 * (pre["size"] + 2), 8 * (pre["size"] + 2)), np.float32) * 0.15
        for k in range(min(64, len(x))):
            r, c = divmod(k, 8)
            s = pre["size"] + 2
            grid[r * s + 1:r * s + 1 + pre["size"], c * s + 1:c * s + 1 + pre["size"]] = x[k]
        (run / "aug_samples.png").write_bytes(to_png_bytes(grid, invert=True, scale=2))

    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]
    with open(run / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(fields)

    best, best_epoch, bad = -1.0, -1, 0
    ckpt_common = {"model_config": net.config, "classes": data.classes, "preprocess": pre, "source_weights": sw_cfg,
                   "data": str(data.root), "args": vars(args)}
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        order = torch.as_tensor(sampler.epoch(), device=dev)
        tl, tc, tn = 0.0, 0, 0
        for s in range(steps_per_epoch):
            idx = order[s * args.batch:(s + 1) * args.batch]
            if len(idx) == 0:
                break
            y = labels[idx]
            with torch.no_grad():
                pts, sid, valid = bank.gather(idx)
                if args.no_aug:
                    x = rasterize_batch(pts, sid, valid, pre["size"], pre["pad"], pre["line_width"])
                else:
                    q, v2, lw, sh = augment(pts, sid, valid, y, table)
                    x = rasterize_batch(q, sid, v2, pre["size"], pre["pad"], lw, sh)
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = net(x)
                wt = w_table[train_src[idx]]
                loss = (F.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing, reduction="none") * wt).sum() / wt.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
            tc += (logits.argmax(1) == y).sum().item()
            tn += len(idx)
        va, vl = evaluate_split(net, vbank, vlabels, pre, amp=amp)
        dt = time.time() - t0
        row = [ep, tl / tn, tc / tn, vl, va, sched.get_last_lr()[0], dt]
        with open(run / "metrics.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{v:.5f}" if isinstance(v, float) else v for v in row])
        mark = ""
        torch.save({**ckpt_common, "model": net.state_dict(), "epoch": ep, "val_acc": va}, run / "last.pt")
        if va > best:
            best, best_epoch, bad = va, ep, 0
            torch.save({**ckpt_common, "model": net.state_dict(), "epoch": ep, "val_acc": va}, run / "best.pt")
            mark = "  *"
        else:
            bad += 1
        print(f"эпоха {ep:>3}: train loss {tl / tn:.4f} acc {tc / tn:.4f} | val loss {vl:.4f} acc {va:.4f} | "
              f"{dt:.1f} с ({tn / dt:,.0f} пр/с){mark}", flush=True)
        if bad >= args.patience:
            print(f"Ранняя остановка: {args.patience} эпох без улучшения.")
            break
    print(f"Лучшая val acc {best:.4f} (эпоха {best_epoch}). Чекпоинт: {run / 'best.pt'}")
    print(f"Дальше: python scripts/evaluate.py --run {run}")


if __name__ == "__main__":
    main()
