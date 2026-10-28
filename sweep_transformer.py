#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在同一划分、同一随机种子下比较数个微调设置（依次单进程运行，避免 GPU 显存堆积）。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROFILES = {
    "baseline": {
        "max_length": 128, "lr": 3e-5, "label_smoothing": 0.0,
        "layerwise_lr_decay": 1.0, "fgm_epsilon": 0.0, "rdrop_alpha": 0.0,
    },
    "longer": {
        "max_length": 192, "lr": 2e-5, "label_smoothing": 0.04,
        "layerwise_lr_decay": 0.92, "fgm_epsilon": 0.0, "rdrop_alpha": 0.0,
    },
    "fgm": {
        "max_length": 192, "lr": 2e-5, "label_smoothing": 0.04,
        "layerwise_lr_decay": 0.92, "fgm_epsilon": 0.5, "rdrop_alpha": 0.0,
    },
    "rdrop": {
        "max_length": 160, "lr": 2e-5, "label_smoothing": 0.04,
        "layerwise_lr_decay": 0.92, "fgm_epsilon": 0.0, "rdrop_alpha": 0.5,
    },
}


def main():
    p = argparse.ArgumentParser(description="RoBERTa 微调配置对比（Macro-F1 排名）")
    p.add_argument("--profiles", default="baseline,longer,fgm", help="可选 baseline,longer,fgm,rdrop")
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--model", default="uer/chinese_roberta_L-8_H-512")
    p.add_argument("--output-dir", default="outputs/sweep")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rerun", action="store_true", help="对已经完成的实验也重新执行")
    p.add_argument("--dry-run", action="store_true", help="只显示命令，不执行")
    args = p.parse_args()
    profiles = [name.strip() for name in args.profiles.split(",")]
    if len(set(profiles)) != len(profiles) or any(name not in PROFILES for name in profiles):
        raise ValueError(f"--profiles 必须从 {list(PROFILES)} 中选择且不能重复")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    script = Path(__file__).with_name("train_transformer.py")
    for name in profiles:
        run_out = out / name
        metrics_file = run_out / "restored_metrics.json"
        command = [
            sys.executable, str(script), "--train", args.train, "--validation", args.validation,
            "--model", args.model, "--output-dir", str(run_out),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--eval-batch-size", str(args.eval_batch_size),
            "--grad-accum", str(args.grad_accum), "--seed", str(args.seed),
        ]
        for key, val in PROFILES[name].items():
            command += [f"--{key.replace('_', '-')}", str(val)]
        print(f"\n===== profile {name} =====\n{' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        if args.rerun or not metrics_file.is_file():
            subprocess.run(command, check=True)
        with metrics_file.open(encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({
            "profile": name, "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"], "best_epoch": metrics["best_epoch"],
            "model_dir": str((run_out / "best_model").resolve()),
            **PROFILES[name],
        })
        pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(
            out / "leaderboard.csv", index=False, encoding="utf-8-sig",
        )
    if rows:
        table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
        print("\n验证 Macro-F1 排序：")
        print(table[["profile", "macro_f1", "accuracy", "best_epoch", "model_dir"]].to_string(index=False))
        print("完整表格:", (out / "leaderboard.csv").resolve())


if __name__ == "__main__":
    main()
