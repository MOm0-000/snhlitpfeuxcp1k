#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep FC-only fine-tuning hyperparameters by repeatedly calling train_fc.py.

Designed for the current project structure:
  outputs/fc_only/train_cls_features.npz
  outputs/fc_only/validation_cls_features.npz

Each run gets its own output directory. After all runs, the script writes:
  <sweep_output>/sweep_results.csv
  <sweep_output>/best_run.json

The best configuration is selected by validation Macro-F1.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_csv_list(text: str, cast):
    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("参数列表不能为空")
    try:
        return [cast(x) for x in items]
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"无法解析参数列表: {text}") from exc


def parse_args():
    p = argparse.ArgumentParser(description="FC-only 超参数 sweep")
    p.add_argument("--train-script", default="train_fc.py")
    p.add_argument("--train-features", default="outputs/fc_only/train_cls_features.npz")
    p.add_argument("--validation-features", default="outputs/fc_only/validation_cls_features.npz")
    p.add_argument("--sweep-output", default="outputs/fc_sweep")

    # 默认给出一组足够写实验报告的对照，不会跑得过多。
    p.add_argument("--lrs", default="0.0005,0.001,0.002")
    p.add_argument("--weight-decays", default="0,0.01")
    p.add_argument("--batch-sizes", default="128")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", default="42")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="即使某个 run 已有 best_metrics.json，也重新训练",
    )
    p.add_argument(
        "--copy-best",
        action="store_true",
        help="将最佳 run 的关键文件复制到 <sweep-output>/best/",
    )
    return p.parse_args()


def load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "best_metrics.json"
    config_path = run_dir / "training_config.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"缺少 {metrics_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少 {config_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return {**config, **metrics}


def run_one(
    python_exe: str,
    train_script: Path,
    train_features: Path,
    validation_features: Path,
    run_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
    overwrite: bool,
):
    metrics_path = run_dir / "best_metrics.json"

    if metrics_path.is_file() and not overwrite:
        print(f"[SKIP] {run_dir} 已存在结果")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        str(train_script),
        "--train-features", str(train_features),
        "--validation-features", str(validation_features),
        "--output-dir", str(run_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--seed", str(seed),
        "--device", device,
    ]

    print("\n" + "=" * 88)
    print(
        f"RUN: lr={lr:g}, weight_decay={weight_decay:g}, "
        f"batch_size={batch_size}, seed={seed}"
    )
    print("CMD:", " ".join(cmd))
    print("=" * 88)

    subprocess.run(cmd, check=True)


def save_csv(rows: list[dict], path: Path):
    columns = [
        "run_name",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "seed",
        "best_epoch",
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "verified_reload",
        "trainable_parameters",
        "encoder_trainable_parameters_during_training",
        "run_dir",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def main():
    args = parse_args()

    train_script = Path(args.train_script)
    train_features = Path(args.train_features)
    validation_features = Path(args.validation_features)
    sweep_output = Path(args.sweep_output)

    if not train_script.is_file():
        raise FileNotFoundError(f"找不到训练脚本: {train_script}")
    if not train_features.is_file():
        raise FileNotFoundError(f"找不到训练特征: {train_features}")
    if not validation_features.is_file():
        raise FileNotFoundError(f"找不到验证特征: {validation_features}")

    lrs = parse_csv_list(args.lrs, float)
    weight_decays = parse_csv_list(args.weight_decays, float)
    batch_sizes = parse_csv_list(args.batch_sizes, int)
    seeds = parse_csv_list(args.seeds, int)

    if any(x <= 0 for x in lrs):
        raise ValueError("learning rate 必须 > 0")
    if any(x < 0 for x in weight_decays):
        raise ValueError("weight decay 必须 >= 0")
    if any(x < 1 for x in batch_sizes):
        raise ValueError("batch size 必须 >= 1")
    if args.epochs < 1:
        raise ValueError("epochs 必须 >= 1")

    sweep_output.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(lrs, weight_decays, batch_sizes, seeds))
    print(f"共 {len(combinations)} 组实验")

    run_dirs = []
    for idx, (lr, wd, bs, seed) in enumerate(combinations, start=1):
        # 文件名避免科学计数法和小数点过多。
        lr_tag = f"{lr:.8f}".rstrip("0").rstrip(".").replace(".", "p")
        wd_tag = f"{wd:.8f}".rstrip("0").rstrip(".").replace(".", "p")
        run_name = f"run_{idx:02d}_lr{lr_tag}_wd{wd_tag}_bs{bs}_s{seed}"
        run_dir = sweep_output / run_name
        run_dirs.append((run_name, run_dir))

        run_one(
            python_exe=sys.executable,
            train_script=train_script,
            train_features=train_features,
            validation_features=validation_features,
            run_dir=run_dir,
            epochs=args.epochs,
            batch_size=bs,
            lr=lr,
            weight_decay=wd,
            seed=seed,
            device=args.device,
            overwrite=args.overwrite,
        )

    rows = []
    for run_name, run_dir in run_dirs:
        info = load_metrics(run_dir)
        rows.append({
            "run_name": run_name,
            "learning_rate": info.get("learning_rate"),
            "weight_decay": info.get("weight_decay"),
            "batch_size": info.get("batch_size"),
            "epochs": info.get("epochs"),
            "seed": info.get("seed"),
            "best_epoch": info.get("best_epoch"),
            "accuracy": info.get("accuracy"),
            "macro_f1": info.get("macro_f1"),
            "macro_precision": info.get("macro_precision"),
            "macro_recall": info.get("macro_recall"),
            "verified_reload": info.get("verified_reload"),
            "trainable_parameters": info.get("trainable_parameters"),
            "encoder_trainable_parameters_during_training": info.get(
                "encoder_trainable_parameters_during_training"
            ),
            "run_dir": str(run_dir),
        })

    rows.sort(
        key=lambda r: (
            float(r["macro_f1"]),
            float(r["accuracy"]),
        ),
        reverse=True,
    )

    results_path = sweep_output / "sweep_results.csv"
    save_csv(rows, results_path)

    best = rows[0]
    best_summary = {
        "selection_metric": "validation_macro_f1",
        "best_run": best,
        "num_runs": len(rows),
        "results_csv": str(results_path),
    }
    with (sweep_output / "best_run.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    if args.copy_best:
        source = Path(best["run_dir"])
        target = sweep_output / "best"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        for name in [
            "classifier_best.pt",
            "classifier_last.pt",
            "history.csv",
            "training_config.json",
            "best_metrics.json",
            "validation_predictions.csv",
        ]:
            src = source / name
            if src.is_file():
                shutil.copy2(src, target / name)

        with (target / "selected_from.json").open("w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)

    print("\n" + "#" * 88)
    print("Sweep 完成，按 validation Macro-F1 排序：")
    for i, row in enumerate(rows, start=1):
        print(
            f"{i:02d}. {row['run_name']} | "
            f"F1={float(row['macro_f1']):.6f} | "
            f"Acc={float(row['accuracy']):.6f} | "
            f"best_epoch={row['best_epoch']}"
        )

    print("\n最佳配置：")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"\n汇总结果: {results_path}")
    print(f"最佳配置摘要: {sweep_output / 'best_run.json'}")
    if args.copy_best:
        print(f"最佳 run 文件副本: {sweep_output / 'best'}")


if __name__ == "__main__":
    main()
