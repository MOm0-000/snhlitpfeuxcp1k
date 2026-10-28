#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键跑完课程 15 种传统模型、预训练对照、多模型预测差异分析。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="MARC 中文情感分类整套实验")
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--model", default="uer/chinese_roberta_L-8_H-512")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--profiles", default="baseline,longer,fgm", help="微调对照方案")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--skip-classical", action="store_true")
    p.add_argument("--skip-transformer", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-ensemble", action="store_true", help="关闭多模型软投票集成")
    args = p.parse_args()
    root = Path(args.output_dir)
    source = Path(__file__).parent
    classical = root / "classical"
    sweep = root / "sweep"

    def execute(name: str, script: str, params: list[str]):
        cmd = [sys.executable, str(source / script), *params]
        print(f"\n========== {name} ==========\n$ {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if not args.skip_classical:
        execute("15 种经典分类器 + 特征优化", "train_classical.py", [
            "--train", args.train, "--validation", args.validation,
            "--output-dir", str(classical),
        ])
    if not args.skip_transformer:
        execute("RoBERTa 多配置微调", "sweep_transformer.py", [
            "--train", args.train, "--validation", args.validation,
            "--model", args.model, "--output-dir", str(sweep),
            "--profiles", args.profiles, "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--eval-batch-size", str(args.eval_batch_size),
            "--grad-accum", str(args.grad_accum),
        ])
    leaderboard = sweep / "leaderboard.csv"
    classical_model = classical / "best_classical.joblib"
    if args.dry_run:
        return
    if classical_model.is_file() and leaderboard.is_file():
        best_transformer = str(pd.read_csv(leaderboard).iloc[0]["model_dir"])
        classical_preds = root / "classical_best_validation.csv"
        transformer_preds = root / "transformer_best_validation.csv"
        execute("最优传统模型验证集预测", "predict_sentiment.py", [
            "--model", str(classical_model), "--input-csv", args.validation,
            "--output", str(classical_preds),
        ])
        execute("最优 RoBERTa 验证集预测", "predict_sentiment.py", [
            "--model", best_transformer, "--input-csv", args.validation,
            "--output", str(transformer_preds),
        ])
        ensemble_preds = root / "ensemble_validation.csv"
        if not args.no_ensemble:
            execute("多模型 soft-voting 集成", "ensemble_predict.py", [
                "--leaderboard", str(leaderboard), "--top-k", "3",
                "--input-csv", args.validation, "--output", str(ensemble_preds),
            ])
            from common import compute_metrics, read_csv
            labels = read_csv(args.validation)["label"].to_numpy()
            ensemble_scores = compute_metrics(labels, pd.read_csv(ensemble_preds)["prediction"].to_numpy())
            single_scores = compute_metrics(labels, pd.read_csv(transformer_preds)["prediction"].to_numpy())
            if ensemble_scores["macro_f1"] > single_scores["macro_f1"]:
                print("集成的验证 Macro-F1 更高，后续误差对比选用集成结果")
                transformer_preds = ensemble_preds
            else:
                print("单模型的验证 Macro-F1 不低于集成，后续误差对比继续使用单模型")
        execute("否定/转折/模型分歧案例分析", "analyze_errors.py", [
            "--validation", args.validation,
            "--classical", str(classical_preds),
            "--transformer", str(transformer_preds),
            "--output-dir", str(root / "analysis"),
        ])
    else:
        print("传统或预训练模型结果尚不存在；跳过跨模型对比分析。")
    print("\n实验文件目录:", root.resolve())


if __name__ == "__main__":
    main()
