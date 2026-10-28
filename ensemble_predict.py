#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个独立微调的 UER 模型 soft-voting；每次只加载一个模型以控制显存。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import compute_metrics, dump_json, print_metrics, read_csv, save_errors, save_predictions
from predict_sentiment import predict_transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="多个 RoBERTa 最佳模型的概率平均集成")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--models", nargs="+", help="空格分隔的 best_model 目录")
    selection.add_argument("--leaderboard", help="通过 sweep_transformer.py 的排行榜加载前 K 个模型")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--input-csv", default="data/validation.csv")
    parser.add_argument("--output", default="outputs/ensemble/validation_predictions.csv")
    parser.add_argument("--weights", help="可选模型投票权重，按模型顺序逗号分隔，例如 1,1,2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("batch-size/top-k 必须 > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if args.leaderboard:
        leaderboard = pd.read_csv(args.leaderboard)
        if "model_dir" not in leaderboard.columns or "macro_f1" not in leaderboard.columns:
            raise ValueError("排行榜文件缺少 model_dir 或 macro_f1 列")
        models = leaderboard.sort_values("macro_f1", ascending=False)["model_dir"].head(args.top_k).tolist()
    else:
        models = args.models
    if not models or len(set(models)) != len(models):
        raise ValueError("模型列表为空或存在重复模型")
    if args.weights:
        weights = np.asarray([float(v.strip()) for v in args.weights.split(",")], dtype=np.float64)
        if len(weights) != len(models) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("weights 数量必须匹配模型数量，全部非负且总和大于 0")
    else:
        weights = np.ones(len(models), dtype=np.float64)
    weights /= weights.sum()
    frame = read_csv(args.input_csv, require_labels=False)
    all_probabilities = []
    configs = []
    for i, model_string in enumerate(models):
        model_dir = Path(model_string)
        cfg_path = model_dir.parent / "training_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"模型缺少 training_config.json，需提供微调时的 max_length: {model_dir}")
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        max_length = int(cfg["max_length"])
        print(f"[Ensemble {i + 1}/{len(models)}] {model_dir}, max_length={max_length}, weight={weights[i]:.3f}", flush=True)
        _, probs = predict_transformer(model_dir, frame["text"].tolist(), args.batch_size, max_length, device)
        all_probabilities.append(probs.astype(np.float64))
        configs.append({"model_dir": str(model_dir.resolve()), "max_length": max_length, "weight": float(weights[i])})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ensemble_probs = np.average(np.stack(all_probabilities), axis=0, weights=weights)
    predictions = (ensemble_probs > 0.5).astype(np.int64)
    save_predictions(frame, predictions, args.output, ensemble_probs)
    metadata = {"models": configs, "type": "probability_soft_voting", "samples": len(frame)}
    if "label" in frame:
        metrics = compute_metrics(frame["label"].to_numpy(), predictions)
        metadata["metrics"] = metrics
        print_metrics("Ensemble", metrics)
        save_errors(frame, predictions, Path(args.output).with_name("ensemble_errors.csv"), ensemble_probs)
    dump_json(metadata, Path(args.output).with_suffix(".metadata.json"))
    print("结果:", Path(args.output).resolve())


if __name__ == "__main__":
    main()
