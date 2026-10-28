#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一恢复 A/B/C/D 的原始/修正训练模型，并评价两套验证标签。

四个模型 × 两种训练集 × 两种验证标签 = 16 行评估结果；实际只加载8个checkpoint。
为了严格区分标签改变与模型改变，对每个checkpoint仅推理一次，
然后用同一份预测分别计算原始验证标签和部分修正验证标签上的指标。

输出只保留：outputs/abcd/comparison.csv、outputs/abcd/predictions.csv。
依赖 train_A.py 的模型定义，不依赖旧 predict_sentiment.py / train_transformer.py。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from train_A import (
    calculate_metrics, load_checkpoint, read_reviews, verify_same_comments,
)


def predict_texts(model, tokenizer, texts: list[str], max_length: int, batch_size: int,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, probabilities = [], []
    for start in tqdm(range(0, len(texts), batch_size), desc="测试", leave=False):
        batch = tokenizer(
            texts[start:start + batch_size], return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        )
        batch = {key: val.to(device) for key, val in batch.items()}
        with torch.inference_mode():
            # 与train_A中保存最佳epoch时一致：float32推理。
            logits = model(**batch).logits.float()
            positive = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            predicted = logits.argmax(dim=-1).cpu().numpy()
        probabilities.append(positive)
        predictions.append(predicted)
    return np.concatenate(predictions), np.concatenate(probabilities)


def parse_args():
    parser = argparse.ArgumentParser(description="ABCD × 原始/修正训练集 × 原始/修正验证集")
    parser.add_argument("--output-root", default="outputs/abcd")
    parser.add_argument("--validation", default="data/validation.csv")
    parser.add_argument("--semantic-validation", default="outputs/label_audit_v2/validation_semantic_partial.csv")
    parser.add_argument("--models", default="A,B,C,D", help="可选A,B,C,D的逗号列表")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--a-original-dir", default=None,
                        help="如不想重训A，可使用旧 outputs/sweep/fgm/best_model")
    parser.add_argument("--a-cleaned-dir", default=None,
                        help="如不想重训A，可使用旧 outputs/fgm_cleaned_train/best_model")
    parser.add_argument("--legacy-max-length", type=int, default=192)
    parser.add_argument("--no-demo", action="store_true", help="不展示最优模型的新评论预测")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size须大于0")
    wanted = [item.strip().upper() for item in args.models.split(",") if item.strip()]
    if not wanted or len(wanted) != len(set(wanted)) or any(item not in "ABCD" for item in wanted):
        raise ValueError("--models只能为不重复的A,B,C,D列表")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch没有检测到CUDA")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    base = read_reviews(args.validation)
    revised = read_reviews(args.semantic_validation)
    verify_same_comments(base, revised, "验证集")
    revised = revised.set_index("id").loc[base["id"].tolist()].reset_index()
    original_labels = base["label"].to_numpy()
    revised_labels = revised["label"].to_numpy()
    changed = int((original_labels != revised_labels).sum())
    print(f"验证集：{len(base)}条；两套标签相差{changed}条；对每个checkpoint只推理一次。")

    output_root = Path(args.output_root)
    rows = []
    pred_frames = []
    candidate_best = None
    demo_cases = [
        "东西很好用，质量不错，值得购买。",
        "质量太差了，一点也不好用，不推荐。",
        "包装很好但是商品有明显缺陷，体验很差。",
        "虽然物流慢了一点，但产品质量很好。",
        "不是不好用，不过性价比确实一般。",
    ]
    for variant in wanted:
        for train_version in ["original", "cleaned"]:
            if variant == "A" and train_version == "original" and args.a_original_dir:
                ckpt = Path(args.a_original_dir)
            elif variant == "A" and train_version == "cleaned" and args.a_cleaned_dir:
                ckpt = Path(args.a_cleaned_dir)
            else:
                ckpt = output_root / variant / train_version / "best_model"
            if not ckpt.is_dir():
                raise FileNotFoundError(f"找不到{variant}/{train_version}权重：{ckpt}。先运行train_{variant}.py。")
            legacy = "A" if variant == "A" else None
            model, tokenizer, info = load_checkpoint(
                ckpt, device, legacy_variant=legacy,
                legacy_max_length=args.legacy_max_length,
            )
            if info["variant"] != variant:
                raise ValueError(f"{ckpt}的variant是{info['variant']}，与请求的{variant}不一致")
            pred, pos_prob = predict_texts(
                model, tokenizer, base["text"].tolist(), int(info["max_length"]),
                args.batch_size, device,
            )
            pred_frames.append(pd.DataFrame({
                "model": variant, "train_version": train_version,
                "id": base["id"].to_numpy(), "text": base["text"].to_numpy(),
                "original_label": original_labels,
                "semantic_label": revised_labels,
                "prediction": pred, "positive_probability": pos_prob,
            }))
            for label_type, labels in [("original", original_labels), ("semantic_partial", revised_labels)]:
                metrics = calculate_metrics(labels, pred)
                cm = metrics["confusion_matrix"]
                rows.append({
                    "model": variant, "train_version": train_version,
                    "validation_labels": label_type,
                    "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "correct": metrics["correct"], "wrong": metrics["wrong"],
                    "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
                    "checkpoint": str(ckpt),
                })
                print(f"{variant}/{train_version} × {label_type}: "
                      f"F1={metrics['macro_f1']:.5f} "
                      f"Acc={metrics['accuracy']:.5f} "
                      f"错={metrics['wrong']}", flush=True)
                if label_type == "original" and (
                    candidate_best is None or metrics["macro_f1"] > candidate_best[0]
                ):
                    candidate_best = (metrics["macro_f1"], variant, train_version, ckpt)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_root / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_frames, ignore_index=True).to_csv(
        output_root / "predictions.csv", index=False, encoding="utf-8-sig",
    )
    print("\n原始验证标签下的成绩排名：")
    print(table[table["validation_labels"] == "original"]
          .sort_values("macro_f1", ascending=False)[
              ["model", "train_version", "accuracy", "macro_f1", "wrong"]
          ].to_string(index=False))
    print("\n已保存：", output_root / "comparison.csv", "和", output_root / "predictions.csv")

    if not args.no_demo and candidate_best is not None:
        _, winner, source, winner_folder = candidate_best
        winner_model, winner_tokenizer, winner_info = load_checkpoint(
            winner_folder, device, legacy_variant="A" if winner == "A" else None,
            legacy_max_length=args.legacy_max_length,
        )
        predictions, probs = predict_texts(
            winner_model, winner_tokenizer, demo_cases,
            int(winner_info["max_length"]), args.batch_size, device,
        )
        print(f"\n新评论预测示例（原始验证集最优模型：{winner}/{source}）：")
        for sample, pred, proba in zip(demo_cases, predictions, probs):
            print(f"{int(pred)} ({'正面' if pred == 1 else '负面'}, "
                  f"正面概率={proba:.4f})  {sample}")


if __name__ == "__main__":
    main()
