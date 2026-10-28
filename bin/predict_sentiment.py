#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对任意中文评论或 CSV 推理；可载入传统最佳模型或微调最佳模型。"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from common import compute_metrics, dump_json, print_metrics, read_csv, save_predictions


def predict_classical(model_file: Path, texts: list[str]):
    from train_classical import get_features

    bundle = joblib.load(model_file)
    features = get_features(
        bundle["kind"], texts, bundle["char_vectorizer"],
        word_vec=bundle.get("word_vectorizer"),
        svd=bundle.get("svd"),
        feature_weights=bundle.get("feature_weights"),
    )
    classifier = bundle["classifier"]
    predictions = classifier.predict(features).astype(int)
    prob = classifier.predict_proba(features)[:, 1] if hasattr(classifier, "predict_proba") else None
    return predictions, prob


@torch.inference_mode()
def predict_transformer(model_dir: Path, texts: list[str], batch_size: int,
                        max_length: int, device: torch.device):
    from transformers import AutoModelForSequenceClassification, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True,
    ).to(device).eval()
    if model.config.num_labels != 2:
        raise ValueError("此实验只支持二分类")
    predictions = []
    positive_probs = []
    for start in range(0, len(texts), batch_size):
        samples = texts[start:start + batch_size]
        encoded = tokenizer(
            samples, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        predictions.append(probabilities.argmax(axis=-1))
        positive_probs.append(probabilities[:, 1])
    return np.concatenate(predictions), np.concatenate(positive_probs)


def parse_args():
    parser = argparse.ArgumentParser(description="从已保存模型预测评论文本或 CSV")
    parser.add_argument("--model", required=True, help="best_model/ 目录，或 best_classical.joblib")
    parser.add_argument("--model-type", choices=["auto", "transformer", "classical"], default="auto")
    parser.add_argument("--input-csv", help="提供 csv 时批量处理：id, text，可带 label")
    parser.add_argument("--text", action="append", default=[], help="可多次传入，以预测多句话")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=None,
                        help="默认尝试从训练配置读取，无法读取则默认 160")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于 0")
    model_file = Path(args.model)
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    kind = ("classical" if model_file.is_file() else "transformer") if args.model_type == "auto" else args.model_type
    if kind == "classical" and not model_file.is_file():
        raise ValueError("传统模型应当指定 best_classical.joblib 文件")
    if kind == "transformer" and not model_file.is_dir():
        raise ValueError("预训练模型应当指定 best_model 文件夹")

    if args.input_csv:
        frame = read_csv(args.input_csv, require_labels=False)
    else:
        texts = args.text or [
            "质量很好，下次还会购买！",
            "包装不错，但是产品质量非常差。",
            "物流虽然慢，但东西很好用。",
            "一点也不好用，不推荐购买。",
            "没有想象中那么差，不过价格确实有点贵。",
        ]
        if any(not text.strip() for text in texts):
            raise ValueError("评论不可为空")
        frame = pd.DataFrame({"id": [f"new_{i:04d}" for i in range(len(texts))], "text": texts})

    if kind == "classical":
        preds, probabilities = predict_classical(model_file, frame["text"].tolist())
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("没有检测到 CUDA")
        device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
        max_length = args.max_length
        if max_length is None:
            # best_model 位于实验输出目录下：先读取目录旁保存的微调配置。
            config_file = model_file.parent / "training_config.json"
            if config_file.is_file():
                import json
                with config_file.open(encoding="utf-8") as f:
                    max_length = int(json.load(f)["max_length"])
            else:
                max_length = 160
                print("未发现训练配置，使用 max_length=160；建议显式传 --max-length")
        preds, probabilities = predict_transformer(
            model_file, frame["text"].tolist(), args.batch_size, max_length, device
        )

    output = save_predictions(frame, preds, args.output, probabilities)
    display_cols = ["text", "predicted_label"] + (["positive_probability"] if probabilities is not None else [])
    print(output[display_cols].head(20).to_string(index=False))
    if "label" in frame:
        metrics = compute_metrics(frame["label"].to_numpy(), preds)
        print_metrics("Evaluation", metrics)
        dump_json(metrics, Path(args.output).with_suffix(".metrics.json"))
    print(f"\n预测结果已保存：{Path(args.output).resolve()}")
    print("注：输出的概率分数（如有）未做专门校准，不应直接当作真实正确概率。")


if __name__ == "__main__":
    main()
