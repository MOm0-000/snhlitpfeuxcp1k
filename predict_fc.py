#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恢复冻结 RoBERTa + 已训练全连接分类头，并完成：
1) validation.csv 重新评价；
2) 正面、负面、否定、转折四类新评论预测。

输出：
  outputs/fc_only/reloaded_validation_metrics.json
  outputs/fc_only/reloaded_validation_predictions.csv
  outputs/fc_only/new_comment_demo.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, BertTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="恢复冻结编码器 + FC 权重并评价/预测")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--model-dir", default="pretrained")
    p.add_argument("--classifier", default="outputs/fc_only/classifier_best.pt")
    p.add_argument("--output-dir", default="outputs/fc_only")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def choose_device(mode: str) -> torch.device:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但没有检测到 CUDA")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_reviews(path: str | Path):
    df = pd.read_csv(path, keep_default_na=False, dtype={"id": str, "text": str})
    if not {"id", "text", "label"}.issubset(df.columns):
        raise ValueError("validation.csv 必须包含 id、text、label")
    labels = pd.to_numeric(df["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("label 必须为 0/1")
    df["label"] = labels.astype(np.int64)
    return df[["id", "text", "label"]].copy()


def metrics(y_true, y_pred):
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
    }


@torch.inference_mode()
def encode_texts(texts, encoder, tokenizer, device, max_length, batch_size):
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="编码"):
        part = texts[start:start + batch_size]
        batch = tokenizer(
            part,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        cls = encoder(**batch).last_hidden_state[:, 0, :]
        features.append(cls.float().cpu())
    return torch.cat(features, dim=0)


@torch.inference_mode()
def classify(features, classifier, device, batch_size):
    preds, probs = [], []
    for start in range(0, len(features), batch_size):
        logits = classifier(features[start:start + batch_size].to(device)).float()
        prob = torch.softmax(logits, dim=-1)
        preds.append(prob.argmax(dim=-1).cpu())
        probs.append(prob[:, 1].cpu())
    return torch.cat(preds).numpy(), torch.cat(probs).numpy()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    classifier_path = Path(args.classifier)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not classifier_path.is_file():
        raise FileNotFoundError(f"找不到分类层权重：{classifier_path}")

    device = choose_device(args.device)
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)

    encoder = AutoModel.from_pretrained(
        model_dir,
        add_pooling_layer=False,
        local_files_only=True,
    )
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device).eval()

    classifier = nn.Linear(int(config.hidden_size), 2).to(device)
    state = torch.load(classifier_path, map_location=device, weights_only=True)
    classifier.load_state_dict(state, strict=True)
    classifier.eval()

    if sum(p.numel() for p in classifier.parameters()) != 1026:
        raise RuntimeError("分类头参数数量不是 1026")

    # 1) 恢复后重新评价 validation
    val = read_reviews(args.validation)
    val_features = encode_texts(
        val["text"].tolist(),
        encoder,
        tokenizer,
        device,
        args.max_length,
        args.batch_size,
    )
    val_pred, val_prob = classify(
        val_features, classifier, device, args.batch_size
    )
    score = metrics(val["label"].to_numpy(), val_pred)

    with (output_dir / "reloaded_validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(score, f, ensure_ascii=False, indent=2)

    result = val[["id", "text", "label"]].copy()
    result["prediction"] = val_pred
    result["positive_probability"] = val_prob
    result["correct"] = result["label"].to_numpy() == val_pred
    result.to_csv(
        output_dir / "reloaded_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("恢复模型后的验证指标：")
    print(json.dumps(score, ensure_ascii=False, indent=2))

    # 2) 指导书要求的四类新评论
    demo = [
        ("正面", "质量很好，物流很快，下次还会购买。"),
        ("负面", "质量非常差，客服态度也不好。"),
        ("否定", "这款商品不是不好，只是价格有点高。"),
        ("转折", "虽然包装很好，但是商品有明显缺陷。"),
    ]
    demo_features = encode_texts(
        [x[1] for x in demo],
        encoder,
        tokenizer,
        device,
        args.max_length,
        args.batch_size,
    )
    demo_pred, demo_prob = classify(
        demo_features, classifier, device, args.batch_size
    )

    lines = []
    for (kind, text), pred, prob in zip(demo, demo_pred, demo_prob):
        label = "正面" if int(pred) == 1 else "负面"
        line = f"{kind}\t{label}\t正面概率={float(prob):.6f}\t{text}"
        lines.append(line)
        print(line)

    (output_dir / "new_comment_demo.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
