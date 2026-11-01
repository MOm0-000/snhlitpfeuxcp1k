#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只训练情感分类全连接层 Linear(512, 2)。

该脚本完全不加载 Transformer；它只读取 extract_features.py 导出的固定特征。
因此训练阶段不可能更新预训练编码器。

输出：
  outputs/fc_only/classifier_best.pt
  outputs/fc_only/classifier_last.pt
  outputs/fc_only/history.csv
  outputs/fc_only/best_metrics.json
  outputs/fc_only/training_config.json
  outputs/fc_only/validation_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import random
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
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    p = argparse.ArgumentParser(description="只训练全连接分类层")
    p.add_argument("--train-features", default="outputs/fc_only/train_cls_features.npz")
    p.add_argument("--validation-features", default="outputs/fc_only/validation_cls_features.npz")
    p.add_argument("--output-dir", default="outputs/fc_only")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(mode: str) -> torch.device:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但没有检测到 CUDA")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_feature_file(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到特征文件：{path}")
    data = np.load(path, allow_pickle=False)
    required = {"features", "labels", "ids"}
    if not required.issubset(data.files):
        raise ValueError(f"{path} 必须包含 features、labels、ids")
    x = np.asarray(data["features"], dtype=np.float32)
    y = np.asarray(data["labels"], dtype=np.int64)
    ids = np.asarray(data["ids"]).astype(str)
    if x.ndim != 2:
        raise ValueError(f"{path} 的 features 必须是二维矩阵")
    if len(x) != len(y) or len(x) != len(ids):
        raise ValueError(f"{path} 中 features/labels/ids 样本数不一致")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path} 的 label 必须为 0/1")
    return x, y, ids


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
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
def predict(model, x: torch.Tensor, device: torch.device, batch_size: int):
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size].to(device))
        out.append(logits.argmax(dim=-1).cpu())
    return torch.cat(out).numpy()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs、batch-size 必须 >=1，lr 必须 >0")

    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train_np, y_train_np, train_ids = load_feature_file(args.train_features)
    x_val_np, y_val_np, val_ids = load_feature_file(args.validation_features)

    if x_train_np.shape[1] != x_val_np.shape[1]:
        raise ValueError("训练/验证特征维度不一致")

    feature_dim = int(x_train_np.shape[1])
    if feature_dim != 512:
        raise ValueError(f"特征维度为 {feature_dim}，预期为 512")

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_val = torch.from_numpy(x_val_np)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # 唯一可训练模块：一个全连接层。
    classifier = nn.Linear(feature_dim, 2).to(device)

    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total = sum(p.numel() for p in classifier.parameters())
    expected = feature_dim * 2 + 2
    if trainable != expected or total != expected:
        raise RuntimeError("分类头参数数目异常")

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    training_config = {
        "method": "Linear probing / frozen encoder + trainable FC head",
        "classifier": f"Linear({feature_dim}, 2)",
        "trainable_parameters": int(trainable),
        "encoder_trainable_parameters_during_training": 0,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "train_features": str(args.train_features),
        "validation_features": str(args.validation_features),
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)

    print(f"设备: {device}")
    print(f"分类头: Linear({feature_dim}, 2)")
    print(f"可训练参数: {trainable}（应为 512*2+2 = 1026）")
    print("Transformer 在本脚本中没有被加载，因此不可能被更新。")

    best_f1 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)
            seen += len(xb)

        val_pred = predict(classifier, x_val, device, args.batch_size)
        score = metrics(y_val_np, val_pred)
        train_loss = total_loss / seen

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "accuracy": score["accuracy"],
            "macro_f1": score["macro_f1"],
            "macro_precision": score["macro_precision"],
            "macro_recall": score["macro_recall"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv", index=False, encoding="utf-8-sig"
        )

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.6f} | "
            f"Acc={score['accuracy']:.4f} | "
            f"Macro-F1={score['macro_f1']:.4f} | "
            f"Macro-P={score['macro_precision']:.4f} | "
            f"Macro-R={score['macro_recall']:.4f}"
        )

        if score["macro_f1"] > best_f1 + 1e-12:
            best_f1 = score["macro_f1"]
            best_epoch = epoch
            torch.save(classifier.state_dict(), output_dir / "classifier_best.pt")

    torch.save(classifier.state_dict(), output_dir / "classifier_last.pt")

    # 恢复最佳模型并重新评价，满足“模型恢复与核对验证成绩”。
    restored = nn.Linear(feature_dim, 2).to(device)
    state = torch.load(output_dir / "classifier_best.pt", map_location=device, weights_only=True)
    restored.load_state_dict(state, strict=True)
    restored_pred = predict(restored, x_val, device, args.batch_size)
    restored_score = metrics(y_val_np, restored_pred)

    best_metrics = {
        "best_epoch": int(best_epoch),
        "verified_reload": abs(restored_score["macro_f1"] - best_f1) < 1e-12,
        **restored_score,
    }
    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({
        "id": val_ids,
        "label": y_val_np,
        "prediction": restored_pred,
        "correct": y_val_np == restored_pred,
    })
    pred_df.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n最佳模型恢复验证：")
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print(f"最佳权重：{output_dir / 'classifier_best.pt'}")


if __name__ == "__main__":
    main()
