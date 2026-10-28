#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验公共逻辑：严格的数据检查、统一的指标和预测结果输出。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABEL_NAMES = {0: "负面", 1: "正面"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: str | Path, require_labels: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据文件：{path.resolve()}")
    frame = pd.read_csv(path, keep_default_na=False, dtype={"id": str, "text": str})
    if "text" not in frame:
        raise ValueError(f"{path} 缺少 text 列")
    if "id" not in frame:
        frame.insert(0, "id", [str(i) for i in range(len(frame))])
    if frame.empty or frame["text"].str.strip().eq("").any():
        raise ValueError(f"{path} 中存在空数据或空评论")
    if frame["id"].str.strip().eq("").any() or frame["id"].duplicated().any():
        raise ValueError(f"{path} 中存在空 ID 或重复 ID")
    if "label" not in frame:
        if require_labels:
            raise ValueError(f"{path} 缺少 label 列")
    else:
        if not require_labels and frame["label"].astype(str).str.strip().eq("").all():
            frame = frame.drop(columns="label")
        else:
            try:
                labels = pd.to_numeric(frame["label"], errors="raise")
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path} 的 label 必须是整数 0 或 1") from e
            if not labels.isin([0, 1]).all():
                raise ValueError(f"{path} 的 label 只能为 0（负面）或 1（正面）")
            frame["label"] = labels.astype(int)
    return frame


def check_split(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    overlap = set(train["id"]) & set(validation["id"])
    if overlap:
        raise ValueError(f"训练集和验证集存在 {len(overlap)} 个相同 id，拒绝训练")
    same_texts = set(train["text"]) & set(validation["text"])
    if same_texts:
        affected = int(validation["text"].isin(same_texts).sum())
        print(f"注意：两集有 {len(same_texts)} 种完全相同的评论文本，验证集涉及 {affected} 条；保留课程原始划分。")
    for name, frame in [("train", train), ("validation", validation)]:
        counts = frame["label"].value_counts().sort_index().to_dict()
        print(f"{name}: {len(frame)} 条；标签分布 {counts}")


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    p_macro, r_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    p_pos, r_pos, f_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "positive_precision": float(p_pos),
        "positive_recall": float(r_pos),
        "positive_f1": float(f_pos),
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["negative_0", "positive_1"],
            zero_division=0, output_dict=True,
        ),
    }


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_predictions(
    frame: pd.DataFrame,
    predictions: Any,
    path: str | Path,
    positive_probs: Any | None = None,
) -> pd.DataFrame:
    preds = np.asarray(predictions, dtype=int)
    if len(preds) != len(frame):
        raise ValueError("预测结果数量与数据集数量不一致")
    cols = ["id", "text"] + (["label"] if "label" in frame else [])
    result = frame[cols].copy()
    result["prediction"] = preds
    result["predicted_label"] = [LABEL_NAMES[i] for i in preds]
    if positive_probs is not None:
        probs = np.asarray(positive_probs, dtype=float)
        if len(probs) != len(result):
            raise ValueError("预测概率数量错误")
        result["positive_probability"] = probs
    if "label" in result:
        result["correct"] = result["label"].to_numpy() == preds
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result


def save_errors(
    frame: pd.DataFrame,
    predictions: Any,
    path: str | Path,
    positive_probs: Any | None = None,
    top_n: int = 250,
) -> None:
    if "label" not in frame:
        return
    preds = np.asarray(predictions, dtype=int)
    mask = preds != frame["label"].to_numpy()
    cols = ["id", "text", "label"]
    wrong = frame.loc[mask, cols].copy()
    wrong["prediction"] = preds[mask]
    if positive_probs is not None:
        probs = np.asarray(positive_probs, dtype=float)[mask]
        wrong["positive_probability"] = probs
        # 越靠前表示模型越自信但预测错误
        confidence = np.maximum(probs, 1.0 - probs)
        wrong["error_confidence"] = confidence
        wrong = wrong.sort_values("error_confidence", ascending=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrong.head(top_n).to_csv(path, index=False, encoding="utf-8-sig")


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(
        f"{name:<32} "
        f"Accuracy={metrics['accuracy']:.4f} "
        f"Macro-F1={metrics['macro_f1']:.4f} "
        f"Macro-P={metrics['macro_precision']:.4f} "
        f"Macro-R={metrics['macro_recall']:.4f}",
        flush=True,
    )
