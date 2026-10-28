#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MARC 情感标签人工核查：候选筛选 -> 填写 decision -> 生成新 CSV。

本版改动：
  * 不需要 reason 列；只需在 decision 中填写 keep / flip / unsure。
  * 不使用 top-k 数量限制。只要模型预测与原标签不一致，且预测置信度
    >= --min-confidence（默认 0.90），就全部纳入人工核查清单。
  * 训练集采用 OOF 预测；验证集采用仅在训练集拟合的模型预测。
  * 验证集也只审核高置信度分歧样本，不再输出全部 4000 条。
  * 原始 train.csv、validation.csv 永不覆盖，且生成文件默认禁止覆盖。

示例（项目根目录）：
  python label_curation.py prepare --train data/train.csv --validation data/validation.csv
  # 在 outputs/label_audit/review_train.csv、review_validation.csv 的 decision 列填写：
  #   flip   = 人工核查后确认要翻转标签（0 -> 1，1 -> 0）
  #   keep   = 核查后原标签正确
  #   unsure = 暂时不能确定（不改）
  #   空白   = 未审核（不改）
  python label_curation.py apply --train data/train.csv --validation data/validation.csv

注意：被模型高置信度预测为另一类，不代表原标签一定错。
      原始标签表示评分极性；修订标签尝试表示评论文本的情感语义。
      对修订后的验证集得到的分数不能当作原始验证指标的提升。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # 兼容较早的 scikit-learn 版本
    StratifiedGroupKFold = None

REQUIRED_COLUMNS = ("id", "text", "label")
REVIEW_COLUMNS = ("id", "text", "original_label", "predicted_label", "confidence", "decision")


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件: {path}")
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        raise ValueError(f"{path}: 必须包含 id、text、label 三列")
    df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].astype(str)
    if df["id"].eq("").any() or df["id"].duplicated().any():
        raise ValueError(f"{path}: id 不得为空或重复")
    if df["text"].str.strip().eq("").any():
        raise ValueError(f"{path}: 发现空文本")
    if not df["label"].astype(str).isin(["0", "1"]).all():
        raise ValueError(f"{path}: label 只能取 0 或 1")
    df["label"] = df["label"].astype(int)
    if df["label"].nunique() != 2:
        raise ValueError(f"{path}: 必须同时具有 0、1 两类")
    return df


def build_model():
    # TF-IDF 词表在各折的训练部分内部拟合，避免使用待预测样本的统计信息。
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=120_000,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        LogisticRegression(C=3.0, solver="liblinear", max_iter=1200),
    )


def positive_probability(model, texts: list[str]) -> np.ndarray:
    # predict_proba 的列按 classes_ 排列，明确取标签 1 的概率。
    classes = model.named_steps["logisticregression"].classes_
    idx = int(np.flatnonzero(classes == 1)[0])
    return model.predict_proba(texts)[:, idx]


def train_oof_probability(train: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    labels = train["label"].to_numpy(dtype=int)
    smallest_class = int(train["label"].value_counts().min())
    if not 2 <= folds <= smallest_class:
        raise ValueError(f"--folds 必须在 2 到 {smallest_class} 之间")

    texts = train["text"].tolist()
    # 同一评论文本即使多次出现，也不能跨训练折与待预测折。
    # 有 StratifiedGroupKFold 时同时尽量保持两类样本比例。
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        splitter = GroupKFold(n_splits=folds)
    groups = train["text"].to_numpy()
    prob = np.full(len(train), np.nan, dtype=np.float64)

    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(texts, labels, groups), start=1):
        print(f"[prepare] 训练集 OOF {fold}/{folds}：拟合 {len(fit_idx)} 条，预测 {len(holdout_idx)} 条", flush=True)
        model = build_model()
        model.fit([texts[i] for i in fit_idx], labels[fit_idx])
        prob[holdout_idx] = positive_probability(model, [texts[i] for i in holdout_idx])

    if not np.isfinite(prob).all():
        raise RuntimeError("部分训练样本没有获得有效 OOF 预测")
    return prob


def select_candidates(df: pd.DataFrame, prob_positive: np.ndarray, min_confidence: float, name: str) -> pd.DataFrame:
    labels = df["label"].to_numpy(dtype=int)
    pred = (prob_positive >= 0.5).astype(int)
    confidence = np.where(pred == 1, prob_positive, 1.0 - prob_positive)
    disagreement = pred != labels
    selected = disagreement & (confidence >= min_confidence)

    candidates = df.loc[selected, ["id", "text", "label"]].copy()
    candidates.rename(columns={"label": "original_label"}, inplace=True)
    candidates["predicted_label"] = pred[selected]
    candidates["confidence"] = np.round(confidence[selected], 6)
    candidates["decision"] = ""
    # 只做排序，不设候选数量上限。
    candidates = candidates.sort_values("confidence", ascending=False, kind="stable")
    candidates = candidates.loc[:, list(REVIEW_COLUMNS)].reset_index(drop=True)
    print(
        f"[prepare] {name}：共 {len(df)} 条，预测与标签分歧 {int(disagreement.sum())} 条；"
        f"置信度 >= {min_confidence:.2f} 的待核查样本 {len(candidates)} 条（不设数量上限）"
    )
    return candidates


def create_csv(df: pd.DataFrame, path: Path) -> None:
    # x 模式：即使同名文件已存在也不会覆盖。
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        df.to_csv(f, index=False)


def prepare(train: pd.DataFrame, validation: pd.DataFrame, out: Path, folds: int, seed: int, min_confidence: float):
    if not 0.5 < min_confidence <= 1.0:
        raise ValueError("--min-confidence 必须大于 0.5 且不大于 1.0，例如 0.90")
    out.mkdir(parents=True, exist_ok=True)
    review_train = out / "review_train.csv"
    review_val = out / "review_validation.csv"
    for path in (review_train, review_val):
        if path.exists():
            raise FileExistsError(f"审核表已存在，拒绝覆盖：{path}。请换一个 --out-dir。")

    train_prob = train_oof_probability(train, folds=folds, seed=seed)
    train_review = select_candidates(train, train_prob, min_confidence, "训练集（OOF）")

    print("[prepare] 训练全部 train.csv 的标签预测器，用于预测未参与训练的 validation.csv", flush=True)
    model = build_model()
    model.fit(train["text"].tolist(), train["label"].to_numpy(dtype=int))
    val_prob = positive_probability(model, validation["text"].tolist())
    val_review = select_candidates(validation, val_prob, min_confidence, "验证集（Holdout）")

    create_csv(train_review, review_train)
    create_csv(val_review, review_val)
    print(f"[prepare] 已生成：{review_train}")
    print(f"[prepare] 已生成：{review_val}")
    print("[prepare] 只需填写 decision 列：flip=翻转，keep=保留，unsure/空白=不变；无需填写 reason。")


def apply_reviews(original: pd.DataFrame, review_file: Path, split: str):
    if not review_file.exists():
        raise FileNotFoundError(f"找不到审核表：{review_file}；请先运行 prepare")
    review = pd.read_csv(review_file, dtype={"id": str}, keep_default_na=False)
    required = {"id", "text", "original_label", "decision"}
    if not required.issubset(review.columns):
        raise ValueError(f"{review_file}: 缺少列：{sorted(required - set(review.columns))}")
    if review["id"].eq("").any() or review["id"].duplicated().any():
        raise ValueError(f"{review_file}: id 不能为空或重复")

    source = original.set_index("id", drop=False)
    changed = original.copy().set_index("id", drop=False)
    edits = []
    n_keep, n_flip, n_unsure, n_blank = 0, 0, 0, 0
    for row in review.itertuples(index=False):
        sid = str(row.id)
        if sid not in source.index:
            raise ValueError(f"{review_file}: 样本 id 不在原始 {split} 中：{sid}")
        old_label = int(source.loc[sid, "label"])
 

        decision = str(row.decision).strip().lower()
        if decision not in {"", "flip", "keep", "unsure"}:
            raise ValueError(f"{review_file}: {sid} 的 decision='{decision}' 无效")
        if decision == "flip":
            new_label = 1 - old_label
            changed.loc[sid, "label"] = new_label
            edits.append({"split": split, "id": sid, "old_label": old_label, "new_label": new_label})
            n_flip += 1
        elif decision == "keep":
            n_keep += 1
        elif decision == "unsure":
            n_unsure += 1
        else:
            n_blank += 1

    print(
        f"[apply] {split}：候选共 {len(review)} 条；"
        f"flip={n_flip}, keep={n_keep}, unsure={n_unsure}, 未审核={n_blank}"
    )
    return changed.reset_index(drop=True), edits


def apply(train: pd.DataFrame, validation: pd.DataFrame, out: Path) -> None:
    train_clean, train_edits = apply_reviews(train, out / "review_train.csv", "train")
    val_clean, val_edits = apply_reviews(validation, out / "review_validation.csv", "validation")

    # 验证集中只有高置信度分歧样本经过候选筛选，未审核所有评论，始终标记为 partial。
    paths = (
        out / "train_semantic.csv",
        out / "validation_semantic_partial.csv",
        out / "changes.csv",
    )
    for path in paths:
        if path.exists():
            raise FileExistsError(f"输出文件已存在，拒绝覆盖：{path}。请先备份旧输出或更换目录。")

    changes = pd.DataFrame(train_edits + val_edits, columns=["split", "id", "old_label", "new_label"])
    create_csv(train_clean, paths[0])
    create_csv(val_clean, paths[1])
    create_csv(changes, paths[2])
    for path in paths:
        print(f"[apply] 已生成：{path}")
    print("[注意] validation_semantic_partial.csv 仅是部分候选人工核查版，并非完整语义金标。")
    print("[注意] 原始 validation.csv 仍应保留用于与原始实验公平比较。")


def main():
    parser = argparse.ArgumentParser(description="标签候选核查与人工翻转：不需要 reason，不覆盖原数据")
    parser.add_argument("action", choices=["prepare", "apply"])
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--validation", default="data/validation.csv")
    parser.add_argument("--out-dir", default="outputs/label_audit")
    parser.add_argument("--folds", type=int, default=5, help="训练集 OOF 折数（默认 5）")
    parser.add_argument("--min-confidence", type=float, default=0.90, help="入选候选的最低预测置信度（默认 0.90），不限制数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = read_dataset(Path(args.train))
    validation = read_dataset(Path(args.validation))
    out = Path(args.out_dir)
    if args.action == "prepare":
        prepare(train, validation, out, args.folds, args.seed, args.min_confidence)
    else:
        apply(train, validation, out)


if __name__ == "__main__":
    main()
