#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比传统模型与预训练模型的分歧、错误及否定/转折案例。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import compute_metrics, dump_json, print_metrics, read_csv

NEGATIONS = ("不", "没", "无", "未", "别", "并非", "从不", "不是", "没有", "不太", "一点也不")
CONTRASTS = ("但是", "不过", "然而", "可是", "虽然", "却", "但", "尽管", "只是", "反而")


def main() -> None:
    p = argparse.ArgumentParser(description="传统 ML 与 RoBERTa 预测结果的误分类分析")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--classical", required=True, help="传统模型预测 CSV（通过 predict_sentiment.py 生成）")
    p.add_argument("--transformer", required=True, help="RoBERTa validation_predictions.csv")
    p.add_argument("--output-dir", default="outputs/analysis")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = read_csv(args.validation)
    pred_cl = pd.read_csv(args.classical, dtype={"id": str}, keep_default_na=False)
    pred_tr = pd.read_csv(args.transformer, dtype={"id": str}, keep_default_na=False)
    for title, frame in [("classical", pred_cl), ("transformer", pred_tr)]:
        if not {"id", "prediction"}.issubset(frame.columns) or frame["id"].duplicated().any():
            raise ValueError(f"{title} 预测文件必须含唯一 id 和 prediction 列")
        if set(frame["id"]) != set(truth["id"]):
            raise ValueError(f"{title} 的 ID 集合和 validation 不完全一致，禁止静默对齐")

    df = truth[["id", "text", "label"]].merge(
        pred_cl[["id", "prediction"]].rename(columns={"prediction": "classical_pred"}), on="id", validate="1:1"
    ).merge(
        pred_tr[["id", "prediction"]].rename(columns={"prediction": "transformer_pred"}), on="id", validate="1:1"
    )
    df["classical_pred"] = df["classical_pred"].astype(int)
    df["transformer_pred"] = df["transformer_pred"].astype(int)
    df["classical_correct"] = df["classical_pred"] == df["label"]
    df["transformer_correct"] = df["transformer_pred"] == df["label"]
    df["disagree"] = df["classical_pred"] != df["transformer_pred"]
    df["negation"] = df["text"].apply(lambda t: any(token in t for token in NEGATIONS))
    df["contrast"] = df["text"].apply(lambda t: any(token in t for token in CONTRASTS))
    df["transformer_wins"] = ~df["classical_correct"] & df["transformer_correct"]
    df["classical_wins"] = df["classical_correct"] & ~df["transformer_correct"]
    df["both_wrong"] = ~df["classical_correct"] & ~df["transformer_correct"]
    df.to_csv(out / "full_comparison.csv", index=False, encoding="utf-8-sig")
    df.loc[df["disagree"]].to_csv(out / "disagreements.csv", index=False, encoding="utf-8-sig")
    df.loc[df["transformer_wins"]].to_csv(out / "transformer_better_examples.csv", index=False, encoding="utf-8-sig")
    df.loc[df["classical_wins"]].to_csv(out / "classical_better_examples.csv", index=False, encoding="utf-8-sig")
    df.loc[df["both_wrong"]].to_csv(out / "both_wrong_examples.csv", index=False, encoding="utf-8-sig")
    overview = {
        "total": len(df), "disagreements": int(df.disagree.sum()),
        "transformer_wins": int(df.transformer_wins.sum()),
        "classical_wins": int(df.classical_wins.sum()),
        "both_wrong": int(df.both_wrong.sum()),
        "overall": {}, "subgroups": {},
    }
    for model in ["classical", "transformer"]:
        scores = compute_metrics(df["label"], df[f"{model}_pred"])
        overview["overall"][model] = scores
        print_metrics(f"{model} overall", scores)
    for group_name, mask in [
        ("contains_negation", df["negation"]),
        ("contains_contrast", df["contrast"]),
        ("contains_both", df["negation"] & df["contrast"]),
        ("contains_neither", ~df["negation"] & ~df["contrast"]),
    ]:
        subset = df.loc[mask]
        if subset.empty:
            continue
        overview["subgroups"][group_name] = {"n": len(subset)}
        print(f"\n群组 {group_name}: n={len(subset)}")
        for model in ["classical", "transformer"]:
            metrics = compute_metrics(subset["label"], subset[f"{model}_pred"])
            overview["subgroups"][group_name][model] = {
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]
            }
            print_metrics(model, metrics)
    dump_json(overview, out / "analysis_summary.json")
    print("\n输出目录:", out.resolve())
    print("注意：否定/转折分组由字符串规则粗略识别，需人工核查示例后撰写结论。")


if __name__ == "__main__":
    main()
