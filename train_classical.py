#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""经典 ML：完成课程 Notebook 指定的全部 15 种分类器，并额外优化 LinearSVC/NB-SVM。

默认完整运行（包含部分较慢的核方法/集成方法）；可用 --fast 调试。
仅在 train 上 fit 各种文本特征，validation 只用于评估和模型选择。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from common import (
    check_split, compute_metrics, dump_json, print_metrics,
    read_csv, save_errors, save_predictions, set_seed,
)

DENSE_METHODS = {
    "NearestCentroid", "KNeighborsClassifier", "SVC_RBF", "DecisionTreeClassifier",
    "RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier", "MLPClassifier",
}
METHOD_NAMES = [
    "MultinomialNB", "ComplementNB", "BernoulliNB", "LogisticRegression",
    "LinearSVC", "RidgeClassifier", "Perceptron", "NearestCentroid",
    "KNeighborsClassifier", "SVC_RBF", "DecisionTreeClassifier", "RandomForestClassifier",
    "AdaBoostClassifier", "GradientBoostingClassifier", "MLPClassifier",
]


def classifiers(seed: int, fast: bool) -> dict:
    """所有模型都有实际配置，不为满足方法数量而重复套用同一模型。"""
    return {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "ComplementNB": ComplementNB(alpha=0.5),
        "BernoulliNB": BernoulliNB(alpha=0.8, binarize=0.0),
        "LogisticRegression": LogisticRegression(C=3.0, solver="liblinear", max_iter=1500, random_state=seed),
        "LinearSVC": LinearSVC(C=2.0, max_iter=5000, random_state=seed),
        "RidgeClassifier": RidgeClassifier(alpha=1.0),
        "Perceptron": Perceptron(alpha=0.0001, max_iter=1500, random_state=seed),
        "NearestCentroid": NearestCentroid(),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=11, weights="distance", algorithm="brute", n_jobs=-1),
        "SVC_RBF": SVC(C=2.0, gamma="scale", cache_size=512, kernel="rbf"),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=24 if not fast else 12, min_samples_leaf=7, random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=150 if not fast else 30, max_depth=24, min_samples_leaf=3,
            n_jobs=-1, random_state=seed,
        ),
        "AdaBoostClassifier": AdaBoostClassifier(n_estimators=110 if not fast else 20, learning_rate=0.5, random_state=seed),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=85 if not fast else 20, learning_rate=0.08, max_depth=2,
            max_features="sqrt", random_state=seed,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(160, 64), activation="relu", alpha=0.001,
            learning_rate_init=0.001, batch_size=256, max_iter=70 if not fast else 14,
            early_stopping=True, n_iter_no_change=7, random_state=seed,
        ),
    }


def char_vectorizer(fast: bool) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), min_df=2,
        max_features=180000 if not fast else 40000,
        sublinear_tf=True, lowercase=True, norm="l2", dtype=np.float32,
    )


def tokenize_jieba(texts: list[str]) -> list[str]:
    """jieba 仅在训练语料上学习词表；预处理为分词后空格连接的字符串。"""
    import jieba
    import re

    token_re = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
    jieba.setLogLevel(30)
    return [" ".join(t.lower() for t in jieba.lcut(text) if token_re.fullmatch(t)) for text in texts]


def get_features(
    kind: str,
    texts: list[str],
    char_vec: TfidfVectorizer,
    word_vec: TfidfVectorizer | None = None,
    svd: TruncatedSVD | None = None,
    feature_weights: np.ndarray | None = None,
):
    x_char = char_vec.transform(texts)
    if kind == "char":
        x = x_char
    elif kind == "lsa":
        if svd is None:
            raise ValueError("LSA 模型缺少 TruncatedSVD")
        x = normalize(svd.transform(x_char).astype(np.float32), norm="l2")
    elif kind in {"char_word", "nb_svm"}:
        if word_vec is None:
            raise ValueError("char_word / nb_svm 缺少词向量转换器")
        x_word = word_vec.transform(tokenize_jieba(texts))
        x = normalize(sparse.hstack(
            [x_char * np.float32(0.78), x_word * np.float32(0.62)], format="csr"
        ), norm="l2")
        if kind == "nb_svm":
            if feature_weights is None:
                raise ValueError("NB-SVM 模型缺少 feature_weights")
            x = x.multiply(feature_weights).tocsr()
    else:
        raise ValueError(f"未知特征类型: {kind}")
    return x


def nbswm_log_ratio(x: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    x_binary = (x > 0).astype(np.float32)
    pos = np.asarray(x_binary[y == 1].sum(axis=0)).ravel() + 1.0
    neg = np.asarray(x_binary[y == 0].sum(axis=0)).ravel() + 1.0
    log_ratio = np.log(pos / pos.sum()) - np.log(neg / neg.sum())
    return log_ratio.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="MARC 情感分类：15 种机器学习方法和优化对照")
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--output-dir", default="outputs/classical")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--svd-components", type=int, default=96)
    p.add_argument("--methods", default="all", help="逗号分隔的方法名称或 all")
    p.add_argument("--no-tuning", action="store_true", help="不额外调参；仍运行选定的经典分类器")
    p.add_argument("--no-word-features", action="store_true", help="优化阶段不使用 jieba 词特征")
    p.add_argument("--fast", action="store_true", help="只抽取部分数据调试；结果不能用于报告")
    args = p.parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_train = read_csv(args.train)
    df_dev = read_csv(args.validation)
    check_split(df_train, df_dev)
    if args.fast:
        df_train = df_train.groupby("label", group_keys=False).sample(
            n=min(1200, df_train.label.value_counts().min()), random_state=args.seed
        ).reset_index(drop=True)
        df_dev = df_dev.groupby("label", group_keys=False).sample(
            n=min(240, df_dev.label.value_counts().min()), random_state=args.seed
        ).reset_index(drop=True)
        print("[FAST 调试模式] 子集上的分数不得写入正式实验报告")
    methods = METHOD_NAMES if args.methods == "all" else [s.strip() for s in args.methods.split(",")]
    if len(set(methods)) != len(methods) or any(name not in METHOD_NAMES for name in methods):
        raise ValueError(f"--methods 无效。合法方法: {', '.join(METHOD_NAMES)}")

    y_train = df_train["label"].to_numpy()
    y_dev = df_dev["label"].to_numpy()
    texts_train = df_train["text"].tolist()
    texts_dev = df_dev["text"].tolist()
    start = time.perf_counter()

    print("训练字符级 TF-IDF（词表仅从 train 学习）...", flush=True)
    char_vec = char_vectorizer(args.fast)
    x_train = char_vec.fit_transform(texts_train)
    x_dev = char_vec.transform(texts_dev)
    print("char TF-IDF 特征:", x_train.shape, "非零项:", x_train.nnz)

    svd = None
    x_train_lsa = x_dev_lsa = None
    if any(m in DENSE_METHODS for m in methods):
        n_components = min(args.svd_components, x_train.shape[0] - 2, x_train.shape[1] - 1)
        if n_components < 2:
            raise ValueError("样本太少，无法使用 SVD")
        print(f"运行 TruncatedSVD（{n_components} 维），供距离/树/核/MLP 使用...", flush=True)
        svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=args.seed)
        x_train_lsa = normalize(svd.fit_transform(x_train).astype(np.float32), norm="l2")
        x_dev_lsa = normalize(svd.transform(x_dev).astype(np.float32), norm="l2")

    rows: list[dict] = []
    best_score = -1.0
    best_data: dict | None = None
    extra_word_vec = None

    def evaluate_one(name: str, clf, kind: str, x_tr, x_va, words=None, ratios=None) -> None:
        nonlocal best_score, best_data
        start_model = time.perf_counter()
        print(f"\n正在训练: {name} ({kind})", flush=True)
        clf.fit(x_tr, y_train)
        pred = clf.predict(x_va).astype(int)
        metrics = compute_metrics(y_dev, pred)
        seconds = round(time.perf_counter() - start_model, 2)
        print_metrics(name, metrics)
        save_predictions(df_dev, pred, out / f"pred_{name}.csv")
        rows.append({
            "method": name, "feature": kind, "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"], "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"], "seconds": seconds,
            "parameters": str(clf.get_params()),
        })
        dump_json(metrics, out / f"metrics_{name}.json")
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_data = {
                "method": name, "kind": kind, "classifier": clf,
                "char_vectorizer": char_vec, "word_vectorizer": words,
                "svd": svd if kind == "lsa" else None,
                "feature_weights": ratios,
                "max_length": None, "label_names": {0: "负面", 1: "正面"},
            }
            print(f"新最优模型：{name} Macro-F1={best_score:.4f}", flush=True)

    models = classifiers(args.seed, args.fast)
    for method in methods:
        kind = "lsa" if method in DENSE_METHODS else "char"
        x_tr = x_train_lsa if kind == "lsa" else x_train
        x_va = x_dev_lsa if kind == "lsa" else x_dev
        evaluate_one(method, models[method], kind, x_tr, x_va)

    # 超出课程 15 种对照方法之外的改进：线性分类器超参、分词 n-gram、NB-SVM。
    # 为避免 15 个基础分类器被过多额外训练掩盖，可使用 --no-tuning 关闭。
    if not args.no_tuning:
        for c in (0.5, 1.0, 4.0, 8.0):
            name = f"Tuned_LinearSVC_C{c:g}"
            evaluate_one(name, LinearSVC(C=c, max_iter=6000, random_state=args.seed), "char", x_train, x_dev)
        for c in (1.0, 4.0):
            name = f"Tuned_LogisticRegression_C{c:g}"
            evaluate_one(name, LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=args.seed), "char", x_train, x_dev)

        if not args.no_word_features:
            try:
                import jieba  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("启用词级特征需要 pip install jieba，或加 --no-word-features") from exc
            print("\n增加训练集派生的 jieba 词级 (1,2)-gram 特征...", flush=True)
            extra_word_vec = TfidfVectorizer(
                analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2),
                min_df=2, max_features=85000 if not args.fast else 15000,
                sublinear_tf=True, lowercase=True, norm="l2", dtype=np.float32,
            )
            text_train_seg = tokenize_jieba(texts_train)
            text_dev_seg = tokenize_jieba(texts_dev)
            wx_train = extra_word_vec.fit_transform(text_train_seg)
            wx_dev = extra_word_vec.transform(text_dev_seg)
            x_mix_train = normalize(sparse.hstack([
                x_train * np.float32(0.78), wx_train * np.float32(0.62)
            ], format="csr"), norm="l2")
            x_mix_dev = normalize(sparse.hstack([
                x_dev * np.float32(0.78), wx_dev * np.float32(0.62)
            ], format="csr"), norm="l2")
            for c in (1.0, 4.0):
                evaluate_one(
                    f"CharWord_LinearSVC_C{c:g}",
                    LinearSVC(C=c, max_iter=6000, random_state=args.seed),
                    "char_word", x_mix_train, x_mix_dev, words=extra_word_vec,
                )
            ratios = nbswm_log_ratio(x_mix_train, y_train)
            x_nb_train = x_mix_train.multiply(ratios).tocsr()
            x_nb_dev = x_mix_dev.multiply(ratios).tocsr()
            for c in (1.0, 4.0):
                evaluate_one(
                    f"NB_SVM_C{c:g}",
                    LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=args.seed),
                    "nb_svm", x_nb_train, x_nb_dev, words=extra_word_vec, ratios=ratios,
                )

    assert best_data is not None
    all_rows = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    all_rows.to_csv(out / "classical_results.csv", index=False, encoding="utf-8-sig")
    joblib.dump(best_data, out / "best_classical.joblib", compress=3)
    best_name = best_data["method"]
    best_predictions = pd.read_csv(out / f"pred_{best_name}.csv", keep_default_na=False)
    save_errors(df_dev, best_predictions["prediction"].to_numpy(), out / "best_errors.csv")
    summary = {
        "best_method": best_name, "best_macro_f1": best_score,
        "best_model_file": str((out / "best_classical.joblib").resolve()),
        "methods_required": len(METHOD_NAMES),
        "methods_completed": len([r for r in rows if r["method"] in METHOD_NAMES]),
        "extra_optimized_methods": len(rows) - len([r for r in rows if r["method"] in METHOD_NAMES]),
        "mode": "fast_debug" if args.fast else "full_train",
        "total_seconds": round(time.perf_counter() - start, 2),
        "train_rows": len(df_train), "validation_rows": len(df_dev),
        "train_path": str(Path(args.train).resolve()), "validation_path": str(Path(args.validation).resolve()),
    }
    dump_json(summary, out / "summary.json")
    print("\n分类器对比（按验证集 Macro-F1 排序）：")
    print(all_rows[["method", "feature", "accuracy", "macro_f1", "seconds"]].to_string(index=False))
    print("\n最优传统模型：", best_name, "分数", round(best_score, 4))
    print("输出位置:", out.resolve())


if __name__ == "__main__":
    main()
