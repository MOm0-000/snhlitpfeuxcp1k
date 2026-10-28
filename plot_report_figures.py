#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文商品评论情感分析：只绘制 3 张适合课程报告的必要图。

用法（将本文件放在 sentiment_experiment 项目根目录）：
    python plot_report_figures.py

自定义目录：
    python plot_report_figures.py --outputs /path/to/outputs

只读取已完成的实验结果，不会重新训练模型：
    outputs/classical/classical_results.csv
    outputs/sweep/leaderboard.csv
    outputs/analysis/analysis_summary.json
    outputs/sweep/<最佳profile>/history.csv   # 可选；有则把训练曲线放入图2右侧

只生成 3 个 PNG，位置：outputs/report_figures/。
依赖：pandas、numpy、matplotlib（以及系统中文字体，建议 Noto Sans CJK SC）。
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Ubuntu 服务器无图形桌面也可运行
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd


# 与最初 train_classical.py 中的 METHOD_NAMES 严格一致。
# 绘图时仅选这 15 个基础模型；所有额外优化不混进“15 模型排名”。
ORIGINAL_15 = [
    "MultinomialNB", "ComplementNB", "BernoulliNB", "LogisticRegression",
    "LinearSVC", "RidgeClassifier", "Perceptron", "NearestCentroid",
    "KNeighborsClassifier", "SVC_RBF", "DecisionTreeClassifier",
    "RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier",
    "MLPClassifier",
]
PROFILE_NAMES = {
    "baseline": "RoBERTa · Baseline",
    "longer": "RoBERTa · Longer",
    "fgm": "RoBERTa · FGM",
    "rdrop": "RoBERTa · R-Drop",
}

# 专门采用低饱和度配色，适合嵌入论文和灰度打印。
C_GRAY = "#8A9AA7"
C_BLUE = "#2D698A"
C_TEAL = "#207A73"
C_ORANGE = "#B47B39"
C_PALE = "#E5EBEF"
C_DARK = "#253341"


def set_fonts() -> None:
    """优先选择可显示中文的已安装字体，不依赖额外的字体文件。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
        "WenQuanYi Micro Hei", "SimHei", "PingFang SC", "Arial Unicode MS",
        "Noto Sans CJK JP", "AR PL UMing CN", "AR PL SungtiL GB",
    ]
    chosen = next((family for family in candidates if family in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        print("[提醒] 未检测到常见中文字体；若中文显示为方框，请安装 fonts-noto-cjk。")
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelcolor": C_DARK,
        "xtick.color": C_DARK,
        "ytick.color": C_DARK,
        "text.color": C_DARK,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 120,
    })


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"未找到实验输出文件：{path}\n请确认 --outputs 指向运行后生成的 outputs 目录。")
    return path


def load_results(outputs: Path):
    classical = pd.read_csv(require_file(outputs / "classical" / "classical_results.csv"),
                            encoding="utf-8-sig")
    leaderboard = pd.read_csv(require_file(outputs / "sweep" / "leaderboard.csv"),
                              encoding="utf-8-sig")
    with require_file(outputs / "analysis" / "analysis_summary.json").open(encoding="utf-8") as f:
        analysis = json.load(f)

    for filename, df, fields in [
        ("classical_results.csv", classical, ("method", "macro_f1")),
        ("leaderboard.csv", leaderboard, ("profile", "macro_f1", "best_epoch")),
    ]:
        missing = set(fields) - set(df.columns)
        if missing:
            raise ValueError(f"{filename} 缺少列：{sorted(missing)}")
        df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="raise")
        if df.empty or not np.isfinite(df["macro_f1"]).all() or not df["macro_f1"].between(0, 1).all():
            raise ValueError(f"{filename} 中 macro_f1 非法或为空")
    if classical["method"].duplicated().any():
        raise ValueError("classical_results.csv 中 method 不唯一，无法判断模型排行")
    if leaderboard["profile"].duplicated().any():
        raise ValueError("leaderboard.csv 中 profile 不唯一")
    for kind in ("classical", "transformer"):
        try:
            cm = np.asarray(analysis["overall"][kind]["confusion_matrix"], dtype=int)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"analysis_summary.json 缺少 {kind} 混淆矩阵") from exc
        if cm.shape != (2, 2) or np.any(cm < 0):
            raise ValueError(f"analysis_summary.json 中 {kind} 混淆矩阵格式错误：{cm.shape}")

    return classical, leaderboard, analysis


def tidy_axes(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C_PALE)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.grid(axis=grid_axis, color=C_PALE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_classical(classical: pd.DataFrame, save_to: Path):
    """图 1：15 种基础传统方法排名 + 最佳额外优化方法参照线。"""
    base = classical[classical["method"].isin(ORIGINAL_15)].sort_values("macro_f1").copy()
    if base.empty:
        raise ValueError("classical_results.csv 没有找到 15 种基础分类器中的任何一种")
    missing = set(ORIGINAL_15) - set(base["method"])
    if missing:
        warnings.warn("未找到以下基础分类器：" + ", ".join(sorted(missing)), stacklevel=2)

    original_best = base.loc[base["macro_f1"].idxmax()]
    # 最优经典方法可能来自额外优化；不应把它假扮成基础 15 种中的一种。
    best = classical.loc[classical["macro_f1"].idxmax()]
    low = max(0.0, float(base["macro_f1"].min()) - 0.025)
    high = min(1.0, max(float(base["macro_f1"].max()), float(best["macro_f1"])) + 0.045)

    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    y = np.arange(len(base))
    ax.scatter(base["macro_f1"], y, s=66, zorder=3, color=C_GRAY)
    top_i = int(np.argmax(base["macro_f1"].to_numpy()))
    ax.scatter([base.iloc[top_i]["macro_f1"]], [y[top_i]], color=C_BLUE, s=100, zorder=4)
    for yi, score in zip(y, base["macro_f1"]):
        # 将数值写在点的左侧，避免与最优 NB-SVM 的竖向参考线相交。
        ax.text(float(score) - 0.003, yi, f"{score * 100:.2f}%",
                va="center", ha="right", fontsize=8.5)
    ax.set_yticks(y, base["method"].tolist(), fontsize=9.3)
    ax.set_ylim(-0.7, len(base) - 0.3)
    ax.set_xlim(low, high)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("验证集 Macro-F1")
    if str(best["method"]) not in set(base["method"]):
        ax.axvline(float(best["macro_f1"]), linestyle="--", linewidth=1.8,
                   color=C_TEAL, zorder=2)
    tidy_axes(ax)
    fig.suptitle("图1　15 种经典机器学习方法的 Macro-F1 对比", y=0.97)
    fig.text(0.27, 0.018,
             f"注：虚线为最优优化方法 {best['method']}（{best['macro_f1']*100:.2f}%）；基础方法使用 char TF-IDF / LSA。",
             fontsize=8.5, color="#52616B")
    fig.subplots_adjust(left=0.30, right=0.98, top=0.91, bottom=0.13)
    fig.savefig(save_to, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return original_best, best


def read_best_history(outputs: Path, best_profile: str):
    # 不使用 leaderboard.csv 中绝对 model_dir，因为可能来自另一台电脑。
    candidate = outputs / "sweep" / best_profile / "history.csv"
    if not candidate.is_file():
        print(f"[提示] 未找到 {candidate}：图2只画参数对照，不画训练曲线。")
        return None
    try:
        history = pd.read_csv(candidate, encoding="utf-8-sig")
        needed = {"epoch", "train_loss", "macro_f1"}
        if not needed.issubset(history.columns):
            raise ValueError("缺少列 " + ", ".join(sorted(needed - set(history.columns))))
        for col in needed:
            history[col] = pd.to_numeric(history[col], errors="raise")
        history = history.sort_values("epoch")
        if history.empty or not np.isfinite(history[list(needed)]).all().all():
            raise ValueError("存在非有限数字，或没有有效的 epoch 记录")
        return history
    except (OSError, ValueError) as exc:
        warnings.warn(f"训练曲线数据不可用：{exc}。仅绘制参数对照。", stacklevel=2)
        return None


def plot_finetuning(classical: pd.DataFrame, leaderboard: pd.DataFrame,
                    outputs: Path, save_to: Path):
    """图 2：微调配置效果；若有 history，将曲线并入同一张图右侧。"""
    best_classical = classical.loc[classical["macro_f1"].idxmax()]
    best_pre = leaderboard.loc[leaderboard["macro_f1"].idxmax()]
    best_profile = str(best_pre["profile"])
    history = read_best_history(outputs, best_profile)

    compare = [(f"最佳传统方法（{best_classical['method']}）", float(best_classical["macro_f1"]))]
    # 按预先设定的实验顺序展示，不把 profile 名称按成绩乱排。
    order = {"baseline": 0, "longer": 1, "fgm": 2, "rdrop": 3}
    sorted_rows = leaderboard.sort_values(
        "profile", key=lambda s: s.map(lambda x: order.get(str(x), 999))
    )
    for _, item in sorted_rows.iterrows():
        name = str(item["profile"])
        compare.append((PROFILE_NAMES.get(name, f"RoBERTa · {name}"), float(item["macro_f1"])))
    labels = [x[0] for x in compare]
    vals = np.array([x[1] for x in compare])

    if history is None:
        fig, ax = plt.subplots(figsize=(9.1, 3.6))
    else:
        fig, (ax, curve_ax) = plt.subplots(
            1, 2, figsize=(13.0, 4.0), gridspec_kw={"width_ratios": [1.2, 1.1]}
        )

    ys = np.arange(len(labels))
    colors = [C_GRAY] + [C_TEAL if str(row["profile"]) == best_profile else C_BLUE
                        for _, row in sorted_rows.iterrows()]
    ax.scatter(vals, ys, s=115, c=colors, zorder=4)
    for i, v in enumerate(vals):
        ax.text(v + 0.0014, ys[i], f"{v * 100:.2f}%", fontsize=9.5,
                va="center", ha="left", fontweight="semibold" if colors[i] == C_TEAL else "normal")
    ax.set_yticks(ys, labels, fontsize=9.3)
    low = max(0, min(vals) - 0.015)
    high = min(1, max(vals) + 0.015)
    ax.set_xlim(low, high)
    ax.set_ylim(-0.55, len(labels) - 0.45)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("验证集 Macro-F1")
    ax.set_title("微调方案对照")
    tidy_axes(ax)

    if history is not None:
        epoch = history["epoch"].to_numpy()
        f1 = history["macro_f1"].to_numpy() * 100
        loss = history["train_loss"].to_numpy()
        curve_ax.plot(epoch, f1, "o-", color=C_TEAL, lw=2, ms=6, label="验证 Macro-F1")
        curve_ax.set_ylabel("验证 Macro-F1（%）", color=C_TEAL)
        curve_ax.tick_params(axis="y", labelcolor=C_TEAL)
        curve_ax.set_xticks(epoch)
        curve_ax.set_xlabel("Epoch")
        curve_ax.set_title(f"最优方案 {best_profile}：训练过程")
        curve_ax.spines["top"].set_visible(False)
        curve_ax.grid(axis="x", color=C_PALE)
        loss_ax = curve_ax.twinx()
        loss_ax.plot(epoch, loss, "s--", color=C_ORANGE, lw=1.6, ms=5, label="训练 Loss")
        loss_ax.set_ylabel("训练 Loss", color=C_ORANGE)
        loss_ax.tick_params(axis="y", labelcolor=C_ORANGE)
        loss_ax.spines["top"].set_visible(False)
        try:
            best_epoch = int(best_pre["best_epoch"])
            if best_epoch in epoch:
                curve_ax.axvline(best_epoch, lw=1, linestyle=":", color=C_GRAY, zorder=0)
        except (ValueError, TypeError):
            pass
        handles1, labels1 = curve_ax.get_legend_handles_labels()
        handles2, labels2 = loss_ax.get_legend_handles_labels()
        curve_ax.legend(handles1 + handles2, labels1 + labels2,
                        loc="lower center", frameon=False, fontsize=8.5)

    fig.suptitle("图2　RoBERTa 微调配置与最佳传统方法的性能对比", y=0.975)
    if history is not None:
        fig.subplots_adjust(left=0.25, right=0.91, top=0.79, bottom=0.18, wspace=0.70)
    else:
        fig.subplots_adjust(left=0.32, right=0.96, top=0.79, bottom=0.20)
    fig.savefig(save_to, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return best_pre, history is not None


def plot_confusion(analysis: dict, classical: pd.DataFrame,
                   leaderboard: pd.DataFrame, save_to: Path):
    """图3：同一组 4,000 验证评论上的两模型混淆矩阵，并行对比。"""
    matrices = [
        np.asarray(analysis["overall"]["classical"]["confusion_matrix"], dtype=int),
        np.asarray(analysis["overall"]["transformer"]["confusion_matrix"], dtype=int),
    ]
    scores = [
        float(analysis["overall"]["classical"]["macro_f1"]),
        float(analysis["overall"]["transformer"]["macro_f1"]),
    ]
    best_cl = classical.loc[classical["macro_f1"].idxmax()]
    best_ro = leaderboard.loc[leaderboard["macro_f1"].idxmax()]
    if abs(scores[0] - float(best_cl["macro_f1"])) > 0.002 or abs(scores[1] - float(best_ro["macro_f1"])) > 0.002:
        warnings.warn("analysis_summary.json 与排行榜的最优 F1 不一致；混淆矩阵代表错误分析脚本所用模型，"
                      "未必是排行榜最佳模型。", stacklevel=2)
        names = ["传统模型（错误分析所用）", "RoBERTa（错误分析所用）"]
    else:
        names = ["最佳传统方法", "最佳 RoBERTa"]

    total = int(matrices[0].sum())
    if matrices[1].sum() != total:
        raise ValueError("两种模型混淆矩阵的总样本数不相等，不能直接比较")
    cmap = plt.get_cmap("Blues")
    vmax = max(int(m.max()) for m in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.15))
    for ax, matrix, name, score in zip(axes, matrices, names, scores):
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks([0, 1], ["负面 0", "正面 1"], fontsize=10)
        ax.set_yticks([0, 1], ["负面 0", "正面 1"], fontsize=10)
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")
        ax.set_title(f"{name}\nMacro-F1 = {score * 100:.2f}%", fontsize=11.5, pad=12)
        for r in range(2):
            row_n = int(matrix[r].sum())
            for c in range(2):
                count = int(matrix[r, c])
                percentage = 100 * count / row_n if row_n else 0.0
                ax.text(c, r, f"{count:,}\n({percentage:.1f}%)",
                        ha="center", va="center", fontsize=11.5,
                        color="white" if count > vmax * 0.57 else C_DARK,
                        fontweight="semibold" if r == c else "normal")
        ax.tick_params(axis="both", length=0)
        for side in ax.spines:
            ax.spines[side].set_visible(False)
        ax.set_aspect("equal")

    wrong_cl = int(matrices[0][0, 1] + matrices[0][1, 0])
    wrong_ro = int(matrices[1][0, 1] + matrices[1][1, 0])
    fig.suptitle("图3　最佳传统模型与 RoBERTa 的混淆矩阵对比", y=0.98)
    fig.text(0.5, 0.025, f"同一验证集（N = {total:,}）：误判数 {wrong_cl} → {wrong_ro}"
             f"，减少 {wrong_cl - wrong_ro} 条", ha="center", fontsize=10, color="#52616B")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.76, bottom=0.16, wspace=0.37)
    fig.savefig(save_to, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return wrong_cl, wrong_ro


def main():
    p = argparse.ArgumentParser(description="从已有实验 outputs 中只生成 3 张报告用图，不重新训练。")
    p.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parent / "outputs",
                   help="实验输出目录（默认：脚本所在目录/outputs）")
    p.add_argument("--figure-dir", type=Path, default=None,
                   help="图片保存目录（默认：<outputs>/report_figures）")
    args = p.parse_args()
    outputs = args.outputs.expanduser().resolve()
    figure_dir = (args.figure_dir.expanduser().resolve() if args.figure_dir
                  else outputs / "report_figures")
    classical, leaderboard, analysis = load_results(outputs)
    set_fonts()
    figure_dir.mkdir(parents=True, exist_ok=True)

    p1 = figure_dir / "fig1_classical_15.png"
    p2 = figure_dir / "fig2_roberta_comparison.png"
    p3 = figure_dir / "fig3_confusion_matrices.png"
    original_best, classical_best = plot_classical(classical, p1)
    roberta_best, has_history = plot_finetuning(classical, leaderboard, outputs, p2)
    wrong_cl, wrong_ro = plot_confusion(analysis, classical, leaderboard, p3)

    print("\n绘图完成：仅生成 3 张 PNG，不生成重复图。")
    for output in (p1, p2, p3):
        print("  -", output)
    print("\n用于报告的实际结果：")
    print(f"  基础15方法最佳：{original_best['method']}, Macro-F1={original_best['macro_f1']:.4%}")
    print(f"  优化传统方法最佳：{classical_best['method']}, Macro-F1={classical_best['macro_f1']:.4%}")
    print(f"  最佳RoBERTa方案：{roberta_best['profile']}, Macro-F1={roberta_best['macro_f1']:.4%}")
    print(f"  RoBERTa比最优传统方法高 {(roberta_best['macro_f1'] - classical_best['macro_f1'])*100:.2f} 个百分点")
    print(f"  同一验证集误判数：传统 {wrong_cl}，RoBERTa {wrong_ro}，减少 {wrong_cl - wrong_ro}")
    print("  图2训练曲线：", "已加入" if has_history else "未加入（未找到对应 history.csv）")


if __name__ == "__main__":
    main()
