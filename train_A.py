#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B/C/D 情感分类实验：单次训练，双验证标签独立选模。

每种模型使用原始/修正训练集分别训练一次（共8次训练）。
每个 epoch 对同一批验证评论只预测一次，分别计算原始/语义修正标签的 Macro-F1；
根据两套标签各自的最佳 Macro-F1，分别保存一个 checkpoint（总计16个）。

四个入口：train_A.py / train_B.py / train_C.py / train_D.py。
B/C/D 只导入本文件的 run()，无需 train_transformer.py、predict_sentiment.py 或 test_abcd.py。
模型始终从项目本地 pretrained/ 加载，不联网。

重要：修正验证标签仅用于附加语义研究，不是独立的测试集。
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, BertTokenizer, get_scheduler
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_ID = "uer/chinese_roberta_L-8_H-512"
LOCAL_PRETRAINED = Path(__file__).resolve().parent / "pretrained"
FGM_VERSION = "current_microbatch_gradient_v2"
VARIANTS = {
    "A": {"pooling": False, "rdrop_alpha": 0.0, "name": "FGM"},
    "B": {"pooling": True,  "rdrop_alpha": 0.0, "name": "FGM + CLS/Mean Pooling"},
    "C": {"pooling": False, "rdrop_alpha": 0.5, "name": "FGM + R-Drop"},
    "D": {"pooling": True,  "rdrop_alpha": 0.5, "name": "FGM + CLS/Mean Pooling + R-Drop"},
}


# --------------- 数据与评价 ---------------

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_reviews(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    df = pd.read_csv(path, dtype={"id": str, "text": str}, keep_default_na=False)
    if not {"id", "text", "label"} <= set(df.columns):
        raise ValueError(f"{path} 必须包含 id、text、label 三列")
    if df.empty or df["id"].duplicated().any() or df["id"].str.strip().eq("").any():
        raise ValueError(f"{path} 存在空ID、重复ID或数据为空")
    if df["text"].str.strip().eq("").any():
        raise ValueError(f"{path} 存在空评论")
    label = pd.to_numeric(df["label"], errors="raise")
    if not label.isin([0, 1]).all():
        raise ValueError(f"{path} 的 label 必须是 0 或 1")
    df["label"] = label.astype(int)
    return df[["id", "text", "label"]].copy()


def verify_same_comments(original: pd.DataFrame, revised: pd.DataFrame, title: str) -> None:
    """修正标签只能改 label；两份数据必须有相同的 id/text。"""
    first = original.set_index("id")["text"].sort_index()
    second = revised.set_index("id")["text"].sort_index()
    if not first.equals(second):
        raise ValueError(f"{title} 的原始版与修正版 id/text 不一致")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    precision, recall, _, _ = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "correct": int((actual == predicted).sum()),
        "wrong": int((actual != predicted).sum()),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_length: int):
        encoded = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        self.inputs = [{key: val[i] for key, val in encoded.items()} for i in range(len(df))]
        self.labels = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {**self.inputs[index], "labels": self.labels[index]}


def make_loader(df, tokenizer, max_length, batch_size, shuffle=False, amp=False, pin_memory=False):
    def collate(rows):
        features = [{k: v for k, v in r.items() if k != "labels"} for r in rows]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt", pad_to_multiple_of=8 if amp else None)
        batch["labels"] = torch.tensor([r["labels"] for r in rows], dtype=torch.long)
        return batch

    return DataLoader(
        ReviewDataset(df, tokenizer, max_length), batch_size=batch_size,
        shuffle=shuffle, collate_fn=collate, pin_memory=pin_memory,
    )


# --------------- 模型及 checkpoint ---------------

class CLSMeanClassifier(nn.Module):
    """B/D：拼接最后一层 CLS 表示和非特殊词元的平均表示。"""

    def __init__(self, backbone, hidden_size, dropout, cls_id, sep_id):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 2)
        self.cls_id, self.sep_id = cls_id, sep_id

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, **kwargs,
        ).last_hidden_state
        cls = hidden[:, 0]
        content = (input_ids != self.cls_id) & (input_ids != self.sep_id)
        if attention_mask is not None:
            content = content & attention_mask.bool()
        mask = content.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        logits = self.classifier(self.dropout(torch.cat([cls, mean], dim=-1)))
        return SequenceClassifierOutput(logits=logits)


def build_model(variant, config, tokenizer, dropout, pretrained=None):
    """pretrained 为本地目录时加载预训练权重；None 用于恢复自定义模型。"""
    if not VARIANTS[variant]["pooling"]:
        if pretrained is None:
            return AutoModelForSequenceClassification.from_config(config)
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained, config=config, local_files_only=True
        )
    if pretrained is None:
        backbone = AutoModel.from_config(config, add_pooling_layer=False)
    else:
        backbone = AutoModel.from_pretrained(
            pretrained, config=config, add_pooling_layer=False, local_files_only=True
        )
    return CLSMeanClassifier(
        backbone, config.hidden_size, dropout, tokenizer.cls_token_id, tokenizer.sep_token_id
    )


def save_checkpoint(model, tokenizer, folder: Path, variant: str, dropout: float, max_length: int):
    folder.mkdir(parents=True, exist_ok=True)
    if VARIANTS[variant]["pooling"]:
        model.backbone.config.save_pretrained(folder)
        torch.save(model.state_dict(), folder / "model_state.pt")
    else:
        model.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    save_json({
        "variant": variant, "pooling": VARIANTS[variant]["pooling"],
        "rdrop_alpha": VARIANTS[variant]["rdrop_alpha"],
        "dropout": dropout, "max_length": max_length,
        "pretrained_model": MODEL_ID,
    }, folder / "variant.json")


def load_checkpoint(folder: Path, device: torch.device,
                    legacy_variant: str | None = None, legacy_max_length: int = 192):
    """与现有 test_abcd.py 接口、checkpoint 格式兼容。"""
    folder = Path(folder)
    info_file = folder / "variant.json"
    if info_file.exists():
        info = json.loads(info_file.read_text(encoding="utf-8"))
    elif legacy_variant == "A":  # 兼容旧 FGM checkpoint
        info = {"variant": "A", "max_length": legacy_max_length, "dropout": 0.1}
    else:
        raise FileNotFoundError(f"{folder} 缺少 variant.json")

    variant = info["variant"]
    tokenizer = BertTokenizer.from_pretrained(folder, local_files_only=True)
    if VARIANTS[variant]["pooling"]:
        config = AutoConfig.from_pretrained(folder, local_files_only=True)
        model = build_model(variant, config, tokenizer, float(info["dropout"]))
        state = torch.load(folder / "model_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True)
    return model.to(device).eval(), tokenizer, info


# --------------- FGM、R-Drop 和优化器 ---------------

class FGM:
    """对词嵌入添加扰动；梯度累积时仅使用当前 micro-batch 的梯度。"""

    def __init__(self, model: nn.Module, epsilon: float):
        self.embedding = next(
            (p for name, p in model.named_parameters() if name.endswith("word_embeddings.weight")), None
        )
        if self.embedding is None:
            raise ValueError("没有找到 word_embeddings.weight，无法进行 FGM")
        self.epsilon = epsilon
        self.backup = None

    def snapshot_grad(self):
        grad = self.embedding.grad
        return None if grad is None else grad.detach().clone()

    def attack(self, previous_grad) -> bool:
        grad = self.embedding.grad
        if grad is None:
            raise RuntimeError("词嵌入没有梯度，FGM 无法生效")
        current_grad = grad.detach() if previous_grad is None else grad.detach() - previous_grad
        norm = current_grad.float().norm()
        if not torch.isfinite(norm).item() or norm.item() == 0:
            return False
        with torch.no_grad():
            self.backup = self.embedding.detach().clone()
            self.embedding.add_(self.epsilon * (current_grad.float() / norm).to(self.embedding.dtype))
        return True

    def restore(self):
        if self.backup is not None:
            with torch.no_grad():
                self.embedding.copy_(self.backup)
            self.backup = None


def optimizer_groups(model, args, n_layers):
    """分类层较大学习率；Transformer 从底层到高层递增。"""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("classifier.", "score.")):
            lr = args.lr * args.head_lr_multiplier
        elif ".embeddings." in name:
            lr = args.lr * args.layerwise_lr_decay ** n_layers
        else:
            match = re.search(r"\.encoder\.layer\.(\d+)\.", name)
            lr = args.lr * args.layerwise_lr_decay ** (n_layers - 1 - int(match[1])) if match else args.lr
        wd = 0 if name.endswith(".bias") or "layernorm" in name.lower() or "layer_norm" in name.lower() else args.weight_decay
        groups.setdefault((lr, wd), []).append(p)
    return [{"params": params, "lr": lr, "weight_decay": wd} for (lr, wd), params in groups.items()]


def loss_for_batch(model, batch, smoothing, rdrop_alpha):
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    labels = batch["labels"]
    logits1 = model(**inputs).logits.float()
    ce1 = F.cross_entropy(logits1, labels, label_smoothing=smoothing)
    if rdrop_alpha == 0:
        return ce1
    logits2 = model(**inputs).logits.float()
    ce2 = F.cross_entropy(logits2, labels, label_smoothing=smoothing)
    p, q = F.log_softmax(logits1, -1), F.log_softmax(logits2, -1)
    symmetric_kl = (F.kl_div(p, q.exp(), reduction="batchmean") +
                    F.kl_div(q, p.exp(), reduction="batchmean")) / 2
    return (ce1 + ce2) / 2 + rdrop_alpha * symmetric_kl


def amp_dtype_for(mode, device):
    if mode == "none" or device.type != "cuda":
        return None
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前显卡不支持 BF16")
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16 if mode == "bf16" else torch.float16


def amp_context(device, amp_dtype):
    return (torch.autocast("cuda", dtype=amp_dtype)
            if device.type == "cuda" and amp_dtype else contextlib.nullcontext())


# --------------- 训练、验证、保存 ---------------

@torch.inference_mode()
def predict_labels(model, loader, device):
    """对验证文本预测一次，随后分别与两套标签比较。"""
    model.eval()  # FP32，训练时和恢复时使用相同的预测精度
    predictions = []
    for batch in tqdm(loader, desc="验证", leave=False):
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}
        predictions.append(model(**inputs).logits.argmax(-1).cpu().numpy())
    return np.concatenate(predictions)


def train_epoch(model, loader, optimizer, scheduler, scaler, fgm, device, amp_dtype, args, alpha):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, samples, steps = 0.0, 0, 0
    for i, batch in enumerate(tqdm(loader, desc="训练", leave=False)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        group_start = (i // args.grad_accum) * args.grad_accum
        group_size = min(args.grad_accum, len(loader) - group_start)

        previous_grad = fgm.snapshot_grad()
        with amp_context(device, amp_dtype):
            loss = loss_for_batch(model, batch, args.label_smoothing, alpha)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Loss 不是有限数，可降低学习率或关闭 FP16")
        scaler.scale(loss / group_size).backward()

        if fgm.attack(previous_grad):
            try:
                with amp_context(device, amp_dtype):
                    adversarial_loss = loss_for_batch(model, batch, args.label_smoothing, alpha)
                scaler.scale(adversarial_loss / group_size).backward()
            finally:
                fgm.restore()

        n = batch["labels"].size(0)
        loss_sum += loss.item() * n  # 记录正常输入损失，不包括额外对抗损失
        samples += n
        if (i + 1) % args.grad_accum == 0 or i + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:  # FP16 溢出时不更新学习率
                scheduler.step()
                steps += 1
            optimizer.zero_grad(set_to_none=True)
    return loss_sum / samples, steps


def train_one(variant, train_name, args, data, valid, semantic_valid):
    """训练一次；对原始、修正两套验证标签分别选取最佳 epoch。"""
    out = Path(args.output_root) / variant / train_name
    train_file = args.train if train_name == "original" else args.clean_train
    labels = {
        "original": valid["label"].to_numpy(),
        "semantic": semantic_valid["label"].to_numpy(),
    }
    # 完成标记同时也是复现依据：避免把旧版单验证结果错误地当成新版结果。
    signature = {
        "experiment_version": "dual_validation_best_epoch_v1",
        "variant": variant,
        "fgm_implementation": FGM_VERSION,
        "train_sha256": file_hash(train_file),
        "validation_sha256": file_hash(args.validation),
        "semantic_validation_sha256": file_hash(args.semantic_validation),
        "model_id": MODEL_ID,
        **{key: value for key, value in vars(args).items()
           if key not in {"output_root", "overwrite", "dataset", "demo"}},
    }
    selected_paths = {key: out / f"best_by_{key}" for key in labels}
    completed = all((folder / "best_model" / "variant.json").is_file() and
                    (folder / "metrics.json").is_file() for folder in selected_paths.values())
    if completed and not args.overwrite:
        config_path = out / "training_config.json"
        if not config_path.is_file():
            raise FileExistsError(f"{out} 已有模型但没有配置记录，请检查或另设输出目录")
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if any(old.get(key) != value for key, value in signature.items()):
            raise FileExistsError(f"{out} 的训练数据或参数不同，请使用新目录，不要混合比较")
        print(f"[{variant}/{train_name}] 已存在完整且配置相同的结果，跳过。", flush=True)
        return
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} 已存在但不完整；如需重训请指定 --overwrite（会删除该目录）")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    save_json(signature, out / "training_config.json")

    print(f"\n{'=' * 65}\n模型 {variant} | {VARIANTS[variant]['name']} | {train_name}"
          f"\n训练 {len(data)} 条，双验证 {len(valid)} 条，输出 {out}\n{'=' * 65}", flush=True)
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("没有检测到 NVIDIA CUDA 显卡")
    dtype = amp_dtype_for(args.amp, device)
    tokenizer = BertTokenizer.from_pretrained(LOCAL_PRETRAINED, local_files_only=True)
    config = AutoConfig.from_pretrained(LOCAL_PRETRAINED, local_files_only=True)
    if config.num_hidden_layers != 8 or config.hidden_size != 512:
        raise ValueError("pretrained/ 不是课程指定的 UER RoBERTa-Medium (8层/512维)")
    if args.max_length > config.max_position_embeddings:
        raise ValueError("max-length 超过预训练模型最大长度")
    config.num_labels = 2
    config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    config.hidden_dropout_prob = args.dropout
    config.classifier_dropout = args.dropout
    model = build_model(variant, config, tokenizer, args.dropout, pretrained=LOCAL_PRETRAINED).to(device)
    if args.grad_checkpointing:
        (model.backbone if VARIANTS[variant]["pooling"] else model).gradient_checkpointing_enable()

    train_loader = make_loader(data, tokenizer, args.max_length, args.batch_size,
                               shuffle=True, amp=dtype is not None, pin_memory=device.type == "cuda")
    # 两份验证 CSV 的 id/text 相同，故只构建一次验证 DataLoader。
    val_loader = make_loader(valid, tokenizer, args.max_length, args.eval_batch_size,
                             pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(optimizer_groups(model, args, config.num_hidden_layers))
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=round(total_steps * args.warmup_ratio),
                              num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    fgm = FGM(model, args.fgm_epsilon)
    best = {key: {"f1": -1.0, "epoch": -1} for key in labels}
    history, stale, started, steps = [], 0, time.perf_counter(), 0
    for epoch in range(1, args.epochs + 1):
        loss, this_steps = train_epoch(model, train_loader, optimizer, scheduler, scaler,
                                      fgm, device, dtype, args, VARIANTS[variant]["rdrop_alpha"])
        steps += this_steps
        pred = predict_labels(model, val_loader, device)
        scores = {key: calculate_metrics(target, pred) for key, target in labels.items()}
        history.append({
            "epoch": epoch, "train_loss": loss,
            **{f"{key}_{metric}": scores[key][metric]
               for key in labels for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall")},
            "optimizer_steps": steps, "elapsed_seconds": round(time.perf_counter() - started, 1),
        })
        pd.DataFrame(history).to_csv(out / "history.csv", index=False, encoding="utf-8-sig")
        print(f"{variant}/{train_name} Epoch {epoch}: Loss={loss:.5f} "
              f"F1(original)={scores['original']['macro_f1']:.5f} "
              f"F1(semantic)={scores['semantic']['macro_f1']:.5f}", flush=True)
        improved = False
        for key in labels:
            f1 = scores[key]["macro_f1"]
            if f1 > best[key]["f1"] + 1e-8:
                best[key] = {"f1": f1, "epoch": epoch}
                save_checkpoint(model, tokenizer, selected_paths[key] / "best_model",
                                variant, args.dropout, args.max_length)
                improved = True
        # 只有两种验证指标都连续没有改善时才触发早停；默认不早停。
        stale = 0 if improved else stale + 1
        if args.patience > 0 and stale >= args.patience:
            print(f"两套验证指标均连续 {stale} 轮未提升，提前停止。", flush=True)
            break

    del model, optimizer, scheduler, scaler, fgm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # 恢复并复核两个最佳 checkpoint，各自在两套标签上重新计算指标。
    for key, folder in selected_paths.items():
        restored, _, _ = load_checkpoint(folder / "best_model", device)
        pred = predict_labels(restored, val_loader, device)
        restored_scores = {name: calculate_metrics(target, pred) for name, target in labels.items()}
        matched = abs(restored_scores[key]["macro_f1"] - best[key]["f1"]) < 1e-7
        save_json({
            "variant": variant, "train_version": train_name, "selected_by": key,
            "best_epoch": best[key]["epoch"], "verified_reload": matched,
            "scores": restored_scores,
            "model_path": str((folder / "best_model").resolve()),
            "training_seconds": round(time.perf_counter() - started, 1),
            "semantic_validation_is_partial": Path(args.semantic_validation).name.endswith("_partial.csv"),
        }, folder / "metrics.json")
        if not matched:
            raise RuntimeError(f"{variant}/{train_name}/{key} 恢复后F1不一致")
        print(f"已恢复：best_by_{key}, epoch={best[key]['epoch']}, "
              f"F1(original)={restored_scores['original']['macro_f1']:.5f}, "
              f"F1(semantic)={restored_scores['semantic']['macro_f1']:.5f}")
        del restored
        if device.type == "cuda":
            torch.cuda.empty_cache()


def update_comparison(output_root: Path):
    """汇总已完成的训练结果。运行四个脚本后累计得到16行，不额外写测试脚本。"""
    rows = []
    for variant in VARIANTS:
        for train_version in ("original", "cleaned"):
            for selected_by in ("original", "semantic"):
                path = output_root / variant / train_version / f"best_by_{selected_by}" / "metrics.json"
                if not path.is_file():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                score_o = result["scores"]["original"]
                score_s = result["scores"]["semantic"]
                rows.append({
                    "model": variant, "train_version": train_version,
                    "selected_by": selected_by, "best_epoch": result["best_epoch"],
                    "original_accuracy": score_o["accuracy"],
                    "original_macro_f1": score_o["macro_f1"],
                    "original_macro_precision": score_o["macro_precision"],
                    "original_macro_recall": score_o["macro_recall"],
                    "semantic_accuracy": score_s["accuracy"],
                    "semantic_macro_f1": score_s["macro_f1"],
                    "semantic_macro_precision": score_s["macro_precision"],
                    "semantic_macro_recall": score_s["macro_recall"],
                    "checkpoint": result["model_path"],
                    "verified_reload": result["verified_reload"],
                })
    if rows:
        dest = output_root / "comparison.csv"
        pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8-sig")
        print(f"已汇总 {len(rows)}/16 个checkpoint：{dest}")


# --------------- 命令行入口 ---------------

def parse_args(variant):
    p = argparse.ArgumentParser(description=f"模型 {variant}：{VARIANTS[variant]['name']}，双验证指标独立选模")
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--label-dir", default="outputs/label_audit",
                   help="人工修正数据目录；应同时包含train_semantic.csv与validation_semantic_partial.csv")
    p.add_argument("--clean-train", default=None, help="可单独覆盖修正训练集路径")
    p.add_argument("--semantic-validation", default=None, help="可单独覆盖修正验证集路径")
    p.add_argument("--dataset", choices=["original", "cleaned", "both"], default="both")
    p.add_argument("--output-root", default="outputs/abcd_dual", help="新目录，避免覆盖之前abcd实验")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--fgm-epsilon", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.04)
    p.add_argument("--layerwise-lr-decay", type=float, default=0.92)
    p.add_argument("--head-lr-multiplier", type=float, default=5.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=0,
                   help="0表示完整训练指定轮数；>0表示两套指标都不提升后早停")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--demo", action="store_true", help="训练后用最佳原始验证F1模型预测几条新评论")
    p.add_argument("--overwrite", action="store_true",
                   help="删除当前模型/训练集组合的原有输出后重训")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum, args.eval_batch_size) < 1 or args.max_length < 3:
        p.error("epochs、batch-size、grad-accum、eval-batch-size 应 >=1，max-length 应 >=3")
    if args.lr <= 0 or args.fgm_epsilon <= 0 or args.head_lr_multiplier <= 0:
        p.error("学习率、FGM epsilon 和分类头学习率倍数必须 >0")
    if not (0 <= args.label_smoothing < 1 and 0 <= args.dropout < 1):
        p.error("label-smoothing 和 dropout 应在 [0,1) 区间")
    if not (0 < args.layerwise_lr_decay <= 1 and 0 <= args.warmup_ratio < 1):
        p.error("layerwise-lr-decay 应在 (0,1]，warmup-ratio 应在 [0,1)")
    if args.patience < 0:
        p.error("patience不能小于0")
    args.clean_train = args.clean_train or str(Path(args.label_dir) / "train_semantic.csv")
    args.semantic_validation = args.semantic_validation or str(
        Path(args.label_dir) / "validation_semantic_partial.csv"
    )
    return args


def print_demo(variant, versions, output_root, device):
    """选原始验证标准下表现最好的checkpoint，展示正负、否定、转折新评论预测。"""
    choices = []
    for version in versions:
        folder = output_root / variant / version / "best_by_original"
        info_path = folder / "metrics.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            choices.append((info["scores"]["original"]["macro_f1"], folder))
    if not choices:
        return
    _, best_folder = max(choices, key=lambda row: row[0])
    model, tokenizer, info = load_checkpoint(best_folder / "best_model", device)
    examples = [
        "质量很好，物流很快，下次还会购买。",
        "质量非常差，客服态度也不好。",
        "这款商品不是不好，只是价格有点高。",
        "虽然包装很好，但是商品有明显缺陷。",
    ]
    encoded = tokenizer(examples, padding=True, truncation=True,
                        max_length=int(info["max_length"]), return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**encoded).logits.float(), dim=-1)[:, 1].cpu().tolist()
    print(f"\n模型 {variant} 新评论预测（{best_folder}）：")
    for text, probability in zip(examples, probs):
        print(f"{'正面' if probability >= 0.5 else '负面'} "
              f"(正面概率={probability:.3f})：{text}")


def run(variant):
    args = parse_args(variant)
    if not LOCAL_PRETRAINED.is_dir():
        raise FileNotFoundError(f"缺少本地预训练模型目录：{LOCAL_PRETRAINED}")
    original = read_reviews(args.train)
    valid = read_reviews(args.validation)
    semantic_valid = read_reviews(args.semantic_validation)
    verify_same_comments(valid, semantic_valid, "验证集")
    # 按ID对齐两套标签，不能假定两个CSV行顺序永远相同。
    semantic_valid = semantic_valid.set_index("id").loc[valid["id"].tolist()].reset_index()
    changed = int((valid["label"].to_numpy() != semantic_valid["label"].to_numpy()).sum())
    print(f"验证集 {len(valid)} 条，两套标签有 {changed} 条差异；两套标签将独立选模。")
    if set(original["id"]) & set(valid["id"]):
        raise ValueError("训练集与验证集存在重复 ID")
    if args.dataset != "original":
        cleaned = read_reviews(args.clean_train)
        verify_same_comments(original, cleaned, "训练集")
        cleaned = cleaned.set_index("id").loc[original["id"].tolist()].reset_index()
    else:
        cleaned = None
    datasets = ("original", "cleaned") if args.dataset == "both" else (args.dataset,)
    for train_name in datasets:
        data = original if train_name == "original" else cleaned
        train_one(variant, train_name, args, data, valid, semantic_valid)
        update_comparison(Path(args.output_root))
    if args.demo:
        device = torch.device("cuda" if (args.device == "cuda" or
                             (args.device == "auto" and torch.cuda.is_available())) else "cpu")
        print_demo(variant, datasets, Path(args.output_root), device)


if __name__ == "__main__":
    run("A")
