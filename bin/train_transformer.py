#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UER 中文 RoBERTa-Medium 二分类微调（纯 .py，无 notebook / Trainer 依赖）。

功能：
- 配套 BertTokenizer / WordPiece，动态 padding，GPU AMP (fp16/bf16)
- 分层学习率衰减 LLRD、分类头学习率倍增、AdamW、warmup + linear/cosine
- 梯度累积、裁剪、可选 FGM 对抗训练、可选 R-Drop、label smoothing
- 每轮验证，按 Macro-F1 保存最优 checkpoint，早停、恢复并复核结果
- 保存历史 loss/指标，验证集全量预测、错误样例、混淆矩阵与运行配置

注：预训练模型本身不含情感二分类头，新增 classifier 会随机初始化，这是正常的。
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BertTokenizer,
    get_scheduler,
)

from common import (
    check_split, compute_metrics, dump_json, print_metrics,
    read_csv, save_errors, save_predictions, set_seed,
)

LABEL_ID2NAME = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL_NAME2ID = {"NEGATIVE": 0, "POSITIVE": 1}


class EncodedReviews(Dataset):
    """预先 tokenize 但不进行 padding；同批数据动态补齐，节省显存。"""

    def __init__(self, tokenizer: BertTokenizer, frame: pd.DataFrame, max_length: int):
        texts = frame["text"].tolist()
        batch = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
        self.features = [{key: values[i] for key, values in batch.items()} for i in range(len(texts))]
        self.labels = frame["label"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {**self.features[index], "labels": self.labels[index]}


class DynamicCollate:
    def __init__(self, tokenizer: BertTokenizer, pad_multiple: int | None = None):
        self.tokenizer = tokenizer
        self.pad_multiple = pad_multiple

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        labels = [r["labels"] for r in rows]
        features = [{k: v for k, v in row.items() if k != "labels"} for row in rows]
        batch = self.tokenizer.pad(
            features, padding=True, pad_to_multiple_of=self.pad_multiple,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


class FGM:
    """Fast Gradient Method：只扰动词嵌入，不改写保存的权重。"""

    def __init__(self, model: torch.nn.Module, epsilon: float):
        self.model = model
        self.epsilon = epsilon
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self) -> bool:
        self.backup.clear()
        for name, param in self.model.named_parameters():
            if not (name.endswith("word_embeddings.weight") and param.requires_grad and param.grad is not None):
                continue
            norm = torch.norm(param.grad)
            if not torch.isfinite(norm).item() or norm.item() == 0:
                continue
            self.backup[name] = param.data.clone()
            param.data.add_(self.epsilon * param.grad / norm)
        return bool(self.backup)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def make_param_groups(model: torch.nn.Module, lr: float, weight_decay: float,
                      layerwise_decay: float, head_multiplier: float) -> list[dict]:
    """低层 Transformer 使用较低学习率，随机初始化分类头使用较大学习率。"""
    n_layers = int(model.config.num_hidden_layers)
    buckets: dict[tuple[float, float], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier.") or name.startswith("score."):
            local_lr = lr * head_multiplier
        elif ".embeddings." in name:
            local_lr = lr * (layerwise_decay ** n_layers)
        else:
            match = re.search(r"\.encoder\.layer\.(\d+)\.", name)
            if match:
                layer_id = int(match.group(1))
                local_lr = lr * (layerwise_decay ** (n_layers - 1 - layer_id))
            else:
                local_lr = lr
        # 不对 bias 和归一化参数应用权重衰减。
        lower_name = name.lower()
        no_decay = (
            name.endswith(".bias") or "layernorm" in lower_name
            or "layer_norm" in lower_name or "norm.weight" in lower_name
        )
        decay = 0.0 if no_decay else weight_decay
        buckets.setdefault((local_lr, decay), []).append(param)
    return [
        {"params": params, "lr": lr_val, "weight_decay": wd}
        for (lr_val, wd), params in buckets.items()
    ]


def amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    return torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" and amp_dtype else contextlib.nullcontext()


def compute_training_loss(model, batch, label_smoothing: float, rdrop_alpha: float):
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    labels = batch["labels"]
    logits = model(**inputs).logits
    first_ce = F.cross_entropy(logits.float(), labels, label_smoothing=label_smoothing)
    if rdrop_alpha <= 0:
        return first_ce
    logits2 = model(**inputs).logits
    second_ce = F.cross_entropy(logits2.float(), labels, label_smoothing=label_smoothing)
    logp = F.log_softmax(logits.float(), dim=-1)
    logq = F.log_softmax(logits2.float(), dim=-1)
    symmetric_kl = 0.5 * (
        F.kl_div(logp, logq.exp(), reduction="batchmean") +
        F.kl_div(logq, logp.exp(), reduction="batchmean")
    )
    return (first_ce + second_ce) * 0.5 + rdrop_alpha * symmetric_kl


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, amp_dtype: torch.dtype | None):
    model.eval()
    true_labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    positive_probs: list[np.ndarray] = []
    for batch in tqdm(loader, desc="  validation", leave=False):
        labels = batch.pop("labels").numpy()
        inputs = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
        with amp_context(device, amp_dtype):
            logits = model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        true_labels.append(labels)
        predictions.append(probs.argmax(axis=-1).astype(np.int64))
        positive_probs.append(probs[:, 1])
    y_true = np.concatenate(true_labels)
    y_pred = np.concatenate(predictions)
    proba = np.concatenate(positive_probs)
    return compute_metrics(y_true, y_pred), y_pred, proba


def choose_amp(mode: str, device: torch.device) -> torch.dtype | None:
    if mode == "none":
        return None
    if device.type != "cuda":
        if mode == "auto":
            return None
        raise ValueError("fp16/bf16 混合精度仅支持 CUDA，请使用 --amp auto 或 --amp none")
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前 GPU 不支持 bf16，请用 --amp fp16")
    return torch.float16 if mode == "fp16" else torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser(description="基于 UER RoBERTa-Medium 的中文情感分类微调")
    p.add_argument("--model", default="uer/chinese_roberta_L-8_H-512", help="HF 模型 ID 或含权重的本地模型目录")
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--output-dir", default="outputs/roberta_medium")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr-multiplier", type=float, default=5.0)
    p.add_argument("--layerwise-lr-decay", type=float, default=0.92)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--fgm-epsilon", type=float, default=0.0, help=">0 开启对抗训练；建议 0.3~0.8")
    p.add_argument("--rdrop-alpha", type=float, default=0.0, help=">0 开启 R-Drop，例如 0.5；计算量约翻倍")
    p.add_argument("--patience", type=int, default=2, help="验证 Macro-F1 连续不提升多少轮后早停，0=关闭")
    p.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-checkpointing", action="store_true", help="降低显存但减慢训练")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for flag in ["epochs", "batch_size", "eval_batch_size", "grad_accum", "max_length"]:
        if getattr(args, flag) < 1:
            raise ValueError(f"--{flag.replace('_', '-')} 必须 >=1")
    if args.max_length < 3:
        raise ValueError("--max-length 需要至少容纳 [CLS]、正文、[SEP]")
    if args.lr <= 0 or not (0 <= args.warmup_ratio < 1) or not (0 < args.layerwise_lr_decay <= 1):
        raise ValueError("无效的学习率、warmup 比例或 layerwise lr decay")
    if not (0 <= args.label_smoothing < 1) or not (0 <= args.dropout < 1):
        raise ValueError("label-smoothing / dropout 必须在 [0,1) 范围内")
    if args.fgm_epsilon < 0 or args.rdrop_alpha < 0:
        raise ValueError("FGM epsilon 和 R-Drop alpha 必须 >=0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch 检测不到 CUDA，请检查驱动与 CUDA 版本")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    amp_dtype = choose_amp(args.amp, device)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print("CUDA GPU:", torch.cuda.get_device_name(0))
    print("Device:", device, "AMP:", str(amp_dtype), "PyTorch:", torch.__version__)

    out = Path(args.output_dir)
    model_dir = out / "best_model"
    out.mkdir(parents=True, exist_ok=True)
    dump_json(vars(args), out / "training_config.json")
    df_train = read_csv(args.train)
    df_val = read_csv(args.validation)
    check_split(df_train, df_val)
    print("载入配套 WordPiece / BERT 模型:", args.model, flush=True)
    tokenizer = BertTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    if args.max_length > getattr(config, "max_position_embeddings", 512):
        raise ValueError("max-length 超过预训练模型的位置嵌入上限")
    config.num_labels = 2
    config.id2label = LABEL_ID2NAME
    config.label2id = LABEL_NAME2ID
    config.hidden_dropout_prob = args.dropout
    config.classifier_dropout = args.dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config, ignore_mismatched_sizes=True,
    ).to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,}；参与更新: {trainable_params:,}；词表大小: {tokenizer.vocab_size}")
    print("Token 示例:", tokenizer.convert_ids_to_tokens(tokenizer("包装好但是质量差")["input_ids"]))

    print("预编码文本（动态 padding 在 DataLoader 中执行）...", flush=True)
    train_data = EncodedReviews(tokenizer, df_train, args.max_length)
    val_data = EncodedReviews(tokenizer, df_val, args.max_length)
    collate = DynamicCollate(tokenizer, pad_multiple=8 if amp_dtype is not None else None)
    loader_train = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    loader_val = DataLoader(
        val_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        make_param_groups(model, args.lr, args.weight_decay,
                          args.layerwise_lr_decay, args.head_lr_multiplier),
        lr=args.lr, eps=1e-8,
    )
    updates_per_epoch = math.ceil(len(loader_train) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = round(total_updates * args.warmup_ratio)
    scheduler = get_scheduler(
        args.scheduler, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_updates,
    )
    # CUDA fp16 需要梯度缩放；CUDA bf16 不需要；CPU 不需要。
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    fgm = FGM(model, args.fgm_epsilon) if args.fgm_epsilon > 0 else None

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    global_step = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        num_seen = 0
        progress = tqdm(enumerate(loader_train), total=len(loader_train), desc=f"Epoch {epoch}/{args.epochs}")
        for idx, batch in progress:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            # 最后不足 grad_accum 个 batch 的组，也保持正确的梯度缩放。
            group_start = (idx // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(loader_train) - group_start)
            with amp_context(device, amp_dtype):
                loss = compute_training_loss(model, batch, args.label_smoothing, args.rdrop_alpha)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"第 {epoch} 轮出现非有限 loss，请减小 LR 或关闭 fp16")
            scaler.scale(loss / group_size).backward()

            if fgm is not None and fgm.attack():
                try:
                    with amp_context(device, amp_dtype):
                        adv_loss = compute_training_loss(model, batch, args.label_smoothing, args.rdrop_alpha)
                    scaler.scale(adv_loss / group_size).backward()
                finally:
                    fgm.restore()

            batch_size = len(batch["labels"])
            epoch_loss += float(loss.detach().item()) * batch_size
            num_seen += batch_size
            should_update = (idx + 1) % args.grad_accum == 0 or (idx + 1) == len(loader_train)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                before_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # fp16 梯度溢出时跳过 scheduler，避免学习率计划与实际更新脱节。
                if scaler.get_scale() >= before_scale:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
            if idx % 20 == 0:
                progress.set_postfix(loss=f"{epoch_loss / max(1, num_seen):.4f}", lr=f"{scheduler.get_last_lr()[-1]:.2g}")

        train_loss = epoch_loss / num_seen
        metrics, y_pred, positive_probs = evaluate(model, loader_val, device, amp_dtype)
        row = {
            "epoch": epoch, "train_loss": train_loss, "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"], "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "positive_precision": metrics["positive_precision"],
            "positive_recall": metrics["positive_recall"],
            "optimizer_step": global_step,
            "lr_last_param_group": scheduler.get_last_lr()[-1],
            "elapsed_seconds": round(time.perf_counter() - start_time, 1),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False, encoding="utf-8-sig")
        print(f"第 {epoch} 轮 train_loss={train_loss:.5f}")
        print_metrics("Validation", metrics)
        if metrics["macro_f1"] > best_f1 + 1e-7:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
            dump_json({
                **metrics, "best_epoch": best_epoch, "model": args.model,
                "max_length": args.max_length, "total_parameters": total_params,
                "trainable_parameters": trainable_params, "amp_dtype": str(amp_dtype),
            }, out / "best_metrics.json")
            save_predictions(df_val, y_pred, out / "validation_predictions.csv", positive_probs)
            save_errors(df_val, y_pred, out / "errors_top250.csv", positive_probs)
            print(f"保存新的最佳 checkpoint：{model_dir} (Macro-F1 {best_f1:.4f})")
        else:
            no_improve += 1
            print(f"Macro-F1 未刷新最佳，已连续 {no_improve} 轮")
            if args.patience > 0 and no_improve >= args.patience:
                print("触发早停")
                break

    print("\n按课程要求：从磁盘恢复最佳模型，重新评估完整验证集。")
    del model, optimizer, scheduler, scaler, fgm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    restored = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True,
    ).to(device)
    new_metrics, new_pred, new_probs = evaluate(restored, loader_val, device, amp_dtype)
    if abs(new_metrics["macro_f1"] - best_f1) > 1e-4:
        raise RuntimeError("恢复后的 Macro-F1 与训练时记录的不一致，请检查权重或精度设置")
    print_metrics(f"Restored best (epoch {best_epoch})", new_metrics)
    dump_json({
        **new_metrics, "best_epoch": best_epoch,
        "best_model": str(model_dir.resolve()), "verified_reload": True,
        "training_seconds": round(time.perf_counter() - start_time, 1),
    }, out / "restored_metrics.json")
    save_predictions(df_val, new_pred, out / "restored_validation_predictions.csv", new_probs)
    print("\n全部完成。最优轮次:", best_epoch, "最佳 Macro-F1:", round(best_f1, 5))
    print("模型:", model_dir.resolve())
    print("结果:", out.resolve())


if __name__ == "__main__":
    main()
