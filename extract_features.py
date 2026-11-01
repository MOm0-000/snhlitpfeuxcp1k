#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从课程指定的 UER 中文 RoBERTa 预训练模型中提取冻结的 [CLS] 特征。

输出：
  outputs/fc_only/train_cls_features.npz
  outputs/fc_only/validation_cls_features.npz
  outputs/fc_only/feature_metadata.json

注意：
- 预训练编码器所有参数均冻结，不发生任何训练。
- 每条评论保存最后一层 [CLS] 的 512 维表示。
- tokenizer 与预训练模型来自同一 model-dir。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, BertTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="提取冻结 RoBERTa 的 CLS 特征")
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--model-dir", default="pretrained")
    p.add_argument("--output-dir", default="outputs/fc_only")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_reviews(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    df = pd.read_csv(path, keep_default_na=False, dtype={"id": str, "text": str})
    required = {"id", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} 必须包含 id、text、label 三列")
    if df.empty:
        raise ValueError(f"{path} 为空")
    if df["id"].str.strip().eq("").any() or df["id"].duplicated().any():
        raise ValueError(f"{path} 存在空 ID 或重复 ID")
    if df["text"].str.strip().eq("").any():
        raise ValueError(f"{path} 存在空评论")
    labels = pd.to_numeric(df["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{path} 的 label 必须为 0 或 1")
    df["label"] = labels.astype(np.int64)
    return df[["id", "text", "label"]].copy()


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.ids = df["id"].astype(str).tolist()
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.ids[i], self.texts[i], self.labels[i]


def choose_device(mode: str) -> torch.device:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但没有检测到 CUDA")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def extract_split(
    name: str,
    df: pd.DataFrame,
    encoder,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    num_workers: int,
    output_path: Path,
):
    dataset = TextDataset(df)

    def collate(rows):
        ids, texts, labels = zip(*rows)
        batch = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return list(ids), batch, np.asarray(labels, dtype=np.int64)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    features, labels_all, ids_all = [], [], []

    encoder.eval()
    for ids, batch, labels in tqdm(loader, desc=f"提取 {name} 特征"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = encoder(**batch)
        cls = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        features.append(cls)
        labels_all.append(labels)
        ids_all.extend(ids)

    x = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(labels_all, axis=0).astype(np.int64, copy=False)
    ids_np = np.asarray(ids_all, dtype=str)

    if x.shape[0] != len(df) or y.shape[0] != len(df):
        raise RuntimeError("提取后的样本数与原 CSV 不一致")

    np.savez_compressed(
        output_path,
        features=x,
        labels=y,
        ids=ids_np,
    )

    print(f"{name}: features={x.shape}, labels={y.shape}, 保存到 {output_path}")
    return {
        "rows": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "label_0": int((y == 0).sum()),
        "label_1": int((y == 1).sum()),
        "file": str(output_path),
    }


def main():
    args = parse_args()

    train_path = Path(args.train)
    val_path = Path(args.validation)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"找不到预训练模型目录：{model_dir}")
    if not (model_dir / "pytorch_model.bin").is_file() and not any(model_dir.glob("*.safetensors")):
        raise FileNotFoundError(
            f"{model_dir} 中没有 pytorch_model.bin 或 safetensors 权重文件"
        )

    train_df = read_reviews(train_path)
    val_df = read_reviews(val_path)

    overlap = set(train_df["id"]) & set(val_df["id"])
    if overlap:
        raise ValueError(f"训练集和验证集有 {len(overlap)} 个重复 ID")

    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)

    if getattr(config, "hidden_size", None) != 512:
        raise ValueError(
            f"预训练模型 hidden_size={getattr(config, 'hidden_size', None)}，"
            "不是课程参考模型的 512"
        )

    if args.max_length > getattr(config, "max_position_embeddings", 512):
        raise ValueError("--max-length 超过模型最大位置长度")

    # 不加载下游分类头，也不使用随机 pooler。
    encoder = AutoModel.from_pretrained(
        model_dir,
        add_pooling_layer=False,
        local_files_only=True,
    )

    # 明确冻结编码器。
    for p in encoder.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    if trainable != 0:
        raise RuntimeError("编码器仍存在可训练参数，拒绝继续")

    device = choose_device(args.device)
    encoder.to(device)
    encoder.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"设备: {device}")
    print(f"编码器总参数: {total:,}")
    print(f"编码器可训练参数: {trainable}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"max_length: {args.max_length}")

    train_info = extract_split(
        "train",
        train_df,
        encoder,
        tokenizer,
        device,
        args.max_length,
        args.batch_size,
        args.num_workers,
        output_dir / "train_cls_features.npz",
    )

    val_info = extract_split(
        "validation",
        val_df,
        encoder,
        tokenizer,
        device,
        args.max_length,
        args.batch_size,
        args.num_workers,
        output_dir / "validation_cls_features.npz",
    )

    metadata = {
        "method": "Frozen pretrained encoder + last-layer CLS feature",
        "model_dir": str(model_dir),
        "model_type": getattr(config, "model_type", None),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "max_length": int(args.max_length),
        "tokenizer_vocab_size": int(len(tokenizer)),
        "encoder_total_parameters": int(total),
        "encoder_trainable_parameters": int(trainable),
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(val_path),
        "train": train_info,
        "validation": val_info,
    }

    with (output_dir / "feature_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"特征元数据已保存：{output_dir / 'feature_metadata.json'}")


if __name__ == "__main__":
    main()
