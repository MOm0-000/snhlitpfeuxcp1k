#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现指导书中 WordPiece、特殊 token、token_type_ids 与 attention_mask 的示例。"""
from __future__ import annotations

import argparse
import json

from transformers import BertTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 UER 中文 WordPiece 编码过程")
    parser.add_argument("--model", default="uer/chinese_roberta_L-8_H-512")
    parser.add_argument("--max-length", type=int, default=24)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    texts = ["包装很好，但是商品质量差。", "这个keyboard很好用。", "物流有点慢，不过东西很好用。", "unaffordable"]
    print("模型:", args.model, "词表大小:", tokenizer.vocab_size)
    for text in texts:
        tokens = tokenizer.tokenize(text)
        print("\n原文:", text)
        print("WordPiece:", list(zip(tokens, tokenizer.convert_tokens_to_ids(tokens))))
    examples = tokenizer(texts[:2], padding=True, truncation=True, max_length=args.max_length)
    for i, sentence in enumerate(texts[:2]):
        print("\n", sentence)
        print(json.dumps({
            "tokens": tokenizer.convert_ids_to_tokens(examples["input_ids"][i]),
            "input_ids": examples["input_ids"][i],
            "attention_mask": examples["attention_mask"][i],
            "token_type_ids": examples.get("token_type_ids", [None] * 2)[i],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
