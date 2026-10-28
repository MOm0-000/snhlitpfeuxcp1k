#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型B：原始 UER RoBERTa-Medium + CLS/Mean Pooling 分类头 + FGM。"""
from train_A import run

if __name__ == "__main__":
    run("B")
