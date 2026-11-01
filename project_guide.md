# FC_finetune 实验项目脚本与结果阅读指南

本文档说明 `FC_finetune` 分支最终有效的实验链路。最终结果口径与实验报告一致：

```text
经典最佳：NB_SVM_C4
Macro-F1 = 85.173%

预训练路线：Frozen RoBERTa + FC
learning_rate = 0.002
weight_decay = 0.01
best_epoch = 18
Macro-F1 = 85.525%
```

## 1. 最终实验链路

```text
data/train.csv + data/validation.csv
        │
        ├── inspect_tokenizer.py
        │       └── outputs/inspect_tokenizer.txt
        │
        ├── train_classical.py
        │       └── outputs/classical/
        │
        └── extract_features.py
                │
                ├── 冻结 UER RoBERTa
                ├── train_cls_features.npz
                └── validation_cls_features.npz
                        │
                        ├── train_fc.py
                        │       └── 单组 FC 训练
                        │
                        └── sweep_fc.py
                                ├── 3 个学习率 × 2 个 weight decay
                                ├── outputs/fc_sweep/run_01...run_06/
                                ├── sweep_results.csv
                                └── outputs/fc_sweep/best/
                                        │
                                        └── predict_fc.py
                                                ├── 从原始 validation 重新编码
                                                ├── 恢复最终 FC 权重
                                                ├── 验证指标与逐条预测
                                                └── 四类新评论预测
```

## 2. 各脚本用途

### `common.py`

服务于经典机器学习流程，提供数据检查、指标计算、预测保存、错误样本保存和随机种子等公共函数。

### `inspect_tokenizer.py`

检查 UER 模型配套 WordPiece tokenizer 的实际编码，包括：

- tokens
- token IDs
- `[CLS]`
- `[SEP]`
- `[PAD]`
- `attention_mask`
- `token_type_ids`

输出：

```text
outputs/inspect_tokenizer.txt
```

### `train_classical.py`

训练经典机器学习分类器。

正式实验：

```text
基础分类器：15 个
额外优化配置：10 个
最佳：NB_SVM_C4
Macro-F1：85.173%
```

结果位于：

```text
outputs/classical/
```

其中包含各方法评价指标、逐条预测、汇总结果、最佳模型和代表性误分类样本。

### `extract_features.py`

加载 UER `chinese_roberta_L-8_H-512`，冻结全部 RoBERTa 参数，使用最后一层 `[CLS]` 向量作为固定 512 维特征。

输出：

```text
outputs/fc_only/train_cls_features.npz
outputs/fc_only/validation_cls_features.npz
outputs/fc_only/feature_metadata.json
```

形状：

```text
train      (20000, 512)
validation (4000, 512)
```

`outputs/fc_only/` 在最终提交中只承担“固定特征文件及元数据”的角色，不再作为最终 FC 模型结果目录。

### `train_fc.py`

读取已经保存的固定特征，仅训练：

```text
Linear(512, 2)
```

可训练参数：

```text
1,026
```

每轮记录训练 loss 和验证集 Accuracy、Macro-Precision、Macro-Recall、Macro-F1，并以 Macro-F1 保存该组最优 epoch。

### `sweep_fc.py`

正式 FC 超参数对照脚本。

配置：

```text
learning_rate = {0.0005, 0.001, 0.002}
weight_decay  = {0, 0.01}
batch_size    = 128
epochs        = 20
seed          = 42
```

共 6 组。

结果汇总：

```text
outputs/fc_sweep/sweep_results.csv
outputs/fc_sweep/best_run.json
```

最终最佳：

```text
run_06_lr0p002_wd0p01_bs128_s42
learning_rate = 0.002
weight_decay = 0.01
best_epoch = 18
Macro-F1 = 85.525%
```

### `predict_fc.py`

最终恢复时使用：

```text
原始冻结 RoBERTa
+
outputs/fc_sweep/best/classifier_best.pt
```

从原始 `validation.csv` 重新进行 tokenizer 编码和 RoBERTa 特征提取，再接最终 FC 权重。

最终恢复指标：

```text
Accuracy         = 85.525%
Macro-Precision  = 85.527%
Macro-Recall     = 85.525%
Macro-F1         = 85.525%
```

混淆矩阵：

```text
[[1704, 296],
 [ 283,1717]]
```

输出：

```text
outputs/fc_sweep/best/reloaded_validation_metrics.json
outputs/fc_sweep/best/reloaded_validation_predictions.csv
outputs/fc_sweep/best/new_comment_demo.txt
```

## 3. 6 组参数对照

```text
lr=0.0005, wd=0     → Macro-F1 85.35%, best epoch 18
lr=0.0005, wd=0.01  → Macro-F1 85.37%, best epoch 18
lr=0.001,  wd=0     → Macro-F1 85.45%, best epoch 15
lr=0.001,  wd=0.01  → Macro-F1 85.45%, best epoch 14
lr=0.002,  wd=0     → Macro-F1 85.50%, best epoch 18
lr=0.002,  wd=0.01  → Macro-F1 85.52%, best epoch 18
```

因此报告及最终提交均以第 6 组作为正式 FC 结果。

## 4. 最终模型相关文件

最终模型目录：

```text
outputs/fc_sweep/best/
```

关键内容包括：

```text
classifier_best.pt
history.csv
training_config.json
best_metrics.json
validation_predictions.csv
reloaded_validation_metrics.json
reloaded_validation_predictions.csv
new_comment_demo.txt
selected_from.json
```

其中：

- `classifier_best.pt`：最终 FC 权重
- `history.csv`：最终配置 20 个 epoch 的训练记录
- `training_config.json`：最终配置
- `best_metrics.json`：训练阶段恢复最佳权重得到的验证指标
- `validation_predictions.csv`：基于保存特征的验证预测
- `reloaded_validation_metrics.json`：从原始文本重新编码后的恢复指标
- `reloaded_validation_predictions.csv`：从原始文本重新编码后的逐条预测
- `new_comment_demo.txt`：正面、负面、否定、转折四类新评论预测
- `selected_from.json`：说明该目录复制自 sweep 的哪一组最优实验

## 5. 最终新评论预测

```text
正面：
质量很好，物流很快，下次还会购买。
→ 正面，正面概率 0.972928

负面：
质量非常差，客服态度也不好。
→ 负面，正面概率 0.003753

否定：
这款商品不是不好，只是价格有点高。
→ 负面，正面概率 0.366510

转折：
虽然包装很好，但是商品有明显缺陷。
→ 正面，正面概率 0.673484
```

这些结果用于说明冻结上下文表示 + 线性分类头对复杂否定和转折关系仍存在局限。

## 6. 两条路线最终比较

```text
NB_SVM_C4
Accuracy         85.175%
Macro-Precision  85.196%
Macro-Recall     85.175%
Macro-F1         85.173%

Frozen RoBERTa + FC
Accuracy         85.525%
Macro-Precision  85.527%
Macro-Recall     85.525%
Macro-F1         85.525%
```

预训练路线 Macro-F1 提高约 0.35 个百分点。

## 7. 提交检查

提交前只需确保以下几点：

1. 最终报告中的 FC 指标是 85.525%，最佳 epoch 为 18。
2. 最终 FC 权重引用 `outputs/fc_sweep/best/classifier_best.pt`。
3. `outputs/fc_only/` 只作为固定特征存放位置，不再混入旧的 85.45% FC 训练结果。
4. 6 组参数实验保留完整记录，`sweep_results.csv` 与报告表格一致。
5. 模型恢复指标与报告中的最终指标一致。
6. 另附实验报告 PDF。
