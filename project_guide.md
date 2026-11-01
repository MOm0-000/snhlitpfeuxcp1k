# FC_finetune 实验项目脚本与结果阅读指南

本文档只说明 `FC_finetune` 分支当前有效的实验链路。

旧的 A/B/C/D、FGM、R-Drop、人工标签修正和全参数微调实验不属于本分支正式结果。

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
                ├── 冻结 UER RoBERTa（0 个可训练编码器参数）
                │
                ├── train_cls_features.npz
                │
                └── validation_cls_features.npz
                        │
                        └── train_fc.py
                                │
                                ├── 只训练 Linear(512,2)
                                ├── 1026 个可训练参数
                                ├── 每轮 validation
                                └── classifier_best.pt
                                        │
                                        └── predict_fc.py
                                                ├── 恢复验证
                                                ├── 逐条 validation 预测
                                                └── 四类新评论预测
```

## 2. 数据

```text
train.csv       20,000 条，0/1 各 10,000
validation.csv   4,000 条，0/1 各 2,000
```

两条路线使用同一数据划分。

## 3. 文件用途

### `common.py`

只服务于经典机器学习脚本，提供数据检查、指标计算、预测保存、错误样本保存和随机种子等公共函数。

### `inspect_tokenizer.py`

检查课程指定 UER 模型的 WordPiece 编码：

```bash
python inspect_tokenizer.py --model pretrained
```

结果：

```text
outputs/inspect_tokenizer.txt
```

### `train_classical.py`

训练经典分类器：

```bash
python train_classical.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --output-dir outputs/classical
```

正式结果：

```text
15 个基础分类器
10 个额外优化配置
最佳 NB_SVM_C4
Macro-F1 = 0.8517277499
```

### `extract_features.py`

加载本地 `pretrained/` 中的 UER RoBERTa，显式冻结全部编码器参数，从最后一层取 `[CLS]` 作为 512 维固定特征。

```bash
python extract_features.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --model-dir pretrained \
  --output-dir outputs/fc_only \
  --max-length 192 \
  --batch-size 64 \
  --device auto
```

结果：

```text
train_cls_features.npz       (20000, 512)
validation_cls_features.npz  (4000, 512)
feature_metadata.json
```

`feature_metadata.json` 记录编码器可训练参数为 0。

### `train_fc.py`

不加载 Transformer，仅读取前一步的固定特征，训练：

```text
Linear(512, 2)
```

共：

```text
1026 trainable parameters
```

命令：

```bash
python train_fc.py \
  --train-features outputs/fc_only/train_cls_features.npz \
  --validation-features outputs/fc_only/validation_cls_features.npz \
  --output-dir outputs/fc_only \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3 \
  --weight-decay 0.01 \
  --device auto
```

每轮计算 validation 指标，并按 Macro-F1 保存最佳：

```text
classifier_best.pt
```

当前最佳：

```text
epoch            14
Accuracy         0.8545
Macro-F1         0.8544990906
Macro-Precision  0.8545088627
Macro-Recall     0.8545
```

### `predict_fc.py`

重新加载原始预训练编码器和 `classifier_best.pt`，重新编码 validation 后核对指标，并测试正面、负面、否定、转折评论。

```bash
python predict_fc.py \
  --validation data/validation.csv \
  --model-dir pretrained \
  --classifier outputs/fc_only/classifier_best.pt \
  --output-dir outputs/fc_only \
  --max-length 192 \
  --batch-size 64 \
  --device auto
```

恢复指标与训练阶段最佳指标一致。

## 4. `outputs/fc_only/` 文件解释

```text
train_cls_features.npz
```

训练集固定 `[CLS]` 特征、标签和 ID。属于需要提交的特征文件。

```text
validation_cls_features.npz
```

验证集固定 `[CLS]` 特征、标签和 ID。属于需要提交的特征文件。

```text
feature_metadata.json
```

记录模型维度、层数、序列长度、数据哈希、样本数量及编码器可训练参数数目。

```text
history.csv
```

20 个 epoch 的训练 loss 与 validation Accuracy / Macro-F1 / Macro-P / Macro-R。

```text
training_config.json
```

记录 FC-only 训练配置；可训练参数为 1026，编码器训练参数为 0。

```text
classifier_best.pt
```

Macro-F1 最高 epoch 的全连接层权重。最终提交重点文件。

```text
best_metrics.json
```

训练结束后重新恢复 `classifier_best.pt` 得到的最佳验证指标；`verified_reload=true`。

```text
validation_predictions.csv
```

`train_fc.py` 恢复最佳 FC 后对已保存 validation 特征的逐条预测。

```text
reloaded_validation_metrics.json
```

`predict_fc.py` 从原始文本重新经冻结 RoBERTa 编码，再接恢复 FC 权重得到的完整验证指标。

```text
reloaded_validation_predictions.csv
```

`predict_fc.py` 对 validation 的逐条文本、标签、预测、正面概率和正确性记录。

```text
new_comment_demo.txt
```

正面、负面、否定、转折四种自拟评论预测。

## 5. 结果一致性

训练阶段最佳模型：

```text
Macro-F1 = 0.8544990906193164
```

重新加载原始编码器 + FC 权重：

```text
Macro-F1 = 0.8544990906193164
```

说明恢复结果与训练阶段记录一致。

## 6. 当前结果分析注意事项

经典最佳 `NB_SVM_C4`：

```text
Macro-F1 = 85.17%
```

FC-only：

```text
Macro-F1 = 85.45%
```

提升约：

```text
+0.28 个百分点
```

因此报告不应写“预训练模型显著碾压经典方法”。

新评论中：

- 简单正面、负面判断正确；
- “不是不好”被预测为负面；
- “虽然……但是商品有明显缺陷”被预测为正面。

这些结果适合用于讨论冻结特征 + 线性分类头对复杂否定和转折关系的局限。

## 7. 推荐提交文件

强烈建议保留：

```text
data/
pretrained/ 中 tokenizer/config 文件
inspect_tokenizer.py
train_classical.py
common.py
extract_features.py
train_fc.py
predict_fc.py
requirements.txt
outputs/inspect_tokenizer.txt
outputs/classical/classical_results.csv
outputs/classical/summary.json
outputs/classical/best_classical.joblib
outputs/classical/best_errors.csv
outputs/fc_only/train_cls_features.npz
outputs/fc_only/validation_cls_features.npz
outputs/fc_only/feature_metadata.json
outputs/fc_only/history.csv
outputs/fc_only/training_config.json
outputs/fc_only/classifier_best.pt
outputs/fc_only/best_metrics.json
outputs/fc_only/reloaded_validation_metrics.json
outputs/fc_only/reloaded_validation_predictions.csv
outputs/fc_only/new_comment_demo.txt
```

原始 `pretrained/pytorch_model.bin` 可保存在本地，不建议提交 GitHub；它属于公开预训练模型原始权重，不是本实验训练得到的权重。
