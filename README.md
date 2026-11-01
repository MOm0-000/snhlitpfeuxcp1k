# 中文商品评论情感分析实验（FC_finetune）

本分支对应哈尔滨工业大学《自然语言处理 C》实验一“基于分类模型的情感分析”的最终实现，包含两条路线：

1. **经典机器学习**：字符/词级 TF-IDF、LSA、课程要求的 15 种基础分类器，以及额外的线性模型、字词融合和 NB-SVM 式优化。
2. **预训练模型路线**：使用 UER `chinese_roberta_L-8_H-512` 作为**冻结特征提取器**，提取最后一层 `[CLS]` 的 512 维特征；训练阶段只更新二分类全连接层 `Linear(512, 2)`。

本分支不再包含旧的 A/B/C/D、FGM、R-Drop、标签清洗或全参数微调路线。

> 课程只提供 `train.csv` 和 `validation.csv`。本仓库中的模型选择和最终指标均在 `validation.csv` 上计算，不将其表述为独立测试集结果。

## 1. 项目结构

```text
.
├── data/
│   ├── train.csv
│   └── validation.csv
├── pretrained/
│   ├── config.json
│   ├── pytorch_model.bin          # 本地准备，不提交 GitHub
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── common.py
├── inspect_tokenizer.py
├── train_classical.py
├── extract_features.py
├── train_fc.py
├── predict_fc.py
├── requirements.txt
├── outputs/
│   ├── inspect_tokenizer.txt
│   ├── classical/
│   └── fc_only/
├── README.md
└── project_guide.md
```

## 2. 数据

课程数据：

```text
data/train.csv        20,000 条，0/1 各 10,000
data/validation.csv    4,000 条，0/1 各 2,000
```

字段：

```text
id,text,label
```

标签：

- `0`：负面；
- `1`：正面；
- 3 星样本不使用。

## 3. 环境

```bash
python -m pip install -r requirements.txt
```

PyTorch 不写入 `requirements.txt`，避免覆盖本机已配置的 CUDA/PyTorch 环境。

检查：

```bash
python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

## 4. 预训练模型

课程参考模型为 UER 中文 RoBERTa-Medium：

```text
uer/chinese_roberta_L-8_H-512
```

本地 `pretrained/` 需要包含原始预训练权重 `pytorch_model.bin`。GitHub 中不提交该大文件。

当前特征提取记录：

```text
hidden_size                 = 512
num_hidden_layers           = 8
tokenizer_vocab_size        = 21128
encoder_total_parameters    = 36,300,800
encoder_trainable_parameters= 0
max_length                  = 192
```

## 5. Tokenizer 编码实验

运行：

```bash
python inspect_tokenizer.py --model pretrained
```

已保存示例：

```text
outputs/inspect_tokenizer.txt
```

其中包括 WordPiece、token ID、`[CLS]`、`[SEP]`、`[PAD]`、`attention_mask` 与 `token_type_ids`。

## 6. 经典机器学习

运行：

```bash
python train_classical.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --output-dir outputs/classical
```

当前正式结果：

```text
课程要求基础分类器：15/15
额外优化配置：10
最佳方法：NB_SVM_C4
验证 Macro-F1：0.8517277499
```

主要输出：

```text
outputs/classical/
├── classical_results.csv
├── best_classical.joblib
├── best_errors.csv
├── summary.json
├── metrics_*.json
└── pred_*.csv
```

## 7. 冻结 RoBERTa 并生成特征文件

运行：

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

模型加载时明确：

- 不加载下游分类头；
- 不使用随机 pooler；
- RoBERTa 编码器全部参数 `requires_grad=False`；
- 使用最后一层 `[CLS]` 向量作为固定 512 维特征。

生成：

```text
outputs/fc_only/train_cls_features.npz
outputs/fc_only/validation_cls_features.npz
outputs/fc_only/feature_metadata.json
```

实际形状：

```text
train       = (20000, 512)
validation  = (4000, 512)
```

这两个 `.npz` 为预训练路线形成的特征文件。

## 8. 只训练全连接层

运行：

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

训练阶段完全不加载 Transformer，只读取固定特征。唯一可训练模块：

```text
Linear(512, 2)
```

可训练参数：

```text
512 × 2 + 2 = 1026
```

实际训练配置：

```text
epochs        = 20
batch_size    = 128
learning_rate = 0.001
weight_decay  = 0.01
seed          = 42
```

每轮在验证集计算 Accuracy、Macro-F1、Macro-Precision、Macro-Recall，并以 Macro-F1 选择最佳 epoch。

当前结果：

```text
best_epoch       = 14
Accuracy         = 0.8545
Macro-F1         = 0.8544990906
Macro-Precision  = 0.8545088627
Macro-Recall     = 0.8545
```

混淆矩阵：

```text
[[1714, 286],
 [ 296, 1704]]
```

主要输出：

```text
outputs/fc_only/
├── history.csv
├── training_config.json
├── classifier_best.pt
├── best_metrics.json
└── validation_predictions.csv
```

其中 `classifier_best.pt` 是需要提交的**微调后全连接层权重文件**。

## 9. 模型恢复与新评论预测

运行：

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

脚本重新加载：

```text
原始冻结 RoBERTa + classifier_best.pt
```

恢复后的验证指标与训练阶段保存的最佳指标完全一致：

```text
Accuracy         = 0.8545
Macro-F1         = 0.8544990906
Macro-Precision  = 0.8545088627
Macro-Recall     = 0.8545
```

输出：

```text
outputs/fc_only/reloaded_validation_metrics.json
outputs/fc_only/reloaded_validation_predictions.csv
outputs/fc_only/new_comment_demo.txt
```

当前四类自拟评论结果：

```text
正面：预测正面，正面概率 0.971894
负面：预测负面，正面概率 0.004803
否定：“这款商品不是不好，只是价格有点高。” → 预测负面，正面概率 0.342949
转折：“虽然包装很好，但是商品有明显缺陷。” → 预测正面，正面概率 0.695706
```

后两条体现了仅使用冻结 `[CLS]` 特征和线性分类头对复杂否定、转折表达仍存在局限。

## 10. 最终结果口径

同一原始验证集：

| 方法 | Macro-F1 |
|---|---:|
| 最佳经典模型 `NB_SVM_C4` | 85.17% |
| 冻结 RoBERTa + FC-only | 85.45% |

FC-only 相比最佳经典方法提高约 **0.28 个百分点**。

这里不能声称预训练模型具有压倒性优势；当前结果更适合分析“冻结上下文表示 + 极少量任务参数”的参数效率，以及其在复杂语义上的局限。

## 11. 提交建议

建议最终提交至少包括：

```text
代码：
  inspect_tokenizer.py
  train_classical.py
  extract_features.py
  train_fc.py
  predict_fc.py
  common.py

特征：
  outputs/fc_only/train_cls_features.npz
  outputs/fc_only/validation_cls_features.npz
  outputs/fc_only/feature_metadata.json

FC 微调权重：
  outputs/fc_only/classifier_best.pt

运行结果：
  outputs/fc_only/history.csv
  outputs/fc_only/best_metrics.json
  outputs/fc_only/reloaded_validation_metrics.json
  outputs/fc_only/reloaded_validation_predictions.csv
  outputs/fc_only/new_comment_demo.txt

经典路线：
  outputs/classical/classical_results.csv
  outputs/classical/summary.json
  outputs/classical/best_classical.joblib
  outputs/classical/best_errors.csv
```

实验报告中的参数、指标与文件内容必须保持一致。
