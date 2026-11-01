# 中文商品评论情感分析实验（FC_finetune）

本分支对应哈尔滨工业大学《自然语言处理 C》实验一“基于分类模型的情感分析”的最终实现，包含经典机器学习与冻结预训练表示 + FC 分类头两条路线。

> 课程只提供 `train.csv` 和 `validation.csv`。本项目中的参数选择、模型比较和最终指标均在 `validation.csv` 上完成，不将其表述为独立测试集成绩。

## 1. 最终实验路线

### 经典机器学习

- 字符级 TF-IDF
- 词级 TF-IDF
- LSA
- 15 种基础分类器
- 10 组额外参数/特征优化
- 最终最佳方法：`NB_SVM_C4`

### 预训练模型

- UER `chinese_roberta_L-8_H-512`
- 使用配套 WordPiece tokenizer
- RoBERTa 编码器全部冻结
- 提取最后一层 `[CLS]` 的 512 维固定特征
- 仅训练 `Linear(512, 2)` 分类头，共 1,026 个可训练参数
- 比较 3 个学习率 × 2 个 weight decay，共 6 组配置
- 最终最佳配置：`learning_rate=0.002`、`weight_decay=0.01`
- 最佳 epoch：18

## 2. 项目结构

```text
.
├── data/
│   ├── train.csv
│   └── validation.csv
├── pretrained/
│   ├── config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── common.py
├── inspect_tokenizer.py
├── train_classical.py
├── extract_features.py
├── train_fc.py
├── sweep_fc.py
├── predict_fc.py
├── requirements.txt
├── outputs/
│   ├── inspect_tokenizer.txt
│   ├── classical/
│   ├── fc_only/
│   │   ├── train_cls_features.npz
│   │   ├── validation_cls_features.npz
│   │   └── feature_metadata.json
│   └── fc_sweep/
│       ├── sweep_results.csv
│       ├── best_run.json
│       ├── run_01_lr0p0005_wd0_bs128_s42/
│       ├── run_02_lr0p0005_wd0p01_bs128_s42/
│       ├── run_03_lr0p001_wd0_bs128_s42/
│       ├── run_04_lr0p001_wd0p01_bs128_s42/
│       ├── run_05_lr0p002_wd0_bs128_s42/
│       ├── run_06_lr0p002_wd0p01_bs128_s42/
│       └── best/
├── README.md
└── project_guide.md
```

提交时另附最终实验报告 PDF。

## 3. 数据

课程数据：

```text
data/train.csv         20,000 条，正负各 10,000
data/validation.csv     4,000 条，正负各 2,000
```

字段：

```text
id,text,label
```

标签：

- `0`：负面
- `1`：正面
- 3 星样本不参与实验

两条路线使用完全相同的数据划分。

## 4. 环境

```bash
python -m pip install -r requirements.txt
```

PyTorch 可根据本机 CUDA 环境单独安装。

检查：

```bash
python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

## 5. 预训练模型与 tokenizer

课程参考模型：

```text
uer/chinese_roberta_L-8_H-512
```

本地运行时需要原始预训练权重 `pytorch_model.bin`，但该公开预训练大文件不提交到 GitHub。仓库保留 tokenizer 和模型配置文件。

当前模型信息：

```text
hidden_size                  = 512
num_hidden_layers            = 8
tokenizer_vocab_size         = 21128
encoder_total_parameters     = 36,300,800
encoder_trainable_parameters = 0
max_length                   = 192
```

Tokenizer 编码检查：

```bash
python inspect_tokenizer.py --model pretrained
```

输出：

```text
outputs/inspect_tokenizer.txt
```

其中记录 WordPiece、token ID、`[CLS]`、`[SEP]`、`[PAD]`、`attention_mask` 和 `token_type_ids` 示例。

## 6. 经典机器学习

运行：

```bash
python train_classical.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --output-dir outputs/classical
```

正式结果：

```text
基础分类器：15/15
额外优化配置：10
最佳方法：NB_SVM_C4
Accuracy：85.175%
Macro-F1：85.173%
```

完整参数、指标和逐条预测结果保存在：

```text
outputs/classical/
```

其中 `classical_results.csv` 汇总各方法结果，`metrics_*.json` 保存各方法评价指标，`pred_*.csv` 保存对应预测结果。

## 7. 冻结 RoBERTa 并提取固定特征

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

模型加载后：

- RoBERTa 编码器参数全部 `requires_grad=False`
- 编码器以 `eval` 模式提取表示
- 使用最后一层 `[CLS]` 向量作为固定 512 维特征

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

这些文件是预训练路线的关键特征文件。

## 8. FC-only 基础训练脚本

`train_fc.py` 仅读取固定特征，不更新 Transformer，仅训练：

```text
Linear(512, 2)
```

可训练参数：

```text
512 × 2 + 2 = 1,026
```

每轮在验证集计算：

- Accuracy
- Macro-Precision
- Macro-Recall
- Macro-F1

并按验证集 Macro-F1 保存该配置的最佳 epoch。

## 9. 6 组 FC 超参数对照

运行：

```bash
python sweep_fc.py \
  --train-features outputs/fc_only/train_cls_features.npz \
  --validation-features outputs/fc_only/validation_cls_features.npz \
  --lrs 0.0005,0.001,0.002 \
  --weight-decays 0,0.01 \
  --batch-sizes 128 \
  --epochs 20 \
  --seeds 42 \
  --device auto \
  --copy-best
```

共 6 组：

| 学习率 | weight decay | 最佳 epoch | Macro-F1 |
|---:|---:|---:|---:|
| 0.0005 | 0 | 18 | 85.35% |
| 0.0005 | 0.01 | 18 | 85.37% |
| 0.001 | 0 | 15 | 85.45% |
| 0.001 | 0.01 | 14 | 85.45% |
| 0.002 | 0 | 18 | 85.50% |
| **0.002** | **0.01** | **18** | **85.52%** |

汇总结果：

```text
outputs/fc_sweep/sweep_results.csv
outputs/fc_sweep/best_run.json
```

最终采用：

```text
run_06_lr0p002_wd0p01_bs128_s42
```

正式最佳模型文件统一放在：

```text
outputs/fc_sweep/best/
```

## 10. 最终 FC 模型结果

最终配置：

```text
epochs          = 20
batch_size      = 128
learning_rate   = 0.002
weight_decay    = 0.01
seed            = 42
best_epoch      = 18
trainable params= 1,026
encoder params updated during training = 0
```

验证集最终指标：

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

最终权重：

```text
outputs/fc_sweep/best/classifier_best.pt
```

训练记录和配置：

```text
outputs/fc_sweep/best/history.csv
outputs/fc_sweep/best/training_config.json
outputs/fc_sweep/best/best_metrics.json
```

## 11. 模型恢复与预测

使用最终最佳权重重新从原始评论编码并验证：

```bash
python predict_fc.py \
  --validation data/validation.csv \
  --model-dir pretrained \
  --classifier outputs/fc_sweep/best/classifier_best.pt \
  --output-dir outputs/fc_sweep/best \
  --max-length 192 \
  --batch-size 64 \
  --device auto
```

恢复后的指标：

```text
Accuracy         = 85.525%
Macro-Precision  = 85.527%
Macro-Recall     = 85.525%
Macro-F1         = 85.525%
```

混淆矩阵仍为：

```text
[[1704, 296],
 [ 283,1717]]
```

说明保存和恢复流程一致。

对应文件：

```text
outputs/fc_sweep/best/reloaded_validation_metrics.json
outputs/fc_sweep/best/reloaded_validation_predictions.csv
outputs/fc_sweep/best/new_comment_demo.txt
```

四类自拟评论的最终预测：

```text
正面：质量很好，物流很快，下次还会购买。
      → 正面，正面概率 0.972928

负面：质量非常差，客服态度也不好。
      → 负面，正面概率 0.003753

否定：这款商品不是不好，只是价格有点高。
      → 负面，正面概率 0.366510

转折：虽然包装很好，但是商品有明显缺陷。
      → 正面，正面概率 0.673484
```

## 12. 最终结果对照

| 方法 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| NB-SVM_C4 | 85.175% | 85.196% | 85.175% | 85.173% |
| Frozen RoBERTa + FC | **85.525%** | **85.527%** | **85.525%** | **85.525%** |

Frozen RoBERTa + FC 的 Macro-F1 比最佳经典模型提高约 **0.35 个百分点**。

预训练表示在本实验中略优，但优势不大。结合误分类和自拟评论可见，冻结编码器、只训练线性分类头的方法对一般正负评论较稳定，但对复杂否定、转折以及评价对象变化仍存在局限。

## 13. 提交材料对应关系

老师要求的材料在本项目中分别由以下内容支撑：

- 数据与特征处理：数据读取/检查代码、`extract_features.py`、固定 `[CLS]` 特征文件及元数据
- 经典机器学习分类：`train_classical.py`、各模型指标、预测结果和汇总对照
- 预训练模型微调：`train_fc.py`、`sweep_fc.py`、6 组训练记录、最终 FC 权重、训练配置
- 模型恢复与预测：`predict_fc.py` 及最终恢复指标、逐条预测、新评论预测
- 实验评测：经典模型结果、6 组 FC 对照、最终两条路线指标对比
- 实验报告：单独提交最终实验报告 PDF

实验报告中的参数、指标与提交文件应保持一致。
