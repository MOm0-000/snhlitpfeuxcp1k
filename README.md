# 中文商品评论情感分析实验

本仓库是哈尔滨工业大学《自然语言处理 C》实验一“基于分类模型的情感分析”的 Python 实现。实验使用课程提供的 MARC 中文商品评论数据，完成经典机器学习分类与 UER 中文 RoBERTa 微调，并保存训练、选模、恢复验证、新评论预测和误差分析结果。

当前 `main` 分支以以下流程为唯一主线：

1. **经典机器学习**：字符级 TF-IDF、LSA、课程要求的 15 种基础分类器，以及线性模型调参、字词特征融合和 NB-SVM 式改进；
2. **预训练模型微调**：UER `chinese_roberta_L-8_H-512`，比较 A/B/C/D 四种训练配置，并补充原始/人工核查标签对照；
3. **最终对比分析**：最佳经典模型 `NB_SVM_C4` 与最终 Transformer `C/cleaned` 在同一原始验证集上比较，并分析否定、转折、模型分歧和代表性错误。

> 课程未提供独立测试集。仓库中的模型选择、对比与结果均为 `validation.csv` 上的验证结果，不将其称为独立测试结果。

## 1. 项目结构

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
├── label_curation.py
├── train_A.py
├── train_B.py
├── train_C.py
├── train_D.py
├── analyze_errors.py
├── requirements.txt
├── outputs/
│   ├── inspect_tokenizer.txt
│   ├── new_comment_demo.txt
│   ├── final_transformer_validation.csv
│   ├── classical/
│   ├── label_audit/
│   ├── abcd_dual/
│   └── analysis/
├── README.md
└── 实验项目脚本与结果阅读指南.md
```

当前分支仅保留最终实验主线；历史 sweep、ensemble 与旧 FGM 分析已从工作树移除，避免与当前结果混淆。

## 2. 数据与评价口径

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

- `0`：负面，来自 1–2 星评论；
- `1`：正面，来自 4–5 星评论；
- 3 星评论不使用。

经典路线和预训练路线使用同一训练/验证划分。主要指标为 Accuracy、Macro-F1、Macro-Precision、Macro-Recall；预训练模型以原始验证标签上的 **Macro-F1** 作为主选模指标。

## 3. 环境

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 不安装 PyTorch，以免覆盖本机已有的 CUDA/PyTorch 环境。请按本机设备单独安装适配版本。

检查环境：

```bash
python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

## 4. 预训练权重说明

`.gitignore` 排除：

```text
*.pt
*.bin
*.safetensors
```

因此 GitHub 中：

- `pretrained/` 只保留配置和 tokenizer 文件，不包含 UER 模型权重；
- `outputs/abcd_dual/**/best_model/` 只保留配置、tokenizer、`variant.json` 等元数据，不包含训练后的大权重文件。

首次在新环境复现实验时，先准备本地权重：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="uer/chinese_roberta_L-8_H-512",
    local_dir="pretrained",
)
PY
```

然后使用 `--overwrite` 重新生成完整 checkpoint：

```bash
python train_A.py --overwrite
python train_B.py --overwrite
python train_C.py --overwrite
python train_D.py --overwrite
```

## 5. Tokenizer 编码实验

```bash
python inspect_tokenizer.py --model pretrained
```

实际运行结果保存在：

```text
outputs/inspect_tokenizer.txt
```

其中记录了 WordPiece、token ID、`[CLS]`、`[SEP]`、`[PAD]`、`attention_mask`、`token_type_ids` 等。词表大小为 21,128；示例 `keyboard` 被切分为 `key` 与 `##board`。

## 6. 经典机器学习

完整运行：

```bash
python train_classical.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --output-dir outputs/classical
```

课程要求的 15 种基础分类器全部完成：

```text
MultinomialNB
ComplementNB
BernoulliNB
LogisticRegression
LinearSVC
RidgeClassifier
Perceptron
NearestCentroid
KNeighborsClassifier
SVC_RBF
DecisionTreeClassifier
RandomForestClassifier
AdaBoostClassifier
GradientBoostingClassifier
MLPClassifier
```

基础字符 TF-IDF：

```text
analyzer      = char
ngram_range   = (2, 4)
min_df        = 2
max_features  = 180000
sublinear_tf  = True
norm          = l2
```

距离、树、核方法和 MLP 等稠密输入模型使用训练集拟合的 96 维 TruncatedSVD。额外实验包括 LinearSVC/LogisticRegression 调参、字符+词级特征融合和 NB-SVM 式对数计数比加权。

当前完整运行结果：

```text
methods_required        = 15
methods_completed       = 15
extra_optimized_methods = 10
best_method             = NB_SVM_C4
best_macro_f1           = 0.8517277499
```

即最佳经典方法验证集 Macro-F1 约 **85.17%**。

输出：

```text
outputs/classical/
├── classical_results.csv
├── best_classical.joblib
├── best_errors.csv
├── summary.json
├── metrics_*.json
└── pred_*.csv
```

`--fast` 仅用于调试，不能把其结果作为正式实验结果。

## 7. 人工标签核查

生成候选：

```bash
python label_curation.py prepare \
  --train data/train.csv \
  --validation data/validation.csv \
  --out-dir outputs/label_audit
```

应用人工审核：

```bash
python label_curation.py apply \
  --train data/train.csv \
  --validation data/validation.csv \
  --out-dir outputs/label_audit
```

流程是“模型筛候选 → 人工阅读 → `keep/flip/unsure` → 生成新 CSV”，不会自动把模型预测当作真值，也不会覆盖课程原始数据。

主要输出：

```text
outputs/label_audit/
├── review_train.csv
├── review_validation.csv
├── train_semantic.csv
├── validation_semantic_partial.csv
└── changes.csv
```

`validation_semantic_partial.csv` 只对候选样本进行了部分人工语义修正，不是完整重新标注的金标准，因此只作为补充实验评价口径。

## 8. 预训练主实验 A/B/C/D

模型：

```text
uer/chinese_roberta_L-8_H-512
```

配置：

| 方案 | FGM | CLS/Mean Pooling | R-Drop |
|---|---:|---:|---:|
| A | ✓ |  |  |
| B | ✓ | ✓ |  |
| C | ✓ |  | ✓ |
| D | ✓ | ✓ | ✓ |

统一默认训练参数：

```text
epochs                 5
batch_size             8
gradient accumulation  4
effective batch        32
eval_batch_size        32
max_length             192
base LR                2e-5
FGM epsilon            0.5
label smoothing        0.04
layerwise LR decay     0.92
head LR multiplier     5.0
weight decay           0.01
warmup ratio           0.10
dropout                0.10
R-Drop alpha           C/D = 0.5
max grad norm          1.0
seed                   42
patience               0
AMP                     auto
```

默认每个方案分别训练：

- `data/train.csv`：`original`
- `outputs/label_audit/train_semantic.csv`：`cleaned`

每个 epoch 对同一 4,000 条 validation 预测一次，再分别按：

- 原始验证标签；
- 部分修正验证标签；

计算指标并独立选择 Macro-F1 最佳 epoch。

运行：

```bash
python train_A.py
python train_B.py
python train_C.py
python train_D.py
```

汇总结果保存在：

```text
outputs/abcd_dual/comparison.csv
```

按**原始验证标签选模**的当前结果：

| 模型 | 训练标签 | 原始验证 Macro-F1 | 部分修正验证 Macro-F1 | 最佳 epoch |
|---|---|---:|---:|---:|
| A | original | 88.32% | 89.10% | 5 |
| A | cleaned  | 88.40% | 89.17% | 5 |
| B | original | 88.22% | 89.05% | 4 |
| B | cleaned  | 88.12% | 88.95% | 4 |
| C | original | 88.25% | 89.02% | 5 |
| C | cleaned  | **88.40%** | **89.22%** | 5 |
| D | original | 88.22% | 89.05% | 4 |
| D | cleaned  | 88.17% | 89.00% | 4 |

A/cleaned 和 C/cleaned 四舍五入后都为 88.40%，但未舍入值中 C/cleaned 略高，因此最终 Transformer 采用：

```text
C / cleaned / best_by_original
```

当前恢复验证结果：

```text
Accuracy        0.884000
Macro-F1        0.883999884
Macro-Precision 0.884001536
Macro-Recall    0.884000
verified_reload True
```

## 9. 新评论预测

最终 C/cleaned checkpoint 已实际运行正面、负面、否定、转折四类新评论，结果保存在：

```text
outputs/new_comment_demo.txt
```

当前结果：

```text
正面 (正面概率=0.972)：质量很好，物流很快，下次还会购买。
负面 (正面概率=0.016)：质量非常差，客服态度也不好。
正面 (正面概率=0.601)：这款商品不是不好，只是价格有点高。
负面 (正面概率=0.283)：虽然包装很好，但是商品有明显缺陷。
```

本地存在完整 checkpoint 时可重新运行：

```bash
python train_C.py --demo
```

## 10. 最终验证预测与误差分析

最终 Transformer 对完整原始 validation 的逐条预测保存在：

```text
outputs/final_transformer_validation.csv
```

重新生成该文件的完整恢复/批量推理代码见《实验项目脚本与结果阅读指南》第 15 节。

正式误差分析运行方式：

```bash
python analyze_errors.py \
  --validation data/validation.csv \
  --classical outputs/classical/pred_NB_SVM_C4.csv \
  --transformer outputs/final_transformer_validation.csv \
  --output-dir outputs/analysis
```

正式输出：

```text
outputs/analysis/
├── analysis_summary.json
├── full_comparison.csv
├── disagreements.csv
├── transformer_better_examples.csv
├── classical_better_examples.csv
└── both_wrong_examples.csv
```

整体结果：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| NB-SVM_C4 | 85.18% | 85.17% |
| C/cleaned | 88.40% | 88.40% |

两模型：

```text
预测不同           355
C/cleaned 正确     242
NB-SVM 正确        113
双方都错误         351
```

关键词规则筛出的子集：

| 子集 | NB-SVM Macro-F1 | C/cleaned Macro-F1 |
|---|---:|---:|
| 含否定关键词 | 84.81% | 88.01% |
| 含转折关键词 | 82.22% | 86.65% |
| 同时含两者 | 82.62% | 86.76% |
| 两者都不含 | 84.48% | 87.60% |

这些分组由字符串规则粗略识别，具体评论仍需人工阅读后解释。

## 11. 推荐结果口径

当前项目的正式主结果统一为：

```text
最佳经典模型：
NB_SVM_C4
Macro-F1 = 85.17%

最终 Transformer：
C / cleaned / best_by_original
Macro-F1 = 88.40%

提升：
约 +3.23 个百分点
```

部分修正验证标签上的 89.22% 只用于标签质量补充分析，不与原始 validation 主指标直接混用。

## 12. 复现顺序

从新环境完整复现时建议：

```text
1. 安装 requirements.txt，并准备 PyTorch
2. 下载 UER 权重到 pretrained/
3. python inspect_tokenizer.py --model pretrained
4. python train_classical.py
5. python label_curation.py prepare
6. 人工填写 review_train.csv / review_validation.csv
7. python label_curation.py apply
8. python train_A.py --overwrite
9. python train_B.py --overwrite
10. python train_C.py --overwrite
11. python train_D.py --overwrite
12. 生成 final_transformer_validation.csv
13. python analyze_errors.py ...
14. python train_C.py --demo
```

仓库已经提交了对应实验 CSV/JSON/文本记录；由于 GitHub 不保存大模型权重，步骤 8–11 在新环境需要重新训练。

更详细的脚本、结果文件和报告取数说明见：

**[实验项目脚本与结果阅读指南.md](./实验项目脚本与结果阅读指南.md)**
