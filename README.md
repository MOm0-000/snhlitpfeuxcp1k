# 中文商品评论情感分析实验

本仓库是哈尔滨工业大学《自然语言处理》实验一的 Python 实现，使用课程提供的 MARC 中文商品评论数据完成二分类情感分析。

当前代码包含两条实验路线：

1. **经典机器学习**：字符级 TF-IDF、LSA、15 种基础分类器，以及线性模型调参、字词特征融合和 NB-SVM 式改进。
2. **预训练模型微调**：基于 UER `chinese_roberta_L-8_H-512`，比较 A/B/C/D 四种配置，并同时研究原始评分标签与部分人工修正标签。

> **重要：当前主线已经不是旧 README 中的 `sweep_transformer.py + train_transformer.py + predict_sentiment.py`。**
> 最新实验入口是根目录下的 `train_A.py`、`train_B.py`、`train_C.py`、`train_D.py`。旧版 Transformer 训练和预测脚本已移入 `bin/`，仅用于保留旧实验流程。

## 1. 数据与评价口径

课程数据位于：

```text
data/
├── train.csv        # 20,000 条，0/1 各 10,000
└── validation.csv   # 4,000 条，0/1 各 2,000
```

CSV 字段为：

```text
id,text,label
```

标签定义：`0=负面`，`1=正面`。

课程没有提供独立测试集，因此仓库中的模型比较均基于 `validation.csv`。验证集同时参与调参和选模，**不能把验证集结果称为独立测试结果**。

主要评价指标为 Accuracy、Macro-F1、Macro-Precision 和 Macro-Recall，其中预训练模型以 **Macro-F1** 作为主要选模指标。

## 2. 当前项目结构

```text
.
├── data/
│   ├── train.csv
│   └── validation.csv
├── pretrained/                 # 配置与 tokenizer；GitHub 不含模型权重
├── train_classical.py          # 15 种经典分类器 + 额外优化
├── inspect_tokenizer.py        # WordPiece / token id / mask 示例
├── label_curation.py           # 人工标签核查与修正
├── train_A.py                  # A：FGM
├── train_B.py                  # B：FGM + CLS/Mean Pooling
├── train_C.py                  # C：FGM + R-Drop
├── train_D.py                  # D：FGM + CLS/Mean Pooling + R-Drop
├── common.py                   # 经典方法公共函数
├── outputs/                    # 已保存指标、CSV、配置和部分模型元数据
├── bin/                        # 旧版 Transformer 流程，非当前主线
├── requirements.txt
├── README.md
└── 实验项目脚本与结果阅读指南.md
```

仓库中还保留了 `run_all.py`、`sweep_transformer.py`、`ensemble_predict.py`、`analyze_errors.py`、`plot_report_figures.py` 等旧流程脚本。它们主要对应之前的 `outputs/sweep` 实验链路，**不是当前 A/B/C/D 双验证标签实验的推荐入口**。

特别是当前版本中，旧 `train_transformer.py` 和 `predict_sentiment.py` 已经移动到 `bin/`，而 `run_all.py` / `sweep_transformer.py` 仍按旧路径寻找它们，因此不要把 `run_all.py` 当作当前版本的一键复现实验入口。

## 3. 环境安装

建议使用 Linux / Ubuntu + Python 3.10 或相近版本。

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` **不会安装 PyTorch**，避免覆盖本机已有的 CUDA 版本。请根据自己的 GPU / CUDA 环境单独安装合适的 PyTorch。

可以先检查环境：

```bash
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

## 4. 重要：GitHub 不包含模型权重

当前 `.gitignore` 会忽略：

```text
*.pt
*.bin
*.safetensors
```

因此：

- `pretrained/` 在 GitHub 中只有配置和 tokenizer 文件，**没有 UER 预训练权重**；
- `outputs/abcd_dual/**/best_model/` 在 GitHub 中也只有配置、tokenizer、`variant.json` 和指标文件，**没有训练后的 checkpoint 权重**。

也就是说，clone 仓库后不能直接恢复 A/B/C/D 模型，需要先准备预训练权重并重新训练。

### 准备 UER 预训练模型

当前 `train_A.py` 固定从项目根目录的 `pretrained/` 离线加载模型。可以使用 `huggingface_hub` 下载：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="uer/chinese_roberta_L-8_H-512",
    local_dir="pretrained",
)
PY
```

下载后 `pretrained/` 中除了配置、词表，还应存在模型权重文件。大权重文件会继续被 `.gitignore` 排除，不需要提交到 GitHub。

## 5. WordPiece 编码实验

查看分词、token id、`[CLS]`、`[SEP]`、`attention_mask` 和 `token_type_ids`：

```bash
python inspect_tokenizer.py --model pretrained
```

如果希望直接从 Hugging Face 加载，也可以：

```bash
python inspect_tokenizer.py --model uer/chinese_roberta_L-8_H-512
```

后者需要网络或本地 Hugging Face 缓存。

## 6. 经典机器学习实验

完整运行 15 种基础分类器及额外优化：

```bash
python train_classical.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --output-dir outputs/classical
```

基础方法包括朴素贝叶斯、逻辑回归、LinearSVC、Ridge、Perceptron、NearestCentroid、KNN、RBF-SVM、决策树、随机森林、AdaBoost、GradientBoosting 和 MLP 等共 15 种。

额外优化包括：

- LinearSVC 多组 `C`；
- LogisticRegression 多组 `C`；
- 字符 + jieba 词级 n-gram 特征融合；
- NB-SVM 式对数计数比加权。

主要输出：

```text
outputs/classical/
├── classical_results.csv
├── best_classical.joblib
├── summary.json
├── best_errors.csv
├── metrics_*.json
└── pred_*.csv
```

当前仓库保存的完整实验结果中：

- 15/15 个基础分类器均已完成；
- 额外完成 10 组优化；
- 最佳经典方法为 `NB_SVM_C4`；
- 验证集 Macro-F1 ≈ **85.17%**。

快速调试可使用：

```bash
python train_classical.py --fast
```

`--fast` 使用抽样子集，**其分数不能写入正式实验报告**。

## 7. 标签人工核查

`label_curation.py` 用独立的字符 TF-IDF + LogisticRegression 模型筛选高置信度、且预测与原标签不一致的样本，然后由人工决定是否翻转标签。

### 第一步：生成待审核样本

```bash
python label_curation.py prepare \
  --train data/train.csv \
  --validation data/validation.csv \
  --out-dir outputs/label_audit
```

默认：

- 训练集使用 5 折 OOF 预测；
- 验证集使用仅在训练集拟合的模型预测；
- 只选预测与原标签不同且置信度 `>=0.90` 的样本。

生成：

```text
outputs/label_audit/
├── review_train.csv
└── review_validation.csv
```

人工只需要填写 `decision`：

- `flip`：翻转标签
- `keep`：保留原标签
- `unsure`：暂不修改
- 空白：未审核，不修改

### 第二步：应用人工审核结果

```bash
python label_curation.py apply \
  --train data/train.csv \
  --validation data/validation.csv \
  --out-dir outputs/label_audit
```

生成：

```text
outputs/label_audit/
├── train_semantic.csv
├── validation_semantic_partial.csv
└── changes.csv
```

注意：

- 原始 CSV 不会被覆盖；
- `validation_semantic_partial.csv` 只修改了经过候选筛选并人工审核的部分样本；
- 它不是完整重新标注的语义金标准；
- 在该文件上得到的分数不能直接当作原始验证集性能提升。

仓库已经提交了一份人工审核后的 `outputs/label_audit/`。如果想从头重新审核，请使用新的 `--out-dir`，因为脚本会拒绝覆盖已有审核文件。

## 8. 当前预训练模型主实验：A / B / C / D

四种模型都基于 UER RoBERTa-Medium：

```text
uer/chinese_roberta_L-8_H-512
```

| 模型 | 配置 |
|---|---|
| A | FGM |
| B | FGM + CLS/Mean Pooling |
| C | FGM + R-Drop |
| D | FGM + CLS/Mean Pooling + R-Drop |

其中：

- FGM 扰动词嵌入；
- B/D 将最后一层 `CLS` 表示与非特殊 token 的 mean pooling 拼接后分类；
- C/D 使用 R-Drop，默认 `alpha=0.5`。

默认训练参数：

```text
epochs                = 5
batch_size            = 8
grad_accum             = 4
effective batch size  = 32
eval_batch_size       = 32
max_length            = 192
learning rate         = 2e-5
FGM epsilon           = 0.5
label smoothing       = 0.04
layerwise LR decay    = 0.92
head LR multiplier    = 5.0
weight decay          = 0.01
warmup ratio          = 0.1
dropout               = 0.1
seed                   = 42
patience               = 0
AMP                    = auto
```

### 双训练集 × 双验证标签

每个 A/B/C/D 脚本默认训练：

- 原始训练集 `data/train.csv`
- 修正训练集 `outputs/label_audit/train_semantic.csv`

每个 epoch 对同一批 4,000 条验证评论只预测一次，然后分别计算：

- 原始验证标签 `data/validation.csv`
- 部分修正标签 `outputs/label_audit/validation_semantic_partial.csv`

两套 Macro-F1 **分别独立选最佳 epoch**，因此每个“模型 × 训练集”最多保存两个 checkpoint：

```text
best_by_original/
best_by_semantic/
```

四个模型 × 两种训练集 × 两种选模标准，共最多 **16 个 checkpoint**；实际只进行 **8 次训练**。

### 运行

```bash
python train_A.py
python train_B.py
python train_C.py
python train_D.py
```

只训练一种训练集：

```bash
python train_C.py --dataset original
python train_C.py --dataset cleaned
```

### clone 后首次复现必须注意

仓库提交了 `outputs/abcd_dual/` 的指标、配置和 tokenizer 元数据，但没有提交模型权重。

当前代码判断一个实验“已完成”时主要检查 `variant.json` 和 `metrics.json`。因此 clone 后直接运行 `python train_A.py` 可能会提示已有结果并跳过训练，但实际上不存在可恢复的权重。

要在新环境重新生成完整 checkpoint，请先准备 `pretrained/` 权重，然后使用：

```bash
python train_A.py --overwrite
python train_B.py --overwrite
python train_C.py --overwrite
python train_D.py --overwrite
```

`--overwrite` 会删除对应模型/训练集的旧输出目录后重训，请确认不需要保留本地已有权重。

## 9. A/B/C/D 输出结构

```text
outputs/abcd_dual/
├── comparison.csv
├── A/
│   ├── original/
│   │   ├── training_config.json
│   │   ├── history.csv
│   │   ├── best_by_original/
│   │   │   ├── metrics.json
│   │   │   └── best_model/
│   │   └── best_by_semantic/
│   │       ├── metrics.json
│   │       └── best_model/
│   └── cleaned/
│       └── ...
├── B/
├── C/
└── D/
```

`comparison.csv` 汇总：模型、训练标签版本、`selected_by`、最佳 epoch、两套验证标签下的 Accuracy / Macro-F1 / Macro-P / Macro-R、checkpoint 路径和恢复验证状态。

`checkpoint` 列中的绝对路径来自实际运行机器，只用于记录；换机器后不具有可移植性。

## 10. 当前已提交的 A/B/C/D 结果

下面只列“按原始验证标签选出的 checkpoint”，避免重复展示同一训练的两个选模记录。

| 模型 | 训练标签 | 原始验证 Macro-F1 | 部分修正验证 Macro-F1 | 最佳 epoch |
|---|---|---:|---:|---:|
| A | original | 88.32% | 89.10% | 5 |
| A | cleaned | 88.40% | 89.17% | 5 |
| B | original | 88.22% | 89.05% | 4 |
| B | cleaned | 88.12% | 88.95% | 4 |
| C | original | 88.25% | 89.02% | 5 |
| C | cleaned | 88.40% | 89.22% | 5 |
| D | original | 88.22% | 89.05% | 4 |
| D | cleaned | 88.17% | 89.00% | 4 |

这些数字来自当前 `outputs/abcd_dual/comparison.csv`。

部分修正验证标签与原始验证标签不是同一个评价标准，因此两列之间的差值**不能全部解释为模型性能提升**。

## 11. 输入新评论并观察预测

当前 A/B/C/D 主线已经把新评论演示集成进训练入口。

例如，本地已经存在完整 C 模型 checkpoint 时：

```bash
python train_C.py --demo
```

脚本会在 C 的 original / cleaned 结果中选择**原始验证 Macro-F1 更高**的 checkpoint，并预测四条新的正面、负面、否定和转折评论。

如果实验结果已经存在且权重完整，参数一致时训练阶段会跳过；如果是 GitHub clone 后的无权重状态，则需要先按前文使用 `--overwrite` 重训。

如需预测自己输入的任意评论，可直接调用 `train_A.py` 中的 `load_checkpoint()`，具体代码见《实验项目脚本与结果阅读指南.md》。

## 12. 旧流程与当前流程的关系

### 当前推荐入口

```text
train_classical.py
inspect_tokenizer.py
label_curation.py
train_A.py
train_B.py
train_C.py
train_D.py
```

### 旧版 / 历史流程

```text
bin/train_transformer.py
bin/predict_sentiment.py
bin/test_abcd.py
sweep_transformer.py
run_all.py
ensemble_predict.py
analyze_errors.py
plot_report_figures.py
outputs/sweep/
outputs/analysis/
```

旧流程中的结果仍可用于核对此前实验和报告，但不要与当前 `outputs/abcd_dual/` 的 A/B/C/D 结果混为同一次实验。

当前仓库中：

- `sweep_transformer.py` 仍按旧位置寻找根目录 `train_transformer.py`；
- `run_all.py` 仍按旧位置寻找根目录 `predict_sentiment.py`；
- 这两个依赖脚本目前已经位于 `bin/`。

因此，**当前版本不要直接使用 `run_all.py` 作为一键入口**。

## 13. 结果复现说明

请区分三类内容：

1. **源代码**：可以直接从 GitHub 获取；
2. **CSV / JSON 指标和历史记录**：仓库保存了实际实验记录；
3. **大模型权重**：由于 `.gitignore` 排除大文件，不在 GitHub 中，需要在本地准备或重新训练。

模型训练存在随机性。当前代码固定 `seed=42`，但不同 PyTorch、CUDA、GPU 或底层算子版本仍可能造成细微差异。

## 14. 更详细的说明

各脚本用途、参数、输出文件、A/B/C/D checkpoint 格式、任意新评论预测代码，以及旧流程与当前流程的对应关系，见：

**[实验项目脚本与结果阅读指南.md](./实验项目脚本与结果阅读指南.md)**
