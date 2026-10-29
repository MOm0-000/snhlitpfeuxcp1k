# 实验项目脚本与结果阅读指南

本文档对应当前 `main` 分支的最终实验结构，只说明当前仍存在、且与最终实验结果一致的代码与输出。历史 sweep、ensemble 与旧 FGM 分析不再作为当前实验主线。

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
        └── label_curation.py
                └── outputs/label_audit/
                        │
                        ├── train_A.py
                        ├── train_B.py
                        ├── train_C.py
                        └── train_D.py
                                └── outputs/abcd_dual/
                                        │
                                        ├── final Transformer:
                                        │   C/cleaned/best_by_original
                                        │
                                        ├── outputs/final_transformer_validation.csv
                                        │
                                        └── analyze_errors.py
                                                └── outputs/analysis/

final Transformer
        └── train_C.py --demo
                └── outputs/new_comment_demo.txt
```

课程没有提供独立测试集，所有主指标均为原始 `validation.csv` 上的验证结果。

---

# 第一部分：数据与公共工具

## 2. 数据

`data/train.csv`：

- 20,000 条；
- 0/1 各 10,000；
- 用于训练和特征拟合。

`data/validation.csv`：

- 4,000 条；
- 0/1 各 2,000；
- 用于模型比较、参数选择、最佳 epoch 选择和误差分析。

字段：

```text
id,text,label
```

标签：

```text
0 = 负面（1–2 星）
1 = 正面（4–5 星）
```

3 星不使用。

## 3. `common.py`

经典机器学习使用的公共函数包括：

- `read_csv()`：读取并检查列、空值、ID、标签；
- `check_split()`：检查训练/验证 ID 重叠并提示完全重复文本；
- `compute_metrics()`：Accuracy、Macro-F1、Macro-P、Macro-R、混淆矩阵等；
- `save_predictions()`：保存逐条预测；
- `save_errors()`：保存误分类；
- `set_seed()`：固定随机种子。

A/B/C/D 主训练逻辑在 `train_A.py` 中独立实现数据检查、指标、checkpoint 和恢复验证。

---

# 第二部分：Tokenizer

## 4. `inspect_tokenizer.py`

用途：

- 使用 UER 模型配套 WordPiece tokenizer；
- 查看 token 切分；
- 查看 token ID；
- 查看 `[CLS]` / `[SEP]` / `[PAD]`；
- 查看 `attention_mask`；
- 查看 `token_type_ids`。

推荐：

```bash
python inspect_tokenizer.py --model pretrained
```

已保存的实际运行记录：

```text
outputs/inspect_tokenizer.txt
```

当前记录中：

```text
vocab size = 21128
keyboard -> key + ##board
[CLS] = 101
[SEP] = 102
[PAD] = 0
```

---

# 第三部分：经典机器学习

## 5. `train_classical.py`

### 5.1 字符 TF-IDF

```text
analyzer      char
ngram_range   (2, 4)
min_df        2
max_features  180000
sublinear_tf  True
norm          l2
dtype         float32
```

向量器只在训练集 `fit`，validation 只 `transform`。

### 5.2 课程要求的 15 种基础分类器

```text
1.  MultinomialNB
2.  ComplementNB
3.  BernoulliNB
4.  LogisticRegression
5.  LinearSVC
6.  RidgeClassifier
7.  Perceptron
8.  NearestCentroid
9.  KNeighborsClassifier
10. SVC_RBF
11. DecisionTreeClassifier
12. RandomForestClassifier
13. AdaBoostClassifier
14. GradientBoostingClassifier
15. MLPClassifier
```

距离、树、RBF-SVM、MLP 等使用训练集拟合的 96 维 TruncatedSVD 表示。

### 5.3 额外优化

默认还运行 10 组额外配置：

- LinearSVC：`C=0.5,1,4,8`
- LogisticRegression：`C=1,4`
- CharWord LinearSVC：`C=1,4`
- NB-SVM 式逻辑回归：`C=1,4`

字词融合使用：

- 字符 TF-IDF `(2,4)`；
- jieba 词级 `(1,2)`；
- 词特征最大 85,000；
- 字符/词特征缩放后拼接并 L2 归一化。

NB-SVM 式方法在融合特征上计算训练集正负类的对数计数比，再进行逻辑回归。

### 5.4 正式运行

```bash
python train_classical.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --output-dir outputs/classical
```

正式结果：

```text
15/15 基础方法完成
10 组额外优化完成
共 25 组
最佳 NB_SVM_C4
Macro-F1 = 0.8517277499
```

### 5.5 输出解释

```text
outputs/classical/classical_results.csv
```

25 组模型总表，包含：

```text
method
feature
accuracy
macro_f1
macro_precision
macro_recall
seconds
parameters
```

```text
outputs/classical/pred_<method>.csv
```

每种方法对 4,000 条 validation 的逐条预测，可用于 classic-vs-classic 案例比较。

```text
outputs/classical/metrics_<method>.json
```

各模型完整指标。

```text
outputs/classical/best_classical.joblib
```

最佳传统模型及其向量器/特征处理器。

```text
outputs/classical/summary.json
```

正式运行摘要。

---

# 第四部分：人工标签核查

## 6. `label_curation.py`

该脚本用于额外的标签质量实验，不改变课程原始 CSV。

流程：

```text
训练模型筛候选
    ↓
人工阅读
    ↓
填写 decision
    ↓
生成修正标签 CSV
```

### 6.1 prepare

```bash
python label_curation.py prepare \
  --train data/train.csv \
  --validation data/validation.csv \
  --out-dir outputs/label_audit
```

训练集候选使用 5 折 OOF，优先使用 `StratifiedGroupKFold`，完全相同文本作为 group。验证集候选由只在 train 上拟合的筛选模型产生。

默认候选条件：

```text
prediction != original label
confidence >= 0.90
```

人工填写：

```text
flip
keep
unsure
空白
```

### 6.2 apply

```bash
python label_curation.py apply \
  --train data/train.csv \
  --validation data/validation.csv \
  --out-dir outputs/label_audit
```

输出：

```text
train_semantic.csv
validation_semantic_partial.csv
changes.csv
```

当前记录中，训练集有 178 条翻转，validation 有 33 条翻转。

`validation_semantic_partial.csv` 不是完整金标准，因此原始 validation 始终是课程主指标。

---

# 第五部分：A/B/C/D 预训练实验

## 7. 模型与本地权重

固定模型：

```text
uer/chinese_roberta_L-8_H-512
```

`train_A.py` 从项目根目录：

```text
pretrained/
```

离线加载。

GitHub 不提交 `.bin/.pt/.safetensors`，因此新环境必须先下载 UER 权重，然后重训。

## 8. A/B/C/D 入口

核心实现都在：

```text
train_A.py
```

`train_B.py`、`train_C.py`、`train_D.py` 是薄入口，调用同一套 `run()`。

配置：

```text
A = FGM
B = FGM + CLS/Mean Pooling
C = FGM + R-Drop
D = FGM + CLS/Mean Pooling + R-Drop
```

### 8.1 FGM

当前版本：

```text
FGM_VERSION = current_microbatch_gradient_v2
```

只扰动 `word_embeddings.weight`。在梯度累积条件下，用当前 micro-batch 新产生的梯度决定扰动方向，随后计算 adversarial loss 并恢复 embedding。

默认：

```text
epsilon = 0.5
```

### 8.2 CLS/Mean Pooling

B/D 将最后一层：

```text
CLS hidden
+
非 [CLS]/[SEP] 正文 token 的 mean pooling
```

拼接为 1024 维向量，再经 dropout + 二分类层。

### 8.3 R-Drop

C/D 对同一 batch 做两次 dropout 前向：

```text
loss = 平均 CE + alpha * symmetric KL
alpha = 0.5
```

## 9. 优化器与统一超参数

```text
epochs                  5
batch size              8
gradient accumulation   4
effective batch         32
eval batch              32
max length              192

base LR                 2e-5
head LR multiplier      5.0
layerwise LR decay      0.92
weight decay            0.01
warmup ratio            0.10

FGM epsilon             0.5
label smoothing         0.04
dropout                 0.10
R-Drop alpha            C/D=0.5

max grad norm           1.0
seed                    42
patience                0
AMP                     auto
```

`patience=0` 表示默认完整训练 5 个 epoch。

分类头使用更高学习率；Transformer 越靠近输入层学习率越低；bias 与 LayerNorm 参数不做 weight decay。

## 10. 双训练标签 × 双验证标签

每个模型默认训练两个版本：

```text
original = data/train.csv
cleaned  = outputs/label_audit/train_semantic.csv
```

每个 epoch 对 4,000 条 validation 只预测一次，再分别按：

```text
original labels = data/validation.csv
semantic labels = outputs/label_audit/validation_semantic_partial.csv
```

计算指标。

两套标签各自独立选择 Macro-F1 最佳 epoch，故每个“模型 × 训练版本”最多有：

```text
best_by_original/
best_by_semantic/
```

四个模型 × 两种训练集 = 8 次训练；两种选模口径最多产生 16 个 checkpoint 记录。

## 11. 正式结果

`outputs/abcd_dual/comparison.csv` 是 A/B/C/D 总表。

按原始验证标签选模：

```text
A/original  88.325%
A/cleaned   88.400%

B/original  88.224%
B/cleaned   88.124%

C/original  88.250%
C/cleaned   88.400%

D/original  88.224%
D/cleaned   88.174%
```

未舍入 Macro-F1：

```text
A/cleaned 0.8839997389994128
C/cleaned 0.8839998839998839
```

因此当前最终 Transformer 定义为：

```text
C/cleaned/best_by_original
```

其恢复验证：

```text
Accuracy        0.884
Macro-F1        0.8839998839998839
Macro-Precision 0.8840015360061441
Macro-Recall    0.884
verified_reload True
```

## 12. `history.csv`

例如：

```text
outputs/abcd_dual/C/cleaned/history.csv
```

保存每个 epoch：

```text
train_loss
original Accuracy/F1/P/R
semantic Accuracy/F1/P/R
optimizer_steps
elapsed_seconds
```

C/cleaned 原始验证 Macro-F1：

```text
epoch 1  86.50%
epoch 2  87.80%
epoch 3  88.25%
epoch 4  88.10%
epoch 5  88.40%
```

训练 loss：

```text
0.4132
0.3205
0.2928
0.2708
0.2556
```

---

# 第六部分：模型恢复、新评论与最终预测

## 13. checkpoint 格式

A/C 使用 Hugging Face 标准分类模型结构；B/D 使用自定义 CLS/Mean 分类器。

恢复 A/B/C/D 时应使用：

```python
from train_A import load_checkpoint
```

不要直接假设所有 checkpoint 都能由 `AutoModelForSequenceClassification.from_pretrained()` 恢复。

## 14. 新评论 demo

运行：

```bash
python train_C.py --demo
```

当前保存的实际输出：

```text
outputs/new_comment_demo.txt
```

包含：

- 正面；
- 负面；
- 否定；
- 转折。

当前四条结果：

```text
正面 0.972：质量很好，物流很快，下次还会购买。
负面 0.016：质量非常差，客服态度也不好。
正面 0.601：这款商品不是不好，只是价格有点高。
负面 0.283：虽然包装很好，但是商品有明显缺陷。
```

## 15. 最终 Transformer validation 预测

当前最终 C/cleaned 对原始 validation 的逐条预测保存在：

```text
outputs/final_transformer_validation.csv
```

字段包含 `id,text,label,prediction,positive_probability,correct`。

如果需要从本地完整 checkpoint 重新生成该文件，可在项目根目录运行：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import torch
from train_A import load_checkpoint

checkpoint = Path("outputs/abcd_dual/C/cleaned/best_by_original/best_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, info = load_checkpoint(checkpoint, device)

df = pd.read_csv("data/validation.csv", dtype={"id": str}, keep_default_na=False)
predictions, probabilities = [], []

for start in range(0, len(df), 32):
    texts = df["text"].iloc[start:start + 32].tolist()
    batch = tokenizer(
        texts, padding=True, truncation=True,
        max_length=int(info["max_length"]), return_tensors="pt"
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**batch).logits.float(), dim=-1)
    predictions.extend(probs.argmax(dim=-1).cpu().tolist())
    probabilities.extend(probs[:, 1].cpu().tolist())

result = df[["id", "text", "label"]].copy()
result["prediction"] = predictions
result["positive_probability"] = probabilities
result["correct"] = result["prediction"] == result["label"]
result.to_csv("outputs/final_transformer_validation.csv", index=False, encoding="utf-8-sig")
PY
```

---

# 第七部分：最终误差分析

## 16. `analyze_errors.py`

正式命令：

```bash
python analyze_errors.py \
  --validation data/validation.csv \
  --classical outputs/classical/pred_NB_SVM_C4.csv \
  --transformer outputs/final_transformer_validation.csv \
  --output-dir outputs/analysis
```

输出：

```text
analysis_summary.json
full_comparison.csv
disagreements.csv
transformer_better_examples.csv
classical_better_examples.csv
both_wrong_examples.csv
```

## 17. 总体结果

```text
NB-SVM_C4
Accuracy   85.175%
Macro-F1   85.173%

C/cleaned
Accuracy   88.400%
Macro-F1   88.400%
```

混淆矩阵：

```text
NB-SVM
[[1728, 272],
 [ 321,1679]]

C/cleaned
[[1770, 230],
 [ 234,1766]]
```

因此：

```text
NB-SVM 错误 593
C/cleaned 错误 464
净减少错误 129
```

预测分歧：

```text
disagreements      355
transformer wins   242
classical wins     113
both wrong         351
```

## 18. 否定与转折子集

字符串规则粗筛：

```text
contains_negation n=3055
NB-SVM F1    84.81%
C/cleaned    88.01%

contains_contrast n=907
NB-SVM F1    82.22%
C/cleaned    86.65%

contains_both n=796
NB-SVM F1    82.62%
C/cleaned    86.76%

contains_neither n=834
NB-SVM F1    84.48%
C/cleaned    87.60%
```

规则只表示“出现相关关键词”，不是人工句法/语义标注，报告中必须保留这一限定。

## 19. 代表性实例

### 19.1 转折：Transformer 正确

`zh_0758336`：

> 板型好，很合身，但是超级硬……还是比较硬。不推荐

真实 0；NB-SVM → 1；C/cleaned → 0。

前半段局部正向，`但是` 后出现持续负面描述并以“不推荐”收束。

### 19.2 否定作用域 + 转折：Transformer 正确

`zh_0589784`：

> 不是学HR，也不做HR，但是通读本书还是对公司基本架构和人力资源管理上形成了基本认识。

真实 1；NB-SVM → 0；C/cleaned → 1。

开头否定的是读者身份而不是书籍质量，真正评价在 `但是` 后。

### 19.3 评价对象变化：经典模型正确

`zh_0745494`：

> 书是好书，大爱阮老师的作品。可是，第二本书没有图片你也卖……

真实 0；NB-SVM → 0；C/cleaned → 1。

评论先赞作品，后批购买到的商品/平台，Transformer 仍可能受强正向前文影响。

### 19.4 两模型都错：评分对象与局部情感不一致

`zh_0111902`：

> 梭罗的书很耐读，文笔也很优美，一星是给书本的印刷！……纸质粗糙……好多错别字……

真实 0；两模型均 → 1。

作品内容为正向，但星级针对印刷质量，体现多评价对象与评分标签之间的复杂关系。

---

# 第八部分：哪些文件是正式报告取数依据

最终报告应优先从以下文件取数：

```text
经典 25 组：
outputs/classical/classical_results.csv

经典最佳：
outputs/classical/summary.json
outputs/classical/metrics_NB_SVM_C4.json
outputs/classical/pred_NB_SVM_C4.csv

A/B/C/D：
outputs/abcd_dual/comparison.csv

最终 C/cleaned 训练曲线：
outputs/abcd_dual/C/cleaned/history.csv

最终逐条预测：
outputs/final_transformer_validation.csv

最终误差分析：
outputs/analysis/analysis_summary.json
outputs/analysis/*.csv

Tokenizer：
outputs/inspect_tokenizer.txt

新评论：
outputs/new_comment_demo.txt

标签修正：
outputs/label_audit/changes.csv
```

正式报告中的主指标统一使用**原始 validation 标签**：

```text
NB_SVM_C4  85.17%
C/cleaned  88.40%
```

`validation_semantic_partial.csv` 的分数只用于补充标签质量实验。

---

# 第九部分：GitHub 与本地 checkpoint 的区别

GitHub 保存的是：

- 代码；
- 数据 CSV；
- 训练历史；
- metrics；
- tokenizer/config 元数据；
- 各类预测与误差分析。

GitHub 不保存：

```text
*.pt
*.bin
*.safetensors
```

所以远端 `best_model/` 不等于可直接加载的完整本地 checkpoint。

新机器上要先下载 UER 权重，并用：

```bash
python train_A.py --overwrite
python train_B.py --overwrite
python train_C.py --overwrite
python train_D.py --overwrite
```

重新生成完整训练权重。

---

# 第十部分：最终结论口径

当前项目的最终实验结论应保持以下口径：

```text
最佳经典方法：NB_SVM_C4
原始 validation Macro-F1 = 85.17%

最终 Transformer：C/cleaned/best_by_original
原始 validation Macro-F1 = 88.40%

提升约 3.23 个百分点。
```

Transformer 在关键词规则筛出的转折子集上的优势更明显：

```text
82.22% -> 86.65%
```

但 `zh_0745494`、`zh_0111902` 等案例说明，评价对象变化和“文本局部情感 vs 星级评分对象”仍然是主要困难。

标签修正实验是补充研究，不改变课程主评价口径；课程没有提供独立测试集，因此不要把 validation 结果表述为 test score。
