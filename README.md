# 中文情感分析实验：Ubuntu + NVIDIA CUDA（纯 Python `.py` 项目）

按课程提供的两个 Notebook 和实验指导书编写。**没有 `.ipynb`，所有训练/验证/预测都是独立 Python 程序。** 本压缩包中已包含你提供的 `data/train.csv` 和 `data/validation.csv`。

## 目录

```text
sentiment_experiment/
├── common.py                  # 数据检查、统一指标、CSV/JSON 输出
├── inspect_tokenizer.py       # UER WordPiece、ID、mask、CLS/SEP 示例
├── train_classical.py         # 完整 15 种机器学习方法 + 性能优化
├── train_transformer.py       # UER RoBERTa-Medium 完整微调
├── sweep_transformer.py       # 多组微调参数对照、模型选择
├── predict_sentiment.py      # 最佳模型恢复、CSV/自由文本预测
├── analyze_errors.py         # 否定/转折、两模型分歧及错误分析
├── ensemble_predict.py       # 不同微调配置 soft-voting 集成
├── run_all.py                # 一键完整运行 + 输出分析
├── requirements.txt
└── data/
    ├── train.csv             # 20000 条，0/1 各 10000
    └── validation.csv        # 4000 条，0/1 各 2000
```

## 环境检查和安装

先进入包含 `.py` 文件的目录：

```bash
cd sentiment_experiment
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO NVIDIA CUDA")'
python -m pip install -r requirements.txt
```

> 项目不在 `requirements.txt` 中安装 PyTorch，因为你已经有 CUDA 对应的 PyTorch，避免意外替换为 CPU 版。当前新版 Transformers 加载 UER 官方旧版 `pytorch_model.bin` 时，可能要求 PyTorch ≥2.6；若启动时提示 PyTorch/transformers 版本不兼容，升级 **匹配自己 CUDA 的官方 PyTorch 构建**，不要直接无脑覆盖系统 CUDA 环境。
>
> 第一次从 Hugging Face 加载 `uer/chinese_roberta_L-8_H-512` 需要网络；之后利用本地缓存或改用 `--model /path/to/pretrained/`。本地目录需要有 `config.json`、模型权重和 `vocab.txt` 等 tokenizer 文件。

## 0. WordPiece 编码实验

```bash
python inspect_tokenizer.py
```

打印 `token -> id`、`[CLS]`、`[SEP]`、`[PAD]`、`attention_mask` 和 `token_type_ids`，对应 Notebook 4.1–4.3 的要求。

## 1. 完整 15 种传统机器学习方法

```bash
python train_classical.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --output-dir outputs/classical
```

默认会运行课程要求的 **全部 15 个方法**：`MultinomialNB`、`ComplementNB`、`BernoulliNB`、`LogisticRegression`、`LinearSVC`、`RidgeClassifier`、`Perceptron`、`NearestCentroid`、`KNeighborsClassifier`、`SVC_RBF`、`DecisionTreeClassifier`、`RandomForestClassifier`、`AdaBoostClassifier`、`GradientBoostingClassifier`、`MLPClassifier`。

统一在训练集构建字符 2–4-gram TF-IDF；对非线性、树、MLP 和近邻等适用低维稠密输入的方法先做 **训练集拟合的 TruncatedSVD（LSA）**；额外运行 LinearSVC/逻辑回归参数对照、jieba 词级 1–2-gram 特征融合、NB-SVM 改进。

输出：`outputs/classical/classical_results.csv`（完整排行榜）、每个模型 `metrics_*.json`/`pred_*.csv`、`best_classical.joblib`（向量化器+最优分类器）以及 `best_errors.csv`。可能较慢的核 SVC、集成方法全部运行完才能得出完整结果。对于 CPU 算法，**NVIDIA GPU 不会自动加速 scikit-learn**。

快速调试（只检查程序，不代表真实成绩）：

```bash
python train_classical.py --fast --no-tuning --output-dir outputs/classical_debug
```

不使用词级特征：`--no-word-features`。只训练指定模型：`--methods LinearSVC,LogisticRegression`。

## 2. 单次 UER RoBERTa-Medium 微调

使用官方 UER 中文 RoBERTa-Medium（`uer/chinese_roberta_L-8_H-512`，**8 层、隐藏维度 512**）；与 Notebook 一致，使用配套 `BertTokenizer`（WordPiece）。

```bash
python train_transformer.py \
  --model uer/chinese_roberta_L-8_H-512 \
  --train data/train.csv \
  --validation data/validation.csv \
  --output-dir outputs/roberta_medium \
  --epochs 5 \
  --max-length 160 \
  --batch-size 16 \
  --grad-accum 2 \
  --eval-batch-size 32 \
  --lr 3e-5 \
  --amp auto
```

默认对 **全部参数微调**，以验证集 **Macro-F1** 选出最佳轮次。默认启用 GPU 混合精度（支持 BF16 则优先 BF16，否则 FP16）、动态 padding、AdamW、warmup、线性调度、层间学习率衰减、分类头高学习率、梯度累积及裁剪、早停。

可选高级实验设置（建议分开做消融，不是所有方法叠加一定提升）：

```bash
# 对抗训练 FGM 与 label smoothing
python train_transformer.py --output-dir outputs/roberta_fgm \
  --max-length 192 --lr 2e-5 --label-smoothing 0.04 --fgm-epsilon 0.5

# 双 dropout 前向一致性 R-Drop（计算成本更高）
python train_transformer.py --output-dir outputs/roberta_rdrop \
  --max-length 160 --lr 2e-5 --rdrop-alpha 0.5
```

单次运行输出：

```text
outputs/roberta_medium/
├── training_config.json             # 超参数
├── history.csv                      # 各轮 loss/验证指标
├── best_model/                      # 最佳模型全部权重、配置、tokenizer
├── best_metrics.json               # 训练时最佳轮次指标
├── restored_metrics.json           # 实际重载 checkpoint 后复核指标
├── validation_predictions.csv      # 最佳验证预测和正类 softmax 分数
├── restored_validation_predictions.csv
└── errors_top250.csv               # 错误样例（按错误置信度降序）
```

**GPU 显存不足**时用更小的每卡 batch，再通过梯度累积保持有效 batch：

```bash
python train_transformer.py --batch-size 4 --grad-accum 8 \
  --eval-batch-size 8 --max-length 128 --grad-checkpointing
```

## 3. 推荐的性能优化实验：统一验证对照

```bash
python sweep_transformer.py \
  --profiles baseline,longer,fgm \
  --epochs 5 --batch-size 16 --grad-accum 2 \
  --output-dir outputs/sweep
```

三个方案：

- `baseline`：长度 128、常规学习率 3e-5、无 FGM，用于对照。
- `longer`：长度 192、2e-5、分层 LR、少量平滑，用于消融。
- `fgm`：在 longer 上增加 FGM 0.5。
- 可增设 `rdrop`：`--profiles baseline,longer,fgm,rdrop`，R-Drop 多次前向会明显增加训练时间。

每种配置都在**相同训练/验证划分**上运行并保存独立模型；排行榜写入 `outputs/sweep/leaderboard.csv`。高 F1 的验证配置不一定在未知测试集上最好：请把独立测试集留到选模之后使用。

## 4. 高性能可选：多个已训练模型的软投票集成

独立跑完 `sweep_transformer.py` 后，可以把若干最佳模型的正面概率平均，再做分类。每次只加载一个 checkpoint 预测，避免一次加载多个模型占满 GPU 显存。

```bash
python ensemble_predict.py \
  --leaderboard outputs/sweep/leaderboard.csv --top-k 3 \
  --input-csv data/validation.csv \
  --output outputs/ensemble/validation_predictions.csv
```

它会生成集成 Macro-F1、预测和错误样例。**集成不一定比最佳单模型更好**；建议在验证集上同时比较。`run_all.py` 默认自动计算集成并选用验证更好的结果进行误差分析，若想省时间加 `--no-ensemble`。

## 5. 预测、恢复验证、误分类分析

```bash
# 恢复最优预训练模型，验证集评估及预测
python predict_sentiment.py \
  --model outputs/roberta_medium/best_model \
  --input-csv data/validation.csv \
  --output outputs/transformer_validation.csv

# 恢复最优传统方法，在同一验证集上预测
python predict_sentiment.py \
  --model outputs/classical/best_classical.joblib \
  --input-csv data/validation.csv \
  --output outputs/classical_validation.csv

# 两模型预测分歧、否定/转折句、双方正确/错误样例
python analyze_errors.py \
  --validation data/validation.csv \
  --classical outputs/classical_validation.csv \
  --transformer outputs/transformer_validation.csv \
  --output-dir outputs/analysis

# 对未标注新句子预测
python predict_sentiment.py --model outputs/roberta_medium/best_model \
  --text '包装很好，但是质量非常差。' \
  --text '物流虽然有点慢，不过商品很好用。' \
  --output outputs/new_reviews.csv
```

若老师之后下发 `data/test.csv`，只需要：

```bash
python predict_sentiment.py --model outputs/roberta_medium/best_model \
  --input-csv data/test.csv --output outputs/test_predictions.csv
```

对无标签 test.csv 只输出预测；有 `label` 时会额外输出 Accuracy、Macro-F1、Precision、Recall 指标。

## 6. 一键运行

```bash
python run_all.py --epochs 5 --profiles baseline,longer,fgm
```

一键脚本会调用上述 `.py` 文件，运行全部 15 种传统方法 + 微调对照 + 自动读取排行榜中验证表现最好的模型 + 集成方案 + 模型分歧分析。`--dry-run` 只显示命令。

## 注意

1. **当前只提供了训练集和验证集**，没有独立测试集；所有实验结果必须由你真实执行后产生，不能把验证成绩称为独立测试成绩。
2. 原始两份 CSV 的 `id` 不重叠，但有少量**文本完全相同的样本跨集合出现**。代码只警告并按课程指定划分运行，避免改动基准；在报告里可说明该限制。
3. CSV 标签按课程定义：0 = 负面，1 = 正面。评价同时报告宏平均 Precision/Recall 和正类 Precision/Recall，避免两种口径混淆。
4. `softmax` 分数未经概率校准。否定/转折的规则标签是粗筛，建议人工阅读错误例句再讨论结论。
5. 性能优化没有预先保证一定提升，只有使用 `leaderboard.csv` 的实际指标才可以得出优化结论。
6. 参考模型：https://huggingface.co/uer/chinese_roberta_L-8_H-512
