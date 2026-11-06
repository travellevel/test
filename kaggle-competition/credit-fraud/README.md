# 信用卡欺诈检测：四种模型、20% 验证集与 ROC 对比

比赛：[Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11/overview)

这个项目实现完整的表格二分类流程：读取数据、按 **80% / 20%** 分层划分、拟合训练部分的预处理、按验证 ROC-AUC 保存最佳模型，重新加载后生成 ROC 图和欺诈概率。已实现 **Embedding + MLP、FT-Transformer、TabNet、CatBoost** 四种模型，各有独立入口并共用验证方案。建议先阅读 MLP，再对照其余方法。

| 文件 | 职责 |
|---|---|
| `train_mlp.py` | 类别 Embedding + MLP，59 维拼接输入 |
| `train_ft_transformer.py` | 使用论文作者的 FT-Transformer，学习字段间注意力 |
| `train_tabnet.py` | 使用 pytorch-tabnet 的决策步骤与稀疏特征选择 |
| `train_catboost.py` | CatBoost 原生类别处理，默认 CPU |
| `tabular_common.py` | 数据检查、固定抽样和划分、预处理、配置、指标与 ROC 绘图 |
| `neural_training.py` | MLP / FT-Transformer 的梯度训练、早停与模型保存加载 |
| `compare_roc.py` | 校验各运行的验证集一致性，绘制叠加 ROC 和导出指标表 |

## 1. 数据与任务

以下统计来自项目中已有的 CSV。

| 项目 | 数值 / 说明 |
|---|---|
| 原始训练集 | 699,960 行，26 列 |
| 比赛测试集 | 299,984 行，25 列 |
| 提交模板 | 299,984 行，列名为 `id,is_fraud` |
| 目标 | `is_fraud`：0 为正常交易，1 为欺诈交易 |
| 欺诈样本 | 11,689 条，约占 1.67% |
| 模型输入 | 24 个特征；`id` 和 `is_fraud` 不进入输入 |
| 缺失情况 | 当前训练集、测试集均无缺失值 |
| 测试集类别 | 当前 5 个类别字段均未出现训练集之外的新类别 |

| 类型 | 特征 |
|---|---|
| 数值，13 个 | `amount_usd`、`hours_since_last_txn`、`txn_count_last_24h`、`distance_from_home_km`、`card_age_months`、`customer_age`、`account_balance_usd`、`cvv_retry_count`、`velocity_score`、`time_of_day_hour`、`day_of_week`、`merchant_risk_score`、`prior_disputes` |
| 类别，5 个 | `merchant_category`、`card_type`、`auth_method`、`channel`、`device_type` |
| 布尔，6 个 | `is_foreign_transaction`、`is_new_merchant`、`used_vpn`、`ip_country_mismatch`、`billing_shipping_mismatch`、`is_ai_generated_scam_attempt` |

### 评价指标：ROC-AUC

ROC-AUC 可以理解为：随机抽取一笔欺诈交易和一笔正常交易，模型给欺诈交易更高分的概率；分数相同时按一半计算。0.5 对应随机排序，1.0 对应完美排序。比赛需要提交每笔交易的连续欺诈概率。

| 容易混淆的地方 | 应如何理解 |
|---|---|
| 准确率很高是否说明模型好？ | 本数据约 98.33% 的交易正常，全部预测正常也能得到很高的准确率，却没有识别欺诈的能力 |
| 是否需要先选 0.5 阈值？ | ROC-AUC 评估连续分数的排序，生成提交时直接输出概率 |
| AUC 高是否说明概率准确？ | AUC 主要反映排序能力；概率校准属于另一项评估 |
| 样例提交能直接作为预测吗？ | 模板中的约 0.0167 是固定值，恰好等于训练集欺诈比例；常数预测的 ROC-AUC 为 0.5 |
| 为什么同时记录 AP？ | Average Precision 更直接反映少数类的查准率与查全率表现；随机排序的参考水平约为正类比例，比赛主指标仍是 ROC-AUC |

## 2. 为什么先使用 MLP

这里的记录是具有混合字段的表格数据，没有可直接使用的图像结构或交易序列。首版使用数值标准化、布尔编码和类别 Embedding，便于理解整个深度学习训练流程。

| 方法 | 基本思路 | 优点 | 局限 | 本项目定位 |
|---|---|---|---|---|
| **Embedding + MLP** | 类别值映射为可学习向量，与数值特征拼接后输入全连接网络 | 代码直观，计算量较小，能练习反向传播、正则化和模型保存 | 对尺度和优化参数较敏感，复杂特征交互需要通过训练学到；低基数类别的 Embedding 未必优于独热编码 | 已实现，建议先学 |
| FT-Transformer | 将各字段转换成 token，用自注意力学习字段交互 | 适合学习表格注意力模型，可表达较复杂的特征关系 | 通常比小型 MLP 消耗更多显存和时间；超参数更多，收益需要验证 | 已实现，作者库 `rtdl-revisiting-models==0.0.2` |
| TabNet | 通过多步特征选择和决策模块完成预测 | 可观察不同步骤的特征选择，提供解释线索 | 实现和调参更复杂，特征选择权重也不能等同于因果解释，效果不保证领先 | 已实现，`pytorch-tabnet==4.1.0` |
| CatBoost | 使用梯度提升树建模，并提供类别特征处理能力 | 混合表格任务的重要参照，通常不需要数值标准化 | 属于树模型，学习内容与神经网络不同；本数据仍需实测 | 已实现，`catboost==1.2.10` |

上述方法没有预先确定的分数顺序。模型比较应固定数据划分和评价方式，避免把划分变化误认为模型提升。

### 网络结构与训练参数

```text
13 个数值特征 ── 训练部分统计量标准化 ──┐
 6 个布尔特征 ── 0/1 编码 ─────────────┼─ 拼接为 59 维
 5 个类别特征 ── 各自 8 维 Embedding ──┘
    → Linear(256) → ReLU → Dropout(0.2)
    → Linear(128) → ReLU → Dropout(0.2)
    → Linear(64)  → ReLU → Dropout(0.2)
    → Linear(1) → logit
    → 预测时 sigmoid → 欺诈概率
```

| 参数 | 默认值 / 行为 |
|---|---|
| 损失 | 不加权 `BCEWithLogitsLoss`，训练时直接输入 logit |
| 优化器 | AdamW，学习率 `1e-3`，权重衰减 `1e-4` |
| 批大小 / 最大轮数 | `4096` / `30` |
| 模型选择 | 验证 ROC-AUC 最高的轮次 |
| 提前停止 | 连续 5 轮验证 AUC 未提升 |
| 初始输出偏置 | MLP / FT-Transformer 使用 **80% 训练部分** 的正类比例初始化，帮助网络从合理的基础概率开始 |
| 类别不平衡 | 保留原始分布；首版不使用重采样、正类权重或 Focal Loss |
| 随机种子 | `42`；固定数据与种子可复现划分，不承诺不同硬件之间训练结果逐位一致 |

加权损失、重采样和 Focal Loss 可以作为后续实验，但改善损失值或少数类召回率不一定改善 ROC-AUC。

### 其余三种方法的实现

| 方法 | 输入与核心结构 | 默认优化参数 |
|---|---|---|
| FT-Transformer | 19 个数值/布尔字段各自线性变换为 token，5 个类别字段各有 Embedding；CLS token 汇总字段交互；2 块、64 维 token、8 个注意力头、PreNorm 与 ReGLU | AdamW，`lr=1e-4`、`weight_decay=1e-5`；作者参数分组将偏置、归一化与 Embedding 排除出衰减；批大小 512，最多 30 轮 |
| TabNet | 预处理后的 24 列矩阵，`cat_idxs` 指定后 5 列类别编码；`n_d=n_a=16`、3 个决策步骤、8 维类别 Embedding，`lambda_sparse=1e-3` | Adam，`lr=0.02`，批大小 1024，虚拟批大小 128，最多 30 轮；交叉熵加稀疏正则，`weights=0` 不重采样 |
| CatBoost | 数值按训练部分中位数填充但不标准化；类别保留字符串；深度 6 的梯度提升树 | `lr=0.05`，最多 1,000 棵树，Logloss，不加类别权重；默认 CPU，支持 `--device cuda` |

四种模型都按验证 **AUC** 选择最佳轮次，连续 5 轮未提升停止。TabNet 的评估列表将 AUC 放在最后，因为库使用最后一个指标进行早停；其 Ghost BatchNorm 将大批次拆分为虚拟批次计算统计量。为避免 BatchNorm 单样本错误，训练批次余数恰好为 1 时会丢弃最后一条；验证和预测保留全部记录。

## 3. 固定使用 80% 训练、20% 验证

本项目将“交叉验证集”按 **固定分层留出验证集** 使用。

```python
fit_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42,
)
```

| 数据部分 | 完整模式行数 | 用途 |
|---|---:|---|
| 原始训练集的 4/5 | 559,968 | 拟合预处理参数、更新网络权重 |
| 原始训练集的 1/5 | 139,992 | 计算验证指标、选择轮次和提前停止 |
| 比赛测试集 | 299,984 | 最佳模型推理，生成提交文件 |

分层抽样使训练部分和验证部分的欺诈比例接近总体比例。完整运行使用 80% 训练部分训练最佳模型，再预测比赛测试集；不在训练后合并验证集重新拟合。四种方法调用同一个划分函数，固定数据、种子和模式时，验证 ID 与顺序完全相同。

已有 `outputs/mlp/`、`outputs/smoke/` 中的 40% 验证结果保留作为历史记录。新结果放在 `outputs/holdout20/` 中。**旧 40% 与新 20% 的 AUC 不能直接作为同一验证实验比较**，比较脚本会拒绝旧比例。

**所有预处理都在划分之后拟合。**

| 输入 | 训练部分拟合内容 | 验证集 / 测试集处理 |
|---|---|---|
| 数值 | 中位数、均值和标准差 | 使用相同参数；非有限值按缺失值处理，全缺失数值列的填充值为 0 |
| 类别 | 神经网络每列独立词表 | 缺失或未知类别使用编码 0；MLP 中为固定零向量，FT-Transformer / TabNet 由库的 Embedding 处理；CatBoost 使用原始字符串并填充缺失标记 |
| 布尔 | 固定的 0/1 转换规则 | 支持 True/False 和 0/1，缺失值填 0，遇到其他值报错 |

## 4. 环境：使用已有的 Conda agent

本项目使用已有的 **Conda `agent` 环境**，不创建额外的虚拟环境。以下命令从项目目录执行：

```powershell
conda activate agent
cd D:/vscode/file/agent/kaggle-competition/credit-fraud
python --version
```

### RTX 5060 Laptop GPU

本机 GPU 为约 8GB 显存的 RTX 5060 Laptop GPU。RTX 50 系列需要支持其架构的 PyTorch 构建，下面固定使用 PyTorch 2.7.1 的 CUDA 12.8 wheel：

```powershell
python -m pip install "torch==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

PyTorch wheel 自带所需的 CUDA 运行库；运行这份脚本通常无需另行安装完整 CUDA Toolkit。仍需有兼容的 NVIDIA 显卡驱动，可使用 `nvidia-smi` 检查。

### CPU 环境

没有 NVIDIA GPU 时，可以安装 CPU 构建：

```powershell
python -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python train_mlp.py --smoke --device cpu
```

依赖清单包含 Matplotlib、三个模型库及通用数据依赖。当前 `agent` 已有 `torch 2.7.1+cu128`，只需执行 `python -m pip install -r requirements.txt` 即可补齐其余包，无需重新安装 PyTorch。每次运行的依赖版本会写入 `config.json`。

## 5. 运行训练

### 小规模跑通

```powershell
python train_mlp.py --smoke
python train_ft_transformer.py --smoke
python train_tabnet.py --smoke
python train_catboost.py --smoke
```

| 内容 | 小规模模式 |
|---|---|
| 数据抽样 | 从原始训练集按标签分层抽取 20,000 条 |
| 模型训练部分 | 16,000 条 |
| 验证部分 | 4,000 条 |
| 测试记录 | 测试集前 2,000 条，与提交模板前 2,000 条对齐 |
| 训练轮数 | 神经网络最多 2 轮，CatBoost 最多 20 轮（均可提前停止） |
| 默认输出目录 | `outputs/holdout20/smoke/<模型名>/` |
| 预测文件 | `submission_smoke.csv`，用于检查流程 |

小规模指标受数据量和训练轮数影响，不能当作完整训练分数或 Kaggle 排名依据。运行模式会写入配置和指标文件。

### 完整数据训练与提交文件

```powershell
python train_mlp.py
python train_ft_transformer.py
python train_tabnet.py
python train_catboost.py
```

神经网络默认自动使用 CUDA，否则使用 CPU；CatBoost 默认 CPU，也可传 `--device cuda`。完整运行按 80% / 20% 划分原始训练集，每种方法输出 `outputs/holdout20/<模型名>/submission.csv`，共 299,984 行预测。上传时在 Kaggle 比赛页面选择对应文件。

可自定义参数：

```powershell
python train_mlp.py --device cuda --epochs 30 --batch-size 4096 --lr 0.001 --patience 5 --seed 42 --output-dir outputs/experiment_01
python train_catboost.py --device cpu --iterations 1000 --lr 0.05 --patience 5
```

PowerShell 没有激活 Conda 时，也可以使用：

```powershell
conda run -n agent --no-capture-output python train_mlp.py --smoke
```

| 参数 | 默认值 / 说明 |
|---|---|
| `--data-dir` | 脚本旁的 `playground-series-s6e11` 目录 |
| `--output-dir` | 完整模式为 `outputs/holdout20/<模型名>`，小规模模式在 `holdout20/smoke/<模型名>` 下 |
| `--epochs` | 神经网络 30；`--smoke` 下上限为 2；CatBoost 使用 `--iterations` |
| `--iterations` | 仅 CatBoost，默认 1000；`--smoke` 下上限为 20 |
| `--batch-size` | MLP 4096，FT-Transformer 512，TabNet 1024；CatBoost 无此参数 |
| `--lr` | MLP 0.001，FT-Transformer 0.0001，TabNet 0.02，CatBoost 0.05 |
| `--patience` | 5 |
| `--seed` | 42 |
| `--device` | `auto`、`cpu`、`cuda`；神经网络默认 auto，CatBoost 默认 cpu；显式请求不可用的 CUDA 会报错 |
| `--smoke` | 启用小规模验证 |

不传路径参数时，默认目录相对于脚本位置解析，可以从其他工作目录启动。手动传入的相对路径按当前工作目录解析。同一输出目录再次运行会覆盖其中同名产物，比较实验时请指定不同目录。

### 输出文件

| 文件 | 内容 |
|---|---|
| `config.json` | 数据路径、划分比例、实际行数、正类比例、特征顺序、超参数、运行设备与依赖版本 |
| `best_model.pt` | MLP / FT-Transformer 的最佳权重、预处理参数、配置与最佳轮次 |
| `best_model.zip` | TabNet 的原生模型保存格式 |
| `best_model.cbm` | CatBoost 的原生模型保存格式 |
| `preprocessor.json` | 当前模型仅用训练部分拟合的预处理参数 |
| `metrics.json` | 每轮损失与验证 AUC、最终验证 AP、最佳轮次和总耗时；神经网络还记录每轮 AP，MLP / FT 记录每轮耗时 |
| `validation_predictions.csv` | 验证集 `id,is_fraud,predicted_probability`，用于重新计算指标和模型对比 |
| `roc_curve.png` | 最佳模型在验证集上的 ROC；小规模模式标题标注 SMOKE |
| `submission.csv` | 完整模式提交文件，列名为 `id,is_fraud` |
| `submission_smoke.csv` | 小规模模式的预测文件 |

训练结束后，脚本会重新加载保存的最佳模型和预处理参数，再生成预测。提交时检查文件行数、列名和 ID 顺序；脚本也会检查模板对齐和概率有限性。

### 多模型 ROC 对比

先完成上面四条小规模训练命令，然后运行下方单行命令：

```powershell
python compare_roc.py --run-dirs outputs/holdout20/smoke/mlp outputs/holdout20/smoke/ft_transformer outputs/holdout20/smoke/tabnet outputs/holdout20/smoke/catboost --output-dir outputs/holdout20/smoke/comparison
```

完整运行后的比较命令：

```powershell
python compare_roc.py --run-dirs outputs/holdout20/mlp outputs/holdout20/ft_transformer outputs/holdout20/tabnet outputs/holdout20/catboost --output-dir outputs/holdout20/comparison
```

比较脚本检查验证比例、种子、运行模式、验证 ID 及真实标签。CSV 行序可以不同，脚本按 ID 对齐；验证样本或标签不同则拒绝比较。输出 `roc_comparison.png` 和按 AUC 降序排列的 `comparison.csv`，表中包含 AUC、AP、行数及模式。

| ROC 图元素 | 含义 |
|---|---|
| 横轴 FPR | 正常交易中被误报为欺诈的比例 |
| 纵轴 TPR | 欺诈交易中被检出的比例，也称召回率 |
| 曲线上的点 | 改变概率判断阈值得到的不同误报/检出组合 |
| 灰色对角线 | 随机排序参照，AUC=0.5 |
| 图例 AUC | 曲线下面积；越高表示整体排序能力越强，不代表实际概率已校准 |
| SMOKE 标记 | 仅用于跑通流程的小样本、少轮次实验 |

绘图使用无窗口的 Matplotlib Agg 后端，图片保存为 PNG，无需单独执行显示命令。图中使用英文坐标避免依赖中文字体；缓存写入结果目录的 `.matplotlib/`。

### 已完成的小规模验证

本机 Conda `agent`、固定种子 42、16,000/4,000 划分；三种神经网络各 2 轮，CatBoost 最多 20 轮并在第 14 轮结束、选择第 9 轮模型：

| 模型 | 验证 ROC-AUC | 验证 AP | 运行设备 |
|---|---:|---:|---|
| MLP | 0.855346 | 0.264491 | CUDA |
| FT-Transformer | 0.893143 | 0.198165 | CUDA |
| TabNet | 0.747487 | 0.129005 | CUDA |
| CatBoost | 0.927715 | 0.399592 | CPU |

四种方法训练预算和收敛速度不同，这些分数用于验证实现，**不能用来确定完整训练后的模型优劣**。AUC 与 AP 的排序也可能不同。实际结果以本次运行的 `metrics.json` 为准。

复核时先查看各目录的 `config.json`，确认 `validation_size=0.2`、`fit_rows=16000`、`validation_rows=4000`，再运行上面的比较命令；它会根据保存的验证概率重新计算 AUC/AP 并检查样本一致性。

## 6. 下次直接使用 KaggleHub 获取数据

### 先加入比赛并完成认证

在网页中登录 Kaggle，打开比赛页面，加入比赛并接受规则。比赛数据的下载权限由 Kaggle 账号及比赛规则决定。

`kagglehub` 已包含在本项目依赖中，也可以单独安装：

```powershell
python -m pip install --upgrade kagglehub
```

在 Python 或 Notebook 中，首次需要认证时执行：

```python
import kagglehub

kagglehub.login()
```

按提示输入从 [Kaggle API 设置](https://www.kaggle.com/settings/api) 获取的 API token。也可在当前 PowerShell 会话中设置环境变量：

```powershell
$env:KAGGLE_API_TOKEN = "替换为你的 token"
```

已有旧版认证配置时，也可使用用户目录下的 `~/.kaggle/kaggle.json`。令牌属于账号凭据，放在用户配置或环境变量中，避免写入项目代码。

### 下载本次比赛并读取三个 CSV

```python
from pathlib import Path

import kagglehub
import pandas as pd

data_dir = Path(
    kagglehub.competition_download("playground-series-s6e11")
)

train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

print("数据目录：", data_dir)
print(train.shape, test.shape, sample_submission.shape)
```

下载函数返回 **本地文件夹路径字符串**，pandas 再从该目录读取文件。默认由 KaggleHub 管理本地缓存，重复调用通常会复用已有文件，无需每次手动下载和解压 ZIP。

将打印出的实际目录传给脚本：

```powershell
python train_mlp.py --data-dir "上一步打印的数据目录" --smoke
```

### 只下载单个文件或主动刷新

```python
import kagglehub
import pandas as pd

# 下载单个文件时，返回的是该文件的本地路径。
train_path = kagglehub.competition_download(
    "playground-series-s6e11",
    path="train.csv",
)
train = pd.read_csv(train_path)

# 需要重新获取数据时使用；平常保留默认缓存即可。
data_dir = kagglehub.competition_download(
    "playground-series-s6e11",
    force_download=True,
)
```

| 遇到的问题 | 检查方向 |
|---|---|
| 401 / 认证失败 | 确认 token 有效，以及 Python 使用的认证配置 |
| 403 / 无权限 | 确认当前账号已加入比赛、接受规则，并有数据访问权限 |
| 404 / 找不到比赛 | 检查比赛标识和账号访问权限 |
| 下载后找不到 CSV | 使用函数实际返回的路径；整场下载返回目录，单文件下载返回文件路径 |

官方参考：[KaggleHub README](https://github.com/Kaggle/kagglehub#readme) · [比赛下载示例](https://github.com/Kaggle/kagglehub#download-competition)

## 7. 后续练习路线

| 顺序 | 实验 | 如何比较 |
|---|---|---|
| 1 | 完整运行四种模型 | 记录固定 20% 验证集的 AUC、AP、最佳轮次、ROC 与耗时 |
| 2 | 数值与时间特征处理 | 尝试金额的 `log1p`、小时与星期的周期编码，使用同一划分比较 |
| 3 | 类别表示与正则化 | 比较 Embedding 与独热编码，调整宽度和 Dropout，每次改变一个主要因素 |
| 4 | 不平衡训练策略 | 比较类别权重、重采样或 Focal Loss，同时查看 AUC 与 AP |
| 5 | 调整四种模型的结构和优化参数 | 保持相同的数据划分，比较收益和计算成本 |
| 6 | 多随机种子与分层交叉验证 | 检查提升是否稳定；采用新验证方案时，为各模型重新建立可比结果 |
| 7 | 概率融合 | 在共同的验证样本或 OOF 预测上选融合权重，再应用于测试集 |

方法阅读：[MLP 与 FT-Transformer 的表格学习研究](https://arxiv.org/abs/2106.11959) · [TabNet](https://arxiv.org/abs/1908.07442) · [CatBoost 文档](https://catboost.ai/docs/) · [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) · [Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

## 8. 没接触过 Python，如何读这份代码

先运行命令观察产物，再沿着下表顺序阅读。代码中的三引号文字叫“文档字符串”，通常写在文件、类或函数开头，解释用途、参数和返回值；`#` 后面是注释，不参与计算。

| 顺序 | 阅读位置 | 需要理解的内容 |
|---|---|---|
| 1 | `train_mlp.py` 的 `main` | 获取终端参数，指定网络和配置，调用公共训练流程 |
| 2 | `tabular_common.py` 的 `parse_training_args`、`prepare_data`、`split_frames` | 如何将 CSV 读成表，如何确保四种方法使用相同划分 |
| 3 | `Preprocessor.fit` / `transform` | fit 学习训练统计量，transform 使用已有参数处理验证/测试 |
| 4 | `FraudMLP.__init__` / `forward` | 网络由哪些层组成，输入矩阵如何变成每行一个分数 |
| 5 | `neural_training.py` 的 `run_neural` | 批次、损失、梯度、权重更新、验证及早停 |
| 6 | `predict` 与 `save_results` | 如何将分数转为概率，恢复最佳模型并输出 CSV 和 ROC |
| 7 | 另外三个训练入口 | FT-Transformer 的字段注意力、TabNet 的特征选择、CatBoost 的树模型 |
| 8 | `compare_roc.py` | 如何检查样本一致性并对比模型 |

| Python 写法 | 在本项目中的意思 |
|---|---|
| `def fit(...):` | 定义名为 fit 的函数，括号里是传入参数，冒号后缩进的内容属于函数 |
| `seed: int = 42` | 参数名 seed，预期整数，省略时采用默认值 42 |
| `-> np.ndarray` | 返回类型提示：函数返回 NumPy 数组 |
| `return result` | 把计算结果交还给调用这个函数的位置 |
| `None` / `True` / `False` | 没有值 / 是 / 否，Python 区分大小写 |
| `class Preprocessor` | 定义预处理器类型；`prep = Preprocessor.fit(...)` 创建一个记住统计量的对象 |
| `self.medians` | 当前预处理器对象保存的中位数，每个对象拥有自己的值 |
| `@classmethod` | 可通过类名调用的方法，例如 `Preprocessor.fit(data.fit)` |
| `@property` | 像属性一样读取方法结果，例如 `prep.category_sizes`，不加括号 |
| `frame["id"]` / `frame.iloc[indices]` | 从表中取 id 列 / 按行位置取若干行 |
| `[:, 1]` | 二维数组中取所有行的第 2 列；Python 编号从 0 起 |
| `[... for ... in ...]` | 列表推导式：对每一项进行同样处理，并收集为列表 |
| `with torch.no_grad():` | 在该缩进代码段中不记录梯度，例如只设置初始偏置 |
| `if __name__ == "__main__":` | 从终端直接运行此文件时执行下面代码；被其他文件导入时不执行 |

例如命令 `python train_mlp.py --smoke --batch-size 1024` 表示：用 Python 执行 MLP 脚本，启用小规模模式，每批最多送入 1,024 条交易；它不是将训练集缩成 1,024 条。一个 epoch 是把所有训练记录遍历一次。

可以在终端查看函数解释：

```powershell
python -c "from tabular_common import split_frames; help(split_frames)"
python train_mlp.py --help
```

修改实验时优先使用命令行参数。不要把验证集传入 `Preprocessor.fit`，也不要把目标列或 ID 加入输入列表，否则会破坏评估约定。
