"""四种模型共用的工具箱；无需直接运行这个文件。

建议先读 train_mlp.py，遇到同名函数再回到这里查看。
主流程：命令参数 → 读取 CSV → 80%/20% 划分 → 训练部分拟合预处理 →
训练模型（在其他文件中）→ 保存指标、ROC 和提交 CSV。

Python 入门提示：
  def 定义函数；class 定义一类对象；return 把结果交给调用者。
  frame 是 pandas 的 DataFrame，可视为一张有行名/列名的 Excel 表。
  np.ndarray 是 NumPy 数组，用于数值计算；Path 是文件/目录路径对象。
  冒号后的类型标注（如 seed: int）帮助阅读，不是给变量赋值。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# __file__ 是当前代码文件的路径，parent 取所在目录，避免依赖终端启动位置。
PROJECT_DIR = Path(__file__).resolve().parent
TARGET = "is_fraud"
VALIDATION_SIZE = 0.2
# 大写变量表示全项目共用的约定。以下列表控制输入字段及其顺序。
NUMERIC_FEATURES = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
CATEGORICAL_FEATURES = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
]
BOOLEAN_FEATURES = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
MODEL_LABELS = {
    "mlp": "Embedding + MLP", "ft_transformer": "FT-Transformer",
    "tabnet": "TabNet", "catboost": "CatBoost",
}


def parse_training_args(
    model: str, *, batch_size: int = 4096, lr: float = 1e-3,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """把终端命令转为配置对象，并检查参数是否合法。

    参数：
        model：模型标识，例如 "mlp"，决定默认目录及可用选项。
        batch_size、lr：该模型默认批大小、学习率，可被命令参数覆盖。
        argv：供程序或测试传入参数列表；None 表示读取当前终端命令。
    返回：Namespace 对象，可以用 args.seed、args.data_dir 等读取配置。
    用法：parse_training_args("mlp", argv=["--smoke", "--device", "cpu"])。
    """
    parser = argparse.ArgumentParser(
        description=f"Train {MODEL_LABELS[model]} with a stratified 80%/20% holdout."
    )
    parser.add_argument("--data-dir", type=Path,
                        default=PROJECT_DIR / "playground-series-s6e11")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        default="cpu" if model == "catboost" else "auto")
    parser.add_argument("--smoke", action="store_true",
                        help="Use <=20,000 train rows / 2,000 test rows; <=2 neural epochs or 20 trees.")
    parser.add_argument("--lr", type=float, default=lr)
    if model == "catboost":
        parser.add_argument("--iterations", type=int, default=1000)
    else:
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--batch-size", type=int, default=batch_size)
    args = parser.parse_args(argv)
    for key in ["patience", "iterations"] if model == "catboost" else [
        "patience", "epochs", "batch_size",
    ]:
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if model == "tabnet" and args.batch_size < 2:
        parser.error("TabNet BatchNorm needs --batch-size >= 2")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.smoke:
        if model == "catboost":
            args.iterations = min(args.iterations, 20)
        else:
            args.epochs = min(args.epochs, 2)
    if args.output_dir is None:
        root = PROJECT_DIR / "outputs" / "holdout20"
        args.output_dir = root / "smoke" / model if args.smoke else root / model
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def seed_everything(seed: int) -> None:
    """设置随机数起点；seed 是整数，通常为 42；函数不返回结果。

    Python、NumPy、PyTorch 都有各自的随机数生成器，因此分别设置。
    固定种子保证数据划分可重现，不保证不同 GPU/库版本的计算逐位相同。
    用法：seed_everything(42)。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def resolve_device(requested: str) -> torch.device:
    """选择计算设备，返回供张量和模型使用的 torch.device 对象。

    requested 可为 "cpu"、"cuda" 或 "auto"；auto 有可用 GPU 就使用 GPU。
    用户明确请求 cuda 但当前环境不支持时抛出 ValueError，不静默换成 CPU。
    用法：device = resolve_device("auto")。
    """
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Check the PyTorch CUDA build/driver or use --device cpu.")
    device = torch.device(
        "cuda" if requested == "cuda" or (
            requested == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}", flush=True)
    return device


def read_data(data_dir: Path, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从一个文件夹读取训练、测试和提交模板三张表，并检查基本格式。

    参数：data_dir 为包含三个 CSV 的目录；smoke=True 时测试/模板仅读前 2,000 行。
    返回：(train, test, sample)，三个 DataFrame，可用多变量同时接收。
    标签、ID 或字段有误时抛出 ValueError，中止错误数据上的训练。
    此处仍读取完整训练表，后续才能从全体记录中进行分层抽样。
    """
    paths = [data_dir / name for name in ["train.csv", "test.csv", "sample_submission.csv"]]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Required file not found: {path}")
    train = pd.read_csv(paths[0])
    test = pd.read_csv(paths[1], nrows=2000 if smoke else None)
    sample = pd.read_csv(paths[2], nrows=2000 if smoke else None)
    for name, frame, required in [
        ("train.csv", train, ["id", TARGET] + FEATURES),
        ("test.csv", test, ["id"] + FEATURES),
    ]:
        missing = sorted(set(required) - set(frame.columns))
        if missing or frame.empty:
            raise ValueError(f"{name}: missing columns {missing} or empty data")
        if frame["id"].isna().any() or not frame["id"].is_unique:
            raise ValueError(f"{name}: IDs must be non-null and unique")
    if list(sample.columns) != ["id", TARGET]:
        raise ValueError(f"sample_submission.csv must have columns: id,{TARGET}")
    if not test["id"].equals(sample["id"]):
        raise ValueError("test.csv and sample_submission.csv IDs/order do not match")
    if train["id"].isin(test["id"]).any():
        raise ValueError("train.csv and test.csv contain overlapping IDs")
    labels = pd.to_numeric(train[TARGET], errors="raise")
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError(f"{TARGET} must contain both binary labels 0 and 1")
    if labels.value_counts().min() < 2:
        raise ValueError("Each class needs at least two rows for stratified splitting")
    train[TARGET] = labels.astype(np.float32)
    return train, test, sample


def split_frames(train: pd.DataFrame, seed: int, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按欺诈标签分层，返回 (80% 训练表, 20% 验证表)。

    train 需含 id 与 is_fraud；seed 固定抽样结果。
    smoke=True 时先从全体记录中分层抽取至多 20,000 行，再划分。
    stratify=train[TARGET] 表示按标签比例抽样；iloc[...] 按行位置取子表。
    四种模型必须调用本函数，才能让比较使用同一批验证交易。
    """
    # 所有模型共用同一抽样顺序；分层抽样让少数类在两部分中保持相近比例。
    if smoke and len(train) > 20_000:
        chosen, _ = train_test_split(
            np.arange(len(train)), train_size=20_000,
            stratify=train[TARGET], random_state=seed,
        )
        train = train.iloc[chosen].reset_index(drop=True)
    fit_indices, val_indices = train_test_split(
        np.arange(len(train)), test_size=VALIDATION_SIZE,
        stratify=train[TARGET], random_state=seed,
    )
    fit_frame, val_frame = train.iloc[fit_indices], train.iloc[val_indices]
    if any(frame[TARGET].nunique() != 2 for frame in [fit_frame, val_frame]):
        raise ValueError("Both train and validation must contain both classes; provide more rows.")
    return fit_frame, val_frame


@dataclass
class DataSplit:
    """保存一次运行的五项数据；@dataclass 自动生成初始化方法。

    fit：参与训练的表；validation：评估用表；test：比赛无标签测试表；
    sample：提交模板；source_rows：抽样前训练表的行数。
    例如 data.fit 表示从 data 对象中读取 fit 属性。
    """
    fit: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    source_rows: int


def prepare_data(args: argparse.Namespace) -> DataSplit:
    """根据命令配置读表并划分，返回集中存放数据的 DataSplit 对象。

    args 来自 parse_training_args，需要有 data_dir、seed、smoke 属性。
    用法：data = prepare_data(args)，之后用 data.validation 访问验证表。
    """
    train, test, sample = read_data(args.data_dir, args.smoke)
    fit, val = split_frames(train, args.seed, args.smoke)
    print(
        f"Rows: train={len(fit):,}, validation={len(val):,}, test={len(test):,}"
        f" | fraud rates: {fit[TARGET].mean():.4%} / {val[TARGET].mean():.4%}",
        flush=True,
    )
    return DataSplit(fit, val, test, sample, len(train))


def numeric_values(frame: pd.DataFrame) -> pd.DataFrame:
    """取出 13 个数值字段，返回数值表，将正负无穷转换为缺失值 NaN。

    frame 为原始表，原表不被修改；缺失值由后续训练部分的中位数填充。
    """
    return frame[NUMERIC_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan)


def boolean_values(frame: pd.DataFrame) -> np.ndarray:
    """将六个布尔字段转换为形状 (记录数, 6) 的浮点数组。

    接受 True/False、0/1 及其字符串，缺失填 0；其他内容会报错。
    例如 True 转为 1.0，False 转为 0.0；column_stack 将六列拼成矩阵。
    """
    result = []
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    for column in BOOLEAN_FEATURES:
        values = frame[column].astype("string").str.strip().str.lower()
        encoded = values.map(mapping)
        invalid = values.notna() & encoded.isna()
        if invalid.any():
            raise ValueError(f"{column}: unexpected boolean value {values[invalid].iloc[0]!r}")
        result.append(encoded.fillna(0).to_numpy(dtype=np.float32))
    return np.column_stack(result)


@dataclass
class Preprocessor:
    """神经网络预处理器：记住训练统计量，并原样应用到其他数据。

    medians 为 13 列的缺失填充值，means/scales 为标准化参数；
    vocabularies 是“字段名 → 类别字符串 → 整数编号”的字典。
    只保存普通 Python 值，方便 JSON 与 weights_only=True 检查点加载。
    用法：prep = Preprocessor.fit(data.fit)；dense, cats = prep.transform(data.test)。
    """

    medians: list[float]
    means: list[float]
    scales: list[float]
    vocabularies: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> Preprocessor:
        """仅根据训练部分 frame 计算统计量，返回一个新的预处理器。

        @classmethod 使调用方式为 Preprocessor.fit(...)；cls 代表当前类。
        fit 意为“从数据学习参数”；这里学习中位数、标准化参数及类别词表。
        """
        # 调用者只能传入 80% 训练部分，防止验证集的分布信息进入预处理。
        values = numeric_values(frame)
        medians = values.median().fillna(0.0)
        scaler = StandardScaler().fit(values.fillna(medians).to_numpy())
        vocabularies = {}
        for column in CATEGORICAL_FEATURES:
            levels = sorted(frame[column].astype("string").dropna().unique().tolist())
            # 0 保留给缺失和未见类别，词表从 1 开始。
            vocabularies[column] = {level: i + 1 for i, level in enumerate(levels)}
        return cls(medians.tolist(), scaler.mean_.tolist(), scaler.scale_.tolist(), vocabularies)

    @property
    def category_sizes(self) -> list[int]:
        """返回每个类别字段需要的编号数量，包含未知类别编码 0。

        @property 允许写 prep.category_sizes，无需添加调用括号。
        例如某字段有 12 个已知类别，这个字段的 Embedding 表需要 13 行。
        """
        return [len(self.vocabularies[c]) + 1 for c in CATEGORICAL_FEATURES]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """用已保存的参数处理 frame，返回 (dense, categories) 两个数组。

        dense 形状为 (记录数, 19)，含标准化数值及布尔值；
        categories 形状为 (记录数, 5)，含类别编号。
        transform 不重新计算统计量，因此验证/测试不会影响训练参数。
        self 代表当前预处理器，self.medians 即它记住的中位数。
        """
        values = numeric_values(frame).fillna(pd.Series(self.medians, index=NUMERIC_FEATURES))
        scaled = (values.to_numpy() - np.asarray(self.means)) / np.asarray(self.scales)
        dense = np.column_stack([scaled, boolean_values(frame)]).astype(np.float32)
        categories = np.column_stack([
            frame[c].astype("string").map(self.vocabularies[c])
            .fillna(0).to_numpy(dtype=np.int64) for c in CATEGORICAL_FEATURES
        ])
        if not np.isfinite(dense).all():
            raise ValueError("Non-finite values remain after numerical preprocessing")
        return np.ascontiguousarray(dense), np.ascontiguousarray(categories)

    def tabnet_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """把 transform 的两组数组拼成 (记录数, 24) 矩阵供 TabNet 使用。

        前 19 列数值/布尔，后 5 列类别编号；编号转为浮点存储但仍表示类别。
        用法：x_train = prep.tabnet_matrix(data.fit)。
        """
        dense, categories = self.transform(frame)
        # TabNet 接受一个矩阵，用 cat_idxs 标明哪些列是类别编码。
        return np.column_stack([dense, categories]).astype(np.float32)


@dataclass
class CatBoostPreprocessor:
    """CatBoost 专用预处理器，仅记住训练部分的数值中位数。

    类别由 CatBoost 自己处理，因此无需拟合类别词表，也无需标准化。
    """
    medians: list[float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> CatBoostPreprocessor:
        """从训练表 frame 计算填充中位数，返回新的 CatBoost 预处理器。"""
        return cls(numeric_values(frame).median().fillna(0.0).tolist())

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """返回含 24 个输入字段的新表：填数值缺失、编码布尔、保留类别字符串。

        参数 frame 可为训练、验证或测试表；结果不含 ID 和目标，列序固定。
        用法：x_val = prep.transform(data.validation)。
        """
        # 树模型无需标准化；保留类别字符串，让 CatBoost 原生处理类别交互。
        result = numeric_values(frame).fillna(pd.Series(self.medians, index=NUMERIC_FEATURES))
        for column in CATEGORICAL_FEATURES:
            result[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
        result[BOOLEAN_FEATURES] = boolean_values(frame)
        return result[FEATURES]


def write_json(path: Path, content: dict) -> None:
    """将字典 content 以 UTF-8 写入 path；父目录需已存在，不返回值。

    JSON 是可阅读的配置格式；禁止 NaN/Infinity，避免保存无效指标。
    用法：write_json(output_dir / "config.json", config)。
    """
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    """读取 UTF-8 JSON 文件 path，返回 Python 字典；与 write_json 配套。"""
    return json.loads(path.read_text(encoding="utf-8"))


def make_config(
    args: argparse.Namespace, data: DataSplit, model: str,
    device: torch.device, parameters: dict,
) -> dict:
    """汇总并写出本次运行配置，返回同一份配置字典。

    args：终端参数；data：划分后的数据；model：模型标识；
    device：实际计算设备；parameters：模型特有的超参数字典。
    自动建立结果目录，记录行数和依赖版本，便于复现实验。
    """
    versions = {"python": sys.version.split()[0]}
    for package in [
        "torch", "numpy", "pandas", "scikit-learn", "matplotlib",
        "rtdl-revisiting-models", "pytorch-tabnet", "catboost",
    ]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    config = {
        **vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir),
        "model": model, "model_label": MODEL_LABELS[model], "device": str(device),
        "validation_size": VALIDATION_SIZE, "source_train_rows": data.source_rows,
        "used_train_rows": len(data.fit) + len(data.validation),
        "fit_rows": len(data.fit), "validation_rows": len(data.validation),
        "test_rows": len(data.test),
        "fit_positive_rate": float(data.fit[TARGET].mean()),
        "validation_positive_rate": float(data.validation[TARGET].mean()),
        "numeric_features": NUMERIC_FEATURES, "boolean_features": BOOLEAN_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES, "parameters": parameters,
        "versions": versions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", config)
    return config


def validate_probabilities(values: np.ndarray, expected_rows: int) -> np.ndarray:
    """验证每行恰有一个概率，返回统一为 float64 的一维数组。

    values 为预测数组；expected_rows 为应有记录数。
    长度错误、NaN、无穷或超出 [0,1] 都抛出 ValueError。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (expected_rows,):
        raise ValueError(f"Expected {expected_rows} probabilities, got shape {values.shape}")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    return values


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """用真实标签 labels 和连续预测 probabilities 计算 AUC 与 AP。

    两个数组必须按相同记录顺序对应，labels 必须含 0、1 两类。
    返回 {"roc_auc": 浮点数, "average_precision": 浮点数}。
    """
    validate_probabilities(probabilities, len(labels))
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def plot_roc(
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    destination: Path, *, smoke: bool, comparison: bool = False,
) -> None:
    """把一条或多条验证 ROC 保存为 PNG，不打开窗口，也不返回值。

    curves：列表，每项为 (显示名称, 真实标签数组, 预测概率数组)。
    destination：图片保存路径；smoke=True 在标题标注小规模实验；
    comparison=True 使用多模型对比标题。星号 * 表示后两项需按名称传入。
    用法：plot_roc([("MLP", y, p)], output / "roc.png", smoke=True)。
    """
    # Agg 不打开 GUI，终端或无桌面环境也能生成图片；英文图注避免字体依赖。
    os.environ.setdefault("MPLCONFIGDIR", str(destination.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, targets, probabilities in curves:
        auc = classification_metrics(targets, probabilities)["roc_auc"]
        # 使用连续概率改变阈值得到 ROC，不先将输出转换为 0/1。
        fpr, tpr, _ = roc_curve(targets, probabilities)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random (AUC=0.5)")
    title = "Validation ROC comparison" if comparison else "Validation ROC"
    ax.set(
        xlabel="False positive rate (FPR)", ylabel="True positive rate (TPR)",
        xlim=(0, 1), ylim=(0, 1.02), title=f"{title}{' [SMOKE]' if smoke else ''}",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    try:
        fig.savefig(destination, dpi=180)
    finally:
        plt.close(fig)


def save_results(
    args: argparse.Namespace, data: DataSplit, config: dict,
    val_probabilities: np.ndarray, test_probabilities: np.ndarray,
    *, best_epoch: int, history: list[dict], elapsed_seconds: float,
    model_file: str,
) -> dict:
    """保存最佳模型的验证/测试预测、ROC、指标，返回指标字典。

    args、data、config 分别是命令参数、数据和运行配置。
    val_probabilities/test_probabilities 是重新加载最佳模型后的概率，
    顺序必须与 data.validation/data.test 一致。
    best_epoch 从 1 起计；history 为每轮记录列表；elapsed_seconds 为
    数据准备、训练和推理的累计耗时（不含本函数绘图时间）；
    model_file 是已经保存的模型文件名。
    """
    val_probabilities = validate_probabilities(val_probabilities, len(data.validation))
    test_probabilities = validate_probabilities(test_probabilities, len(data.test))
    # 指标、ROC 和 CSV 全部来自重新加载后的同一个最佳模型。
    scores = classification_metrics(data.validation[TARGET].to_numpy(), val_probabilities)
    predictions = data.validation[["id", TARGET]].copy()
    predictions["predicted_probability"] = val_probabilities
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    submission = data.sample[["id"]].copy()
    submission[TARGET] = test_probabilities
    name = "submission_smoke.csv" if args.smoke else "submission.csv"
    submission.to_csv(args.output_dir / name, index=False)
    plot_roc(
        [(config["model_label"], data.validation[TARGET].to_numpy(), val_probabilities)],
        args.output_dir / "roc_curve.png", smoke=args.smoke,
    )
    metrics = {
        "model": config["model"], "smoke": args.smoke,
        "validation_size": VALIDATION_SIZE, "best_epoch": best_epoch,
        "best_validation_roc_auc": scores["roc_auc"],
        "best_validation_average_precision": scores["average_precision"],
        "fit_rows": len(data.fit), "validation_rows": len(data.validation),
        "test_rows": len(data.test), "history": history,
        "elapsed_seconds": elapsed_seconds, "model_file": model_file,
        "submission_file": name, "validation_predictions_file": "validation_predictions.csv",
        "roc_file": "roc_curve.png",
    }
    write_json(args.output_dir / "metrics.json", metrics)
    print(f"Best epoch/iteration: {best_epoch} | AUC={scores['roc_auc']:.6f}"
          f" | AP={scores['average_precision']:.6f}", flush=True)
    print(f"Saved results: {args.output_dir}", flush=True)
    if args.smoke:
        print("SMOKE: limited data/iterations; CSV is not a full competition submission.", flush=True)
    return metrics


def save_preprocessor(path: Path, preprocessor: Preprocessor | CatBoostPreprocessor) -> None:
    """把预处理器的字段转换为普通字典并存为 JSON，便于后续恢复。

    path 是文件路径；preprocessor 是两种预处理器之一，| 表示“或者”。
    """
    write_json(path, asdict(preprocessor))


def entrypoint(main: Callable[[], None]) -> None:
    """调用脚本的 main 函数，将常见错误转换为简明终端提示。

    参数 main 是“函数本身”，传入时不加括号；本函数中 main() 才执行它。
    Callable[[], None] 表示该函数不接收参数、也没有返回结果。
    成功正常结束；失败以状态码 1 退出，供终端或自动化程序判断。
    """
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run python -m pip install -r requirements.txt",
              file=sys.stderr)
        sys.exit(1)
