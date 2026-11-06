"""校验验证集一致性后，绘制多个模型的 ROC 并导出指标表。

本文件不训练模型，仅读取已生成的 config.json 和 validation_predictions.csv。
用法：python compare_roc.py --run-dirs 运行目录A 运行目录B --output-dir 对比目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_common import (
    PROJECT_DIR, TARGET, VALIDATION_SIZE, classification_metrics, entrypoint,
    plot_roc, read_json, validate_probabilities,
)


def load_run(directory: Path) -> tuple[dict, pd.DataFrame]:
    """读取一个结果目录，返回 (配置字典, 按 ID 排序的验证预测表)。

    directory 为训练输出目录；检查列名、行数、标签、概率和验证比例。
    若目录来自旧 40% 划分或存在无效内容，抛出 ValueError，避免错误比较。
    """
    config = read_json(directory / "config.json")
    if config.get("validation_size") != VALIDATION_SIZE:
        raise ValueError(f"{directory}: expected validation_size=0.2; old 40% runs cannot be compared.")
    for key in ["model", "seed", "smoke", "validation_rows"]:
        if key not in config:
            raise ValueError(f"{directory}: config missing {key}")
    predictions = pd.read_csv(directory / "validation_predictions.csv")
    if list(predictions.columns) != ["id", TARGET, "predicted_probability"]:
        raise ValueError(f"{directory}: invalid validation prediction columns")
    if predictions.empty or len(predictions) != config["validation_rows"]:
        raise ValueError(f"{directory}: validation row count does not match config")
    if predictions["id"].isna().any() or not predictions["id"].is_unique:
        raise ValueError(f"{directory}: validation IDs must be non-null and unique")
    if predictions[TARGET].isna().any() or set(predictions[TARGET].unique()) != {0, 1}:
        raise ValueError(f"{directory}: invalid validation labels")
    validate_probabilities(predictions["predicted_probability"].to_numpy(), len(predictions))
    # 先检查唯一性，再排序；避免不同 CSV 行序造成预测和标签错位。
    return config, predictions.sort_values("id").reset_index(drop=True)


def compare_runs(run_dirs: list[Path], output_dir: Path) -> pd.DataFrame:
    """将多个结果目录作公平对比，返回按 AUC 降序排列的指标表。

    run_dirs 至少两个；output_dir 接收 roc_comparison.png 与 comparison.csv。
    比较必须满足种子、验证比例、模式、验证 ID 和标签一致；
    允许同样记录以不同顺序保存在 CSV 中，因为本函数会先按 ID 对齐。
    用法：compare_runs([Path("目录A"), Path("目录B")], Path("对比目录"))。
    """
    if len(run_dirs) < 2:
        raise ValueError("Provide at least two run directories")
    reference_config = None
    reference = None
    curves, rows, labels_seen = [], [], set()
    for directory in run_dirs:
        config, predictions = load_run(directory)
        if reference is None:
            reference_config, reference = config, predictions
        else:
            for key in ["validation_size", "seed", "smoke"]:
                if config[key] != reference_config[key]:
                    raise ValueError(f"{directory}: mismatched {key}")
            if not predictions["id"].equals(reference["id"]):
                raise ValueError(f"{directory}: validation IDs do not match")
            if not np.array_equal(predictions[TARGET], reference[TARGET]):
                raise ValueError(f"{directory}: validation labels do not match")
        label = config.get("model_label", config["model"])
        if label in labels_seen:
            label = f"{label} ({directory.name})"
        if label in labels_seen:
            raise ValueError("Duplicate run labels; use distinct run directories")
        labels_seen.add(label)
        targets = predictions[TARGET].to_numpy()
        probabilities = predictions["predicted_probability"].to_numpy()
        scores = classification_metrics(targets, probabilities)
        curves.append((label, targets, probabilities))
        rows.append({
            "model": config["model"], "label": label, **scores,
            "validation_rows": len(predictions), "smoke": config["smoke"],
            "validation_size": config["validation_size"], "seed": config["seed"],
            "run_dir": str(directory.resolve()),
        })
    # 全部输入通过校验后才写输出，避免产生不完整的对比结果。
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(curves, output_dir / "roc_comparison.png",
             smoke=reference_config["smoke"], comparison=True)
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    table.to_csv(output_dir / "comparison.csv", index=False)
    return table


def main() -> None:
    """读取 --run-dirs 与 --output-dir，生成图表并在终端显示指标。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_DIR / "outputs" / "holdout20" / "comparison")
    args = parser.parse_args()
    table = compare_runs(args.run_dirs, args.output_dir)
    print(table[["label", "roc_auc", "average_precision", "validation_rows", "smoke"]]
          .to_string(index=False))
    print(f"Saved comparison: {args.output_dir.resolve()}")


if __name__ == "__main__":
    # main 不加括号是把函数传给错误处理器，由错误处理器调用它。
    entrypoint(main)
