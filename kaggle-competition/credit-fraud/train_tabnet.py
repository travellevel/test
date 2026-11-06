"""TabNet：通过多个决策步骤的稀疏特征选择完成二分类。

用法：python train_tabnet.py --smoke；完整训练省略 --smoke。
主要步骤：公共数据准备 → 类别列位置标记 → 库内训练与早停 →
保存/加载最佳模型 → 验证和测试概率 → 公共 ROC 输出。
"""

from __future__ import annotations

import time

import numpy as np
import torch

from tabular_common import (
    BOOLEAN_FEATURES, NUMERIC_FEATURES, TARGET, Preprocessor,
    entrypoint, make_config, parse_training_args, prepare_data, read_json,
    resolve_device, save_preprocessor, save_results, seed_everything,
)


def main() -> None:
    """独立的 TabNet 执行入口，无参数/返回值，参数由终端读取。

    n_d：每个决策步骤的输出维度；n_a：选择特征时的表示维度；
    n_steps：决策步骤数；lambda_sparse：稀疏选择正则项强度。
    训练由第三方库负责，下面主要配置输入、优化和评估约定。
    """
    from pytorch_tabnet.metrics import Metric
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.metrics import average_precision_score

    class AveragePrecision(Metric):
        """让 TabNet 的评估系统认识 AP；Metric 是库提供的指标基类。"""

        def __init__(self) -> None:
            """声明指标名和优化方向；_maximize=True 表示越大越好。"""
            self._name = "average_precision"
            self._maximize = True

        def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
            """真实标签 y_true 与两列概率 y_score → 一个 AP 浮点数。

            __call__ 使指标对象可像函数调用；[:,1] 取所有行的正类概率列。
            此项目检查标签只有 0/1，TabNet 对两类按 0、1 顺序排列。
            """
            return float(average_precision_score(y_true, y_score[:, 1]))

    args = parse_training_args("tabnet", batch_size=1024, lr=0.02)
    started = time.perf_counter()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data = prepare_data(args)
    preprocessor = Preprocessor.fit(data.fit)
    x_train = preprocessor.tabnet_matrix(data.fit)
    x_val = preprocessor.tabnet_matrix(data.validation)
    y_train = data.fit[TARGET].to_numpy(dtype=np.int64)
    y_val = data.validation[TARGET].to_numpy(dtype=np.int64)
    # 矩阵的前 19 列是数值/布尔，最后 5 列是类别编号；cat_dims 包含未知编码 0。
    first_category = len(NUMERIC_FEATURES) + len(BOOLEAN_FEATURES)
    virtual_batch_size = min(128, args.batch_size)
    parameters = {
        "n_d": 16, "n_a": 16, "n_steps": 3, "cat_emb_dim": 8,
        "cat_idxs": list(range(first_category, first_category + 5)),
        "cat_dims": preprocessor.category_sizes, "virtual_batch_size": virtual_batch_size,
        "lambda_sparse": 1e-3, "mask_type": "sparsemax", "weights": 0,
        "loss": "cross_entropy + sparse regularization",
        "implementation": "pytorch-tabnet==4.1.0",
    }
    config = make_config(args, data, "tabnet", device, parameters)
    save_preprocessor(args.output_dir / "preprocessor.json", preprocessor)
    # 实例化分类器；optimizer_fn 指定优化器类，optimizer_params 是其参数字典。
    model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, cat_idxs=parameters["cat_idxs"],
        cat_dims=preprocessor.category_sizes, cat_emb_dim=8,
        lambda_sparse=1e-3, mask_type="sparsemax",
        optimizer_fn=torch.optim.Adam, optimizer_params={"lr": args.lr},
        seed=args.seed, device_name=str(device), verbose=1,
    )
    # Ghost BatchNorm 将大批次分成虚拟小批次来计算统计量。
    # 最后一个指标用于早停，所以将 auc 放在最后，且只传入验证集。
    model.fit(
        x_train, y_train, eval_set=[(x_val, y_val)], eval_name=["validation"],
        eval_metric=[AveragePrecision, "auc"], max_epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        virtual_batch_size=virtual_batch_size, weights=0, num_workers=0,
        # 仅在余数为 1 时丢弃最后单条训练记录，避免 BatchNorm 无法估计方差。
        drop_last=len(x_train) % args.batch_size == 1,
        pin_memory=device.type == "cuda", compute_importance=False,
    )
    raw_history = model.history.history
    history = [
        {
            "epoch": i + 1, "train_loss": float(loss),
            "validation_roc_auc": float(raw_history["validation_auc"][i]),
            "validation_average_precision": float(raw_history["validation_average_precision"][i]),
        }
        for i, loss in enumerate(raw_history["loss"])
    ]
    best_epoch = int(model.best_epoch) + 1  # 库从 0 计数，对外统一从 1 计数。
    # 库在训练结束时自动恢复最佳权重，然后使用原生 ZIP 格式保存。
    saved_path = model.save_model(str(args.output_dir / "best_model"))
    restored = TabNetClassifier(device_name=str(device), verbose=0)
    restored.load_model(saved_path)
    restored.batch_size = args.batch_size
    preprocessor = Preprocessor(**read_json(args.output_dir / "preprocessor.json"))
    # predict_proba 的列对应 classes_，显式取欺诈标签 1 的概率。
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    val_probabilities = restored.predict_proba(
        preprocessor.tabnet_matrix(data.validation)
    )[:, positive_column]
    test_probabilities = restored.predict_proba(
        preprocessor.tabnet_matrix(data.test)
    )[:, positive_column]
    save_results(
        args, data, config, val_probabilities, test_probabilities,
        best_epoch=best_epoch, history=history,
        elapsed_seconds=time.perf_counter() - started, model_file="best_model.zip",
    )


if __name__ == "__main__":
    # 直接运行时调用 main，常见数据或依赖错误由公共 entrypoint 解释。
    entrypoint(main)
