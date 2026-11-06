"""CatBoost 梯度提升树；原生处理类别字段，默认在 CPU 运行。

用法：python train_catboost.py --smoke；GPU 可加 --device cuda。
与神经网络不同，它逐轮增加决策树来纠正预测误差，不使用反向传播训练循环。
"""

from __future__ import annotations

import os
import time

import numpy as np

from tabular_common import (
    CATEGORICAL_FEATURES, TARGET, CatBoostPreprocessor, entrypoint,
    make_config, parse_training_args, prepare_data, read_json, resolve_device,
    save_preprocessor, save_results, seed_everything,
)


def main() -> None:
    """执行 CatBoost 数据准备、训练、最佳模型保存加载和评估。

    无函数参数/返回值；运行参数来自 parse_training_args。
    iterations 是最多增加的树数；depth 是每棵树的深度；
    learning_rate 控制每棵新树的贡献；eval_metric 决定验证和早停指标。
    """
    from catboost import CatBoostClassifier

    args = parse_training_args("catboost", lr=0.05)
    started = time.perf_counter()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data = prepare_data(args)
    preprocessor = CatBoostPreprocessor.fit(data.fit)
    x_train = preprocessor.transform(data.fit)
    x_val = preprocessor.transform(data.validation)
    # CatBoost 对类别进行原生处理，不把任意类别编码当成有序数值。
    parameters = {
        "depth": 6, "loss_function": "Logloss", "eval_metric": "AUC",
        "task_type": "GPU" if device.type == "cuda" else "CPU",
        "class_weights": None, "implementation": "catboost==1.2.10",
    }
    config = make_config(args, data, "catboost", device, parameters)
    save_preprocessor(args.output_dir / "preprocessor.json", preprocessor)
    model = CatBoostClassifier(
        iterations=args.iterations, depth=6, learning_rate=args.lr,
        loss_function="Logloss", eval_metric="AUC", random_seed=args.seed,
        task_type=parameters["task_type"],
        thread_count=min(8, os.cpu_count() or 1),
        allow_writing_files=False, verbose=1 if args.smoke else 50,
    )
    # 早停使用验证 AUC；use_best_model 会保留最佳迭代之前的树。
    model.fit(
        x_train, data.fit[TARGET].astype(int), cat_features=CATEGORICAL_FEATURES,
        eval_set=(x_val, data.validation[TARGET].astype(int)),
        early_stopping_rounds=args.patience, use_best_model=True,
    )
    evaluations = model.get_evals_result()
    history = [
        {
            "iteration": i + 1,
            "train_loss": float(evaluations["learn"]["Logloss"][i]),
            "validation_roc_auc": float(auc),
        }
        for i, auc in enumerate(evaluations["validation"]["AUC"])
    ]
    best_epoch = int(model.get_best_iteration()) + 1
    model_path = args.output_dir / "best_model.cbm"
    model.save_model(str(model_path))
    # 创建一个新分类器再加载，确认磁盘中的模型可以独立恢复。
    restored = CatBoostClassifier()
    restored.load_model(str(model_path))
    preprocessor = CatBoostPreprocessor(**read_json(args.output_dir / "preprocessor.json"))
    positive_column = int(np.flatnonzero(restored.classes_ == 1)[0])
    val_probabilities = restored.predict_proba(
        preprocessor.transform(data.validation)
    )[:, positive_column]
    test_probabilities = restored.predict_proba(
        preprocessor.transform(data.test)
    )[:, positive_column]
    save_results(
        args, data, config, val_probabilities, test_probabilities,
        best_epoch=best_epoch, history=history,
        elapsed_seconds=time.perf_counter() - started, model_file=model_path.name,
    )


if __name__ == "__main__":
    # 只有从终端执行本文件时才开始训练。
    entrypoint(main)
