"""MLP 与 FT-Transformer 共用的二分类训练循环，无需直接运行。

train_mlp.py / train_ft_transformer.py 创建模型后调用 run_neural。
阅读顺序：make_loader（分批送数据）→ 训练循环（更新权重）→
predict（输出概率）→ 恢复最佳模型 → 保存 ROC 和 CSV。
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tabular_common import (
    TARGET, Preprocessor, classification_metrics, make_config, prepare_data,
    resolve_device, save_preprocessor, save_results, seed_everything,
    validate_probabilities,
)


def make_loader(
    frame: pd.DataFrame, preprocessor: Preprocessor, args: argparse.Namespace,
    device: torch.device, training: bool = False,
) -> DataLoader:
    """把一张表转换成按批读取的 DataLoader。

    frame：原始表；preprocessor：只用训练部分拟合的预处理器；
    args：含 batch_size 和 seed 的配置；device：CPU/GPU；
    training=True 会附带标签并打乱样本，False 保持原顺序用于预测。
    返回的每个批次包含 dense、categories，以及训练时的 labels。
    Tensor 是 PyTorch 的数值数组，from_numpy 将 NumPy 数组交给 PyTorch。
    """
    dense, categories = preprocessor.transform(frame)
    tensors = [torch.from_numpy(dense), torch.from_numpy(categories)]
    if training:
        tensors.append(torch.from_numpy(frame[TARGET].to_numpy(dtype=np.float32)))
    return DataLoader(
        TensorDataset(*tensors), batch_size=args.batch_size, shuffle=training,
        num_workers=0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """按 loader 顺序计算所有记录的欺诈概率，返回一维 NumPy 数组。

    model 为已加载权重的网络；device 为模型所在设备。
    logits 是尚未限制范围的分数，sigmoid 将其转换成 [0,1] 概率。
    @torch.inference_mode() 是装饰器，让整个函数在无需梯度的模式下执行。
    """
    # eval 关闭 Dropout；inference_mode 关闭梯度记录，降低推理内存开销。
    model.eval()
    probabilities = []
    for dense, categories in loader:
        logits = model(
            dense.to(device, non_blocking=True),
            categories.to(device, non_blocking=True),
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return validate_probabilities(np.concatenate(probabilities), len(loader.dataset))


def run_neural(
    args: argparse.Namespace, model_name: str,
    model_factory: Callable[[list[int]], nn.Module], parameters: dict,
) -> None:
    """完成一个神经网络从读数据到输出结果的全部流程，不返回值。

    args：parse_training_args 解析出的配置；
    model_name：模型标识；model_factory：用类别数量创建模型的函数/类，
    例如 FraudMLP；parameters：结构、权重衰减等模型配置字典。
    训练期间只更新训练部分的权重，验证部分只用于指标和早停。
    用法见 train_mlp.py 中 main() 调用。
    """
    started = time.perf_counter()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data = prepare_data(args)
    preprocessor = Preprocessor.fit(data.fit)
    train_loader = make_loader(data.fit, preprocessor, args, device, training=True)
    val_loader = make_loader(data.validation, preprocessor, args, device)
    val_labels = data.validation[TARGET].to_numpy()
    model = model_factory(preprocessor.category_sizes).to(device)
    prior = float(data.fit[TARGET].mean())
    # 各模型暴露输出层，利用训练部分的正类先验初始化不平衡任务的起始概率。
    with torch.no_grad():
        model.output_layer.bias.fill_(math.log(prior / (1 - prior)))
    weight_decay = parameters["weight_decay"]
    # FT-Transformer 作者实现将偏置、归一化和特征 Embedding 排除出权重衰减。
    # hasattr 检查是否提供特殊分组方法；MLP 没有，就更新全部参数。
    groups = model.parameter_groups() if hasattr(model, "parameter_groups") else model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    parameters = {
        **parameters, "category_sizes": preprocessor.category_sizes,
        "loss": "unweighted BCEWithLogitsLoss",
    }
    config = make_config(args, data, model_name, device, parameters)
    save_preprocessor(args.output_dir / "preprocessor.json", preprocessor)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    history = []
    # 从负无穷开始，让第一轮必然保存；stale_epochs 记录连续未提升次数。
    best_auc, best_epoch, stale_epochs = -math.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        for dense, categories, labels in train_loader:
            dense, categories, labels = [
                tensor.to(device, non_blocking=True) for tensor in (dense, categories, labels)
            ]
            # BCEWithLogitsLoss 内部完成 sigmoid，训练阶段不能提前 sigmoid。
            optimizer.zero_grad(set_to_none=True)  # 清除上一批次的梯度。
            loss = criterion(model(dense, categories), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; check data or lower --lr")
            loss.backward()  # 反向传播：计算各权重对损失的影响（梯度）。
            optimizer.step()  # 根据梯度和学习率更新权重。
            total_loss += float(loss.item()) * len(labels)
        scores = classification_metrics(val_labels, predict(model, val_loader, device))
        auc = scores["roc_auc"]
        record = {
            "epoch": epoch, "train_loss": total_loss / len(data.fit),
            "validation_roc_auc": auc,
            "validation_average_precision": scores["average_precision"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(f"Epoch {epoch:02d}/{args.epochs}: loss={record['train_loss']:.6f} "
              f"val_auc={auc:.6f} val_ap={scores['average_precision']:.6f} "
              f"time={record['seconds']:.1f}s", flush=True)
        # 保存标准依据是排序指标 AUC，而非训练损失或固定阈值的准确率。
        if auc > best_auc:
            best_auc, best_epoch, stale_epochs = auc, epoch, 0
            torch.save({
                "format_version": 2, "config": config,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "preprocessor": asdict(preprocessor), "best_epoch": epoch,
                "validation_roc_auc": auc,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping: {stale_epochs} epochs without AUC improvement.", flush=True)
                break

    # 最后一轮未必最好，必须重新加载最高验证 AUC 对应的权重和预处理。
    # map_location="cpu" 让文件先加载到内存，再把新网络放到指定设备。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    preprocessor = Preprocessor(**checkpoint["preprocessor"])
    model = model_factory(preprocessor.category_sizes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loader = make_loader(data.validation, preprocessor, args, device)
    test_loader = make_loader(data.test, preprocessor, args, device)
    val_probabilities = predict(model, val_loader, device)
    test_probabilities = predict(model, test_loader, device)
    save_results(
        args, data, config, val_probabilities, test_probabilities,
        best_epoch=best_epoch, history=history,
        elapsed_seconds=time.perf_counter() - started, model_file=checkpoint_path.name,
    )
