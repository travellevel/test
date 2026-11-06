"""类别 Embedding + MLP；固定分层 80% 训练、20% 验证。

终端用法：python train_mlp.py --smoke（小规模）或 python train_mlp.py（完整）。
本文件定义网络结构，neural_training.py 负责训练，tabular_common.py 负责数据和图。
"""

from __future__ import annotations

import torch
from torch import nn

from neural_training import run_neural
from tabular_common import (
    BOOLEAN_FEATURES, NUMERIC_FEATURES, entrypoint, parse_training_args,
)

EMBEDDING_DIM = 8
DROPOUT = 0.2


class FraudMLP(nn.Module):
    """一个把交易字段映射为欺诈分数的神经网络。

    nn.Module 是 PyTorch 网络的基类；继承后可自动管理权重与梯度。
    创建方式：model = FraudMLP([13, 6, 6, 6, 9])。
    """

    def __init__(self, category_sizes: list[int]) -> None:
        """初始化网络，category_sizes 是五个字段各自的类别数量（含未知值）。

        __init__ 在创建对象时自动执行；super().__init__() 初始化基类。
        nn.Embedding 像可学习的查表，每个类别编号取出一个 8 维向量；
        nn.Linear 是全连接层，ReLU 提供非线性，Dropout 在训练时随机丢弃部分输出。
        """
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, EMBEDDING_DIM, padding_idx=0) for size in category_sizes
        ])
        # 13 个标准化数值 + 6 个布尔值 + 5×8 维类别向量，共 59 维。
        width = len(NUMERIC_FEATURES) + len(BOOLEAN_FEATURES)
        width += len(category_sizes) * EMBEDDING_DIM
        layers = []
        for hidden in [256, 128, 64]:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(DROPOUT)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    @property
    def output_layer(self) -> nn.Linear:
        """返回最后一个全连接层，供公共训练循环初始化输出偏置。

        self.network[-1] 表示序列最后一项；属性可写为 model.output_layer。
        """
        return self.network[-1]

    def forward(self, dense: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        """执行一次前向计算，返回每条记录的原始分数 logit。

        dense 为 (批大小,19)，categories 为 (批大小,5)。
        torch.cat(..., dim=1) 按列拼接；squeeze(1) 将 (批大小,1) 变为一维。
        PyTorch 调用 model(dense, categories) 时会自动进入 forward。
        """
        # 各类别字段使用独立 Embedding；编码 0 对应固定的零向量。
        embeddings = [layer(categories[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.network(torch.cat([dense, *embeddings], dim=1)).squeeze(1)


def main() -> None:
    """脚本入口：读取命令参数并启动 MLP 训练，没有参数及返回值。"""
    args = parse_training_args("mlp")
    run_neural(args, "mlp", FraudMLP, {
        "hidden_dims": [256, 128, 64], "embedding_dim": EMBEDDING_DIM,
        "dropout": DROPOUT, "weight_decay": 1e-4,
    })


if __name__ == "__main__":
    # 直接运行本文件才启动训练；被其他文件 import 时只加载定义。
    entrypoint(main)
