"""使用论文作者的 FT-Transformer，学习表格各字段之间的注意力交互。

用法：python train_ft_transformer.py --smoke；完整训练省略 --smoke。
token 是一个字段的向量表示，注意力让模型根据其他字段调整该表示。
"""

from __future__ import annotations

import torch
from torch import nn

from neural_training import run_neural
from tabular_common import (
    BOOLEAN_FEATURES, NUMERIC_FEATURES, entrypoint, parse_training_args,
)


class FraudFTTransformer(nn.Module):
    """对作者库作薄封装，使它可以使用与 MLP 相同的训练循环。"""

    def __init__(self, category_sizes: list[int]) -> None:
        """category_sizes 为五个类别字段的编号数量；初始化作者实现。

        n_cont_features：数值/布尔字段数量；cat_cardinalities：类别数量；
        d_out=1：每条记录输出一个分数；n_blocks：注意力块层数；
        d_block：每个字段向量的维度；attention_n_heads：注意力头数。
        dropout 参数控制训练时的随机丢弃比例，用于抑制过拟合。
        """
        super().__init__()
        from rtdl_revisiting_models import FTTransformer

        # 每个数值字段使用独立线性变换得到 token；类别字段使用独立 Embedding。
        # 作者实现还包含 CLS token、PreNorm 和 ReGLU，避免用普通 Transformer 冒充。
        self.transformer = FTTransformer(
            n_cont_features=len(NUMERIC_FEATURES) + len(BOOLEAN_FEATURES),
            cat_cardinalities=category_sizes, d_out=1,
            n_blocks=2, d_block=64, attention_n_heads=8, attention_dropout=0.2,
            ffn_d_hidden=None, ffn_d_hidden_multiplier=2.0,
            ffn_dropout=0.1, residual_dropout=0.0,
        )

    @property
    def output_layer(self) -> nn.Linear:
        """取出作者模型的最终线性层，供训练循环初始化正类先验偏置。"""
        return self.transformer.backbone.output[-1]

    def parameter_groups(self) -> list[dict]:
        """返回两组待更新权重，分别指定正常衰减和零衰减。

        无输入参数；返回可直接交给 AdamW 优化器的参数分组列表。
        """
        return self.transformer.make_parameter_groups()

    def forward(self, dense: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        """输入 (批大小,19) 数值矩阵和 (批大小,5) 类别矩阵，返回一维 logit。

        CLS 是额外的汇总 token，它在注意力层中收集各字段信息。
        用法：logits = model(dense, categories)，概率转换在公共预测函数中执行。
        """
        # token 之间进行自注意力，CLS 的表示最终转换为一个二分类 logit。
        return self.transformer(dense, categories).squeeze(1)


def main() -> None:
    """读取本模型默认参数，调用公共训练流程；本函数无参数/返回值。"""
    args = parse_training_args("ft_transformer", batch_size=512, lr=1e-4)
    run_neural(args, "ft_transformer", FraudFTTransformer, {
        "n_blocks": 2, "d_block": 64, "attention_n_heads": 8,
        "attention_dropout": 0.2, "ffn_d_hidden_multiplier": 2.0,
        "ffn_dropout": 0.1, "residual_dropout": 0.0, "weight_decay": 1e-5,
        "implementation": "rtdl-revisiting-models==0.0.2",
    })


if __name__ == "__main__":
    # 直接执行脚本时启动；import 该文件时不会训练。
    entrypoint(main)
