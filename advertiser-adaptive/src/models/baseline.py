"""
baseline.py
全局单模型 DNN，不区分 domain，作为对比 baseline。
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .base_model import BaseModel, MLP


class Baseline(BaseModel):
    """
    全局 DNN baseline：所有特征 embedding concat → MLP → sigmoid。
    不感知 domain_indicator，作为多场景模型的对比基准。

    Args:
        feature_cols: 特征列名列表
        vocab_size: embedding 词表大小
        embed_dim: embedding 维度
        hidden_dims: MLP 隐层维度列表
        dropout: Dropout 比率
    """

    def __init__(
        self,
        feature_cols: List[str],
        vocab_size: int = 100_000,
        embed_dim: int = 8,
        domain_num: int = 4,  # 接收但不使用，保持接口统一
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__(feature_cols, vocab_size, embed_dim)
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]
        self.mlp = MLP(self.input_dim, hidden_dims, dropout=dropout, output_layer=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.embedding(x)              # [batch, input_dim]
        out = self.mlp(emb)                  # [batch, 1]
        return self.sigmoid(out).squeeze(1)  # [batch]
