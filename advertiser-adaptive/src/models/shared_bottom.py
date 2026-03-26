"""
shared_bottom.py
Shared-Bottom 多场景模型：共享 bottom MLP + 各 domain 独立 tower。
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .base_model import BaseModel, MLP


class SharedBottom(BaseModel):
    """
    Shared-Bottom：所有 domain 共享底层 MLP，各 domain 有独立 tower。

    Args:
        feature_cols: 特征列名列表
        vocab_size: embedding 词表大小
        embed_dim: embedding 维度
        domain_num: domain 数量
        bottom_dims: shared bottom MLP 隐层维度
        tower_dims: 各 domain tower MLP 隐层维度
        dropout: Dropout 比率
    """

    def __init__(
        self,
        feature_cols: List[str],
        vocab_size: int = 100_000,
        embed_dim: int = 8,
        domain_num: int = 4,
        bottom_dims: List[int] = None,
        tower_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__(feature_cols, vocab_size, embed_dim)
        if bottom_dims is None:
            bottom_dims = [512, 256]
        if tower_dims is None:
            tower_dims = [128, 64]

        self.domain_num = domain_num
        self.bottom = MLP(self.input_dim, bottom_dims, dropout=dropout, output_layer=False)
        self.towers = nn.ModuleList([
            MLP(bottom_dims[-1], tower_dims, dropout=dropout, output_layer=True)
            for _ in range(domain_num)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        domain_id = x["domain_indicator"]          # [batch]
        emb = self.embedding(x)                    # [batch, input_dim]
        bottom_out = self.bottom(emb)              # [batch, bottom_dims[-1]]

        ys, masks = [], []
        for d in range(self.domain_num):
            masks.append(domain_id == d)
            tower_out = self.towers[d](bottom_out)  # [batch, 1]
            ys.append(self.sigmoid(tower_out))       # [batch, 1]

        final = torch.zeros_like(ys[0])
        for d in range(self.domain_num):
            final = torch.where(masks[d].unsqueeze(1), ys[d], final)
        return final.squeeze(1)  # [batch]
