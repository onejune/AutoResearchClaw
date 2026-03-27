"""
mmoe.py
MMoE (Multi-gate Mixture-of-Experts)：多 expert + 每个 domain 独立 gate + tower。
参考：KDD 2018 "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts"
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .base_model import BaseModel, MLP


class MMoE(BaseModel):
    """
    MMoE 多场景模型。

    Args:
        feature_cols: 特征列名列表
        vocab_size: embedding 词表大小
        embed_dim: embedding 维度
        domain_num: domain 数量
        n_expert: expert 数量
        expert_dims: expert MLP 隐层维度
        tower_dims: tower MLP 隐层维度
        dropout: Dropout 比率
    """

    def __init__(
        self,
        feature_cols: List[str],
        vocab_size: int = 100_000,
        embed_dim: int = 8,
        domain_num: int = 4,
        n_expert: int = 4,
        expert_dims: List[int] = None,
        tower_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__(feature_cols, vocab_size, embed_dim)
        if expert_dims is None:
            expert_dims = [256, 128]
        if tower_dims is None:
            tower_dims = [128, 64]

        self.domain_num = domain_num
        self.n_expert = n_expert

        self.experts = nn.ModuleList([
            MLP(self.input_dim, expert_dims, dropout=dropout, output_layer=False)
            for _ in range(n_expert)
        ])
        # 每个 domain 一个 gate（softmax over experts）
        self.gates = nn.ModuleList([
            nn.Linear(self.input_dim, n_expert)
            for _ in range(domain_num)
        ])
        self.towers = nn.ModuleList([
            MLP(expert_dims[-1], tower_dims, dropout=dropout, output_layer=True)
            for _ in range(domain_num)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        domain_id = x["domain_indicator"]  # [batch]
        emb = self.embedding(x)            # [batch, input_dim]

        # Expert 输出：[batch, n_expert, expert_dim]
        expert_outs = torch.stack(
            [expert(emb) for expert in self.experts], dim=1
        )

        ys, masks = [], []
        for d in range(self.domain_num):
            masks.append(domain_id == d)
            gate_w = torch.softmax(self.gates[d](emb), dim=-1)  # [batch, n_expert]
            # 加权求和 expert 输出
            pooled = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)  # [batch, expert_dim]
            tower_out = self.towers[d](pooled)  # [batch, 1]
            ys.append(self.sigmoid(tower_out))

        final = torch.zeros_like(ys[0])
        for d in range(self.domain_num):
            final = torch.where(masks[d].unsqueeze(1), ys[d], final)
        return final.squeeze(1)  # [batch]
