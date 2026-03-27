"""
ple.py
PLE (Progressive Layered Extraction)：CGC 结构，domain-specific expert + shared expert。
参考：RecSys 2020 "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model"
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .base_model import BaseModel, MLP


class CGCLayer(nn.Module):
    """
    单层 Customized Gate Control (CGC)。

    Args:
        input_dim: 输入维度
        domain_num: domain 数量
        n_expert_specific: 每个 domain 的专属 expert 数量
        n_expert_shared: 共享 expert 数量
        expert_dims: expert MLP 隐层维度
        dropout: Dropout 比率
        is_last: 是否为最后一层（最后一层不需要 shared gate）
    """

    def __init__(
        self,
        input_dim: int,
        domain_num: int,
        n_expert_specific: int,
        n_expert_shared: int,
        expert_dims: List[int],
        dropout: float = 0.3,
        is_last: bool = False,
    ):
        super().__init__()
        self.domain_num = domain_num
        self.n_expert_specific = n_expert_specific
        self.n_expert_shared = n_expert_shared
        self.is_last = is_last
        expert_out_dim = expert_dims[-1]

        # domain-specific experts
        self.specific_experts = nn.ModuleList([
            MLP(input_dim, expert_dims, dropout=dropout, output_layer=False)
            for _ in range(domain_num * n_expert_specific)
        ])
        # shared experts
        self.shared_experts = nn.ModuleList([
            MLP(input_dim, expert_dims, dropout=dropout, output_layer=False)
            for _ in range(n_expert_shared)
        ])
        # domain-specific gates（选 specific + shared experts）
        gate_input_num = n_expert_specific + n_expert_shared
        self.specific_gates = nn.ModuleList([
            nn.Linear(input_dim, gate_input_num)
            for _ in range(domain_num)
        ])
        # shared gate（选所有 experts，仅非最后层）
        if not is_last:
            total_experts = domain_num * n_expert_specific + n_expert_shared
            self.shared_gate = nn.Linear(input_dim, total_experts)

        self.output_dim = expert_out_dim

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            inputs: [domain_0_input, ..., domain_N_input, shared_input]，长度 domain_num+1
        Returns:
            outputs: 同结构，长度 domain_num+1（最后层无 shared output）
        """
        # 计算所有 specific expert 输出
        specific_outs = [
            self.specific_experts[i](inputs[i // self.n_expert_specific])
            for i in range(self.domain_num * self.n_expert_specific)
        ]
        # 计算 shared expert 输出
        shared_outs = [expert(inputs[-1]) for expert in self.shared_experts]

        cgc_outs = []
        for d in range(self.domain_num):
            # 当前 domain 的 specific experts
            d_specific = specific_outs[d * self.n_expert_specific:(d + 1) * self.n_expert_specific]
            candidates = d_specific + shared_outs  # list of [batch, expert_dim]
            candidates_t = torch.stack(candidates, dim=1)  # [batch, n_sp+n_sh, expert_dim]
            gate_w = torch.softmax(self.specific_gates[d](inputs[d]), dim=-1)  # [batch, n_sp+n_sh]
            pooled = (gate_w.unsqueeze(-1) * candidates_t).sum(dim=1)  # [batch, expert_dim]
            cgc_outs.append(pooled)

        if not self.is_last:
            all_outs = specific_outs + shared_outs
            all_t = torch.stack(all_outs, dim=1)  # [batch, total, expert_dim]
            gate_w = torch.softmax(self.shared_gate(inputs[-1]), dim=-1)  # [batch, total]
            shared_pooled = (gate_w.unsqueeze(-1) * all_t).sum(dim=1)
            cgc_outs.append(shared_pooled)

        return cgc_outs


class PLE(BaseModel):
    """
    PLE 多场景模型，支持多层 CGC。

    Args:
        feature_cols: 特征列名列表
        vocab_size: embedding 词表大小
        embed_dim: embedding 维度
        domain_num: domain 数量
        n_level: CGC 层数
        n_expert_specific: 每个 domain 的专属 expert 数量
        n_expert_shared: 共享 expert 数量
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
        n_level: int = 1,
        n_expert_specific: int = 2,
        n_expert_shared: int = 1,
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
        self.n_level = n_level

        # CGC 层（最后一层 is_last=True）
        self.cgc_layers = nn.ModuleList()
        in_dim = self.input_dim
        for i in range(n_level):
            is_last = (i == n_level - 1)
            self.cgc_layers.append(CGCLayer(
                input_dim=in_dim,
                domain_num=domain_num,
                n_expert_specific=n_expert_specific,
                n_expert_shared=n_expert_shared,
                expert_dims=expert_dims,
                dropout=dropout,
                is_last=is_last,
            ))
            in_dim = expert_dims[-1]

        self.towers = nn.ModuleList([
            MLP(in_dim, tower_dims, dropout=dropout, output_layer=True)
            for _ in range(domain_num)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        domain_id = x["domain_indicator"]  # [batch]
        emb = self.embedding(x)            # [batch, input_dim]

        # 初始输入：每个 domain + shared 都用同一个 emb
        ple_inputs = [emb] * (self.domain_num + 1)

        for cgc in self.cgc_layers:
            ple_inputs = cgc(ple_inputs)

        ys, masks = [], []
        for d in range(self.domain_num):
            masks.append(domain_id == d)
            tower_out = self.towers[d](ple_inputs[d])  # [batch, 1]
            ys.append(self.sigmoid(tower_out))

        final = torch.zeros_like(ys[0])
        for d in range(self.domain_num):
            final = torch.where(masks[d].unsqueeze(1), ys[d], final)
        return final.squeeze(1)  # [batch]
