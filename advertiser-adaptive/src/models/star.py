"""
star.py
STAR (Star Topology Adaptive Recommender)：
  - Domain Normalization（共享 γ/β + domain-specific γ/β）
  - Star Topology FCN：W = W_shared ⊙ W_domain（element-wise 乘法）
  - 辅助网络（全量数据训练，防长尾 domain 过拟合）
参考：CIKM 2021 "One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction"
"""
from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn import Parameter

from .base_model import BaseModel, MLP


class STAR(BaseModel):
    """
    STAR 多场景模型，核心方案。

    Args:
        feature_cols: 特征列名列表
        vocab_size: embedding 词表大小
        embed_dim: embedding 维度
        domain_num: domain 数量
        fcn_dims: Star FCN 隐层维度（不含输入/输出层）
        aux_dims: 辅助网络隐层维度
        dropout: Dropout 比率（用于辅助网络）
    """

    def __init__(
        self,
        feature_cols: List[str],
        vocab_size: int = 100_000,
        embed_dim: int = 8,
        domain_num: int = 4,
        fcn_dims: List[int] = None,
        aux_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__(feature_cols, vocab_size, embed_dim)
        if fcn_dims is None:
            fcn_dims = [512, 256, 64]
        if aux_dims is None:
            aux_dims = [256, 64]

        self.domain_num = domain_num
        self.eps = 1e-6

        # FCN 层维度：[input_dim, *fcn_dims, 1]
        layer_dims = [self.input_dim] + fcn_dims + [1]
        self.n_layers = len(layer_dims) - 1

        # ── Domain Normalization ──────────────────────────────────────────
        # 共享 γ/β
        self.dn_shared_gamma = Parameter(torch.ones(self.input_dim))
        self.dn_shared_beta = Parameter(torch.zeros(self.input_dim))
        # domain-specific γ/β
        self.dn_domain_gamma = nn.ParameterList([
            Parameter(torch.ones(self.input_dim)) for _ in range(domain_num)
        ])
        self.dn_domain_beta = nn.ParameterList([
            Parameter(torch.zeros(self.input_dim)) for _ in range(domain_num)
        ])

        # ── Star Topology FCN ─────────────────────────────────────────────
        # 共享参数
        self.shared_w = nn.ParameterList([
            Parameter(torch.empty(layer_dims[i], layer_dims[i + 1]))
            for i in range(self.n_layers)
        ])
        self.shared_b = nn.ParameterList([
            Parameter(torch.zeros(layer_dims[i + 1]))
            for i in range(self.n_layers)
        ])
        # domain-specific 参数（初始化为全 1/0，使初始行为等同于共享参数）
        self.domain_w = nn.ParameterList([
            nn.ParameterList([
                Parameter(torch.ones(layer_dims[i], layer_dims[i + 1]))
                for i in range(self.n_layers)
            ])
            for _ in range(domain_num)
        ])
        self.domain_b = nn.ParameterList([
            nn.ParameterList([
                Parameter(torch.zeros(layer_dims[i + 1]))
                for i in range(self.n_layers)
            ])
            for _ in range(domain_num)
        ])
        self.bn_layers = nn.ModuleList([
            nn.ModuleList([nn.BatchNorm1d(layer_dims[i + 1]) for i in range(self.n_layers)])
            for _ in range(domain_num)
        ])

        # ── 辅助网络（全量数据，防长尾过拟合）────────────────────────────
        self.aux_net = MLP(self.input_dim, aux_dims, dropout=dropout, output_layer=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for w in self.shared_w:
            nn.init.kaiming_uniform_(w)

    def _domain_norm(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """对 embedding 做 Domain Normalization。"""
        mean = x.mean(dim=0)
        var = ((x - mean) ** 2).mean(dim=0)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        gamma = self.dn_shared_gamma * self.dn_domain_gamma[d]
        beta = self.dn_shared_beta + self.dn_domain_beta[d]
        return gamma * x_norm + beta

    def _star_fcn(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Star Topology FCN：W = W_shared ⊙ W_domain。"""
        h = x
        for i in range(self.n_layers):
            w = self.shared_w[i] * self.domain_w[d][i]   # element-wise 乘法
            b = self.shared_b[i] + self.domain_b[d][i]
            h = h @ w + b
            h = self.bn_layers[d][i](h)
            if i < self.n_layers - 1:
                h = self.relu(h)
        return h  # [batch, 1]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        domain_id = x["domain_indicator"]  # [batch]
        emb = self.embedding(x)            # [batch, input_dim]

        # 辅助网络（全量数据，不区分 domain）
        aux_out = self.aux_net(emb)        # [batch, 1]

        ys, masks = [], []
        for d in range(self.domain_num):
            masks.append(domain_id == d)
            # Domain Normalization
            normed = self._domain_norm(emb, d)
            # Star FCN
            main_out = self._star_fcn(normed, d)  # [batch, 1]
            ys.append(main_out)

        # 按 domain 路由，合并结果
        main_final = torch.zeros_like(ys[0])
        for d in range(self.domain_num):
            main_final = torch.where(masks[d].unsqueeze(1), ys[d], main_final)

        out = self.sigmoid(main_final + aux_out)  # [batch, 1]
        return out.squeeze(1)  # [batch]
