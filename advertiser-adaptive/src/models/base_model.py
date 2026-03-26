"""
base_model.py
所有多场景模型的基类，定义共享 EmbeddingLayer 和统一接口。
"""
from typing import Dict, List

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    为每个特征列创建独立的 Embedding table。
    forward 返回所有特征 embedding 拼接后的向量 [batch, total_embed_dim]。
    """

    def __init__(self, feature_cols: List[str], vocab_size: int, embed_dim: int):
        super().__init__()
        self.feature_cols = feature_cols
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for col in feature_cols
        })
        self.output_dim = embed_dim * len(feature_cols)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: {feature_name: LongTensor[batch]}
        Returns:
            Tensor [batch, embed_dim * num_features]
        """
        embs = [self.embeddings[col](x[col]) for col in self.feature_cols]
        return torch.cat(embs, dim=-1)  # [batch, total_embed_dim]


class MLP(nn.Module):
    """通用 MLP 模块，支持 BN、Dropout、可选输出层。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        output_layer: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        act_fn = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "tanh": nn.Tanh}.get(activation, nn.ReLU)

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        if output_layer:
            layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim if not output_layer else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseModel(nn.Module):
    """
    多场景模型基类。

    子类必须实现 forward(x) -> Tensor[batch]（sigmoid 概率值）。
    x 格式：{feature_name: LongTensor[batch], "domain_indicator": LongTensor[batch]}
    """

    def __init__(self, feature_cols: List[str], vocab_size: int, embed_dim: int):
        super().__init__()
        self.feature_cols = feature_cols
        self.embedding = EmbeddingLayer(feature_cols, vocab_size, embed_dim)
        self.input_dim = self.embedding.output_dim

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
