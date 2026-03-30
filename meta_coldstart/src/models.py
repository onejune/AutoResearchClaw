"""
基础模型：Embedding + MLP

作为所有方法（Global / Finetune / MAML / ANIL）的共享骨干。
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for feat, vocab_size in vocab_sizes.items()
        })
        self.features = list(vocab_sizes.keys())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features) → (B, n_features * embed_dim)"""
        embeds = []
        for i, feat in enumerate(self.features):
            embeds.append(self.embeddings[feat](x[:, i]))   # (B, embed_dim)
        return torch.cat(embeds, dim=-1)                     # (B, n_feat * embed_dim)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x)).squeeze(-1)


class PurchaseModel(nn.Module):
    """Embedding + MLP，输出 purchase logit"""

    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int,
                 mlp_dims: List[int], dropout: float = 0.1):
        super().__init__()
        n_feat    = len(vocab_sizes)
        input_dim = n_feat * embed_dim

        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.mlp       = MLP(input_dim, mlp_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)    # (B, n_feat * embed_dim)
        return self.mlp(emb)       # (B,) logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.binary_cross_entropy_with_logits(logits, y)
