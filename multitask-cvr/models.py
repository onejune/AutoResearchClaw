"""
models.py - 多任务学习模型

包含：
  - EmbeddingLayer  稀疏特征 embedding + 数值特征拼接
  - MLP             多层感知机（BN + Dropout）
  - BaseMultiTaskModel  基类
  - SharedBottom    共享底层 MLP
  - ESMM            Entire Space Multi-Task Model
  - MMoE            Multi-gate Mixture-of-Experts
  - ESCM2           ESMM + 反事实正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# 底层组件
# ─────────────────────────────────────────────────────────────

class EmbeddingLayer(nn.Module):
    """
    稀疏特征 embedding + 数值特征拼接。

    Parameters
    ----------
    sparse_vocab : dict  {feat_name: vocab_size}
    sparse_feats : list  稀疏特征名列表（顺序固定）
    dense_dim    : int   数值特征维度
    embedding_dim: int   每个稀疏特征的 embedding 维度
    """

    def __init__(
        self,
        sparse_vocab: Dict[str, int],
        sparse_feats: List[str],
        dense_dim: int,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.sparse_feats  = sparse_feats
        self.embedding_dim = embedding_dim
        self.dense_dim     = dense_dim

        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
            for feat, vocab_size in sparse_vocab.items()
            if feat in sparse_feats
        })

        self.output_dim = len(sparse_feats) * embedding_dim + dense_dim

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x : dict 包含 sparse 特征（long tensor）和可选的 "dense"（float tensor）
        返回 shape: (batch, output_dim)
        """
        # 确定 device
        ref_key = list(x.keys())[0]
        device = x[ref_key].device

        emb_list = []
        for feat in self.sparse_feats:
            if feat in self.embeddings:
                emb_list.append(self.embeddings[feat](x[feat]))   # (B, emb_dim)
            else:
                # 缺失特征用零向量
                B = x[ref_key].size(0)
                emb_list.append(torch.zeros(B, self.embedding_dim, device=device))

        parts = emb_list
        if self.dense_dim > 0 and "dense" in x:
            parts = emb_list + [x["dense"]]   # (B, dense_dim)
        return torch.cat(parts, dim=-1)   # (B, output_dim)


class MLP(nn.Module):
    """
    多层感知机，支持 BatchNorm + Dropout。

    Parameters
    ----------
    input_dim  : int
    hidden_dims: tuple/list  各隐层维度
    output_dim : int | None  None 则最后一层为 hidden_dims[-1]，不接输出层
    dropout    : float
    use_bn     : bool
    output_activation : str | None  "sigmoid" | "relu" | None
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        output_dim: Optional[int] = 1,
        dropout: float = 0.1,
        use_bn: bool = True,
        output_activation: Optional[str] = "sigmoid",
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        self.hidden = nn.Sequential(*layers)
        self.out_dim = in_dim

        if output_dim is not None:
            self.output_layer = nn.Linear(in_dim, output_dim)
            self.out_dim = output_dim
        else:
            self.output_layer = None

        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        if self.output_layer is not None:
            h = self.output_layer(h)
        if self.output_activation == "sigmoid":
            h = torch.sigmoid(h)
        elif self.output_activation == "relu":
            h = F.relu(h)
        return h


# ─────────────────────────────────────────────────────────────
# 基类
# ─────────────────────────────────────────────────────────────

class BaseMultiTaskModel(nn.Module):
    """所有多任务模型的基类"""

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys: "ctr", "cvr", "ctcvr"  (all shape: (B,))
        """
        raise NotImplementedError

    def compute_loss(
        self,
        preds: Dict[str, torch.Tensor],
        ctr_label: torch.Tensor,
        cvr_label: torch.Tensor,
        ctcvr_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys: "total", "ctr", "cvr", "ctcvr"
        """
        raise NotImplementedError

    @staticmethod
    def _bce(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(pred.squeeze(-1), label, reduction="mean")


# ─────────────────────────────────────────────────────────────
# 1. SharedBottom
# ─────────────────────────────────────────────────────────────

class SharedBottom(BaseMultiTaskModel):
    """
    EmbeddingLayer → MLP(shared) → [MLP(ctr_tower), MLP(cvr_tower)]
    Loss = BCE(ctr) + BCE(cvr)
    """

    def __init__(self, feature_info: Dict, config):
        super().__init__()
        self.embedding = EmbeddingLayer(
            sparse_vocab  = feature_info["sparse_vocab"],
            sparse_feats  = feature_info["sparse_feats"],
            dense_dim     = feature_info["dense_dim"],
            embedding_dim = config.embedding_dim,
        )
        input_dim = self.embedding.output_dim

        self.shared_mlp = MLP(
            input_dim, config.mlp_dims, output_dim=None,
            dropout=config.dropout, output_activation=None,
        )
        tower_in = self.shared_mlp.out_dim

        self.ctr_tower = MLP(tower_in, (64,), output_dim=1, dropout=0.0, output_activation="sigmoid")
        self.cvr_tower = MLP(tower_in, (64,), output_dim=1, dropout=0.0, output_activation="sigmoid")

    def forward(self, x):
        emb  = self.embedding(x)
        h    = self.shared_mlp(emb)
        p_ctr = self.ctr_tower(h).squeeze(-1)
        p_cvr = self.cvr_tower(h).squeeze(-1)
        p_ctcvr = p_ctr * p_cvr
        return {"ctr": p_ctr, "cvr": p_cvr, "ctcvr": p_ctcvr}

    def compute_loss(self, preds, ctr_label, cvr_label, ctcvr_label):
        loss_ctr = self._bce(preds["ctr"], ctr_label)
        loss_cvr = self._bce(preds["cvr"], cvr_label)
        total    = loss_ctr + loss_cvr
        return {"total": total, "ctr": loss_ctr, "cvr": loss_cvr, "ctcvr": torch.tensor(0.0)}


# ─────────────────────────────────────────────────────────────
# 2. ESMM
# ─────────────────────────────────────────────────────────────

class ESMM(BaseMultiTaskModel):
    """
    Entire Space Multi-Task Model

    EmbeddingLayer → [MLP(ctr_tower), MLP(ctcvr_tower)]
    p_cvr = p_ctcvr / (p_ctr + eps)
    Loss = BCE(p_ctr, y_ctr) + BCE(p_ctcvr, y_ctcvr)
    """

    def __init__(self, feature_info: Dict, config):
        super().__init__()
        self.embedding = EmbeddingLayer(
            sparse_vocab  = feature_info["sparse_vocab"],
            sparse_feats  = feature_info["sparse_feats"],
            dense_dim     = feature_info["dense_dim"],
            embedding_dim = config.embedding_dim,
        )
        input_dim = self.embedding.output_dim

        self.ctr_tower   = MLP(input_dim, config.mlp_dims, output_dim=1,
                               dropout=config.dropout, output_activation="sigmoid")
        self.ctcvr_tower = MLP(input_dim, config.mlp_dims, output_dim=1,
                               dropout=config.dropout, output_activation="sigmoid")

    def forward(self, x):
        emb      = self.embedding(x)
        p_ctr    = self.ctr_tower(emb).squeeze(-1)
        p_ctcvr  = self.ctcvr_tower(emb).squeeze(-1)
        p_cvr    = p_ctcvr / (p_ctr + 1e-8)
        p_cvr    = torch.clamp(p_cvr, 0.0, 1.0)
        return {"ctr": p_ctr, "cvr": p_cvr, "ctcvr": p_ctcvr}

    def compute_loss(self, preds, ctr_label, cvr_label, ctcvr_label):
        loss_ctr   = self._bce(preds["ctr"],   ctr_label)
        loss_ctcvr = self._bce(preds["ctcvr"], ctcvr_label)
        total      = loss_ctr + loss_ctcvr
        return {"total": total, "ctr": loss_ctr, "cvr": torch.tensor(0.0), "ctcvr": loss_ctcvr}


# ─────────────────────────────────────────────────────────────
# 3. MMoE
# ─────────────────────────────────────────────────────────────

class MMoE(BaseMultiTaskModel):
    """
    Multi-gate Mixture-of-Experts

    EmbeddingLayer → K Expert MLPs → [Gate_ctr, Gate_cvr] → [Tower_ctr, Tower_cvr]
    Loss = BCE(ctr) + BCE(cvr)
    """

    def __init__(self, feature_info: Dict, config):
        super().__init__()
        self.embedding = EmbeddingLayer(
            sparse_vocab  = feature_info["sparse_vocab"],
            sparse_feats  = feature_info["sparse_feats"],
            dense_dim     = feature_info["dense_dim"],
            embedding_dim = config.embedding_dim,
        )
        input_dim  = self.embedding.output_dim
        n_experts  = config.n_experts
        expert_dim = config.mlp_dims[-1]   # 每个 expert 输出维度

        # K 个 Expert（每个是一个小 MLP）
        self.experts = nn.ModuleList([
            MLP(input_dim, config.mlp_dims, output_dim=None,
                dropout=config.dropout, output_activation=None)
            for _ in range(n_experts)
        ])
        expert_out_dim = self.experts[0].out_dim

        # 2 个 Gate（CTR / CVR）
        self.gate_ctr = nn.Linear(input_dim, n_experts)
        self.gate_cvr = nn.Linear(input_dim, n_experts)

        # Tower
        self.ctr_tower = MLP(expert_out_dim, (64,), output_dim=1, dropout=0.0, output_activation="sigmoid")
        self.cvr_tower = MLP(expert_out_dim, (64,), output_dim=1, dropout=0.0, output_activation="sigmoid")

    def forward(self, x):
        emb = self.embedding(x)   # (B, D)

        # Expert 输出
        expert_outs = torch.stack([e(emb) for e in self.experts], dim=1)  # (B, K, E)

        # Gate
        gate_ctr = F.softmax(self.gate_ctr(emb), dim=-1).unsqueeze(-1)   # (B, K, 1)
        gate_cvr = F.softmax(self.gate_cvr(emb), dim=-1).unsqueeze(-1)   # (B, K, 1)

        # 加权求和
        h_ctr = (expert_outs * gate_ctr).sum(dim=1)   # (B, E)
        h_cvr = (expert_outs * gate_cvr).sum(dim=1)   # (B, E)

        p_ctr   = self.ctr_tower(h_ctr).squeeze(-1)
        p_cvr   = self.cvr_tower(h_cvr).squeeze(-1)
        p_ctcvr = p_ctr * p_cvr
        return {"ctr": p_ctr, "cvr": p_cvr, "ctcvr": p_ctcvr}

    def compute_loss(self, preds, ctr_label, cvr_label, ctcvr_label):
        loss_ctr = self._bce(preds["ctr"], ctr_label)
        loss_cvr = self._bce(preds["cvr"], cvr_label)
        total    = loss_ctr + loss_cvr
        return {"total": total, "ctr": loss_ctr, "cvr": loss_cvr, "ctcvr": torch.tensor(0.0)}


# ─────────────────────────────────────────────────────────────
# 4. ESCM2
# ─────────────────────────────────────────────────────────────

class ESCM2(BaseMultiTaskModel):
    """
    ESMM + 反事实正则化（Counterfactual Regularization）

    同 ESMM 结构，额外惩罚未点击样本上的高 CVR 预测：
      L_CR = mean(p_cvr[y_ctr==0]^2)
      Total = BCE(p_ctr, y_ctr) + BCE(p_ctcvr, y_ctcvr) + λ * L_CR
    """

    def __init__(self, feature_info: Dict, config):
        super().__init__()
        self.lam = config.escm2_lambda

        self.embedding = EmbeddingLayer(
            sparse_vocab  = feature_info["sparse_vocab"],
            sparse_feats  = feature_info["sparse_feats"],
            dense_dim     = feature_info["dense_dim"],
            embedding_dim = config.embedding_dim,
        )
        input_dim = self.embedding.output_dim

        self.ctr_tower   = MLP(input_dim, config.mlp_dims, output_dim=1,
                               dropout=config.dropout, output_activation="sigmoid")
        self.ctcvr_tower = MLP(input_dim, config.mlp_dims, output_dim=1,
                               dropout=config.dropout, output_activation="sigmoid")

    def forward(self, x):
        emb     = self.embedding(x)
        p_ctr   = self.ctr_tower(emb).squeeze(-1)
        p_ctcvr = self.ctcvr_tower(emb).squeeze(-1)
        p_cvr   = torch.clamp(p_ctcvr / (p_ctr + 1e-8), 0.0, 1.0)
        return {"ctr": p_ctr, "cvr": p_cvr, "ctcvr": p_ctcvr}

    def compute_loss(self, preds, ctr_label, cvr_label, ctcvr_label):
        loss_ctr   = self._bce(preds["ctr"],   ctr_label)
        loss_ctcvr = self._bce(preds["ctcvr"], ctcvr_label)

        # 反事实正则化：对未点击样本惩罚高 CVR 预测
        non_click_mask = (ctr_label == 0)
        if non_click_mask.sum() > 0:
            loss_cr = (preds["cvr"][non_click_mask] ** 2).mean()
        else:
            loss_cr = torch.tensor(0.0, device=preds["cvr"].device)

        total = loss_ctr + loss_ctcvr + self.lam * loss_cr
        return {
            "total":  total,
            "ctr":    loss_ctr,
            "cvr":    loss_cr,
            "ctcvr":  loss_ctcvr,
        }


# ─────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "shared_bottom": SharedBottom,
    "esmm":          ESMM,
    "mmoe":          MMoE,
    "escm2":         ESCM2,
}


def build_model(feature_info: Dict, config) -> BaseMultiTaskModel:
    name = config.model_name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}，可选: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](feature_info, config)
