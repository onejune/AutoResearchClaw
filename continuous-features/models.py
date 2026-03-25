"""
DeepFM backbone，支持替换连续特征编码器

结构：
- FM 部分：二阶特征交叉
- Deep 部分：MLP(256, 128, 64)
- 连续特征 embedding + 类别特征 embedding 拼接后输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_encoders import BaseEncoder


class DeepFM(nn.Module):
    def __init__(
        self,
        feature_encoder: BaseEncoder,
        cat_vocab_sizes: list,
        embedding_dim: int = 16,
        n_continuous: int = None,
        n_categorical: int = None,
    ):
        """
        Args:
            feature_encoder: 连续特征编码器（任意 BaseEncoder 子类）
            cat_vocab_sizes: 类别特征的词表大小列表（长度 = n_categorical）
            embedding_dim: 类别特征 embedding 维度（FM embedding）
            n_continuous: 连续特征数量（可选，用于文档/校验，实际由 encoder 决定）
            n_categorical: 类别特征数量（可选，用于文档/校验）
        """
        super().__init__()
        self.feature_encoder = feature_encoder
        self.embedding_dim = embedding_dim
        self.n_cat = len(cat_vocab_sizes)

        # 类别特征 embedding（用于 FM 和 Deep）
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            for vocab_size in cat_vocab_sizes
        ])

        # 连续特征 encoder 输出维度
        cont_out_dim = feature_encoder.output_dim
        self.cont_out_dim = cont_out_dim
        self.has_cont = cont_out_dim > 0

        # FM 一阶：对连续特征做线性变换 + 类别特征 embedding 求和
        # 连续特征一阶项（无连续特征时跳过）
        self.cont_linear = nn.Linear(cont_out_dim, 1, bias=False) if self.has_cont else None
        # 类别特征一阶项（每个类别 embedding 求和到 1 维）
        self.cat_linear = nn.ModuleList([
            nn.Embedding(vocab_size, 1, padding_idx=0)
            for vocab_size in cat_vocab_sizes
        ])

        # FM 二阶：将连续特征投影到 embedding_dim（无连续特征时跳过）
        self.cont_fm_proj = nn.Linear(cont_out_dim, embedding_dim, bias=False) if self.has_cont else None

        # Deep 部分
        # 输入：连续特征 encoder 输出 + 类别特征 embedding 拼接
        deep_input_dim = cont_out_dim + self.n_cat * embedding_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cont: (batch, 13) 连续特征
            x_cat:  (batch, 26) 类别特征（整数索引）
        Returns:
            (batch,) sigmoid 概率
        """
        # ── 连续特征编码 ──
        cont_emb = self.feature_encoder(x_cont)  # (batch, cont_out_dim) or (batch, 0)

        # ── 类别特征 embedding ──
        cat_embs = [self.cat_embeddings[i](x_cat[:, i]) for i in range(self.n_cat)]
        cat_emb_stack = torch.stack(cat_embs, dim=1)  # (batch, n_cat, emb_dim)

        # ════════════════════
        # FM 一阶项
        # ════════════════════
        cat_linear_out = [self.cat_linear[i](x_cat[:, i]) for i in range(self.n_cat)]
        fm1_cat = torch.stack(cat_linear_out, dim=1).sum(dim=1)  # (batch, 1)
        if self.has_cont:
            fm1_cont = self.cont_linear(cont_emb)  # (batch, 1)
            fm1 = fm1_cont + fm1_cat
        else:
            fm1 = fm1_cat

        # ════════════════════
        # FM 二阶项
        # ════════════════════
        if self.has_cont:
            cont_fm = self.cont_fm_proj(cont_emb).unsqueeze(1)  # (batch, 1, emb_dim)
            all_embs = torch.cat([cont_fm, cat_emb_stack], dim=1)  # (batch, 1+n_cat, emb_dim)
        else:
            all_embs = cat_emb_stack  # (batch, n_cat, emb_dim)

        sum_of_embs = all_embs.sum(dim=1)
        sum_of_sq = (all_embs ** 2).sum(dim=1)
        fm2 = 0.5 * (sum_of_embs ** 2 - sum_of_sq).sum(dim=-1, keepdim=True)  # (batch, 1)

        # ════════════════════
        # Deep 部分
        # ════════════════════
        cat_flat = cat_emb_stack.reshape(cat_emb_stack.shape[0], -1)  # (batch, n_cat*emb_dim)
        if self.has_cont:
            deep_input = torch.cat([cont_emb, cat_flat], dim=-1)
        else:
            deep_input = cat_flat
        deep_out = self.deep(deep_input)  # (batch, 1)

        # ════════════════════
        # 合并输出
        # ════════════════════
        logit = fm1 + fm2 + deep_out  # (batch, 1)
        return torch.sigmoid(logit.squeeze(-1))  # (batch,)
