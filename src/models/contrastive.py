"""
exp005: Contrastive ID Learning (SIGIR 2023)

核心思想:
- 使用对比学习增强 ID embedding 的表示能力
- 同类 ID（如同一类别的广告）embedding 应该相近
- 不同类 ID embedding 应该远离
- 解决 ID embedding 空间稀疏、缺乏结构的问题

参考论文: "Contrastive Learning for Sequential Recommendation" (SIGIR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import random


class ContrastiveLoss(nn.Module):
    """对比学习损失函数 (InfoNCE)"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        anchor: torch.Tensor,      # (batch, dim)
        positive: torch.Tensor,    # (batch, dim)
        negatives: torch.Tensor    # (batch, num_neg, dim)
    ) -> torch.Tensor:
        """
        InfoNCE Loss
        
        Args:
            anchor: anchor embedding
            positive: positive sample embedding
            negatives: negative samples embedding
        
        Returns:
            loss: scalar
        """
        # 归一化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # 正样本相似度
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (batch,)
        
        # 负样本相似度
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # (batch, num_neg)
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (batch, 1 + num_neg)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class ContrastiveEmbedding(nn.Module):
    """
    带对比学习的 Embedding 层
    
    在训练时，对同一类别的 ID 进行对比学习，
    使 embedding 空间更有结构。
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_negatives: int = 16,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        
        # Embedding 层
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 投影头（用于对比学习）
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 对比损失
        self.contrastive_loss = ContrastiveLoss(temperature)
        
        # 初始化
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """正常前向传播"""
        return self.embedding(x)
    
    def compute_contrastive_loss(
        self,
        ids: torch.Tensor,           # (batch,)
        group_ids: torch.Tensor      # (batch,) 同组 ID 应该相近
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            ids: ID 输入
            group_ids: 分组 ID（同组视为正样本）
        
        Returns:
            loss: 对比损失
        """
        batch_size = ids.size(0)
        
        # 获取 embedding 并投影
        emb = self.embedding(ids)
        proj = self.projection(emb)
        
        # 构造正负样本
        # 正样本：同一 group 的其他 ID
        # 负样本：不同 group 的 ID
        
        total_loss = 0
        count = 0
        
        unique_groups = group_ids.unique()
        
        for group in unique_groups:
            mask = group_ids == group
            group_indices = mask.nonzero(as_tuple=True)[0]
            
            if len(group_indices) < 2:
                continue
            
            # 随机选择 anchor 和 positive
            perm = torch.randperm(len(group_indices))
            anchor_idx = group_indices[perm[0]]
            positive_idx = group_indices[perm[1]]
            
            anchor = proj[anchor_idx].unsqueeze(0)
            positive = proj[positive_idx].unsqueeze(0)
            
            # 负样本：其他 group 的样本
            neg_mask = group_ids != group
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            
            if len(neg_indices) < self.num_negatives:
                continue
            
            neg_perm = torch.randperm(len(neg_indices))[:self.num_negatives]
            negatives = proj[neg_indices[neg_perm]].unsqueeze(0)  # (1, num_neg, dim)
            
            loss = self.contrastive_loss(anchor, positive, negatives)
            total_loss += loss
            count += 1
        
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=ids.device)


class ContrastiveWideDeep(nn.Module):
    """带对比学习的 WideDeep 模型"""
    
    def __init__(
        self,
        feature_config: Dict[str, int],
        contrastive_features: Dict[str, str],  # {id_feature: group_feature}
        embedding_size: int = 64,
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3,
        contrastive_weight: float = 0.1
    ):
        """
        Args:
            feature_config: {feature_name: vocab_size}
            contrastive_features: {要做对比学习的特征: 分组依据特征}
            embedding_size: embedding 维度
            dnn_hidden_units: DNN 隐藏层
            dropout: dropout 比例
            contrastive_weight: 对比损失权重
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.contrastive_features = contrastive_features
        self.embedding_size = embedding_size
        self.contrastive_weight = contrastive_weight
        
        # 普通 embedding
        self.embeddings = nn.ModuleDict()
        self.contrastive_embeddings = nn.ModuleDict()
        
        for name, vocab_size in feature_config.items():
            if name in contrastive_features:
                self.contrastive_embeddings[name] = ContrastiveEmbedding(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_size
                )
            else:
                self.embeddings[name] = nn.Embedding(
                    vocab_size + 1, embedding_size, padding_idx=0
                )
        
        # DNN
        dnn_input_dim = len(feature_config) * embedding_size
        
        layers = []
        prev_dim = dnn_input_dim
        for hidden_dim in dnn_hidden_units:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.dnn = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 初始化
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor],
        compute_contrastive: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: {feature_name: (batch_size,)}
            compute_contrastive: 是否计算对比损失
        
        Returns:
            logits: (batch_size,)
            contrastive_loss: scalar
        """
        embeddings = []
        contrastive_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 普通 embedding
        for name, emb_layer in self.embeddings.items():
            if name in features:
                embeddings.append(emb_layer(features[name]))
        
        # 对比学习 embedding
        for name, emb_layer in self.contrastive_embeddings.items():
            if name in features:
                emb = emb_layer(features[name])
                embeddings.append(emb)
                
                # 计算对比损失
                if compute_contrastive and self.training:
                    group_feat = self.contrastive_features[name]
                    if group_feat in features:
                        cl_loss = emb_layer.compute_contrastive_loss(
                            features[name],
                            features[group_feat]
                        )
                        contrastive_loss = contrastive_loss + cl_loss
        
        # 拼接
        x = torch.cat(embeddings, dim=-1)
        
        # DNN
        x = self.dnn(x)
        logits = self.output_layer(x).squeeze(-1)
        
        return logits, contrastive_loss * self.contrastive_weight
    
    def get_parameter_stats(self) -> Dict:
        """获取参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "total_params": total_params,
            "contrastive_features": list(self.contrastive_features.keys()),
            "contrastive_weight": self.contrastive_weight
        }
