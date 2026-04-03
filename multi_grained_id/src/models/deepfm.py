"""
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (IJCAI 2017)

核心思想:
- FM 组件：学习二阶特征交互
- Deep 组件：学习高阶特征交互
- 共享 Embedding：FM 和 Deep 共享输入

参考论文:
- "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FMComponent(nn.Module):
    """
    Factorization Machine 组件
    学习二阶特征交互
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, num_fields, embedding_dim]
        
        Returns:
            fm_output: [batch_size, 1]
        """
        # 一阶项：线性组合 (这里简化，只用 embedding sum)
        fm_linear = embeddings.sum(dim=1)  # [batch_size, embedding_dim]
        
        # 二阶项： pairwise interactions
        sum_of_square = torch.sum(embeddings ** 2, dim=1)  # [batch_size, embedding_dim]
        square_of_sum = torch.sum(embeddings, dim=1) ** 2  # [batch_size, embedding_dim]
        
        # FM output: 0.5 * (square_of_sum - sum_of_square)
        fm_interactions = 0.5 * (square_of_sum - sum_of_square)  # [batch_size, embedding_dim]
        
        # 组合一阶和二阶
        fm_output = fm_linear + fm_interactions
        fm_output = fm_output.sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        return fm_output


class DeepComponent(nn.Module):
    """
    Deep 组件
    学习高阶特征交互
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_fields * embedding_dim]
        
        Returns:
            deep_output: [batch_size, 1]
        """
        return self.mlp(x)


class DeepFM(nn.Module):
    """
    DeepFM 模型
    
    FM 组件 + Deep 组件，共享 Embedding
    """
    
    def __init__(
        self,
        field_dims: Dict[str, int],
        embedding_dim: int = 64,
        hidden_units: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.num_fields = len(field_dims)
        
        # Embedding 层 (FM 和 Deep 共享)
        self.embeddings = nn.ModuleDict({
            field_name: nn.Embedding(field_dim, embedding_dim)
            for field_name, field_dim in field_dims.items()
        })
        
        # 初始化
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        
        # FM 组件
        self.fm = FMComponent()
        
        # Deep 组件
        self.deep = DeepComponent(
            input_dim=self.num_fields * embedding_dim,
            hidden_units=hidden_units,
            dropout=dropout
        )
    
    def get_embeddings(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        获取所有特征的 embedding
        
        Args:
            features: {field_name: [batch_size]}
        
        Returns:
            embeddings: [batch_size, num_fields, embedding_dim]
        """
        batch_size = next(iter(features.values())).size(0)
        embeddings = torch.zeros(
            batch_size, self.num_fields, self.embedding_dim,
            device=next(iter(features.values())).device
        )
        
        for i, field_name in enumerate(self.field_dims.keys()):
            if field_name in features:
                embeddings[:, i, :] = self.embeddings[field_name](features[field_name])
        
        return embeddings
    
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            features: {field_name: [batch_size]}
        
        Returns:
            logits: [batch_size, 1]
        """
        # 获取共享 embedding
        embeddings = self.get_embeddings(features)  # [batch_size, num_fields, embedding_dim]
        
        # FM 组件
        fm_output = self.fm(embeddings)  # [batch_size, 1]
        
        # Deep 组件
        deep_input = embeddings.view(embeddings.size(0), -1)  # [batch_size, num_fields * embedding_dim]
        deep_output = self.deep(deep_input)  # [batch_size, 1]
        
        # 组合 FM 和 Deep
        logits = fm_output + deep_output
        
        return logits.squeeze(-1)  # [batch_size]
    
    def predict(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """预测概率"""
        logits = self.forward(features)
        return torch.sigmoid(logits)  # [batch_size]
