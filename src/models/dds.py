"""
exp002: Data Distribution Search (DDS) - WWW 2022

核心思想:
- 根据特征的数据分布自动搜索最优的 embedding 维度
- 使用可微分的搜索方法，端到端优化
- 高频特征分配更多维度，低频特征分配更少维度

参考论文: "Automated Embedding Size Search in Deep Recommender Systems" (WWW 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class DDSEmbedding(nn.Module):
    """
    Data Distribution Search Embedding
    
    为每个特征学习一个软性的维度选择分布，
    通过 Gumbel-Softmax 实现可微分的离散选择。
    """
    
    def __init__(
        self,
        vocab_size: int,
        candidate_dims: List[int] = [8, 16, 32, 64, 128],
        temperature: float = 1.0,
        hard: bool = False
    ):
        """
        Args:
            vocab_size: 特征的词表大小
            candidate_dims: 候选维度列表
            temperature: Gumbel-Softmax 温度
            hard: 是否使用 hard 选择（训练时用 soft，推理时用 hard）
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.candidate_dims = candidate_dims
        self.max_dim = max(candidate_dims)
        self.temperature = temperature
        self.hard = hard
        
        # 为每个候选维度创建 embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, dim, padding_idx=0)
            for dim in candidate_dims
        ])
        
        # 维度选择的 logits（可学习）
        self.dim_logits = nn.Parameter(torch.zeros(len(candidate_dims)))
        
        # 投影层：将不同维度投影到统一的 max_dim
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.max_dim) if dim != self.max_dim else nn.Identity()
            for dim in candidate_dims
        ])
        
        # 初始化
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 ID (batch_size,)
        
        Returns:
            output: (batch_size, max_dim)
            dim_weights: (num_candidates,) 维度选择权重
        """
        # 计算维度选择权重
        if self.training:
            dim_weights = F.gumbel_softmax(
                self.dim_logits, 
                tau=self.temperature, 
                hard=self.hard
            )
        else:
            dim_weights = F.softmax(self.dim_logits, dim=-1)
        
        # 获取各维度的 embedding 并投影
        outputs = []
        for emb, proj in zip(self.embeddings, self.projections):
            emb_out = emb(x)  # (batch, dim_i)
            proj_out = proj(emb_out)  # (batch, max_dim)
            outputs.append(proj_out)
        
        # 加权融合
        outputs = torch.stack(outputs, dim=-1)  # (batch, max_dim, num_candidates)
        output = torch.einsum('bdc,c->bd', outputs, dim_weights)  # (batch, max_dim)
        
        return output, dim_weights
    
    def get_selected_dim(self) -> int:
        """获取当前选择的维度（取概率最大的）"""
        idx = self.dim_logits.argmax().item()
        return self.candidate_dims[idx]
    
    def get_dim_distribution(self) -> Dict[int, float]:
        """获取维度选择的概率分布"""
        probs = F.softmax(self.dim_logits, dim=-1).detach().cpu().numpy()
        return {dim: float(prob) for dim, prob in zip(self.candidate_dims, probs)}


class DDSWideDeep(nn.Module):
    """带 DDS 自动维度搜索的 WideDeep 模型"""
    
    def __init__(
        self,
        feature_config: Dict[str, int],
        candidate_dims: List[int] = [8, 16, 32, 64, 128],
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3,
        temperature: float = 1.0
    ):
        """
        Args:
            feature_config: {feature_name: vocab_size}
            candidate_dims: 候选维度列表
            dnn_hidden_units: DNN 隐藏层维度
            dropout: dropout 比例
            temperature: Gumbel-Softmax 温度
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.candidate_dims = candidate_dims
        self.max_dim = max(candidate_dims)
        
        # 为每个特征创建 DDS Embedding
        self.dds_embeddings = nn.ModuleDict({
            name: DDSEmbedding(
                vocab_size=vocab_size,
                candidate_dims=candidate_dims,
                temperature=temperature
            )
            for name, vocab_size in feature_config.items()
        })
        
        # DNN 输入维度
        dnn_input_dim = len(feature_config) * self.max_dim
        
        # DNN 层
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
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {feature_name: (batch_size,)}
        
        Returns:
            logits: (batch_size,)
        """
        embeddings = []
        
        for name, emb_layer in self.dds_embeddings.items():
            if name in features:
                emb, _ = emb_layer(features[name])
                embeddings.append(emb)
        
        # 拼接所有 embedding
        x = torch.cat(embeddings, dim=-1)
        
        # DNN
        x = self.dnn(x)
        logits = self.output_layer(x).squeeze(-1)
        
        return logits
    
    def get_dimension_stats(self) -> Dict[str, Dict]:
        """获取所有特征的维度选择统计"""
        stats = {}
        for name, emb_layer in self.dds_embeddings.items():
            stats[name] = {
                "selected_dim": emb_layer.get_selected_dim(),
                "distribution": emb_layer.get_dim_distribution()
            }
        return stats
    
    def get_parameter_stats(self) -> Dict:
        """获取参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        emb_params = sum(
            sum(p.numel() for p in emb.parameters())
            for emb in self.dds_embeddings.values()
        )
        
        # 计算有效维度（取概率最大的）
        effective_dims = {
            name: emb.get_selected_dim()
            for name, emb in self.dds_embeddings.items()
        }
        
        return {
            "total_params": total_params,
            "embedding_params": emb_params,
            "effective_dims": effective_dims,
            "avg_effective_dim": sum(effective_dims.values()) / len(effective_dims)
        }
