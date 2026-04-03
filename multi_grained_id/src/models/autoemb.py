"""
exp001: AutoEmb (KDD 2021) - 简化版

核心思想:
- 不是所有 ID 特征都需要相同的 embedding 维度
- 根据特征频次/重要性自动分配不同维度
- 在表达能力和参数量之间找平衡

简化实现:
- 用启发式规则替代强化学习搜索（降低复杂度）
- 按 vocab 大小和样本频次分档分配维度
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class AutoEmbEmbedding(nn.Module):
    """自动化维度分配的 Embedding"""
    
    def __init__(
        self,
        vocab_size: int,
        suggested_dim: int,
        dim_search_space: List[int] = [8, 16, 32, 64, 128]
    ):
        """
        Args:
            vocab_size: 词表大小
            suggested_dim: 推荐的初始维度
            dim_search_space: 可选维度空间
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.suggested_dim = suggested_dim
        
        # 找到最接近的可用维度
        self.embedding_dim = min(dim_search_space, key=lambda x: abs(x - suggested_dim))
        
        # 创建 embedding
        self.embedding = nn.Embedding(vocab_size + 1, self.embedding_dim, padding_idx=0)
        
        # 初始化
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class AutoEmbWideDeep(nn.Module):
    """带自动化维度分配的 WideDeep 模型"""
    
    def __init__(
        self,
        feature_config: Dict[str, int],  # {feat: vocab_size}
        dimension_config: Dict[str, int],  # {feat: embedding_dim}
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3
    ):
        """
        Args:
            feature_config: {feature_name: vocab_size}
            dimension_config: {feature_name: embedding_dim}
            dnn_hidden_units: DNN 隐藏层维度列表
            dropout: dropout 比例
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.dimension_config = dimension_config
        
        # 为每个特征创建对应维度的 embedding
        self.embeddings = nn.ModuleDict({
            feat: AutoEmbEmbedding(
                vocab_size=vocab_size,
                suggested_dim=dimension_config.get(feat, 64)
            )
            for feat, vocab_size in feature_config.items()
        })
        
        # 计算 deep 输入维度（各特征维度之和）
        deep_input_dim = sum(emb.embedding_dim for emb in self.embeddings.values())
        
        # Deep 网络
        deep_layers = []
        prev_dim = deep_input_dim
        for hidden_unit in dnn_hidden_units:
            deep_layers.append(nn.Linear(prev_dim, hidden_unit))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_unit
        
        self.deep_network = nn.Sequential(*deep_layers)
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, 1)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedded_features = []
        
        for feat, idx_tensor in features.items():
            if feat in self.embeddings:
                emb = self.embeddings[feat](idx_tensor)
                embedded_features.append(emb)
        
        # 拼接（注意：不同特征的维度可能不同）
        deep_input = torch.cat(embedded_features, dim=-1)
        
        # Deep 网络
        deep_out = self.deep_network(deep_input)
        
        # 输出
        logits = self.output_layer(deep_out).squeeze(-1)
        
        return logits
    
    def get_parameter_stats(self) -> Dict:
        """获取参数统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        embedding_params = sum(emb.embedding.num_parameters() for emb in self.embeddings.values())
        
        dim_distribution = {}
        for feat, emb in self.embeddings.items():
            dim = emb.embedding_dim
            dim_distribution[dim] = dim_distribution.get(dim, 0) + 1
        
        return {
            "total_params": total_params,
            "embedding_params": embedding_params,
            "dnn_params": total_params - embedding_params,
            "dim_distribution": dim_distribution,
            "avg_embedding_dim": sum(emb.embedding_dim for emb in self.embeddings.values()) / len(self.embeddings)
        }


def suggest_dimensions_heuristic(
    feature_config: Dict[str, int],
    base_dim: int = 64,
    dim_search_space: List[int] = [8, 16, 32, 64, 128]
) -> Dict[str, int]:
    """
    启发式维度分配规则
    
    规则:
    - vocab > 1M: 32 维（超稀疏，高维易过拟合）
    - vocab > 100k: 64 维
    - vocab > 10k: 64 维
    - vocab > 1k: 32 维
    - vocab <= 1k: 16 维（密集特征，低维足够）
    """
    suggestions = {}
    
    for feat, vocab_size in feature_config.items():
        if vocab_size > 1000000:
            suggestions[feat] = 32
        elif vocab_size > 100000:
            suggestions[feat] = 64
        elif vocab_size > 10000:
            suggestions[feat] = 64
        elif vocab_size > 1000:
            suggestions[feat] = 32
        else:
            suggestions[feat] = 16
    
    # 确保维度在搜索空间内
    for feat in suggestions:
        suggestions[feat] = min(dim_search_space, key=lambda x: abs(x - suggestions[feat]))
    
    return suggestions


def compare_parameter_efficiency(
    feature_config: Dict[str, int],
    fixed_dim: int = 64,
    auto_dims: Dict[str, int] = None
) -> Dict:
    """对比固定维度 vs 自动分配的参数效率"""
    
    if auto_dims is None:
        auto_dims = suggest_dimensions_heuristic(feature_config)
    
    # 固定维度方案
    fixed_params = sum((vocab + 1) * fixed_dim for vocab in feature_config.values())
    
    # 自动分配方案
    auto_params = sum((vocab + 1) * auto_dims[feat] for feat, vocab in feature_config.items())
    
    # 各维度分布
    dim_counts = {}
    for dim in auto_dims.values():
        dim_counts[dim] = dim_counts.get(dim, 0) + 1
    
    return {
        "fixed_dim_scheme": {
            "dim": fixed_dim,
            "total_embedding_params": fixed_params,
            "params_mb": fixed_params * 4 / 1024**2  # float32
        },
        "auto_dim_scheme": {
            "dims": auto_dims,
            "total_embedding_params": auto_params,
            "params_mb": auto_params * 4 / 1024**2,
            "dim_distribution": dim_counts
        },
        "savings": {
            "param_reduction": (fixed_params - auto_params) / fixed_params * 100,
            "mb_saved": (fixed_params - auto_params) * 4 / 1024**2
        }
    }
