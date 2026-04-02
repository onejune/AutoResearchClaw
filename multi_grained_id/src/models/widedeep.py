"""
WideDeep 基线模型

简单的 Wide&Deep 实现，用于作为其他实验的 baseline。
"""

import torch
import torch.nn as nn
from typing import Dict, List


class WideDeep(nn.Module):
    """标准的 WideDeep 模型（只用 Deep 部分）"""
    
    def __init__(
        self,
        feature_config: Dict[str, int],  # {feat: vocab_size}
        embedding_size: int = 64,
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3
    ):
        """
        Args:
            feature_config: {feature_name: vocab_size}
            embedding_size: 所有特征统一使用这个维度
            dnn_hidden_units: DNN 隐藏层维度列表
            dropout: dropout 比例
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.embedding_size = embedding_size
        
        # 为每个特征创建固定维度的 embedding
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)
            for feat, vocab_size in feature_config.items()
        })
        
        # 初始化
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        
        # 计算 deep 输入维度
        deep_input_dim = len(feature_config) * embedding_size
        
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
        
        # 拼接
        deep_input = torch.cat(embedded_features, dim=-1)
        
        # Deep 网络
        deep_out = self.deep_network(deep_input)
        
        # 输出
        logits = self.output_layer(deep_out).squeeze(-1)
        
        return logits
    
    def get_parameter_stats(self) -> Dict:
        """获取参数统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        embedding_params = sum(emb.num_parameters() for emb in self.embeddings.values())
        
        return {
            "total_params": total_params,
            "embedding_params": embedding_params,
            "dnn_params": total_params - embedding_params,
            "embedding_dim": self.embedding_size,
            "num_features": len(self.feature_config)
        }
