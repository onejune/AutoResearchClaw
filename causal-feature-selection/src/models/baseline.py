"""
Baseline 模型 - WideDeep / DeepFM
复现 rec-autopilot 最优配置
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class WideDeepBaseline(nn.Module):
    """
    Wide&Deep Baseline Model
    
    复现 rec-autopilot 最优配置:
    - dnn_hidden_units: [1024, 512, 256, 128]
    - embedding_size: 8
    - dropout: 0.3
    - lr: 5e-5
    """
    
    def __init__(
        self,
        feature_config: Dict[str, int],  # {feature_name: vocab_size}
        embedding_size: int = 8,
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3,
        wide_features: List[str] = None  # 用于 wide 部分的特征
    ):
        super().__init__()
        
        self.feature_config = feature_config
        self.embedding_size = embedding_size
        self.wide_features = wide_features or []
        
        # Embedding 层
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)
            for name, vocab_size in feature_config.items()
        })
        
        # Wide 部分（如果使用）
        self.wide_net = None
        if self.wide_features:
            wide_input_dim = len(self.wide_features)
            self.wide_net = nn.Sequential(
                nn.Linear(wide_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        # Deep 部分输入维度
        deep_input_dim = len(feature_config) * embedding_size
        
        # Deep 部分 DNN
        dnn_layers = []
        prev_dim = deep_input_dim
        for hidden_unit in dnn_hidden_units:
            dnn_layers.append(nn.Linear(prev_dim, hidden_unit))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_unit
        
        dnn_layers.append(nn.Linear(prev_dim, 1))
        self.deep_net = nn.Sequential(*dnn_layers)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: dict of {feature_name: tensor(batch_size)}
            
        Returns:
            logits: (batch_size, 1)
        """
        # Embedding lookup
        embedded_features = []
        for name, idx_tensor in features.items():
            if name in self.embeddings:
                embedded = self.embeddings[name](idx_tensor)  # (batch, embed_dim)
                embedded_features.append(embedded)
        
        # Concat all embeddings
        deep_input = torch.cat(embedded_features, dim=-1)  # (batch, total_embed_dim)
        
        # Deep part
        deep_out = self.deep_net(deep_input)  # (batch, 1)
        
        # Wide part (if exists)
        if self.wide_net is not None:
            wide_input = torch.stack([
                features[name].float() for name in self.wide_features if name in features
            ], dim=-1)
            wide_out = self.wide_net(wide_input)
            logits = wide_out + deep_out
        else:
            logits = deep_out
        
        return logits.squeeze(-1)


class DeepFMBaseline(nn.Module):
    """
    DeepFM Baseline Model
    
    FM 部分处理低阶特征交互，Deep 部分处理高阶交互
    """
    
    def __init__(
        self,
        feature_config: Dict[str, int],
        embedding_size: int = 8,
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.feature_config = feature_config
        self.embedding_size = embedding_size
        self.num_features = len(feature_config)
        
        # Shared embedding
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)
            for name, vocab_size in feature_config.items()
        })
        
        # FM part
        self.fm_bias = nn.Parameter(torch.zeros(1))
        
        # Linear part (first order)
        self.linear = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, 1, padding_idx=0)
            for name, vocab_size in feature_config.items()
        })
        
        # Deep part
        deep_input_dim = self.num_features * embedding_size
        dnn_layers = []
        prev_dim = deep_input_dim
        for hidden_unit in dnn_hidden_units:
            dnn_layers.append(nn.Linear(prev_dim, hidden_unit))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_unit
        dnn_layers.append(nn.Linear(prev_dim, 1))
        self.deep_net = nn.Sequential(*dnn_layers)
        
    def _fm_second_order(self, embedded_features: List[torch.Tensor]) -> torch.Tensor:
        """
        FM 二阶交互：1/2 * sum((sum(e_i*x_i))^2 - sum(e_i^2 * x_i^2))
        使用 field-aware 的简化版本
        """
        concatenated = torch.cat(embedded_features, dim=1)  # (batch, num_features * embed_dim)
        
        square_of_sum = torch.sum(concatenated, dim=1) ** 2
        sum_of_square = torch.sum(concatenated ** 2, dim=1)
        
        interaction = 0.5 * (square_of_sum - sum_of_square)
        return interaction.sum(dim=1, keepdim=True)  # (batch, 1)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Embedding lookup
        embedded_features = [
            self.embeddings[name](idx_tensor)
            for name, idx_tensor in features.items()
        ]
        
        # Linear part (first order)
        linear_out = sum([
            self.linear[name](idx_tensor).squeeze(-1)
            for name, idx_tensor in features.items()
        ])
        
        # FM part (second order)
        fm_out = self._fm_second_order(embedded_features)
        
        # Deep part
        deep_input = torch.cat(embedded_features, dim=-1)
        deep_out = self.deep_net(deep_input)
        
        # Combine
        logits = self.fm_bias + linear_out + fm_out.squeeze(-1) + deep_out.squeeze(-1)
        return logits


def get_baseline_model(
    model_type: str = "widedeep",
    feature_config: Dict[str, int] = None,
    **kwargs
) -> nn.Module:
    """
    获取 baseline 模型工厂函数
    
    Args:
        model_type: "widedeep" or "deepfm"
        feature_config: {feature_name: vocab_size}
        **kwargs: 模型超参
        
    Returns:
        模型实例
    """
    if model_type == "widedeep":
        return WideDeepBaseline(feature_config=feature_config, **kwargs)
    elif model_type == "deepfm":
        return DeepFMBaseline(feature_config=feature_config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    torch.manual_seed(42)
    
    # 模拟特征配置
    feature_config = {
        "feat_1": 1000,
        "feat_2": 500,
        "feat_3": 200,
        "feat_4": 10000,
        "feat_5": 5000,
    }
    
    # 测试 WideDeep
    model = WideDeepBaseline(
        feature_config=feature_config,
        embedding_size=8,
        dnn_hidden_units=[1024, 512, 256, 128],
        dropout=0.3
    )
    
    # 模拟输入
    batch_size = 32
    dummy_features = {
        name: torch.randint(1, vocab_size, (batch_size,))
        for name, vocab_size in feature_config.items()
    }
    
    output = model(dummy_features)
    print(f"WideDeep output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试 DeepFM
    model_dfm = DeepFMBaseline(feature_config=feature_config)
    output_dfm = model_dfm(dummy_features)
    print(f"\nDeepFM output shape: {output_dfm.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_dfm.parameters()):,}")
