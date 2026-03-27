"""
ChorusCVR 的 Tower 模块
包含 CTR Tower, CVR Tower, unCVR Tower
"""
import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        use_bn: bool = True,
        output_activation: str = "sigmoid"
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            elif activation.lower() == "gelu":
                layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "softmax":
            layers.append(nn.Softmax(dim=-1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class Tower(nn.Module):
    """单个预测 Tower"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.tower = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout_rate=dropout_rate,
            activation=activation,
            output_activation="sigmoid"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (batch_size, input_dim)
        Returns:
            预测概率 (batch_size,)
        """
        return self.tower(x)


class SharedBottom(nn.Module):
    """共享底层网络"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x)


class EmbeddingLayer(nn.Module):
    """特征嵌入层"""
    
    def __init__(
        self,
        sparse_feature_dims: dict,  # {feature_name: vocab_size}
        dense_feature_num: int,
        embedding_dim: int = 16
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.sparse_feature_names = list(sparse_feature_dims.keys())
        self.dense_feature_num = dense_feature_num
        
        # 稀疏特征嵌入
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embedding_dim)
            for name, vocab_size in sparse_feature_dims.items()
        })
        
        # 稠密特征线性变换
        if dense_feature_num > 0:
            self.dense_transform = nn.Linear(dense_feature_num, embedding_dim)
        else:
            self.dense_transform = None
        
        # 计算输出维度
        self.output_dim = len(sparse_feature_dims) * embedding_dim
        if dense_feature_num > 0:
            self.output_dim += embedding_dim
        
        self._init_weights()
    
    def _init_weights(self):
        for embedding in self.embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
        if self.dense_transform is not None:
            nn.init.xavier_uniform_(self.dense_transform.weight)
            nn.init.zeros_(self.dense_transform.bias)
    
    def forward(
        self,
        sparse_features: dict,  # {feature_name: tensor of indices}
        dense_features: torch.Tensor = None  # (batch_size, dense_feature_num)
    ) -> torch.Tensor:
        """
        Args:
            sparse_features: 稀疏特征字典
            dense_features: 稠密特征张量
        Returns:
            拼接后的嵌入向量 (batch_size, output_dim)
        """
        embeddings = []
        
        # 稀疏特征嵌入
        for name in self.sparse_feature_names:
            if name in sparse_features:
                emb = self.embeddings[name](sparse_features[name])
                embeddings.append(emb)
        
        # 稠密特征变换
        if self.dense_transform is not None and dense_features is not None:
            dense_emb = self.dense_transform(dense_features)
            embeddings.append(dense_emb)
        
        # 拼接所有嵌入
        return torch.cat(embeddings, dim=-1)
