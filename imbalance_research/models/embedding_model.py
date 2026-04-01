"""支持类别特征的 CTR/CVR 预估模型"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict


class EmbeddingMLP(nn.Module):
    """
    带 Embedding 的 MLP 模型 - 适合纯类别特征
    
    每个特征列用独立的 Embedding 层，然后拼接后输入 MLP
    """
    def __init__(self, 
                 feature_dims: List[int],  # 每个特征的 cardinality
                 embedding_dims: List[int] = None,  # 每个特征的 embedding dim
                 embed_dim: int = None,  # 统一 embedding dim (如果指定则覆盖 embedding_dims)
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3,
                 activation: str = 'relu'):
        super().__init__()
        
        # 处理 embedding 维度
        if embed_dim is not None:
            # 使用统一的 embed_dim
            embedding_dims = [embed_dim] * len(feature_dims)
        elif embedding_dims is None:
            # 自动计算 embedding dim: min(50, sqrt(cardinality/2))
            import math
            embedding_dims = [min(50, max(8, int(math.sqrt(d // 2)))) for d in feature_dims]
        
        # 创建 Embedding 层
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=emb_dim, padding_idx=0)
            for dim, emb_dim in zip(feature_dims, embedding_dims)
        ])
        
        # 计算总 embedding 维度
        total_emb_dim = sum(embedding_dims)
        
        self.activation = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'gelu': nn.GELU(),
            'swish': lambda x: x * torch.sigmoid(x)
        }[activation]
        
        # MLP 部分
        layers = []
        prev_dim = total_emb_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        # 注意：这里不加 Sigmoid，Loss 会用 BCEWithLogitsLoss
        
        self.mlp = nn.Sequential(*layers)
        
        print(f"📊 Embedding 统计:")
        print(f"   特征数：{len(feature_dims)}")
        print(f"   Embedding 维度：{embedding_dims[:5]}... (共{len(embedding_dims)}个)")
        print(f"   总 Embedding 维度：{total_emb_dim}")
        emb_params = sum(e.weight.numel() for e in self.embeddings)
        print(f"   Embedding 参数量：{emb_params:,} ({emb_params/1e6:.2f}M)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_features]，整数索引
        Returns:
            [batch_size] 的预测概率
        """
        # Embedding 查找
        embeddings = []
        for i, embedding_layer in enumerate(self.embeddings):
            feat_idx = x[:, i].long()
            emb = embedding_layer(feat_idx)  # [batch_size, emb_dim]
            embeddings.append(emb)
        
        # 拼接所有 embedding
        concat = torch.cat(embeddings, dim=-1)  # [batch_size, total_emb_dim]
        
        # MLP
        output = self.mlp(concat)
        
        return output.squeeze(-1)  # 返回 logits，sigmoid 在 loss 中处理


class DeepFMModel(nn.Module):
    """
    DeepFM 简化版 - 同时学习低阶和高阶特征交互
    
    适合：类别特征，需要显式特征交叉
    """
    factorization_machine_dim = 8
    
    def __init__(self,
                 feature_dims: List[int],
                 embedding_dim: int = 16,
                 mlp_hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_features = len(feature_dims)
        
        # FM 部分的 Embedding
        self.fm_embedding = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=self.factorization_machine_dim, padding_idx=0)
            for dim in feature_dims
        ])
        
        # Deep 部分的 Embedding
        self.deep_embedding = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=embedding_dim, padding_idx=0)
            for dim in feature_dims
        ])
        
        total_emb_dim = embedding_dim * self.num_features
        
        # Deep Network
        deep_layers = []
        prev_dim = total_emb_dim
        for hidden_dim in mlp_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        deep_layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.deep_network = nn.Sequential(*deep_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_features]
        """
        batch_size = x.size(0)
        
        # === FM Part ===
        fm_embeddings = []
        for i, embedding_layer in enumerate(self.fm_embedding):
            feat_idx = x[:, i].long()
            emb = embedding_layer(feat_idx)  # [batch_size, fm_dim]
            fm_embeddings.append(emb)
        
        fm_concat = torch.stack(fm_embeddings, dim=1)  # [batch_size, num_features, fm_dim]
        
        # Inner Product
        fm_output = 0.0
        for i in range(self.num_features):
            for j in range(i + 1, self.num_features):
                fm_output += (fm_concat[:, i] * fm_concat[:, j]).sum(dim=-1, keepdim=True)
        
        # === Deep Part ===
        deep_embeddings = []
        for i, embedding_layer in enumerate(self.deep_embedding):
            feat_idx = x[:, i].long()
            emb = embedding_layer(feat_idx)
            deep_embeddings.append(emb)
        
        deep_concat = torch.cat(deep_embeddings, dim=-1)  # [batch_size, total_emb_dim]
        deep_output = self.deep_network(deep_concat)
        
        # Combine
        output = fm_output + deep_output
        
        return output.squeeze(-1)


def create_model_with_embeddings(model_type: str, 
                                 feature_dims: List[int],
                                 **kwargs) -> nn.Module:
    """
    创建支持类别特征的模型
    
    Args:
        model_type: 'embedding_mlp' | 'embedding_widedeep' | 'mlp' | 'widedeep' | 'deepfm'
        feature_dims: 每个特征的 cardinality (类别数)
        **kwargs: 其他模型参数
    
    Returns:
        模型实例
    """
    # 兼容别名
    model_type_map = {
        'mlp': 'embedding_mlp',
        'widedeep': 'embedding_mlp',  # 暂时用 MLP 替代
        'wide_deep': 'embedding_mlp',
        'embedding_mlp': 'embedding_mlp',
        'embedding_widedeep': 'embedding_mlp',  # 暂时用 MLP 替代
        'deepfm': 'deepfm',
    }
    
    actual_type = model_type_map.get(model_type)
    if actual_type is None:
        raise ValueError(f"未知的模型类型：{model_type}，支持的类型：{list(model_type_map.keys())}")
    
    if actual_type == 'embedding_mlp':
        return EmbeddingMLP(feature_dims=feature_dims, **kwargs)
    elif actual_type == 'deepfm':
        return DeepFMModel(feature_dims=feature_dims, **kwargs)
    else:
        raise ValueError(f"未知的模型类型：{actual_type}")


# ============ 测试 ============

if __name__ == "__main__":
    print("=== 类别特征模型测试 ===\n")
    
    # 模拟数据：10 个特征，cardinality 分别为 [100, 200, 50, 1000, 500, 300, 800, 200, 150, 400]
    feature_dims = [100, 200, 50, 1000, 500, 300, 800, 200, 150, 400]
    batch_size = 32
    
    # 随机生成类别索引
    x = torch.randint(0, max(feature_dims), (batch_size, len(feature_dims)))
    
    # 1. EmbeddingMLP
    print("1. EmbeddingMLP:")
    model1 = create_model_with_embeddings(
        model_type='embedding_mlp',
        feature_dims=feature_dims,
        hidden_dims=[128, 64]
    )
    out1 = model1(x)
    print(f"   Input: {x.shape} → Output: {out1.shape}\n")
    
    # 2. DeepFM
    print("2. DeepFM:")
    model2 = create_model_with_embeddings(
        model_type='deepfm',
        feature_dims=feature_dims,
        embedding_dim=16
    )
    out2 = model2(x)
    print(f"   Input: {x.shape} → Output: {out2.shape}\n")
    
    print("✅ 类别特征模型测试通过！")
