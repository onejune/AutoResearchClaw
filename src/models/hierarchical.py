"""
exp003: Hierarchical Embeddings (CIKM 2021)

核心思想:
- 为 ID 特征同时学习细粒度和粗粒度 embedding
- 用门控网络动态融合两种表示
- 稀疏 ID 自动偏向粗粒度（泛化强），密集 ID 偏向细粒度（信息丰富）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class HierarchicalEmbedding(nn.Module):
    """层次化 Embedding 层"""
    
    def __init__(
        self,
        fine_vocab_size: int,
        coarse_vocab_size: int,
        embedding_dim: int = 64,
        gate_hidden_dim: int = 32
    ):
        """
        Args:
            fine_vocab_size: 细粒度 vocab 大小（如 adid 数量）
            coarse_vocab_size: 粗粒度 vocab 大小（如 category 数量）
            embedding_dim: embedding 维度
            gate_hidden_dim: 门控网络隐藏层维度
        """
        super().__init__()
        
        # 细粒度 embedding（如 adid）
        self.fine_embedding = nn.Embedding(fine_vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 粗粒度 embedding（如 category）
        self.coarse_embedding = nn.Embedding(coarse_vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 门控网络：决定融合比例
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化
        nn.init.xavier_uniform_(self.fine_embedding.weight)
        nn.init.xavier_uniform_(self.coarse_embedding.weight)
    
    def forward(self, fine_ids: torch.Tensor, coarse_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fine_ids: 细粒度 ID (batch_size,)
            coarse_ids: 粗粒度 ID (batch_size,)
        
        Returns:
            fused_embedding: (batch_size, embedding_dim)
        """
        # 获取 embedding
        fine_emb = self.fine_embedding(fine_ids)  # (batch, dim)
        coarse_emb = self.coarse_embedding(coarse_ids)  # (batch, dim)
        
        # 计算门控权重
        combined = torch.cat([fine_emb, coarse_emb], dim=-1)  # (batch, dim*2)
        gate = self.gate_network(combined)  # (batch, 1)
        
        # 融合
        fused_emb = gate * fine_emb + (1 - gate) * coarse_emb
        
        return fused_emb, gate


class HierarchicalWideDeep(nn.Module):
    """带层次化 Embedding 的 WideDeep 模型"""
    
    def __init__(
        self,
        feature_config: Dict[str, int],
        hierarchical_pairs: Dict[str, str],  # {fine_feat: coarse_feat}
        embedding_size: int = 64,
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3
    ):
        """
        Args:
            feature_config: {feature_name: vocab_size}
            hierarchical_pairs: {细粒度特征：粗粒度特征}
            embedding_size: embedding 维度
            dnn_hidden_units: DNN 隐藏层维度列表
            dropout: dropout 比例
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.hierarchical_pairs = hierarchical_pairs
        self.embedding_size = embedding_size
        
        # 识别哪些特征有层次结构
        self.hierarchical_features = set(hierarchical_pairs.keys())
        self.non_hierarchical_features = set(feature_config.keys()) - self.hierarchical_features
        
        # 1. 层次化 Embedding（用于有层次结构的特征）
        self.hierarchical_embeddings = nn.ModuleDict()
        for fine_feat, coarse_feat in hierarchical_pairs.items():
            if fine_feat in feature_config and coarse_feat in feature_config:
                self.hierarchical_embeddings[fine_feat] = HierarchicalEmbedding(
                    fine_vocab_size=feature_config[fine_feat],
                    coarse_vocab_size=feature_config[coarse_feat],
                    embedding_dim=embedding_size
                )
        
        # 2. 普通 Embedding（用于无层次结构的特征）
        self.normal_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)
            for feat, vocab_size in feature_config.items()
            if feat not in self.hierarchical_features
        })
        
        # 3. Deep 部分
        deep_input_dim = len(feature_config) * embedding_size
        deep_layers = []
        prev_dim = deep_input_dim
        for hidden_unit in dnn_hidden_units:
            deep_layers.append(nn.Linear(prev_dim, hidden_unit))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_unit
        
        self.deep_network = nn.Sequential(*deep_layers)
        
        # 4. 输出层
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 初始化
        for emb in self.normal_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {feature_name: tensor(batch_size)}
                     对于层次化特征，还需要 {feat}_coarse
        
        Returns:
            logits: (batch_size,) CTR 预测 logits
        """
        embedded_features = []
        
        # 处理层次化特征
        for fine_feat in self.hierarchical_features:
            coarse_feat = self.hierarchical_pairs[fine_feat]
            
            fine_ids = features.get(fine_feat)
            coarse_ids = features.get(f"{fine_feat}_coarse")
            
            if fine_ids is not None and coarse_ids is not None:
                fused_emb, _ = self.hierarchical_embeddings[fine_feat](fine_ids, coarse_ids)
                embedded_features.append(fused_emb)
            elif fine_ids is not None:
                # 只有细粒度，退化为普通 embedding
                emb = self.hierarchical_embeddings[fine_feat].fine_embedding(fine_ids)
                embedded_features.append(emb)
        
        # 处理普通特征
        for feat in self.non_hierarchical_features:
            if feat in features:
                emb = self.normal_embeddings[feat](features[feat])
                embedded_features.append(emb)
        
        # 拼接所有 embedding
        deep_input = torch.cat(embedded_features, dim=-1)
        
        # Deep 网络
        deep_out = self.deep_network(deep_input)
        
        # 输出
        logits = self.output_layer(deep_out).squeeze(-1)
        
        return logits
    
    def get_gate_stats(self, features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """获取门控统计信息（用于分析）"""
        stats = {}
        
        for fine_feat in self.hierarchical_features:
            coarse_feat = self.hierarchical_pairs[fine_feat]
            fine_ids = features.get(fine_feat)
            coarse_ids = features.get(f"{fine_feat}_coarse")
            
            if fine_ids is not None and coarse_ids is not None:
                _, gate = self.hierarchical_embeddings[fine_feat](fine_ids, coarse_ids)
                stats[fine_feat] = {
                    "mean_gate": float(gate.mean()),
                    "std_gate": float(gate.std())
                }
        
        return stats
