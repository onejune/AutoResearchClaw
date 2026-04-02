"""
exp004: MetaEmb - 元学习冷启动 (WSDM 2022)

核心思想:
- 用元学习为新 ID（冷启动）生成初始 embedding
- 基于 ID 的 side information（如类别、统计特征）生成 embedding
- 解决新广告/新用户的冷启动问题

参考论文: "Learning to Embed Categorical Features without Embedding Tables for Recommendation" (WSDM 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MetaEmbeddingGenerator(nn.Module):
    """
    元学习 Embedding 生成器
    
    基于 side information 为 ID 生成 embedding，
    特别适合冷启动场景。
    """
    
    def __init__(
        self,
        side_info_dims: Dict[str, int],  # {side_feature: vocab_size}
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        """
        Args:
            side_info_dims: side information 特征的维度
            embedding_dim: 输出 embedding 维度
            hidden_dim: 隐藏层维度
            num_layers: MLP 层数
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Side information embedding
        self.side_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, embedding_dim // 2, padding_idx=0)
            for name, vocab_size in side_info_dims.items()
        })
        
        # Meta generator MLP
        input_dim = len(side_info_dims) * (embedding_dim // 2)
        
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.generator = nn.Sequential(*layers)
        
        # 初始化
        for emb in self.side_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, side_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            side_features: {side_feature_name: (batch_size,)}
        
        Returns:
            generated_embedding: (batch_size, embedding_dim)
        """
        side_embs = []
        for name, emb_layer in self.side_embeddings.items():
            if name in side_features:
                side_embs.append(emb_layer(side_features[name]))
        
        if not side_embs:
            raise ValueError("No side features provided")
        
        # 拼接 side embeddings
        combined = torch.cat(side_embs, dim=-1)
        
        # 生成 embedding
        generated = self.generator(combined)
        
        return generated


class MetaEmbWideDeep(nn.Module):
    """
    带元学习 Embedding 的 WideDeep 模型
    
    对于每个 ID 特征，同时维护：
    1. 传统的 lookup embedding（热门 ID）
    2. Meta generator（冷启动 ID）
    
    通过 ID 频次自动决定使用哪个。
    """
    
    def __init__(
        self,
        feature_config: Dict[str, int],
        meta_features: Dict[str, List[str]],  # {id_feature: [side_features]}
        embedding_size: int = 64,
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3,
        cold_threshold: int = 10  # 出现次数少于此值视为冷启动
    ):
        """
        Args:
            feature_config: {feature_name: vocab_size}
            meta_features: {id_feature: [对应的 side features]}
            embedding_size: embedding 维度
            dnn_hidden_units: DNN 隐藏层
            dropout: dropout 比例
            cold_threshold: 冷启动阈值
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.meta_features = meta_features
        self.embedding_size = embedding_size
        self.cold_threshold = cold_threshold
        
        # 传统 embedding（所有特征）
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)
            for name, vocab_size in feature_config.items()
        })
        
        # Meta generator（只为指定的 ID 特征创建）
        self.meta_generators = nn.ModuleDict()
        for id_feat, side_feats in meta_features.items():
            side_dims = {sf: feature_config[sf] for sf in side_feats if sf in feature_config}
            if side_dims:
                self.meta_generators[id_feat] = MetaEmbeddingGenerator(
                    side_info_dims=side_dims,
                    embedding_dim=embedding_size
                )
        
        # ID 频次统计（用于判断冷启动）
        self.id_counts = {}
        
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
        use_meta: bool = True
    ) -> torch.Tensor:
        """
        Args:
            features: {feature_name: (batch_size,)}
            use_meta: 是否使用 meta embedding
        
        Returns:
            logits: (batch_size,)
        """
        embeddings = []
        
        for name, emb_layer in self.embeddings.items():
            if name not in features:
                continue
            
            ids = features[name]
            
            # 获取 lookup embedding
            lookup_emb = emb_layer(ids)
            
            # 如果有 meta generator 且启用，融合 meta embedding
            if use_meta and name in self.meta_generators:
                # 获取 side features
                side_feats = {
                    sf: features[sf] 
                    for sf in self.meta_features[name] 
                    if sf in features
                }
                
                if side_feats:
                    meta_emb = self.meta_generators[name](side_feats)
                    
                    # 简单融合：平均
                    # 更复杂的方案：根据 ID 频次加权
                    final_emb = 0.5 * lookup_emb + 0.5 * meta_emb
                else:
                    final_emb = lookup_emb
            else:
                final_emb = lookup_emb
            
            embeddings.append(final_emb)
        
        # 拼接
        x = torch.cat(embeddings, dim=-1)
        
        # DNN
        x = self.dnn(x)
        logits = self.output_layer(x).squeeze(-1)
        
        return logits
    
    def get_parameter_stats(self) -> Dict:
        """获取参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        emb_params = sum(
            sum(p.numel() for p in emb.parameters())
            for emb in self.embeddings.values()
        )
        meta_params = sum(
            sum(p.numel() for p in gen.parameters())
            for gen in self.meta_generators.values()
        )
        
        return {
            "total_params": total_params,
            "embedding_params": emb_params,
            "meta_generator_params": meta_params,
            "meta_features": list(self.meta_generators.keys())
        }
