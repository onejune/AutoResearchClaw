"""
exp006: FiBiNET + AutoFIS (RecSys 2019 + KDD 2020)

核心思想:
- FiBiNET: 用 SENet 动态学习特征重要性
- AutoFIS: 自动搜索有效的特征交互
- 解决特征重要性不均和无效交互的问题

参考论文:
- "FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate Prediction" (RecSys 2019)
- "AutoFIS: Automatic Feature Interaction Selection in Factorization Models for Click-Through Rate Prediction" (KDD 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import itertools


class SENetLayer(nn.Module):
    """
    Squeeze-and-Excitation Network 层
    
    动态学习每个特征的重要性权重
    """
    
    def __init__(
        self,
        num_features: int,
        reduction_ratio: int = 4
    ):
        """
        Args:
            num_features: 特征数量
            reduction_ratio: 压缩比例
        """
        super().__init__()
        
        reduced_dim = max(1, num_features // reduction_ratio)
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_features, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_features, embedding_dim)
        
        Returns:
            weighted_x: (batch, num_features, embedding_dim)
            weights: (batch, num_features)
        """
        batch_size = x.size(0)
        
        # Squeeze: global average pooling
        # (batch, num_features, dim) -> (batch, num_features)
        squeezed = x.mean(dim=-1)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        weights = self.excitation(squeezed)  # (batch, num_features)
        
        # Scale
        weighted_x = x * weights.unsqueeze(-1)
        
        return weighted_x, weights


class BilinearInteraction(nn.Module):
    """
    双线性特征交互层
    
    比简单的内积交互更有表达能力
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_features: int,
        bilinear_type: str = "field_all"  # "field_each", "field_all", "field_interaction"
    ):
        """
        Args:
            embedding_dim: embedding 维度
            num_features: 特征数量
            bilinear_type: 双线性类型
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.bilinear_type = bilinear_type
        
        num_pairs = num_features * (num_features - 1) // 2
        
        if bilinear_type == "field_each":
            # 每个特征一个 W
            self.W = nn.ParameterList([
                nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
                for _ in range(num_features)
            ])
        elif bilinear_type == "field_all":
            # 所有特征共享一个 W
            self.W = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
        elif bilinear_type == "field_interaction":
            # 每对交互一个 W
            self.W = nn.ParameterList([
                nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
                for _ in range(num_pairs)
            ])
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: list of (batch, dim) tensors
        
        Returns:
            interactions: (batch, num_pairs * dim)
        """
        interactions = []
        pair_idx = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                vi = embeddings[i]  # (batch, dim)
                vj = embeddings[j]  # (batch, dim)
                
                if self.bilinear_type == "field_each":
                    # vi^T * Wi * vj
                    Wi = self.W[i]
                    interaction = vi @ Wi * vj  # (batch, dim)
                elif self.bilinear_type == "field_all":
                    interaction = vi @ self.W * vj
                elif self.bilinear_type == "field_interaction":
                    Wij = self.W[pair_idx]
                    interaction = vi @ Wij * vj
                    pair_idx += 1
                
                interactions.append(interaction)
        
        return torch.cat(interactions, dim=-1)


class AutoFISLayer(nn.Module):
    """
    AutoFIS: 自动特征交互选择
    
    用 Gumbel-Softmax 学习哪些特征交互是有效的
    """
    
    def __init__(
        self,
        num_features: int,
        temperature: float = 1.0
    ):
        """
        Args:
            num_features: 特征数量
            temperature: Gumbel-Softmax 温度
        """
        super().__init__()
        
        self.num_features = num_features
        self.temperature = temperature
        
        # 每对特征交互的选择 logits
        num_pairs = num_features * (num_features - 1) // 2
        self.interaction_logits = nn.Parameter(torch.zeros(num_pairs, 2))  # [keep, drop]
    
    def forward(self, interactions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            interactions: (batch, num_pairs * dim)
        
        Returns:
            masked_interactions: (batch, num_pairs * dim)
            selection_probs: (num_pairs,) 每对交互被选中的概率
        """
        num_pairs = self.interaction_logits.size(0)
        
        # Gumbel-Softmax 选择
        if self.training:
            selection = F.gumbel_softmax(
                self.interaction_logits,
                tau=self.temperature,
                hard=False
            )[:, 0]  # (num_pairs,) 选择 keep 的概率
        else:
            selection = F.softmax(self.interaction_logits, dim=-1)[:, 0]
        
        # 应用 mask
        # interactions: (batch, num_pairs * dim)
        # 需要将 selection 扩展到每个 dim
        dim_per_pair = interactions.size(-1) // num_pairs
        
        # (num_pairs,) -> (num_pairs * dim,)
        selection_expanded = selection.repeat_interleave(dim_per_pair)
        
        masked = interactions * selection_expanded.unsqueeze(0)
        
        return masked, selection
    
    def get_selected_interactions(self, feature_names: List[str]) -> List[Tuple[str, str, float]]:
        """获取被选中的特征交互对"""
        probs = F.softmax(self.interaction_logits, dim=-1)[:, 0].detach().cpu().numpy()
        
        pairs = []
        idx = 0
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                pairs.append((feature_names[i], feature_names[j], float(probs[idx])))
                idx += 1
        
        # 按概率排序
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


class FiBiNETWideDeep(nn.Module):
    """FiBiNET + AutoFIS WideDeep 模型"""
    
    def __init__(
        self,
        feature_config: Dict[str, int],
        embedding_size: int = 64,
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3,
        bilinear_type: str = "field_all",
        use_senet: bool = True,
        use_autofis: bool = True,
        senet_reduction: int = 4,
        autofis_temperature: float = 1.0
    ):
        """
        Args:
            feature_config: {feature_name: vocab_size}
            embedding_size: embedding 维度
            dnn_hidden_units: DNN 隐藏层
            dropout: dropout 比例
            bilinear_type: 双线性类型
            use_senet: 是否使用 SENet
            use_autofis: 是否使用 AutoFIS
            senet_reduction: SENet 压缩比例
            autofis_temperature: AutoFIS 温度
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.feature_names = list(feature_config.keys())
        self.embedding_size = embedding_size
        self.use_senet = use_senet
        self.use_autofis = use_autofis
        
        num_features = len(feature_config)
        num_pairs = num_features * (num_features - 1) // 2
        
        # Embedding 层
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)
            for name, vocab_size in feature_config.items()
        })
        
        # SENet 层
        if use_senet:
            self.senet = SENetLayer(num_features, senet_reduction)
        
        # 双线性交互层
        self.bilinear = BilinearInteraction(
            embedding_size, num_features, bilinear_type
        )
        
        # AutoFIS 层
        if use_autofis:
            self.autofis = AutoFISLayer(num_features, autofis_temperature)
        
        # DNN 输入维度
        dnn_input_dim = num_features * embedding_size + num_pairs * embedding_size
        
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
        
        # 初始化
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {feature_name: (batch_size,)}
        
        Returns:
            logits: (batch_size,)
        """
        # 获取所有 embedding
        emb_list = []
        for name in self.feature_names:
            if name in features:
                emb_list.append(self.embeddings[name](features[name]))
            else:
                # 填充零向量
                batch_size = next(iter(features.values())).size(0)
                emb_list.append(torch.zeros(batch_size, self.embedding_size, device=next(self.parameters()).device))
        
        # Stack: (batch, num_features, dim)
        emb_stack = torch.stack(emb_list, dim=1)
        
        # SENet 重加权
        if self.use_senet:
            emb_stack, senet_weights = self.senet(emb_stack)
        
        # 双线性交互
        interactions = self.bilinear(emb_list)
        
        # AutoFIS 选择
        if self.use_autofis:
            interactions, autofis_probs = self.autofis(interactions)
        
        # 拼接
        emb_flat = emb_stack.view(emb_stack.size(0), -1)
        x = torch.cat([emb_flat, interactions], dim=-1)
        
        # DNN
        x = self.dnn(x)
        logits = self.output_layer(x).squeeze(-1)
        
        return logits
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取 SENet 学到的特征重要性"""
        if not self.use_senet:
            return {}
        
        # 用一个 dummy input 获取权重
        # 这里简化处理，直接返回 excitation 层的偏置作为近似
        weights = self.senet.excitation[-2].bias.detach().cpu().numpy()
        
        return {name: float(w) for name, w in zip(self.feature_names, weights)}
    
    def get_selected_interactions(self) -> List[Tuple[str, str, float]]:
        """获取 AutoFIS 选中的特征交互"""
        if not self.use_autofis:
            return []
        
        return self.autofis.get_selected_interactions(self.feature_names)
    
    def get_parameter_stats(self) -> Dict:
        """获取参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "total_params": total_params,
            "use_senet": self.use_senet,
            "use_autofis": self.use_autofis,
            "num_features": len(self.feature_config),
            "num_interactions": len(self.feature_config) * (len(self.feature_config) - 1) // 2
        }
