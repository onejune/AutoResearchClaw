"""
ChorusCVR 模型实现
论文: ChorusCVR: Chorus Supervision for Entire Space Post-Click Conversion Rate Modeling
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .towers import EmbeddingLayer, SharedBottom, Tower


class ChorusCVR(nn.Module):
    """
    ChorusCVR 模型
    
    架构:
        Input -> Embedding -> SharedBottom -> [CTR Tower, CVR Tower, unCVR Tower]
        
    输出:
        - pCTR: 点击率预测
        - pCVR: 转化率预测  
        - pUnCVR: 未转化率预测
        - pCTCVR: pCTR * pCVR
        - pCTunCVR: pCTR * pUnCVR
    """
    
    def __init__(
        self,
        sparse_feature_dims: dict,
        dense_feature_num: int,
        embedding_dim: int = 16,
        shared_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        # 1. Embedding 层
        self.embedding = EmbeddingLayer(
            sparse_feature_dims=sparse_feature_dims,
            dense_feature_num=dense_feature_num,
            embedding_dim=embedding_dim
        )
        
        # 2. 共享底层网络
        self.shared_bottom = SharedBottom(
            input_dim=self.embedding.output_dim,
            hidden_dims=shared_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        tower_input_dim = self.shared_bottom.output_dim
        
        # 3. CTR Tower
        self.ctr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # 4. CVR Tower
        self.cvr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # 5. unCVR Tower (NDM 模块核心)
        self.uncvr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
    
    def forward(
        self,
        sparse_features: dict,
        dense_features: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            sparse_features: 稀疏特征字典 {feature_name: (batch_size,)}
            dense_features: 稠密特征 (batch_size, dense_num)
            
        Returns:
            预测结果字典:
                - pCTR: (batch_size,)
                - pCVR: (batch_size,)
                - pUnCVR: (batch_size,)
                - pCTCVR: (batch_size,)
                - pCTunCVR: (batch_size,)
        """
        # Embedding
        emb = self.embedding(sparse_features, dense_features)
        
        # 共享底层
        shared_repr = self.shared_bottom(emb)
        
        # 三个 Tower 预测
        pCTR = self.ctr_tower(shared_repr)
        pCVR = self.cvr_tower(shared_repr)
        pUnCVR = self.uncvr_tower(shared_repr)
        
        # 组合预测
        pCTCVR = pCTR * pCVR          # CTCVR = CTR * CVR
        pCTunCVR = pCTR * pUnCVR      # CTunCVR = CTR * unCVR
        
        return {
            'pCTR': pCTR,
            'pCVR': pCVR,
            'pUnCVR': pUnCVR,
            'pCTCVR': pCTCVR,
            'pCTunCVR': pCTunCVR
        }
    
    def predict(
        self,
        sparse_features: dict,
        dense_features: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """推理时使用"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sparse_features, dense_features)
        return {k: v.cpu() for k, v in outputs.items()}


class ESMM(nn.Module):
    """
    ESMM 基线模型 (用于对比)
    只有 CTR 和 CVR 两个 Tower
    """
    
    def __init__(
        self,
        sparse_feature_dims: dict,
        dense_feature_num: int,
        embedding_dim: int = 16,
        shared_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.embedding = EmbeddingLayer(
            sparse_feature_dims=sparse_feature_dims,
            dense_feature_num=dense_feature_num,
            embedding_dim=embedding_dim
        )
        
        self.shared_bottom = SharedBottom(
            input_dim=self.embedding.output_dim,
            hidden_dims=shared_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        tower_input_dim = self.shared_bottom.output_dim
        
        self.ctr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        self.cvr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
    
    def forward(
        self,
        sparse_features: dict,
        dense_features: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        emb = self.embedding(sparse_features, dense_features)
        shared_repr = self.shared_bottom(emb)
        
        pCTR = self.ctr_tower(shared_repr)
        pCVR = self.cvr_tower(shared_repr)
        pCTCVR = pCTR * pCVR
        
        return {
            'pCTR': pCTR,
            'pCVR': pCVR,
            'pCTCVR': pCTCVR
        }
