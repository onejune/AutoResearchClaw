"""
CVR 预测基线模型实现
包含论文中对比的所有基线: ESMM, ESCM2-IPW, ESCM2-DR, DCMT, DDPO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .towers import EmbeddingLayer, SharedBottom, Tower, MLP


class ESMM(nn.Module):
    """
    ESMM: Entire Space Multi-Task Model (SIGIR 2018)
    
    核心思想: CTCVR = CTR * CVR, 在全空间学习
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
    
    def forward(self, sparse_features: dict, dense_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
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


class ESCM2(nn.Module):
    """
    ESCM2: Entire Space Counterfactual Multi-Task Model (SIGIR 2022)
    
    支持两种模式:
    - IPW: 使用 Inverse Propensity Weighting
    - DR: 使用 Doubly Robust 方法
    """
    
    def __init__(
        self,
        sparse_feature_dims: dict,
        dense_feature_num: int,
        embedding_dim: int = 16,
        shared_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        activation: str = "relu",
        mode: str = "ipw"  # "ipw" or "dr"
    ):
        super().__init__()
        self.mode = mode
        
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
        
        # CTR Tower
        self.ctr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # CVR Tower
        self.cvr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # DR 模式需要额外的 imputation tower
        if mode == "dr":
            self.imputation_tower = Tower(
                input_dim=tower_input_dim,
                hidden_dims=tower_hidden_dims,
                dropout_rate=dropout_rate,
                activation=activation
            )
    
    def forward(self, sparse_features: dict, dense_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        emb = self.embedding(sparse_features, dense_features)
        shared_repr = self.shared_bottom(emb)
        
        pCTR = self.ctr_tower(shared_repr)
        pCVR = self.cvr_tower(shared_repr)
        pCTCVR = pCTR * pCVR
        
        outputs = {
            'pCTR': pCTR,
            'pCVR': pCVR,
            'pCTCVR': pCTCVR
        }
        
        if self.mode == "dr":
            # Imputation for DR
            pCVR_imp = self.imputation_tower(shared_repr)
            outputs['pCVR_imp'] = pCVR_imp
        
        return outputs


class DCMT(nn.Module):
    """
    DCMT: Direct Entire-Space Causal Multi-Task Framework (ICDE 2023)
    
    核心思想: 引入 counterfactual CVR 任务
    counterfactual CVR: 假设未点击样本为正样本, 转化样本为负样本
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
        
        # CTR Tower
        self.ctr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # CVR Tower
        self.cvr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # Counterfactual CVR Tower
        self.cf_cvr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
    
    def forward(self, sparse_features: dict, dense_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        emb = self.embedding(sparse_features, dense_features)
        shared_repr = self.shared_bottom(emb)
        
        pCTR = self.ctr_tower(shared_repr)
        pCVR = self.cvr_tower(shared_repr)
        pCTCVR = pCTR * pCVR
        
        # Counterfactual CVR
        pCF_CVR = self.cf_cvr_tower(shared_repr)
        
        return {
            'pCTR': pCTR,
            'pCVR': pCVR,
            'pCTCVR': pCTCVR,
            'pCF_CVR': pCF_CVR
        }


class DDPO(nn.Module):
    """
    DDPO: Direct Dual Propensity Optimization (SIGIR 2024)
    
    核心思想: 使用额外的 CVR tower 生成软标签
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
        
        # CTR Tower
        self.ctr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # Main CVR Tower
        self.cvr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # Extra CVR Tower (for soft label generation)
        self.extra_cvr_tower = Tower(
            input_dim=tower_input_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation
        )
    
    def forward(self, sparse_features: dict, dense_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        emb = self.embedding(sparse_features, dense_features)
        shared_repr = self.shared_bottom(emb)
        
        pCTR = self.ctr_tower(shared_repr)
        pCVR = self.cvr_tower(shared_repr)
        pCTCVR = pCTR * pCVR
        
        # Extra CVR for soft labels
        pCVR_extra = self.extra_cvr_tower(shared_repr)
        
        return {
            'pCTR': pCTR,
            'pCVR': pCVR,
            'pCTCVR': pCTCVR,
            'pCVR_extra': pCVR_extra
        }
