#!/usr/bin/env python3
"""
对比学习 CVR 模型
参考 multitask 项目的模型架构，添加对比学习组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class EmbeddingLayer(nn.Module):
    """Embedding 层"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embed_dim)
            for feat, vocab_size in vocab_sizes.items()
        })
        self.embed_dim = embed_dim
        
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeds = [self.embeddings[feat](x[feat]) for feat in self.embeddings if feat in x]
        return torch.cat(embeds, dim=-1)


class TowerNetwork(nn.Module):
    """塔网络"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_type: str = 'classification'):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(dims[i+1])
            ])
        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)
        self.output_type = output_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.output_type == 'classification':
            return torch.sigmoid(out).squeeze(-1)
        else:
            return F.relu(out).squeeze(-1)


class ProjectionHead(nn.Module):
    """对比学习投影头"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class BaselineCVR(nn.Module):
    """基线 CVR 模型"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 32, 
                 hidden_dims: List[int] = [256, 128]):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        # 实际嵌入维度 = 实际特征数 * embed_dim
        # 这里在 forward 时动态计算
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.tower = None  # 延迟初始化
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embed = self.embedding(x)
        
        # 延迟初始化 tower
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = TowerNetwork(input_dim, self.hidden_dims).to(embed.device)
        
        p_cvr = self.tower(embed)
        return {'purchase': p_cvr}


class ContrastiveCVR(nn.Module):
    """对比学习增强的 CVR 模型"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 32,
                 hidden_dims: List[int] = [256, 128], proj_dim: int = 64,
                 temperature: float = 0.1):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.tower = None
        self.projection = None
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embed = self.embedding(x)
        
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = TowerNetwork(input_dim, self.hidden_dims).to(embed.device)
            self.projection = ProjectionHead(input_dim, self.hidden_dims[0], self.proj_dim).to(embed.device)
        
        p_cvr = self.tower(embed)
        z = self.projection(embed)
        return {'purchase': p_cvr, 'z': z}
    
    def contrastive_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """监督对比损失 - 同类标签为正例"""
        device = z.device
        batch_size = z.shape[0]
        
        # 相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature
        
        # 标签 mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        
        # 计算 loss
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 只在有正例的样本上计算
        mask_sum = mask.sum(1)
        valid_mask = mask_sum > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        mean_log_prob = (mask * log_prob).sum(1) / (mask_sum + 1e-8)
        loss = -mean_log_prob[valid_mask].mean()
        
        return loss


class BusinessTypeContrastiveCVR(nn.Module):
    """基于 business_type 的对比学习 CVR 模型"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 32,
                 hidden_dims: List[int] = [256, 128], proj_dim: int = 64,
                 temperature: float = 0.1):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.tower = None
        self.projection = None
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embed = self.embedding(x)
        
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = TowerNetwork(input_dim, self.hidden_dims).to(embed.device)
            self.projection = ProjectionHead(input_dim, self.hidden_dims[0], self.proj_dim).to(embed.device)
        
        p_cvr = self.tower(embed)
        z = self.projection(embed)
        return {'purchase': p_cvr, 'z': z}
    
    def contrastive_loss(self, z: torch.Tensor, business_type_ids: torch.Tensor) -> torch.Tensor:
        """基于 business_type 的对比损失 - 同 business_type 为正例"""
        device = z.device
        batch_size = z.shape[0]
        
        sim = torch.matmul(z, z.T) / self.temperature
        
        bt_ids = business_type_ids.view(-1, 1)
        mask = torch.eq(bt_ids, bt_ids.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mask_sum = mask.sum(1)
        valid_mask = mask_sum > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        mean_log_prob = (mask * log_prob).sum(1) / (mask_sum + 1e-8)
        loss = -mean_log_prob[valid_mask].mean()
        
        return loss


class UserContrastiveCVR(nn.Module):
    """基于用户行为的对比学习 CVR 模型 - 同用户点击样本为正例"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 32,
                 hidden_dims: List[int] = [256, 128], proj_dim: int = 64,
                 temperature: float = 0.1):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.tower = None
        self.projection = None
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embed = self.embedding(x)
        
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = TowerNetwork(input_dim, self.hidden_dims).to(embed.device)
            self.projection = ProjectionHead(input_dim, self.hidden_dims[0], self.proj_dim).to(embed.device)
        
        p_cvr = self.tower(embed)
        z = self.projection(embed)
        return {'purchase': p_cvr, 'z': z}
    
    def contrastive_loss(self, z: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        """基于用户的对比损失 - 同用户为正例"""
        device = z.device
        batch_size = z.shape[0]
        
        sim = torch.matmul(z, z.T) / self.temperature
        
        uids = user_ids.view(-1, 1)
        mask = torch.eq(uids, uids.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mask_sum = mask.sum(1)
        valid_mask = mask_sum > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        mean_log_prob = (mask * log_prob).sum(1) / (mask_sum + 1e-8)
        loss = -mean_log_prob[valid_mask].mean()
        
        return loss


class AugmentContrastiveCVR(nn.Module):
    """增强对比学习 CVR 模型 - 同样本不同 dropout 为正例"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 32,
                 hidden_dims: List[int] = [256, 128], proj_dim: int = 64,
                 temperature: float = 0.1, dropout_rate: float = 0.3):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.tower = None
        self.projection = None
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embed = self.embedding(x)
        
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = TowerNetwork(input_dim, self.hidden_dims).to(embed.device)
            self.projection = ProjectionHead(input_dim, self.hidden_dims[0], self.proj_dim).to(embed.device)
        
        p_cvr = self.tower(embed)
        
        # 生成两个视图（不同 dropout）
        if self.training:
            z1 = self.projection(self.dropout(embed))
            z2 = self.projection(self.dropout(embed))
            return {'purchase': p_cvr, 'z1': z1, 'z2': z2}
        else:
            z = self.projection(embed)
            return {'purchase': p_cvr, 'z': z}
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss - 同样本两视图为正例"""
        device = z1.device
        batch_size = z1.shape[0]
        
        # 拼接两视图
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        
        # 相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]
        
        # 正例 mask：(i, i+B) 和 (i+B, i) 互为正例
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = 1
            pos_mask[i + batch_size, i] = 1
        
        # 移除对角线
        logits_mask = 1 - torch.eye(2 * batch_size, device=device)
        
        # InfoNCE loss
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 只取正例位置
        loss = -(pos_mask * log_prob).sum() / pos_mask.sum()
        
        return loss


def build_model(model_name: str, vocab_sizes: Dict[str, int], config) -> nn.Module:
    """构建模型"""
    if model_name == 'baseline':
        return BaselineCVR(vocab_sizes, config.embed_dim, config.hidden_dims)
    elif model_name == 'contrastive':
        return ContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims, 
                              config.proj_dim, config.temperature)
    elif model_name == 'bt_contrastive':
        return BusinessTypeContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims,
                                          config.proj_dim, config.temperature)
    elif model_name == 'user_contrastive':
        return UserContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims,
                                  config.proj_dim, config.temperature)
    elif model_name == 'augment_contrastive':
        return AugmentContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims,
                                     config.proj_dim, config.temperature)
    else:
        raise ValueError(f'Unknown model: {model_name}')
