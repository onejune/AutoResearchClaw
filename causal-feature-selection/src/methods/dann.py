# Domain Adversarial Neural Network (DANN)

import torch
import torch.nn as nn
from typing import Dict, List

class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层 - 前向传播恒等，反向传播反转梯度"""
    
    @staticmethod
    def forward(ctx, x, lambda_coeff):
        ctx.lambda_coeff = lambda_coeff
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_coeff * grad_output, None


class DANNModel(nn.Module):
    """Domain Adversarial Neural Network for CTR Prediction"""
    
    def __init__(
        self,
        feature_config: Dict[str, int],
        embedding_size: int = 8,
        dnn_hidden_units: List[int] = [1024, 512, 256, 128],
        domain_hidden_units: List[int] = [256, 128, 1],
        dropout: float = 0.3
    ):
        """
        Args:
            feature_config: {feature_name: vocab_size}
            embedding_size: embedding 维度
            dnn_hidden_units: 特征提取器 DNN 结构
            domain_hidden_units: 域分类器 DNN 结构
            dropout: dropout 比例
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.embedding_size = embedding_size
        
        # Embedding 层
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)
            for name, vocab_size in feature_config.items()
        })
        
        # 特征提取器（共享）
        deep_input_dim = len(feature_config) * embedding_size
        feature_layers = []
        prev_dim = deep_input_dim
        for hidden_unit in dnn_hidden_units:
            feature_layers.append(nn.Linear(prev_dim, hidden_unit))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_unit
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        feature_dim = dnn_hidden_units[-1]
        
        # 标签分类器（CTR 预测）
        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 域分类器
        domain_layers = []
        prev_dim = feature_dim
        for hidden_unit in domain_hidden_units:
            domain_layers.append(nn.Linear(prev_dim, hidden_unit))
            if hidden_unit != 1:
                domain_layers.append(nn.ReLU())
                domain_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_unit
        
        self.domain_classifier = nn.Sequential(*domain_layers)
        
        self.grl = GradientReversalLayer.apply
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        lambda_coeff: float = 1.0,
        return_domain: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {feature_name: tensor(batch_size)}
            lambda_coeff: GRL 系数
            return_domain: 是否返回域预测
        
        Returns:
            logits: (batch_size,) CTR 预测 logits
            domain_logits: (batch_size,) 域预测 logits（如果 return_domain=True）
        """
        # Embedding lookup
        embedded_features = []
        for name, idx_tensor in features.items():
            if name in self.embeddings:
                embedded = self.embeddings[name](idx_tensor)
                embedded_features.append(embedded)
        
        # Concat all embeddings
        deep_input = torch.cat(embedded_features, dim=-1)
        
        # Feature extraction
        features_out = self.feature_extractor(deep_input)
        
        # Label prediction
        label_logits = self.label_classifier(features_out).squeeze(-1)
        
        result = {"logits": label_logits}
        
        # Domain prediction (with GRL)
        if return_domain:
            reversed_features = self.grl(features_out, lambda_coeff)
            domain_logits = self.domain_classifier(reversed_features).squeeze(-1)
            result["domain_logits"] = domain_logits
        
        return result


class DANNTrainer:
    """DANN 训练器"""
    
    def __init__(
        self,
        model: DANNModel,
        device: str = "cuda",
        lr: float = 5e-5,
        domain_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.domain_weight = domain_weight
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.label_criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(
        self,
        source_loader,  # 有标签数据
        target_loader,  # 无标签数据
        epoch: int = 1,
        total_epochs: int = 1,
        lambda_schedule: str = "linear"
    ):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_label_loss = 0.0
        total_domain_loss = 0.0
        n_batches = 0
        
        target_iter = iter(target_loader)
        
        for batch_idx, (source_features, source_labels) in enumerate(source_loader):
            # 获取目标域数据
            try:
                target_features, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_features, _ = next(target_iter)
            
            # 移到设备
            source_features = {k: v.to(self.device) for k, v in source_features.items()}
            source_labels = source_labels.to(self.device).float()
            target_features = {k: v.to(self.device) for k, v in target_features.items()}
            
            # 计算当前 lambda（渐进式）
            if lambda_schedule == "linear":
                # 从 0 线性增长到 domain_weight
                progress = batch_idx / len(source_loader)
                current_lambda = self.domain_weight * progress
            else:
                current_lambda = self.domain_weight
            
            # 拼接源域和目标域
            batch_size = source_labels.size(0)
            target_size = list(target_features.values())[0].size(0)
            
            # 域标签：源域=0，目标域=1
            domain_labels = torch.cat([
                torch.zeros(batch_size),
                torch.ones(target_size)
            ]).to(self.device)
            
            # 拼接特征
            combined_features = {}
            for k in source_features:
                combined_features[k] = torch.cat([source_features[k], target_features[k]], dim=0)
            
            # Forward
            outputs = self.model(combined_features, lambda_coeff=current_lambda, return_domain=True)
            
            # 标签损失（只有源域有标签）
            label_logits_source = outputs["logits"][:batch_size]
            label_loss = self.label_criterion(label_logits_source, source_labels)
            
            # 域损失
            domain_logits = outputs["domain_logits"]
            domain_loss = self.domain_criterion(domain_logits, domain_labels)
            
            # 总损失
            loss = label_loss + current_lambda * domain_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_label_loss += label_loss.item()
            total_domain_loss += domain_loss.item()
            n_batches += 1
        
        return {
            "train_loss": total_loss / n_batches,
            "label_loss": total_label_loss / n_batches,
            "domain_loss": total_domain_loss / n_batches
        }
