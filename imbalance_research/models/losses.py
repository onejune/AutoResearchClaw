"""各种 Focal Loss 及其变体实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class FocalLoss(nn.Module):
    """
    标准 Focal Loss
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: 类别权重，可以是标量或 [pos_weight, neg_weight]
        gamma: 聚焦参数，控制难易样本权重差异
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, 
                 alpha: Union[float, List[float]] = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # PT = p if y=1 else 1-p
        pt = torch.exp(-bce_loss)
        
        # Focal weight
        focal_weight = torch.pow(1 - pt, self.gamma)
        
        # Alpha weight
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets.long()]
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BalancedFocalLoss(nn.Module):
    """
    Balanced Focal Loss - 自动根据类别频率调整 alpha
    
    适用于极度不平衡场景（如 1:1000+）
    """
    def __init__(self,
                 gamma: float = 2.0,
                 sampling_strategy: str = 'frequency',
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.sampling_strategy = sampling_strategy
        self.reduction = reduction
        self.register_buffer('class_counts', torch.zeros(2))
    
    def update_class_counts(self, targets: torch.Tensor):
        """更新类别统计"""
        pos_count = targets.sum().item()
        neg_count = targets.numel() - pos_count
        self.class_counts[0] = neg_count  # class 0
        self.class_counts[1] = pos_count  # class 1
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 更新统计（训练时）
        if self.training:
            self.update_class_counts(targets)
        
        # 计算动态 alpha
        total = self.class_counts.sum() + 1e-8
        pos_ratio = self.class_counts[1] / total
        
        # 正样本越少，alpha 越大
        alpha_pos = 1.0 - pos_ratio
        alpha_neg = pos_ratio
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Focal weight
        focal_weight = torch.pow(1 - pt, self.gamma)
        
        # Alpha weight
        alpha_t = alpha_pos * targets + alpha_neg * (1 - targets)
        
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss - 正负样本使用不同的 gamma
    
    适用于正负样本难度分布明显不同的场景
    """
    def __init__(self,
                 gamma_pos: float = 2.0,
                 gamma_neg: float = 1.0,
                 alpha: float = 0.25,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # 正负样本不同的 gamma
        gamma_t = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        focal_weight = torch.pow(1 - pt, gamma_t)
        
        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DynamicFocalLoss(nn.Module):
    """
    Dynamic Focal Loss - 训练过程中动态调整参数
    
    策略：
    - gamma 随 epoch 衰减（从聚焦难样本 → 平衡整体）
    - alpha 根据当前 batch 的正负比动态调整
    """
    def __init__(self,
                 gamma_init: float = 3.0,
                 gamma_end: float = 1.0,
                 total_epochs: int = 100,
                 alpha_strategy: str = 'auto',  # 'auto' | 'fixed'
                 fixed_alpha: float = 0.25,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma_init = gamma_init
        self.gamma_end = gamma_end
        self.total_epochs = total_epochs
        self.alpha_strategy = alpha_strategy
        self.fixed_alpha = fixed_alpha
        self.reduction = reduction
        
        self.current_epoch = 0
        self.gamma = gamma_init
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch（每个 epoch 开始时调用）"""
        self.current_epoch = epoch
        
        # 线性衰减 gamma
        progress = epoch / max(self.total_epochs - 1, 1)
        self.gamma = self.gamma_init - (self.gamma_init - self.gamma_end) * progress
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Focal weight
        focal_weight = torch.pow(1 - pt, self.gamma)
        
        # Alpha weight
        if self.alpha_strategy == 'auto':
            # 根据当前 batch 正负比动态调整
            pos_ratio = targets.mean()
            alpha_pos = 1.0 - pos_ratio
            alpha_neg = pos_ratio
            alpha_t = alpha_pos * targets + alpha_neg * (1 - targets)
        else:
            alpha_t = self.fixed_alpha * targets + (1 - self.fixed_alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def get_current_gamma(self) -> float:
        """获取当前 gamma 值"""
        return self.gamma


class GroupFocalLoss(nn.Module):
    """
    Group Focal Loss - 按组（如用户 ID）计算 Focal Loss
    
    类似 GAUC 的思想，避免头部用户主导训练
    """
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 group_by: Optional[str] = None,  # 用于标识分组字段
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.group_by = group_by
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, 
                targets: torch.Tensor,
                groups: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: 模型预测 [batch_size]
            targets: 真实标签 [batch_size]
            groups: 分组 ID [batch_size], 如果为 None 则退化为普通 Focal Loss
        """
        if groups is None:
            # 退化为标准 Focal Loss
            focal = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
            return focal(inputs, targets)
        
        # 按组计算 Loss
        unique_groups = groups.unique()
        group_losses = []
        
        for group_id in unique_groups:
            mask = groups == group_id
            group_inputs = inputs[mask]
            group_targets = targets[mask]
            
            # 组内计算 Focal Loss
            bce_loss = F.binary_cross_entropy_with_logits(
                group_inputs, group_targets, reduction='none'
            )
            pt = torch.exp(-bce_loss)
            focal_weight = torch.pow(1 - pt, self.gamma)
            alpha_t = self.alpha * group_targets + (1 - self.alpha) * (1 - group_targets)
            group_loss = alpha_t * focal_weight * bce_loss
            
            group_losses.append(group_loss.mean())
        
        # 平均所有组的 Loss
        return torch.stack(group_losses).mean()


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss + Label Smoothing
    
    防止概率输出过于极端，提升校准度
    """
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 epsilon: float = 0.1,  # Label smoothing 参数
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        smoothed_targets = (1 - self.epsilon) * targets + self.epsilon / 2
        
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        
        focal_weight = torch.pow(1 - pt, self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============ Loss Factory ============

def create_focal_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Focal Loss 工厂函数
    
    Args:
        loss_type: 'bce' | 'focal' | 'balanced' | 'asymmetric' | 'dynamic' | 'group' | 'smoothed'
        **kwargs: 对应 Loss 的参数
    
    Returns:
        Loss 模块实例
    """
    loss_classes = {
        'bce': nn.BCEWithLogitsLoss,  # 接受 logits，内部做 sigmoid
        'focal': FocalLoss,
        'balanced': BalancedFocalLoss,
        'asymmetric': AsymmetricFocalLoss,
        'dynamic': DynamicFocalLoss,
        'group': GroupFocalLoss,
        'smoothed': FocalLossWithLabelSmoothing,
    }
    
    if loss_type not in loss_classes:
        raise ValueError(f"未知的 Loss 类型：{loss_type}. 可选：{list(loss_classes.keys())}")
    
    # BCEWithLogitsLoss 不需要额外参数
    if loss_type == 'bce':
        return loss_classes['bce']()
    
    return loss_classes[loss_type](**kwargs)


# ============ 测试 ============

if __name__ == "__main__":
    # 模拟数据
    inputs = torch.randn(32, requires_grad=True)
    targets = torch.randint(0, 2, (32,)).float()
    
    print("=== Focal Loss 家族测试 ===\n")
    
    # 1. 标准 Focal Loss
    loss1 = FocalLoss(alpha=0.25, gamma=2.0)
    l1 = loss1(inputs, targets)
    print(f"1. FocalLoss: {l1.item():.4f}")
    
    # 2. Balanced Focal Loss
    loss2 = BalancedFocalLoss(gamma=2.0)
    l2 = loss2(inputs, targets)
    print(f"2. BalancedFocalLoss: {l2.item():.4f}")
    
    # 3. Asymmetric Focal Loss
    loss3 = AsymmetricFocalLoss(gamma_pos=2.0, gamma_neg=1.0)
    l3 = loss3(inputs, targets)
    print(f"3. AsymmetricFocalLoss: {l3.item():.4f}")
    
    # 4. Dynamic Focal Loss
    loss4 = DynamicFocalLoss(gamma_init=3.0, total_epochs=100)
    loss4.set_epoch(0)
    l4 = loss4(inputs, targets)
    print(f"4. DynamicFocalLoss (epoch 0, gamma={loss4.gamma:.2f}): {l4.item():.4f}")
    
    loss4.set_epoch(50)
    l4_50 = loss4(inputs, targets)
    print(f"   DynamicFocalLoss (epoch 50, gamma={loss4.gamma:.2f}): {l4_50.item():.4f}")
    
    # 5. Group Focal Loss
    groups = torch.randint(0, 5, (32,))  # 5 个用户组
    loss5 = GroupFocalLoss(gamma=2.0)
    l5 = loss5(inputs, targets, groups)
    print(f"5. GroupFocalLoss: {l5.item():.4f}")
    
    # 6. Focal Loss with Label Smoothing
    loss6 = FocalLossWithLabelSmoothing(gamma=2.0, epsilon=0.1)
    l6 = loss6(inputs, targets)
    print(f"6. FocalLossWithLabelSmoothing: {l6.item():.4f}")
    
    # 7. Factory 测试
    loss7 = create_focal_loss('focal', alpha=0.25, gamma=2.0)
    l7 = loss7(inputs, targets)
    print(f"7. Factory (focal): {l7.item():.4f}")
    
    print("\n✅ 所有 Loss 测试通过！")
