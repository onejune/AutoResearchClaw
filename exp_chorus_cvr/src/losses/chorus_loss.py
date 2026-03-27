"""
ChorusCVR 损失函数实现

总损失: L = L_ctcvr + L_cvr_IPW + L_ctuncvr + L_uncvr_IPW + L_align_IPW

论文公式对应:
- L_ctcvr: Eq.(3) - 全空间 CTCVR 损失
- L_cvr_IPW: Eq.(4) - 点击空间 CVR 损失 (IPW加权)
- L_ctuncvr: Eq.(7) - 全空间 CTunCVR 损失 (NDM核心)
- L_uncvr_IPW: Eq.(9) - 点击空间 unCVR 损失 (IPW加权)
- L_align_IPW: Eq.(10) - 软标签对齐损失 (SAM核心)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    二元交叉熵损失 (数值稳定版本)
    
    Args:
        pred: 预测概率 (batch_size,)
        target: 真实标签 (batch_size,)
        eps: 数值稳定性小量
    Returns:
        逐样本损失 (batch_size,)
    """
    pred = torch.clamp(pred, eps, 1 - eps)
    return -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)


class ChorusCVRLoss(nn.Module):
    """
    ChorusCVR 完整损失函数
    """
    
    def __init__(
        self,
        loss_weights: Dict[str, float] = None,
        ipw_clip_min: float = 0.01,
        ipw_clip_max: float = 1.0
    ):
        super().__init__()
        
        self.loss_weights = loss_weights or {
            'ctcvr': 1.0,
            'cvr_ipw': 1.0,
            'ctuncvr': 1.0,
            'uncvr_ipw': 1.0,
            'align_ipw': 1.0,
        }
        
        self.ipw_clip_min = ipw_clip_min
        self.ipw_clip_max = ipw_clip_max
    
    def _clip_propensity(self, propensity: torch.Tensor) -> torch.Tensor:
        """裁剪倾向性分数，防止除以过小的值"""
        return torch.clamp(propensity, self.ipw_clip_min, self.ipw_clip_max)
    
    def compute_ctcvr_loss(
        self,
        pCTCVR: torch.Tensor,
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> torch.Tensor:
        """
        L_ctcvr: 全空间 CTCVR 损失 (Eq.3)
        在全空间 D 上计算
        """
        ctcvr_label = click_label * conversion_label
        loss = binary_cross_entropy(pCTCVR, ctcvr_label)
        return loss.mean()
    
    def compute_cvr_ipw_loss(
        self,
        pCVR: torch.Tensor,
        pCTR: torch.Tensor,
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> torch.Tensor:
        """
        L_cvr_IPW: 点击空间 CVR 损失 (Eq.4)
        只在点击样本上计算，用 1/pCTR 加权
        """
        # 只选择点击样本
        click_mask = click_label > 0.5
        
        if click_mask.sum() == 0:
            return torch.tensor(0.0, device=pCVR.device)
        
        pCVR_clicked = pCVR[click_mask]
        pCTR_clicked = pCTR[click_mask]
        conversion_clicked = conversion_label[click_mask]
        
        # IPW 加权
        propensity = self._clip_propensity(pCTR_clicked)
        loss = binary_cross_entropy(pCVR_clicked, conversion_clicked) / propensity
        
        return loss.mean()
    
    def compute_ctuncvr_loss(
        self,
        pCTunCVR: torch.Tensor,
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> torch.Tensor:
        """
        L_ctuncvr: 全空间 CTunCVR 损失 (Eq.7) - NDM 核心
        CTunCVR 正样本: 点击但未转化 (o=1, r=0)
        CTunCVR 负样本: 未点击 (o=0) + 点击且转化 (o=1, r=1)
        """
        ctuncvr_label = click_label * (1 - conversion_label)
        loss = binary_cross_entropy(pCTunCVR, ctuncvr_label)
        return loss.mean()
    
    def compute_uncvr_ipw_loss(
        self,
        pUnCVR: torch.Tensor,
        pCTR: torch.Tensor,
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> torch.Tensor:
        """
        L_uncvr_IPW: 点击空间 unCVR 损失 (Eq.9)
        unCVR 标签 = 1 - conversion
        """
        click_mask = click_label > 0.5
        
        if click_mask.sum() == 0:
            return torch.tensor(0.0, device=pUnCVR.device)
        
        pUnCVR_clicked = pUnCVR[click_mask]
        pCTR_clicked = pCTR[click_mask]
        uncvr_label = 1 - conversion_label[click_mask]
        
        # IPW 加权
        propensity = self._clip_propensity(pCTR_clicked)
        loss = binary_cross_entropy(pUnCVR_clicked, uncvr_label) / propensity
        
        return loss.mean()
    
    def compute_align_ipw_loss(
        self,
        pCVR: torch.Tensor,
        pUnCVR: torch.Tensor,
        pCTR: torch.Tensor,
        click_label: torch.Tensor
    ) -> torch.Tensor:
        """
        L_align_IPW: 软标签对齐损失 (Eq.10) - SAM 核心
        
        在点击空间: CVR ≈ 1 - unCVR, 用 1/pCTR 加权
        在未点击空间: CVR ≈ 1 - unCVR, 用 1/(1-pCTR) 加权
        
        互监督: CVR 和 unCVR 互相作为软标签
        """
        click_mask = click_label > 0.5
        unclick_mask = ~click_mask
        
        loss = torch.tensor(0.0, device=pCVR.device)
        
        # 点击空间对齐
        if click_mask.sum() > 0:
            pCVR_c = pCVR[click_mask]
            pUnCVR_c = pUnCVR[click_mask]
            pCTR_c = pCTR[click_mask]
            
            propensity_c = self._clip_propensity(pCTR_c)
            
            # CVR 用 1-unCVR 作为软标签 (stop gradient)
            soft_cvr_label = (1 - pUnCVR_c).detach()
            loss_cvr_align = binary_cross_entropy(pCVR_c, soft_cvr_label) / propensity_c
            
            # unCVR 用 1-CVR 作为软标签 (stop gradient)
            soft_uncvr_label = (1 - pCVR_c).detach()
            loss_uncvr_align = binary_cross_entropy(pUnCVR_c, soft_uncvr_label) / propensity_c
            
            loss = loss + loss_cvr_align.mean() + loss_uncvr_align.mean()
        
        # 未点击空间对齐
        if unclick_mask.sum() > 0:
            pCVR_uc = pCVR[unclick_mask]
            pUnCVR_uc = pUnCVR[unclick_mask]
            pCTR_uc = pCTR[unclick_mask]
            
            # 用 1/(1-pCTR) 加权
            propensity_uc = self._clip_propensity(1 - pCTR_uc)
            
            soft_cvr_label_uc = (1 - pUnCVR_uc).detach()
            loss_cvr_align_uc = binary_cross_entropy(pCVR_uc, soft_cvr_label_uc) / propensity_uc
            
            soft_uncvr_label_uc = (1 - pCVR_uc).detach()
            loss_uncvr_align_uc = binary_cross_entropy(pUnCVR_uc, soft_uncvr_label_uc) / propensity_uc
            
            loss = loss + loss_cvr_align_uc.mean() + loss_uncvr_align_uc.mean()
        
        return loss
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 ChorusCVR 总损失
        
        Args:
            model_outputs: 模型输出字典
                - pCTR, pCVR, pUnCVR, pCTCVR, pCTunCVR
            click_label: 点击标签 (batch_size,)
            conversion_label: 转化标签 (batch_size,)
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        pCTR = model_outputs['pCTR']
        pCVR = model_outputs['pCVR']
        pUnCVR = model_outputs['pUnCVR']
        pCTCVR = model_outputs['pCTCVR']
        pCTunCVR = model_outputs['pCTunCVR']
        
        # 计算各项损失
        loss_ctcvr = self.compute_ctcvr_loss(pCTCVR, click_label, conversion_label)
        loss_cvr_ipw = self.compute_cvr_ipw_loss(pCVR, pCTR, click_label, conversion_label)
        loss_ctuncvr = self.compute_ctuncvr_loss(pCTunCVR, click_label, conversion_label)
        loss_uncvr_ipw = self.compute_uncvr_ipw_loss(pUnCVR, pCTR, click_label, conversion_label)
        loss_align_ipw = self.compute_align_ipw_loss(pCVR, pUnCVR, pCTR, click_label)
        
        # 加权求和
        total_loss = (
            self.loss_weights['ctcvr'] * loss_ctcvr +
            self.loss_weights['cvr_ipw'] * loss_cvr_ipw +
            self.loss_weights['ctuncvr'] * loss_ctuncvr +
            self.loss_weights['uncvr_ipw'] * loss_uncvr_ipw +
            self.loss_weights['align_ipw'] * loss_align_ipw
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'ctcvr': loss_ctcvr.item(),
            'cvr_ipw': loss_cvr_ipw.item(),
            'ctuncvr': loss_ctuncvr.item(),
            'uncvr_ipw': loss_uncvr_ipw.item(),
            'align_ipw': loss_align_ipw.item(),
        }
        
        return total_loss, loss_dict


class ESMMLoss(nn.Module):
    """ESMM 基线损失函数"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pCTR = model_outputs['pCTR']
        pCVR = model_outputs['pCVR']
        pCTCVR = model_outputs['pCTCVR']
        
        # CTR 损失 (全空间)
        loss_ctr = binary_cross_entropy(pCTR, click_label).mean()
        
        # CTCVR 损失 (全空间)
        ctcvr_label = click_label * conversion_label
        loss_ctcvr = binary_cross_entropy(pCTCVR, ctcvr_label).mean()
        
        total_loss = loss_ctr + loss_ctcvr
        
        loss_dict = {
            'total': total_loss.item(),
            'ctr': loss_ctr.item(),
            'ctcvr': loss_ctcvr.item(),
        }
        
        return total_loss, loss_dict
