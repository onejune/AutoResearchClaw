"""
基线模型的损失函数实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """二元交叉熵损失"""
    pred = torch.clamp(pred, eps, 1 - eps)
    return -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)


class ESMMLoss(nn.Module):
    """ESMM 损失函数"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pCTR = model_outputs['pCTR']
        pCTCVR = model_outputs['pCTCVR']
        
        # CTR 损失 (全空间)
        loss_ctr = binary_cross_entropy(pCTR, click_label).mean()
        
        # CTCVR 损失 (全空间)
        ctcvr_label = click_label * conversion_label
        loss_ctcvr = binary_cross_entropy(pCTCVR, ctcvr_label).mean()
        
        total_loss = loss_ctr + loss_ctcvr
        
        return total_loss, {
            'total': total_loss.item(),
            'ctr': loss_ctr.item(),
            'ctcvr': loss_ctcvr.item(),
        }


class ESCM2Loss(nn.Module):
    """
    ESCM2 损失函数
    支持 IPW 和 DR 两种模式
    """
    
    def __init__(self, mode: str = "ipw", ipw_clip_min: float = 0.01, ipw_clip_max: float = 1.0):
        super().__init__()
        self.mode = mode
        self.ipw_clip_min = ipw_clip_min
        self.ipw_clip_max = ipw_clip_max
    
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
        
        # CVR IPW 损失 (点击空间)
        click_mask = click_label > 0.5
        loss_cvr_ipw = torch.tensor(0.0, device=pCVR.device)
        
        if click_mask.sum() > 0:
            pCVR_clicked = pCVR[click_mask]
            pCTR_clicked = pCTR[click_mask]
            conversion_clicked = conversion_label[click_mask]
            
            propensity = torch.clamp(pCTR_clicked, self.ipw_clip_min, self.ipw_clip_max)
            loss_cvr_ipw = (binary_cross_entropy(pCVR_clicked, conversion_clicked) / propensity).mean()
        
        total_loss = loss_ctr + loss_ctcvr + loss_cvr_ipw
        
        loss_dict = {
            'total': total_loss.item(),
            'ctr': loss_ctr.item(),
            'ctcvr': loss_ctcvr.item(),
            'cvr_ipw': loss_cvr_ipw.item(),
        }
        
        # DR 模式额外损失
        if self.mode == "dr" and 'pCVR_imp' in model_outputs:
            pCVR_imp = model_outputs['pCVR_imp']
            # Imputation loss on all samples
            loss_imp = binary_cross_entropy(pCVR_imp, conversion_label).mean()
            total_loss = total_loss + loss_imp
            loss_dict['imputation'] = loss_imp.item()
            loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class DCMTLoss(nn.Module):
    """
    DCMT 损失函数
    包含 counterfactual CVR 约束
    """
    
    def __init__(self, cf_weight: float = 1.0, ipw_clip_min: float = 0.01, ipw_clip_max: float = 1.0):
        super().__init__()
        self.cf_weight = cf_weight
        self.ipw_clip_min = ipw_clip_min
        self.ipw_clip_max = ipw_clip_max
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pCTR = model_outputs['pCTR']
        pCVR = model_outputs['pCVR']
        pCTCVR = model_outputs['pCTCVR']
        pCF_CVR = model_outputs['pCF_CVR']
        
        # CTR 损失
        loss_ctr = binary_cross_entropy(pCTR, click_label).mean()
        
        # CTCVR 损失
        ctcvr_label = click_label * conversion_label
        loss_ctcvr = binary_cross_entropy(pCTCVR, ctcvr_label).mean()
        
        # CVR IPW 损失 (点击空间)
        click_mask = click_label > 0.5
        loss_cvr_ipw = torch.tensor(0.0, device=pCVR.device)
        
        if click_mask.sum() > 0:
            pCVR_clicked = pCVR[click_mask]
            pCTR_clicked = pCTR[click_mask]
            conversion_clicked = conversion_label[click_mask]
            
            propensity = torch.clamp(pCTR_clicked, self.ipw_clip_min, self.ipw_clip_max)
            loss_cvr_ipw = (binary_cross_entropy(pCVR_clicked, conversion_clicked) / propensity).mean()
        
        # Counterfactual CVR 损失
        # CF-CVR: 未点击为正样本, 转化为负样本
        # 约束: CVR = 1 - CF_CVR
        cf_cvr_label = (1 - click_label)  # 未点击为正
        # 对于转化样本, CF_CVR 应该为 0
        cf_cvr_label = cf_cvr_label * (1 - conversion_label) + (1 - conversion_label) * click_label * 0
        
        # 简化: 在全空间约束 CVR ≈ 1 - CF_CVR
        loss_cf = torch.mean((pCVR - (1 - pCF_CVR.detach())) ** 2) + \
                  torch.mean((pCF_CVR - (1 - pCVR.detach())) ** 2)
        
        total_loss = loss_ctr + loss_ctcvr + loss_cvr_ipw + self.cf_weight * loss_cf
        
        return total_loss, {
            'total': total_loss.item(),
            'ctr': loss_ctr.item(),
            'ctcvr': loss_ctcvr.item(),
            'cvr_ipw': loss_cvr_ipw.item(),
            'cf': loss_cf.item(),
        }


class DDPOLoss(nn.Module):
    """
    DDPO 损失函数
    使用额外 CVR tower 生成软标签
    """
    
    def __init__(self, soft_weight: float = 1.0, ipw_clip_min: float = 0.01, ipw_clip_max: float = 1.0):
        super().__init__()
        self.soft_weight = soft_weight
        self.ipw_clip_min = ipw_clip_min
        self.ipw_clip_max = ipw_clip_max
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        click_label: torch.Tensor,
        conversion_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pCTR = model_outputs['pCTR']
        pCVR = model_outputs['pCVR']
        pCTCVR = model_outputs['pCTCVR']
        pCVR_extra = model_outputs['pCVR_extra']
        
        # CTR 损失
        loss_ctr = binary_cross_entropy(pCTR, click_label).mean()
        
        # CTCVR 损失
        ctcvr_label = click_label * conversion_label
        loss_ctcvr = binary_cross_entropy(pCTCVR, ctcvr_label).mean()
        
        # CVR IPW 损失 (点击空间)
        click_mask = click_label > 0.5
        loss_cvr_ipw = torch.tensor(0.0, device=pCVR.device)
        loss_extra_cvr = torch.tensor(0.0, device=pCVR.device)
        
        if click_mask.sum() > 0:
            pCVR_clicked = pCVR[click_mask]
            pCVR_extra_clicked = pCVR_extra[click_mask]
            pCTR_clicked = pCTR[click_mask]
            conversion_clicked = conversion_label[click_mask]
            
            propensity = torch.clamp(pCTR_clicked, self.ipw_clip_min, self.ipw_clip_max)
            loss_cvr_ipw = (binary_cross_entropy(pCVR_clicked, conversion_clicked) / propensity).mean()
            
            # Extra CVR tower 在点击空间学习
            loss_extra_cvr = binary_cross_entropy(pCVR_extra_clicked, conversion_clicked).mean()
        
        # 软标签约束: 在未点击空间, 用 extra CVR 作为软标签
        unclick_mask = ~click_mask
        loss_soft = torch.tensor(0.0, device=pCVR.device)
        
        if unclick_mask.sum() > 0:
            pCVR_unclicked = pCVR[unclick_mask]
            soft_label = pCVR_extra[unclick_mask].detach()
            loss_soft = binary_cross_entropy(pCVR_unclicked, soft_label).mean()
        
        total_loss = loss_ctr + loss_ctcvr + loss_cvr_ipw + loss_extra_cvr + self.soft_weight * loss_soft
        
        return total_loss, {
            'total': total_loss.item(),
            'ctr': loss_ctr.item(),
            'ctcvr': loss_ctcvr.item(),
            'cvr_ipw': loss_cvr_ipw.item(),
            'extra_cvr': loss_extra_cvr.item(),
            'soft': loss_soft.item(),
        }
