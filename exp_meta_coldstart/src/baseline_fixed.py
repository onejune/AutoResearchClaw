"""
修复后的基线方法实现 - 解决采样偏差问题

1. GlobalModel   - 全量混合训练，不做任何适配
2. FineTune      - 全量预训练 → K 条样本 fine-tune（工业最常用）
3. PerTaskModel  - 每个 campaignset 单独训练（数据充足上界）
4. CrossTransfer - 同广告主历史 campaignset 参数迁移（强业务 baseline）
"""
import copy
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .data   import Task, IVRDataset, TaskBuilder
from .models import PurchaseModel

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = model.loss(X, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n if n > 0 else 0.0


# ─────────────────────────────────────────────────────────
# 1. Global Model
# ─────────────────────────────────────────────────────────

def train_global_model(builder: TaskBuilder, cfg: Config, device) -> PurchaseModel:
    """全量数据训练一个全局模型"""
    model = PurchaseModel(builder.vocab_sizes, cfg.embedding_dim,
                          cfg.mlp_dims, cfg.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = builder.global_loader(cfg.batch_size)

    for ep in range(cfg.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        logger.info(f"[Global] epoch={ep+1} loss={loss:.4f}")

    return model


# ─────────────────────────────────────────────────────────
# 2. Pretrain + Fine-tune (修复版本：保持分布一致)
# ─────────────────────────────────────────────────────────

class FineTuner:
    def __init__(self, pretrained_model: PurchaseModel, cfg: Config, device):
        self.pretrained = pretrained_model
        self.cfg = cfg
        self.device = device

    def adapt(self, task: Task, k_shot: Optional[int] = None,
              finetune_epochs: int = 5, finetune_lr: float = 1e-3) -> PurchaseModel:
        """用 support set fine-tune 预训练模型 - 修复采样分布"""
        model = copy.deepcopy(self.pretrained).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=finetune_lr)

        k = k_shot or self.cfg.k_shot
        support_df = task.support_df.head(k)
        
        # 修复：使用相同的 batch_size 和 shuffle 策略，保持分布一致性
        from torch.utils.data import DataLoader
        loader = DataLoader(
            IVRDataset(support_df, self.pretrained.embedding.features),
            batch_size=min(64, len(support_df)), shuffle=True  # shuffle=True 保持分布
        )

        model.train()
        for _ in range(finetune_epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = model.loss(X, y)
                loss.backward()
                # 添加梯度裁剪防止过拟合
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        return model


# ─────────────────────────────────────────────────────────
# 3. Per-Task Model（上界）
# ─────────────────────────────────────────────────────────

def train_per_task_model(task: Task, vocab_sizes, cfg: Config, device,
                         use_full: bool = True) -> PurchaseModel:
    """只用该 task 的数据训练（use_full=True 用全量，False 用 support set）"""
    model = PurchaseModel(vocab_sizes, cfg.embedding_dim,
                          cfg.mlp_dims, cfg.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr*0.1)  # 降低学习率避免过拟合

    from torch.utils.data import DataLoader
    if use_full:
        # 使用 support + query 的全量数据
        full_df = task.support_df.append(task.query_df, ignore_index=True)
        loader = DataLoader(
            IVRDataset(full_df, list(vocab_sizes.keys())),
            batch_size=min(256, len(full_df)), shuffle=True
        )
        epochs = max(cfg.epochs, 5)  # 适量训练
    else:
        # 仅使用 support set（冷启动场景）
        loader = DataLoader(
            IVRDataset(task.support_df, list(vocab_sizes.keys())),
            batch_size=min(64, len(task.support_df)), shuffle=True
        )
        epochs = max(cfg.epochs, 10)   # 冷启动需要更多训练

    for ep in range(epochs):
        train_one_epoch(model, loader, optimizer, device)

    return model


# ─────────────────────────────────────────────────────────
# 4. Cross-CampaignSet Transfer（同广告主迁移）- 修复版本
# ─────────────────────────────────────────────────────────

class CrossTransfer:
    """
    同 business_type 下，用其他 campaignset 的预训练模型初始化，
    再在 support set 上 fine-tune。
    """
    def __init__(self, task_models: dict, cfg: Config, device):
        """task_models: {campaignset_id: model}"""
        self.task_models = task_models
        self.cfg = cfg
        self.device = device

    def adapt(self, task: Task, k_shot: Optional[int] = None,
              finetune_epochs: int = 3) -> PurchaseModel:
        # 找同 business_type 的其他 task 模型，取第一个
        donor = None
        for csid, m in self.task_models.items():
            # 确保是同类型的业务
            if hasattr(m, 'business_type') and m.business_type == task.business_type:
                if csid != task.campaignset_id:
                    donor = m
                    break
        
        # 如果没找到同类型，找任意一个
        if donor is None:
            for csid, m in self.task_models.items():
                if csid != task.campaignset_id:
                    donor = m
                    break

        if donor is None:
            # 没有 donor 模型，回退到预训练模型
            from .baseline import train_global_model
            # 这里需要外部传入 builder，暂时简化
            donor = self.task_models[next(iter(self.task_models.keys()))]

        finetuner = FineTuner(donor, self.cfg, self.device)
        return finetuner.adapt(task, k_shot, finetune_epochs, finetune_lr=5e-4)  # 降低学习率
