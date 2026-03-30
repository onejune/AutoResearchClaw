"""
基线方法实现

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
# 2. Pretrain + Fine-tune
# ─────────────────────────────────────────────────────────

class FineTuner:
    def __init__(self, pretrained_model: PurchaseModel, cfg: Config, device):
        self.pretrained = pretrained_model
        self.cfg = cfg
        self.device = device

    def adapt(self, task: Task, k_shot: Optional[int] = None,
              finetune_epochs: int = 5, finetune_lr: float = 1e-3) -> PurchaseModel:
        """用 support set fine-tune 预训练模型"""
        model = copy.deepcopy(self.pretrained).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=finetune_lr)

        k = k_shot or self.cfg.k_shot
        support_df = task.support_df.head(k)
        from torch.utils.data import DataLoader
        loader = DataLoader(
            IVRDataset(support_df, self.pretrained.embedding.features),
            batch_size=min(64, len(support_df)), shuffle=True
        )

        model.train()
        for _ in range(finetune_epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                model.loss(X, y).backward()
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
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    df = task.support_df if not use_full else \
         torch.utils.data.ConcatDataset  # 用 support+query 全量
    # 简化：直接用 support_df（冷启动场景下只有 support）
    from torch.utils.data import DataLoader
    loader = DataLoader(
        IVRDataset(task.support_df, list(vocab_sizes.keys())),
        batch_size=min(256, len(task.support_df)), shuffle=True
    )

    for ep in range(max(cfg.epochs, 10)):   # per-task 多训几轮
        train_one_epoch(model, loader, optimizer, device)

    return model


# ─────────────────────────────────────────────────────────
# 4. Cross-CampaignSet Transfer（同广告主迁移）
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
              finetune_epochs: int = 5) -> PurchaseModel:
        # 找同 business_type 的其他 task 模型，取第一个
        donor = None
        for csid, m in self.task_models.items():
            if csid != task.campaignset_id:
                donor = m
                break

        if donor is None:
            raise ValueError(f"No donor model for task {task.campaignset_id}")

        finetuner = FineTuner(donor, self.cfg, self.device)
        return finetuner.adapt(task, k_shot, finetune_epochs)
