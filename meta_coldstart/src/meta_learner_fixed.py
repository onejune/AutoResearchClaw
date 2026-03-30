#!/usr/bin/env python3
"""
改进的 MAML 实现，修复了梯度更新问题
"""
import copy
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .config  import Config
from .data    import Task, IVRDataset
from .models  import PurchaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def inner_loop(model: PurchaseModel, support_loader: DataLoader,
               inner_lr: float, inner_steps: int, device,
               first_order: bool = False) -> PurchaseModel:
    """
    在 support set 上做 inner loop 梯度更新。
    first_order=True → FOMAML（不追踪二阶梯度）
    返回适配后的模型（不修改原模型）
    """
    # 使用 deepcopy 创建模型副本
    adapted = copy.deepcopy(model)
    adapted.train()

    for step in range(inner_steps):
        for X, y in support_loader:
            X, y = X.to(device), y.to(device)
            
            # 清零梯度
            adapted.zero_grad()
            
            # 计算损失
            loss = adapted.loss(X, y)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss,
                [p for p in adapted.parameters() if p.requires_grad],
                retain_graph=True,
                create_graph=not first_order,   # MAML 需要二阶图
                allow_unused=True,
            )
            
            # 手动更新参数（使用 torch.no_grad 确保不破坏计算图）
            params = [p for p in adapted.parameters() if p.requires_grad]
            for param, grad in zip(params, grads):
                if grad is not None:
                    # 创建新的参数张量并替换
                    updated_param = param - inner_lr * grad
                    param.data.copy_(updated_param)

            break  # 每步只用一个 batch

    return adapted


def inner_loop_head_only(model: PurchaseModel, support_loader: DataLoader,
                          inner_lr: float, inner_steps: int, device) -> PurchaseModel:
    """ANIL：只更新 MLP head 层，embedding 和 MLP body 冻结"""
    # 使用 deepcopy 创建模型副本
    adapted = copy.deepcopy(model)
    adapted.train()

    # 只对 head 层做 inner loop
    head_params = list(adapted.mlp.head.parameters())

    for step in range(inner_steps):
        for X, y in support_loader:
            X, y = X.to(device), y.to(device)
            
            adapted.zero_grad()
            loss = adapted.loss(X, y)
            
            grads = torch.autograd.grad(loss, head_params, 
                                        retain_graph=True,
                                        create_graph=True,
                                        allow_unused=True)
            
            for param, grad in zip(head_params, grads):
                if grad is not None:
                    updated_param = param - inner_lr * grad
                    param.data.copy_(updated_param)
            
            break

    return adapted


# ─────────────────────────────────────────────────────────
# 1 & 2. MAML / FOMAML
# ─────────────────────────────────────────────────────────

class MAML:
    def __init__(self, vocab_sizes: dict, cfg: Config, device,
                 first_order: bool = False):
        self.cfg         = cfg
        self.device      = device
        self.first_order = first_order
        self.name        = "FOMAML" if first_order else "MAML"

        self.model = PurchaseModel(
            vocab_sizes, cfg.embedding_dim, cfg.mlp_dims, cfg.dropout
        ).to(device)
        self.meta_optimizer = optim.Adam(
            self.model.parameters(), lr=cfg.meta_lr
        )

    def meta_train(self, train_tasks: List[Task], features: List[str]):
        cfg = self.cfg
        logger.info(f"[{self.name}] meta_epochs={cfg.meta_epochs}, "
                    f"inner_steps={cfg.inner_steps}, inner_lr={cfg.inner_lr}")

        for epoch in range(cfg.meta_epochs):
            # 随机采样 meta_batch_size 个 task
            import random
            batch_tasks = random.sample(train_tasks,
                                        min(cfg.meta_batch_size, len(train_tasks)))

            self.meta_optimizer.zero_grad()
            outer_losses = []

            for task in batch_tasks:
                support_loader = task.support_loader(features, batch_size=64)
                query_loader   = task.query_loader(features, batch_size=256)
                if query_loader is None:
                    continue  # 跳过空 query set

                # Inner loop：在 support set 上适配
                adapted = inner_loop(
                    self.model, support_loader,
                    cfg.inner_lr, cfg.inner_steps,
                    self.device, first_order=self.first_order
                )

                # Outer loss：在 query set 上计算
                if query_loader is None:
                    continue
                adapted.train()
                task_outer_losses = []
                batch_count = 0
                for X, y in query_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    loss = adapted.loss(X, y)
                    task_outer_losses.append(loss)
                    batch_count += 1
                    # 如果 query set 很大，限制使用的 batch 数以控制计算量
                    if batch_count >= 3:  # 每个 task 最多用 3 个 batch
                        break
                
                if task_outer_losses:
                    task_outer_loss = torch.stack(task_outer_losses).mean()
                    outer_losses.append(task_outer_loss)

            if outer_losses:
                # 计算总体损失
                total_outer_loss = torch.stack(outer_losses).mean()
                
                # 反向传播
                total_outer_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 更新参数
                self.meta_optimizer.step()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"[{self.name}] epoch={epoch+1}/{cfg.meta_epochs} "
                                f"outer_loss={total_outer_loss.item():.4f}")
            else:
                logger.warning(f"No valid tasks in batch for epoch {epoch+1}")

    def adapt(self, task: Task, features: List[str],
              k_shot: Optional[int] = None) -> PurchaseModel:
        """测试时：在 support set 上做 inner loop 适配"""
        support_loader = task.support_loader(features, batch_size=64)
        adapted = inner_loop(
            self.model, support_loader,
            self.cfg.inner_lr, self.cfg.inner_steps,
            self.device, first_order=True   # 测试时用一阶即可
        )
        return adapted


# ─────────────────────────────────────────────────────────
# 3. ANIL
# ─────────────────────────────────────────────────────────

class ANIL:
    """Almost No Inner Loop：只在 head 层做 inner loop"""
    def __init__(self, vocab_sizes: dict, cfg: Config, device):
        self.cfg    = cfg
        self.device = device

        self.model = PurchaseModel(
            vocab_sizes, cfg.embedding_dim, cfg.mlp_dims, cfg.dropout
        ).to(device)
        self.meta_optimizer = optim.Adam(
            self.model.parameters(), lr=cfg.meta_lr
        )

    def meta_train(self, train_tasks: List[Task], features: List[str]):
        cfg = self.cfg
        logger.info(f"[ANIL] meta_epochs={cfg.meta_epochs}")

        for epoch in range(cfg.meta_epochs):
            import random
            batch_tasks = random.sample(train_tasks,
                                        min(cfg.meta_batch_size, len(train_tasks)))

            self.meta_optimizer.zero_grad()
            outer_losses = []

            for task in batch_tasks:
                support_loader = task.support_loader(features, batch_size=64)
                query_loader   = task.query_loader(features, batch_size=256)
                if query_loader is None:
                    continue

                adapted = inner_loop_head_only(
                    self.model, support_loader,
                    cfg.inner_lr, cfg.inner_steps, self.device
                )

                adapted.train()
                task_outer_losses = []
                batch_count = 0
                for X, y in query_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    loss = adapted.loss(X, y)
                    task_outer_losses.append(loss)
                    batch_count += 1
                    # 如果 query set 很大，限制使用的 batch 数以控制计算量
                    if batch_count >= 3:  # 每个 task 最多用 3 个 batch
                        break
                
                if task_outer_losses:
                    task_outer_loss = torch.stack(task_outer_losses).mean()
                    outer_losses.append(task_outer_loss)

            if outer_losses:
                total_outer_loss = torch.stack(outer_losses).mean()
                total_outer_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.meta_optimizer.step()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"[ANIL] epoch={epoch+1}/{cfg.meta_epochs} "
                                f"outer_loss={total_outer_loss.item():.4f}")
            else:
                logger.warning(f"No valid tasks in batch for epoch {epoch+1}")

    def adapt(self, task: Task, features: List[str],
              k_shot: Optional[int] = None) -> PurchaseModel:
        support_loader = task.support_loader(features, batch_size=64)
        return inner_loop_head_only(
            self.model, support_loader,
            self.cfg.inner_lr, self.cfg.inner_steps, self.device
        )


# ─────────────────────────────────────────────────────────
# 4. ProtoNet（原型网络）
# ─────────────────────────────────────────────────────────

class ProtoNet:
    """
    原型网络：
    - 正类原型 = support set 中正样本 embedding 的均值
    - 负类原型 = support set 中负样本 embedding 的均值
    - 预测 = 与正类原型的距离（转化为概率）
    """
    def __init__(self, vocab_sizes: dict, cfg: Config, device):
        self.cfg    = cfg
        self.device = device

        # 只用 embedding + MLP body 作为特征提取器，不用 head
        self.model = PurchaseModel(
            vocab_sizes, cfg.embedding_dim, cfg.mlp_dims, cfg.dropout
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.meta_lr)

    def meta_train(self, train_tasks: List[Task], features: List[str]):
        """用 episodic training 训练特征提取器"""
        cfg = self.cfg
        logger.info(f"[ProtoNet] meta_epochs={cfg.meta_epochs}")

        for epoch in range(cfg.meta_epochs):
            import random
            batch_tasks = random.sample(train_tasks,
                                        min(cfg.meta_batch_size, len(train_tasks)))

            self.optimizer.zero_grad()
            total_losses = []

            for task in batch_tasks:
                loss = self._episode_loss(task, features)
                if loss is not None:
                    total_losses.append(loss)

            if total_losses:
                avg_loss = torch.stack(total_losses).mean()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"[ProtoNet] epoch={epoch+1}/{cfg.meta_epochs} "
                                f"loss={avg_loss.item():.4f}")
            else:
                logger.warning(f"No valid tasks in batch for ProtoNet epoch {epoch+1}")

    def _get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """提取 embedding（不经过 head）"""
        emb = self.model.embedding(x)
        return self.model.mlp.net(emb)   # MLP body 输出

    def _episode_loss(self, task: Task, features: List[str]):
        support_df = task.support_df
        pos_df = support_df[support_df["purchase"] == 1]
        neg_df = support_df[support_df["purchase"] == 0]
        if len(pos_df) == 0 or len(neg_df) == 0:
            return None

        from .data import IVRDataset
        import numpy as np

        def to_tensor(df):
            return torch.tensor(df[features].values, dtype=torch.long).to(self.device)

        if len(pos_df) == 0 or len(neg_df) == 0:
            return None

        pos_x = to_tensor(pos_df)
        neg_x = to_tensor(neg_df)

        self.model.train()
        pos_emb  = self._get_embedding(pos_x)   # (n_pos, d)
        neg_emb  = self._get_embedding(neg_x)   # (n_neg, d)

        proto_pos = pos_emb.mean(0)   # (d,)
        proto_neg = neg_emb.mean(0)   # (d,)

        # query set
        query_df = task.query_df.head(200)   # 限制大小加速
        if len(query_df) == 0:
            return None
            
        query_x  = to_tensor(query_df)
        query_y  = torch.tensor(query_df["purchase"].values,
                                dtype=torch.float32).to(self.device)

        query_emb = self._get_embedding(query_x)   # (n_q, d)

        # 距离：负类距离 - 正类距离 → logit
        dist_pos = torch.norm(query_emb - proto_pos, dim=-1)
        dist_neg = torch.norm(query_emb - proto_neg, dim=-1)

        logits = dist_neg - dist_pos   # 离正类近 → 高 logit
        return F.binary_cross_entropy_with_logits(logits, query_y)

    def adapt(self, task: Task, features: List[str],
              k_shot: Optional[int] = None) -> "ProtoPredictor":
        """返回一个基于原型的预测器"""
        support_df = task.support_df
        pos_df = support_df[support_df["purchase"] == 1]
        neg_df = support_df[support_df["purchase"] == 0]

        def to_tensor(df):
            return torch.tensor(df[features].values, dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            proto_pos = self._get_embedding(to_tensor(pos_df)).mean(0) if len(pos_df) > 0 else None
            proto_neg = self._get_embedding(to_tensor(neg_df)).mean(0) if len(neg_df) > 0 else None

        return ProtoPredictor(self, proto_pos, proto_neg, features, self.device)


class ProtoPredictor:
    """ProtoNet 适配后的预测器，接口与 PurchaseModel 兼容"""
    def __init__(self, proto_net: ProtoNet, proto_pos, proto_neg, features, device):
        self.proto_net = proto_net
        self.proto_pos = proto_pos
        self.proto_neg = proto_neg
        self.features  = features
        self.device    = device

    def eval(self): return self
    def train(self): return self

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.proto_net.model.eval()
        with torch.no_grad():
            emb = self.proto_net._get_embedding(x)
            if self.proto_pos is None or self.proto_neg is None:
                return torch.full((len(x),), 0.5, device=self.device)
            dist_pos = torch.norm(emb - self.proto_pos, dim=-1)
            dist_neg = torch.norm(emb - self.proto_neg, dim=-1)
            logits   = dist_neg - dist_pos
            return torch.sigmoid(logits)