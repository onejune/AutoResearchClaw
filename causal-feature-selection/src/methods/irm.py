"""
IRM: Invariant Risk Minimization
Arjovsky et al., 2019 (https://arxiv.org/abs/1907.02893)

核心思想:
  将训练数据划分为多个"环境"（如不同 business_type 组），
  在最小化各环境 ERM loss 的同时，加入"不变性惩罚"：
  要求每个环境上的最优线性分类器是同一个（即 w=1 时梯度为 0）

损失函数:
  L_IRM = Σ_e L_ERM^e + λ · Σ_e ||∇_{w|w=1} L^e(w·Φ)||²

实现说明:
  - 不变性惩罚通过对 dummy scalar w=1 求梯度实现
  - 每个 batch 从各环境独立采样，保证环境均衡
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


def irm_penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算单个环境的 IRM 不变性惩罚

    原理: 在 w=1 处对 loss 关于 dummy scalar w 求梯度，
          梯度越大说明该环境的最优 w 偏离 1 越远（不稳定）

    Args:
        logits: 模型输出 logits (batch,)
        labels: 真实标签 (batch,)

    Returns:
        penalty: scalar tensor
    """
    # dummy scalar，requires_grad=True
    scale = torch.ones(1, requires_grad=True, device=logits.device)
    loss = nn.functional.binary_cross_entropy_with_logits(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


class IRMTrainer:
    """
    IRM 多环境训练器

    用法:
        trainer = IRMTrainer(model, environments, penalty_weight=10.0)
        trainer.train(n_epochs=1)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 5e-5,
        penalty_weight: float = 1.0,
        penalty_anneal_iters: int = 500,
    ):
        """
        Args:
            model: CTR 模型
            device: cuda / cpu
            lr: 学习率
            penalty_weight: IRM 惩罚系数 λ
            penalty_anneal_iters: 前 N 步只用 ERM，之后加入 IRM 惩罚（warm-up）
        """
        self.model = model.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.penalty_weight = penalty_weight
        self.penalty_anneal_iters = penalty_anneal_iters

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.global_step = 0

        print(f"IRMTrainer initialized:")
        print(f"  device={self.device}, lr={lr}, λ={penalty_weight}, anneal_iters={penalty_anneal_iters}")

    def train_step(
        self,
        env_batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Dict[str, float]:
        """
        单步训练（多环境联合）

        Args:
            env_batches: 每个环境的 (features, labels) batch 列表

        Returns:
            {"erm_loss": ..., "irm_penalty": ..., "total_loss": ...}
        """
        self.model.train()
        self.optimizer.zero_grad()

        erm_losses = []
        penalties = []

        for features, labels in env_batches:
            features = {k: v.to(self.device) for k, v in features.items()}
            labels = labels.to(self.device)

            logits = self.model(features)
            erm_loss = self.criterion(logits, labels)
            erm_losses.append(erm_loss)

            # IRM 惩罚（warm-up 期间跳过）
            if self.global_step >= self.penalty_anneal_iters:
                penalty = irm_penalty(logits, labels)
                penalties.append(penalty)

        # 合并 loss
        total_erm = torch.stack(erm_losses).mean()

        if penalties:
            total_penalty = torch.stack(penalties).mean()
            # λ 退火：前期用 1.0，之后用完整 λ
            lam = self.penalty_weight
            total_loss = total_erm + lam * total_penalty
        else:
            total_penalty = torch.tensor(0.0)
            total_loss = total_erm

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.global_step += 1

        return {
            "erm_loss": total_erm.item(),
            "irm_penalty": total_penalty.item() if penalties else 0.0,
            "total_loss": total_loss.item()
        }

    def train_epoch(
        self,
        env_loaders: List,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个 epoch（多环境轮流采样）

        Args:
            env_loaders: 每个环境的 DataLoader 列表
            epoch: 当前 epoch 编号

        Returns:
            epoch 平均指标
        """
        from tqdm import tqdm

        # 将所有 loader 转为迭代器
        env_iters = [iter(loader) for loader in env_loaders]
        n_steps = min(len(loader) for loader in env_loaders)

        total_erm = 0.0
        total_penalty = 0.0
        total_loss = 0.0

        pbar = tqdm(range(n_steps), desc=f"Epoch {epoch} IRM")
        for step in pbar:
            # 从每个环境取一个 batch
            env_batches = []
            for env_iter in env_iters:
                try:
                    batch = next(env_iter)
                    env_batches.append(batch)
                except StopIteration:
                    break

            if len(env_batches) < len(env_loaders):
                break

            metrics = self.train_step(env_batches)
            total_erm += metrics["erm_loss"]
            total_penalty += metrics["irm_penalty"]
            total_loss += metrics["total_loss"]

            if step % 100 == 0:
                pbar.set_postfix({
                    "erm": f"{metrics['erm_loss']:.4f}",
                    "pen": f"{metrics['irm_penalty']:.4f}",
                    "λ_active": self.global_step >= self.penalty_anneal_iters
                })

        return {
            "train_erm_loss": total_erm / n_steps,
            "train_irm_penalty": total_penalty / n_steps,
            "train_total_loss": total_loss / n_steps
        }
