"""
trainer.py - 统一训练/评估循环

支持：
  - Adam 优化器
  - Early Stopping
  - GPU 自动检测（cuda / mps / cpu）
  - 每 epoch 打印 loss + AUC
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from evaluate import compute_metrics

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopping:
    """监控验证集 AUC（越大越好），patience 轮无提升则停止"""

    def __init__(self, patience: int = 3):
        self.patience   = patience
        self.best_score = -np.inf
        self.counter    = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    统一训练器。

    Parameters
    ----------
    model  : BaseMultiTaskModel
    config : Config
    """

    def __init__(self, model, config):
        self.config  = config
        self.device  = _get_device()
        self.model   = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        logger.info("Trainer 初始化 | 设备: %s | 模型: %s", self.device, type(model).__name__)

    # ── 训练 ──────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict[str, List]:
        """
        训练模型，支持 early stopping。

        Returns
        -------
        history : dict  {"train_loss": [...], "val_ctr_auc": [...], ...}
        """
        history = {
            "train_loss":    [],
            "val_ctr_auc":   [],
            "val_cvr_auc":   [],
            "val_ctcvr_auc": [],
        }
        es = EarlyStopping(patience=self.config.early_stopping_patience)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_ctr_auc"].append(val_metrics["ctr_auc"])
            history["val_cvr_auc"].append(val_metrics["cvr_auc"])
            history["val_ctcvr_auc"].append(val_metrics["ctcvr_auc"])

            print(
                f"  Epoch {epoch:02d}/{self.config.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"CTR AUC: {val_metrics['ctr_auc']:.4f} | "
                f"CVR AUC: {val_metrics['cvr_auc']:.4f} | "
                f"CTCVR AUC: {val_metrics['ctcvr_auc']:.4f}"
            )

            # Early stopping（以 CTR AUC 为监控指标）
            if es.step(val_metrics["ctr_auc"]):
                print(f"  Early stopping at epoch {epoch}")
                break

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for features, ctr_label, cvr_label, ctcvr_label in loader:
            features    = {k: v.to(self.device) for k, v in features.items()}
            ctr_label   = ctr_label.to(self.device)
            cvr_label   = cvr_label.to(self.device)
            ctcvr_label = ctcvr_label.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(features)
            losses = self.model.compute_loss(preds, ctr_label, cvr_label, ctcvr_label)
            losses["total"].backward()
            self.optimizer.step()

            total_loss += losses["total"].item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ── 评估 ──────────────────────────────────────────────────

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        在给定 DataLoader 上评估模型。

        Returns
        -------
        {"ctr_auc": float, "cvr_auc": float, "ctcvr_auc": float}
        """
        self.model.eval()

        all_ctr_labels   = []
        all_cvr_labels   = []
        all_ctcvr_labels = []
        all_ctr_preds    = []
        all_cvr_preds    = []
        all_ctcvr_preds  = []

        with torch.no_grad():
            for features, ctr_label, cvr_label, ctcvr_label in loader:
                features    = {k: v.to(self.device) for k, v in features.items()}
                preds = self.model(features)

                all_ctr_labels.append(ctr_label.numpy())
                all_cvr_labels.append(cvr_label.numpy())
                all_ctcvr_labels.append(ctcvr_label.numpy())
                all_ctr_preds.append(preds["ctr"].cpu().numpy())
                all_cvr_preds.append(preds["cvr"].cpu().numpy())
                all_ctcvr_preds.append(preds["ctcvr"].cpu().numpy())

        return compute_metrics(
            np.concatenate(all_ctr_labels),
            np.concatenate(all_cvr_labels),
            np.concatenate(all_ctcvr_labels),
            np.concatenate(all_ctr_preds),
            np.concatenate(all_cvr_preds),
            np.concatenate(all_ctcvr_preds),
        )
