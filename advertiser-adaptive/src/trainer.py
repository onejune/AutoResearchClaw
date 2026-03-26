"""
trainer.py
训练循环：支持分 domain 评估 AUC/PCOC/LogLoss，早停，实验结果自动保存。
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.metrics import evaluate_by_domain, format_metrics_table
from .utils.logger import get_logger

logger = get_logger("trainer")

DOMAIN_NAMES = {0: "shein", 1: "aliexpress", 2: "shopee", 3: "lazada"}


class EarlyStopper:
    """基于整体 AUC 的早停。"""

    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_auc = 0.0
        self.best_state = None
        self.counter = 0

    def step(self, auc: float, model_state: dict) -> bool:
        """返回 True 表示应该停止训练。"""
        if auc > self.best_auc:
            self.best_auc = auc
            self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """
    通用训练器，支持多场景模型。

    Args:
        model: PyTorch 模型
        device: 训练设备（"cpu" / "cuda:0"）
        lr: 学习率
        weight_decay: L2 正则
        early_stop_patience: 早停 patience
        exp_dir: 实验结果保存目录
        exp_name: 实验名称
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stop_patience: int = 3,
        exp_dir: str = "./experiments",
        exp_name: str = "exp",
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
        self.early_stopper = EarlyStopper(patience=early_stop_patience)
        self.exp_dir = os.path.join(exp_dir, exp_name)
        self.exp_name = exp_name
        os.makedirs(self.exp_dir, exist_ok=True)

    def train_epoch(self, loader: DataLoader) -> float:
        """训练一个 epoch，返回平均 loss。"""
        self.model.train()
        total_loss, n_batch = 0.0, 0
        for x_dict, labels in loader:
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            labels = labels.to(self.device)
            preds = self.model(x_dict)
            loss = self.criterion(preds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        return total_loss / max(n_batch, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, dict]:
        """在验证集上评估，返回分 domain 指标。"""
        self.model.eval()
        all_labels, all_preds, all_domains = [], [], []
        for x_dict, labels in loader:
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            preds = self.model(x_dict)
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_domains.append(x_dict["domain_indicator"].cpu().numpy())

        labels_arr = np.concatenate(all_labels)
        preds_arr = np.concatenate(all_preds)
        domains_arr = np.concatenate(all_domains)

        return evaluate_by_domain(labels_arr, preds_arr, domains_arr, DOMAIN_NAMES)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
    ) -> Dict[str, dict]:
        """
        完整训练流程，返回最佳 epoch 的验证指标。
        """
        logger.info(f"[{self.exp_name}] 开始训练，epochs={epochs}")
        best_metrics = {}
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            overall_auc = val_metrics.get("Overall", {}).get("auc", 0.0)

            overall_pcoc = val_metrics.get("Overall", {}).get("pcoc", 0.0)
            logger.info(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_auc={overall_auc:.4f} | val_pcoc={overall_pcoc:.4f}"
            )
            logger.info("\n" + format_metrics_table(val_metrics))

            if self.early_stopper.step(overall_auc, self.model.state_dict()):
                logger.info(f"早停触发，best_auc={self.early_stopper.best_auc:.4f}")
                break
            else:
                best_metrics = val_metrics

        # 恢复最优权重
        if self.early_stopper.best_state:
            self.model.load_state_dict(self.early_stopper.best_state)
            best_metrics = self.evaluate(val_loader)

        elapsed = round((time.time() - t0) / 60, 1)
        best_pcoc = best_metrics.get("Overall", {}).get("pcoc", 0.0) if best_metrics else 0.0
        logger.info(
            f"训练完成，耗时 {elapsed} 分钟 | "
            f"best_auc={self.early_stopper.best_auc:.4f} | best_pcoc={best_pcoc:.4f}"
        )

        self._save_results(best_metrics, elapsed)
        return best_metrics

    def _save_results(self, metrics: Dict[str, dict], elapsed_min: float):
        """保存实验结果到 experiments/{exp_name}/metrics.json，更新 leaderboard.json。"""
        result = {
            "exp_name": self.exp_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_min": elapsed_min,
            "metrics": metrics,
        }
        # 保存单次实验结果
        metrics_path = os.path.join(self.exp_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"实验结果已保存：{metrics_path}")

        # 更新 leaderboard
        lb_path = os.path.join(os.path.dirname(self.exp_dir), "leaderboard.json")
        self._update_leaderboard(lb_path, result)

    def _update_leaderboard(self, lb_path: str, result: dict):
        """更新 leaderboard.json，按整体 AUC 降序排列。"""
        if os.path.exists(lb_path):
            with open(lb_path) as f:
                lb = json.load(f)
        else:
            lb = {"experiments": []}

        # 更新或插入
        overall = result["metrics"].get("Overall", {})
        overall_auc = overall.get("auc", 0.0)
        overall_pcoc = overall.get("pcoc", 0.0)
        entry = {
            "exp_name": result["exp_name"],
            "timestamp": result["timestamp"],
            "overall_auc": overall_auc,
            "overall_pcoc": overall_pcoc,
            "elapsed_min": result["elapsed_min"],
            "domain_metrics": {
                k: {"auc": v.get("auc", 0.0), "pcoc": v.get("pcoc", 0.0)}
                for k, v in result["metrics"].items()
                if k != "Overall"
            },
        }
        # 替换同名实验
        lb["experiments"] = [e for e in lb["experiments"] if e["exp_name"] != entry["exp_name"]]
        lb["experiments"].append(entry)
        lb["experiments"].sort(key=lambda e: e["overall_auc"], reverse=True)
        lb["best"] = lb["experiments"][0] if lb["experiments"] else {}

        with open(lb_path, "w") as f:
            json.dump(lb, f, indent=2, ensure_ascii=False)
        logger.info(f"Leaderboard 已更新：{lb_path}")

    def save_model(self, path: Optional[str] = None):
        """保存模型权重。"""
        if path is None:
            path = os.path.join(self.exp_dir, "model.pt")
        torch.save(self.model.state_dict(), path)
        logger.info(f"模型已保存：{path}")
