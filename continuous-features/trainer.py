"""训练器"""
import time
import logging
import torch
import torch.nn as nn
from evaluate import compute_auc

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.criterion = nn.BCELoss()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_cont, x_cat, labels in train_loader:
            x_cont = x_cont.to(self.device)
            x_cat = x_cat.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x_cont, x_cat)
            loss = self.criterion(preds, labels)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(self, train_loader, test_loader):
        """
        训练模型，返回 (best_auc, total_time_seconds)
        支持 early stopping（patience=config.patience）
        """
        best_auc = 0.0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            test_auc = compute_auc(self.model, test_loader, self.device)
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"AUC: {test_auc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            print(
                f"  Epoch {epoch}/{self.config.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"AUC: {test_auc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if test_auc > best_auc:
                best_auc = test_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        total_time = time.time() - start_time
        return best_auc, total_time
