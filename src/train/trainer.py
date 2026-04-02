"""
统一训练器 - 支持动态 GPU/CPU 切换

核心功能:
- 自动检测 GPU 可用性
- 动态调整 batch size 和 num_workers
- 混合精度训练 (AMP)
- 梯度累积（模拟大 batch）
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def get_device(preferred: str = "cuda") -> torch.device:
    """智能设备选择"""
    if preferred == "cuda" and torch.cuda.is_available():
        # 检查 GPU 显存
        try:
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3  # GB
            if free_mem < 2:
                print(f"⚠️  GPU 显存不足 ({free_mem:.1f}GB), 切换到 CPU")
                return torch.device("cpu")
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)} ({free_mem:.1f}GB free)")
            return torch.device("cuda")
        except:
            return torch.device("cuda")
    else:
        cpu_count = os.cpu_count() or 4
        print(f"✅ Using CPU ({cpu_count} cores)")
        return torch.device("cpu")


class DynamicTrainer:
    """支持动态资源调整的训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        max_batch_size: int = 512,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False
    ):
        """
        Args:
            model: 模型
            device: "cuda" 或 "cpu" (None 则自动选择)
            lr: 学习率
            weight_decay: 权重衰减
            max_batch_size: 最大 batch size（根据显存动态调整）
            gradient_accumulation_steps: 梯度累积步数
            use_amp: 是否使用混合精度训练
        """
        self.device = get_device(device) if device is None else torch.device(device)
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and self.device.type == "cuda"
        
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 混合精度 scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 动态 batch size
        self.batch_size = self._estimate_batch_size(max_batch_size)
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Mixed precision: {self.use_amp}")
    
    def _estimate_batch_size(self, max_batch: int) -> int:
        """估算合适的 batch size"""
        if self.device.type == "cpu":
            return min(max_batch, 256)  # CPU 保守一点
        
        try:
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            if free_mem < 4:
                return min(max_batch, 128)
            elif free_mem < 8:
                return min(max_batch, 256)
            else:
                return max_batch
        except:
            return max_batch
    
    def train_epoch(
        self,
        dataloader,
        epoch: int = 1,
        total_epochs: int = 1
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            labels = batch["label"].to(self.device).float()
            
            # 混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1
            
            # 更新进度条
            if batch_idx % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item()*self.gradient_accumulation_steps:.4f}"})
        
        return {"train_loss": total_loss / n_batches}
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """评估"""
        from sklearn.metrics import roc_auc_score, log_loss
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_bts = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            labels = batch["label"].cpu().numpy()
            
            outputs = self.model(features)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())
            
            if "business_type" in batch:
                all_bts.extend(batch["business_type"].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 整体指标
        pcoc = float(all_preds.mean() / all_labels.mean()) if all_labels.mean() > 0 else float("nan")
        metrics = {
            "auc": float(roc_auc_score(all_labels, all_preds)),
            "pcoc": pcoc,
            "logloss": float(log_loss(all_labels, all_preds))
        }
        
        # 分 BT 指标
        if all_bts:
            all_bts = np.array(all_bts)
            bt_metrics = {}
            for bt_id in np.unique(all_bts):
                mask = all_bts == bt_id
                if mask.sum() < 100:
                    continue
                bt_preds = all_preds[mask]
                bt_labels = all_labels[mask]
                bt_pcoc = float(bt_preds.mean() / bt_labels.mean()) if bt_labels.mean() > 0 else float("nan")
                
                # 计算 AUC（需要两个类别都存在）
                n_pos = bt_labels.sum()
                n_neg = len(bt_labels) - n_pos
                if n_pos > 0 and n_neg > 0:
                    bt_auc = float(roc_auc_score(bt_labels, bt_preds))
                else:
                    bt_auc = float("nan")  # 只有一个类别，AUC 无定义
                
                bt_metrics[f"bt_{int(bt_id)}"] = {
                    "count": int(mask.sum()),
                    "auc": bt_auc,
                    "pcoc": bt_pcoc
                }
            metrics["bt_grouped"] = bt_metrics
        
        return metrics
