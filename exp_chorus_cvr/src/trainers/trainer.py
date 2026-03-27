"""
ChorusCVR 训练器
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time


class ChorusCVRTrainer:
    """ChorusCVR 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cuda',
        log_interval: int = 100
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.log_interval = log_interval
        
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        self.train_history = []
        self.val_history = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0
        loss_components = {k: 0 for k in ['ctcvr', 'cvr_ipw', 'ctuncvr', 'uncvr_ipw', 'align_ipw']}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (sparse_features, dense_features, click_labels, conversion_labels) in enumerate(pbar):
            # 移动到设备
            sparse_features = {k: v.to(self.device) for k, v in sparse_features.items()}
            if dense_features is not None:
                dense_features = dense_features.to(self.device)
            click_labels = click_labels.to(self.device)
            conversion_labels = conversion_labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(sparse_features, dense_features)
            
            # 计算损失
            loss, loss_dict = self.loss_fn(outputs, click_labels, conversion_labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss_dict['total']
            for k in loss_components:
                if k in loss_dict:
                    loss_components[k] += loss_dict[k]
            num_batches += 1
            
            # 日志
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # 计算平均损失
        metrics = {
            'loss': total_loss / num_batches,
        }
        for k, v in loss_components.items():
            metrics[f'loss_{k}'] = v / num_batches
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        prefix: str = 'val'
    ) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        all_pCTR = []
        all_pCVR = []
        all_pCTCVR = []
        all_click_labels = []
        all_conversion_labels = []
        
        total_loss = 0
        num_batches = 0
        
        for sparse_features, dense_features, click_labels, conversion_labels in tqdm(data_loader, desc=f'Evaluating {prefix}'):
            sparse_features = {k: v.to(self.device) for k, v in sparse_features.items()}
            if dense_features is not None:
                dense_features = dense_features.to(self.device)
            click_labels = click_labels.to(self.device)
            conversion_labels = conversion_labels.to(self.device)
            
            outputs = self.model(sparse_features, dense_features)
            
            # 计算损失
            loss, _ = self.loss_fn(outputs, click_labels, conversion_labels)
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测
            all_pCTR.append(outputs['pCTR'].cpu().numpy())
            all_pCVR.append(outputs['pCVR'].cpu().numpy())
            all_pCTCVR.append(outputs['pCTCVR'].cpu().numpy())
            all_click_labels.append(click_labels.cpu().numpy())
            all_conversion_labels.append(conversion_labels.cpu().numpy())
        
        # 拼接
        all_pCTR = np.concatenate(all_pCTR)
        all_pCVR = np.concatenate(all_pCVR)
        all_pCTCVR = np.concatenate(all_pCTCVR)
        all_click_labels = np.concatenate(all_click_labels)
        all_conversion_labels = np.concatenate(all_conversion_labels)
        
        # 计算指标
        metrics = {
            f'{prefix}_loss': total_loss / num_batches,
        }
        
        # CTR AUC
        try:
            metrics[f'{prefix}_ctr_auc'] = roc_auc_score(all_click_labels, all_pCTR)
        except:
            metrics[f'{prefix}_ctr_auc'] = 0.5
        
        # CVR AUC (只在点击样本上计算)
        click_mask = all_click_labels > 0.5
        if click_mask.sum() > 0 and all_conversion_labels[click_mask].sum() > 0:
            try:
                metrics[f'{prefix}_cvr_auc'] = roc_auc_score(
                    all_conversion_labels[click_mask],
                    all_pCVR[click_mask]
                )
            except:
                metrics[f'{prefix}_cvr_auc'] = 0.5
        else:
            metrics[f'{prefix}_cvr_auc'] = 0.5
        
        # CTCVR AUC (全空间)
        ctcvr_labels = all_click_labels * all_conversion_labels
        if ctcvr_labels.sum() > 0:
            try:
                metrics[f'{prefix}_ctcvr_auc'] = roc_auc_score(ctcvr_labels, all_pCTCVR)
            except:
                metrics[f'{prefix}_ctcvr_auc'] = 0.5
        else:
            metrics[f'{prefix}_ctcvr_auc'] = 0.5
        
        # PCOC (Predicted Click-Over-Conversion ratio)
        # PCOC = mean(pCVR) / mean(conversion | click)
        if click_mask.sum() > 0:
            actual_cvr = all_conversion_labels[click_mask].mean()
            pred_cvr = all_pCVR[click_mask].mean()
            if actual_cvr > 0:
                metrics[f'{prefix}_pcoc'] = pred_cvr / actual_cvr
            else:
                metrics[f'{prefix}_pcoc'] = 1.0
        else:
            metrics[f'{prefix}_pcoc'] = 1.0
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 1,
        early_stop_patience: int = 3
    ) -> Dict[str, List[float]]:
        """完整训练流程"""
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*50}")
            
            # 训练
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            train_time = time.time() - start_time
            
            # 验证
            val_metrics = self.evaluate(val_loader, prefix='val')
            
            # 更新学习率
            self.scheduler.step(val_metrics['val_cvr_auc'])
            
            # 记录历史
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # 打印指标
            print(f"\nTrain Time: {train_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val CTR-AUC: {val_metrics['val_ctr_auc']:.4f}")
            print(f"Val CVR-AUC: {val_metrics['val_cvr_auc']:.4f}")
            print(f"Val CTCVR-AUC: {val_metrics['val_ctcvr_auc']:.4f}")
            print(f"Val PCOC: {val_metrics['val_pcoc']:.4f}")
            
            # 早停检查
            if val_metrics['val_cvr_auc'] > best_val_auc:
                best_val_auc = val_metrics['val_cvr_auc']
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        # 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train': self.train_history,
            'val': self.val_history
        }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
