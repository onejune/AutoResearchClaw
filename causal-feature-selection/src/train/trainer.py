"""
训练器 - 统一训练接口

支持:
- Baseline 训练 (WideDeep, DeepFM)
- 跨域评估 (按 business_type 分组)
- Early stopping
- 模型保存/加载
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from tqdm import tqdm


class Trainer:
    """CTR 模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 5e-5,
        weight_decay: float = 0.0
    ):
        self.model = model.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss (BCE for binary classification)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-7
        )
        
        print(f"Trainer initialized on {self.device}")
        print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for batch_features, batch_labels in pbar:
            # 移到设备
            batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
            batch_labels = batch_labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(batch_features)
            loss = self.criterion(logits, batch_labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            "train_loss": total_loss / len(train_loader),
            "train_auc": roc_auc_score(all_labels, all_preds),
            "train_logloss": log_loss(all_labels, all_preds),
            "train_acc": accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        prefix: str = "val"
    ) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_bts = []  # 记录 business_type
        
        for batch_features, batch_labels in tqdm(eval_loader, desc=f"{prefix} Eval"):
            batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
            
            logits = self.model(batch_features)
            preds = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
            
            # 如果有 business_type 信息，也记录下来
            if "business_type" in batch_features:
                all_bts.extend(batch_features["business_type"].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 整体指标
        pcoc = float(all_preds.mean() / all_labels.mean()) if all_labels.mean() > 0 else float("nan")
        metrics = {
            f"{prefix}_auc": roc_auc_score(all_labels, all_preds),
            f"{prefix}_pcoc": pcoc,
            f"{prefix}_logloss": log_loss(all_labels, all_preds),
            f"{prefix}_acc": accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        }
        
        # 按 business_type 分组评估
        if all_bts:
            all_bts = np.array(all_bts)
            bt_metrics = {}
            
            for bt_id in np.unique(all_bts):
                mask = all_bts == bt_id
                if mask.sum() < 100:  # 样本太少跳过
                    continue
                
                bt_preds = all_preds[mask]
                bt_labels = all_labels[mask]
                bt_pcoc = float(bt_preds.mean() / bt_labels.mean()) if bt_labels.mean() > 0 else float("nan")
                
                bt_metrics[f"bt_{int(bt_id)}"] = {
                    "count": int(mask.sum()),
                    "auc": float(roc_auc_score(bt_labels, bt_preds)),
                    "pcoc": bt_pcoc,
                    "positive_rate": float(bt_labels.mean())
                }
            
            metrics["bt_grouped"] = bt_metrics
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        save_path: str = None
    ) -> Dict:
        """
        完整训练流程
        
        Returns:
            训练历史 + 最佳模型指标
        """
        best_val_auc = 0.0
        patience_counter = 0
        history = {
            "train": [],
            "val": [],
            "best_val_auc": 0.0,
            "best_epoch": 0
        }
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print("="*60)
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch+1)
            history["train"].append(train_metrics)
            
            print(f"\nTrain Metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # 验证
            val_metrics = self.evaluate(val_loader, prefix="val")
            history["val"].append(val_metrics)
            
            print(f"\nVal Metrics:")
            for k, v in val_metrics.items():
                if k != "bt_grouped":
                    print(f"  {k}: {v:.4f}")
            print(f"  [PCOC 说明: 1.0=校准良好, >1=高估, <1=低估]")
            
            # 按 BT 分组结果
            if "bt_grouped" in val_metrics:
                print(f"\nVal by Business Type:")
                print(f"  {'BT':<8} {'Count':<10} {'AUC':<8} {'PCOC':<8} {'PosRate':<8}")
                print(f"  {'-'*42}")
                for bt_id, bt_m in sorted(val_metrics["bt_grouped"].items()):
                    print(f"  {bt_id:<8} {bt_m['count']:<10,} {bt_m['auc']:<8.4f} {bt_m['pcoc']:<8.4f} {bt_m['positive_rate']:<8.4f}")
            
            # Early stopping
            if val_metrics["val_auc"] > best_val_auc:
                best_val_auc = val_metrics["val_auc"]
                history["best_val_auc"] = best_val_auc
                history["best_epoch"] = epoch + 1
                patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_auc": best_val_auc,
                    }, save_path)
                    print(f"\n✅ Best model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"\n⚠️  No improvement, patience: {patience_counter}/{early_stopping_patience}")
            
            # 学习率更新
            self.scheduler.step()
            
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Val AUC: {checkpoint['val_auc']:.4f}")


def main_test():
    """测试训练流程"""
    import sys
    from pathlib import Path
    
    # 添加 src 到路径
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    
    from data.loader import IVRDataLoader, create_dataloader, IVRCTCVRDataset
    from models.baseline import WideDeepBaseline
    
    print("=" * 60)
    print("Testing Training Pipeline")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] Loading data...")
    data_loader = IVRDataLoader()
    
    # 跨域划分
    train_df, test_df = data_loader.create_cross_domain_split(
        train_bts=[0, 1, 2],  # 只用前 3 个 BT，快速测试
        test_bts=[3, 4, 5]
    )
    
    # 采样 10% 快速测试
    train_df = train_df.sample(frac=0.1, random_state=42)
    test_df = test_df.sample(frac=0.1, random_state=42)
    
    # 保存为临时 parquet
    train_df.to_parquet("/tmp/train_sample.parquet", index=False)
    test_df.to_parquet("/tmp/test_sample.parquet", index=False)
    
    # 创建 Dataset
    train_dataset = IVRCTCVRDataset(
        parquet_path="/tmp/train_sample.parquet",
        feature_cols=data_loader.feature_cols,
        label_col="click_label"
    )
    
    test_dataset = IVRCTCVRDataset(
        parquet_path="/tmp/test_sample.parquet",
        feature_cols=data_loader.feature_cols,
        label_col="click_label"
    )
    
    # DataLoader
    train_loader = create_dataloader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = create_dataloader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # 2. 构建模型
    print("\n[2/4] Building model...")
    feature_config = data_loader.get_feature_config()
    
    model = WideDeepBaseline(
        feature_config=feature_config,
        embedding_size=8,
        dnn_hidden_units=[256, 128, 64],  # 简化版，快速测试
        dropout=0.3
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # 3. 训练
    print("\n[3/4] Training...")
    trainer = Trainer(
        model=model,
        device="cuda",
        lr=5e-5,
        weight_decay=0.01
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=3,
        early_stopping_patience=2,
        save_path="/tmp/baseline_test.pt"
    )
    
    # 4. 最终评估
    print("\n[4/4] Final evaluation...")
    final_metrics = trainer.evaluate(test_loader, prefix="test")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Val AUC: {history['best_val_auc']:.4f} (Epoch {history['best_epoch']})")
    print(f"Test AUC: {final_metrics['test_auc']:.4f}")
    print(f"Test LogLoss: {final_metrics['test_logloss']:.4f}")
    
    return history, final_metrics


if __name__ == "__main__":
    history, metrics = main_test()
