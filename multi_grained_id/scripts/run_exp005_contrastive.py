#!/usr/bin/env python3
"""
exp005: Contrastive ID Learning - 对比学习增强实验

使用对比学习增强 ID embedding 的表示能力。
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.loader import IVRDataLoader, collate_fn
from src.models.contrastive import ContrastiveWideDeep
from src.utils.hardware_monitor import HardwareMonitor


def load_shared_baseline():
    """加载公共 Baseline 结果"""
    baseline_dir = Path("results/shared_baseline")
    
    if not baseline_dir.exists():
        return None, None, None
    
    with open(baseline_dir / "results.json") as f:
        results = json.load(f)
    
    with open(baseline_dir / "feature_config.json") as f:
        feature_config = json.load(f)
    
    bt_grouped = {}
    bt_file = baseline_dir / "bt_grouped.json"
    if bt_file.exists():
        with open(bt_file) as f:
            bt_grouped = json.load(f)
    
    return results, feature_config, bt_grouped


class ContrastiveTrainer:
    """支持对比学习的训练器"""
    
    def __init__(self, model, lr, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        self.model.train()
        total_loss = 0
        total_bce = 0
        total_cl = 0
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
        
        for batch in pbar:
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            labels = batch["label"].to(self.device).float()
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits, cl_loss = self.model(features, compute_contrastive=True)
                    bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
                    loss = bce_loss + cl_loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, cl_loss = self.model(features, compute_contrastive=True)
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = bce_loss + cl_loss
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_cl += cl_loss.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "bce": f"{bce_loss.item():.4f}",
                "cl": f"{cl_loss.item():.4f}"
            })
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_bts = []
        
        import numpy as np
        from sklearn.metrics import roc_auc_score, log_loss
        
        with torch.no_grad():
            for batch in test_loader:
                features = {k: v.to(self.device) for k, v in batch["features"].items()}
                labels = batch["label"]
                
                logits, _ = self.model(features, compute_contrastive=False)
                preds = torch.sigmoid(logits).cpu().numpy()
                
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
                
                if "business_type" in batch["features"]:
                    all_bts.extend(batch["features"]["business_type"].numpy().tolist())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            "auc": float(roc_auc_score(all_labels, all_preds)),
            "pcoc": float(all_preds.mean() / all_labels.mean()) if all_labels.mean() > 0 else float("nan"),
            "logloss": float(log_loss(all_labels, all_preds)),
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
                
                n_pos = bt_labels.sum()
                n_neg = len(bt_labels) - n_pos
                if n_pos > 0 and n_neg > 0:
                    bt_auc = float(roc_auc_score(bt_labels, bt_preds))
                else:
                    bt_auc = float("nan")
                
                bt_metrics[f"bt_{int(bt_id)}"] = {
                    "count": int(mask.sum()),
                    "auc": bt_auc,
                    "pcoc": bt_pcoc
                }
            metrics["bt_grouped"] = bt_metrics
        
        return metrics


def main():
    print("=" * 60)
    print("exp005: Contrastive ID Learning 对比学习实验")
    print("=" * 60)
    
    config = {
        "label_col": "ctcvr_label",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 1,
        "embedding_size": 64,
        "dnn_hidden_units": [1024, 512, 256, 128],
        "dropout": 0.3,
        "contrastive_weight": 0.1,
        "seed": 42,
    }
    
    # 定义对比学习特征对：{ID 特征: 分组依据特征}
    # 同一 group 的 ID 被视为正样本对
    contrastive_features = {
        "demand_pkgname": "business_type",  # 同一业务类型的包名应该相近
        "campaignid": "campaignsetid",      # 同一广告组的计划应该相近
    }
    
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    # 智能选择设备
    print("\n[1/4] 选择训练设备...")
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    print(f"选中设备: {device} ({device_info.name})")
    
    # 加载公共 Baseline
    print("\n[2/4] 加载 Baseline...")
    baseline_results, feature_config, baseline_bt = load_shared_baseline()
    if baseline_results:
        print(f"✅ 已加载公共 Baseline: AUC={baseline_results['auc']:.4f}")
    else:
        print("⚠️ 公共 Baseline 不存在，请先运行 train_shared_baseline.py")
        return
    
    # 加载数据
    print("\n[3/4] 加载数据...")
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
    train_dataset, test_dataset = data_loader.load_datasets()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    feature_config_int = {k: int(v) for k, v in feature_config.items()}
    
    # 过滤有效的对比特征
    valid_contrastive = {
        k: v for k, v in contrastive_features.items()
        if k in feature_config_int and v in feature_config_int
    }
    print(f"有效的对比学习特征配置: {valid_contrastive}")
    
    # 创建 Contrastive 模型
    print("\n[4/4] 训练 Contrastive 模型...")
    model = ContrastiveWideDeep(
        feature_config=feature_config_int,
        contrastive_features=valid_contrastive,
        embedding_size=config["embedding_size"],
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
        contrastive_weight=config["contrastive_weight"],
    )
    
    trainer = ContrastiveTrainer(
        model=model,
        lr=config["lr"],
        device=device,
    )
    
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config["epochs"])
    
    results = trainer.evaluate(test_loader)
    training_time = time.time() - start_time
    
    param_stats = model.get_parameter_stats()
    
    # 保存结果
    results_dir = Path("results/exp005_contrastive")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), results_dir / "model_contrastive.pt")
    
    final_results = {
        "experiment": "exp005_contrastive",
        "config": config,
        "contrastive_features": valid_contrastive,
        "device": str(device),
        "baseline": baseline_results,
        "baseline_bt": baseline_bt,
        "contrastive": {
            "auc": results["auc"],
            "pcoc": results["pcoc"],
            "logloss": results["logloss"],
            "training_time_sec": training_time,
            "bt_grouped": results.get("bt_grouped", {}),
        },
        "parameter_stats": param_stats,
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'Baseline':<15} {'Contrastive':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {baseline_results['auc']:<15.4f} {results['auc']:<15.4f} {(results['auc'] - baseline_results['auc']):>+15.4f}")
    print(f"{'PCOC':<15} {baseline_results['pcoc']:<15.4f} {results['pcoc']:<15.4f} {(results['pcoc'] - baseline_results['pcoc']):>+15.4f}")
    
    print("\n" + "=" * 60)
    print(f"✅ 实验完成！结果已保存到 {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
