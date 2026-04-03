"""
exp003: Hierarchical Embeddings (CIKM 2021)

实验设计:
- 对比普通 WideDeep vs 层次化 Embedding
- 重点观察样本少的 BT 的效果
- 分析门控权重的分布（细粒度 vs 粗粒度偏好）

数据集: IVR Sample v16 CTCVR
- 训练集: /mnt/data/.../ivr_sample_v16_ctcvr_sample/train/
- 测试集: /mnt/data/.../ivr_sample_v16_ctcvr_sample/test/
- 所有特征都是类别特征，已编码好
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.loader import IVRDataLoader, create_dataloader
from models.hierarchical import HierarchicalWideDeep
from train.trainer import DynamicTrainer

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp003_hierarchical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 配置
# ============================================================
LABEL_COL = "ctcvr_label"
BATCH_SIZE = 512
LR = 5e-5
EPOCHS = 1
DEVICE = "cuda"  # 自动降级到 CPU

# 广告 ID 层级结构（由细到粗）：
# campaignid → campaignsetid → offerid → demand_pkgname → business_type
# 层次化特征对：细粒度 → 粗粒度
HIERARCHICAL_PAIRS = {
    "campaignid": "campaignsetid",      # 广告计划 → 广告组
    "offerid": "demand_pkgname",        # offer → 包名
    "demand_pkgname": "business_type",  # 包名 → 业务类型
}


# ============================================================
# 评估函数
# ============================================================
@torch.no_grad()
def evaluate_with_gates(model, dataloader, device, has_hierarchical=True):
    """评估并收集门控统计"""
    from sklearn.metrics import roc_auc_score, log_loss
    
    model.eval()
    all_preds, all_labels, all_bts = [], [], []
    gate_stats = {}
    if has_hierarchical and hasattr(model, 'hierarchical_features'):
        gate_stats = {feat: [] for feat in model.hierarchical_features}
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        features = {k: v.to(device) for k, v in batch["features"].items()}
        labels = batch["label"].cpu().numpy()
        
        outputs = model(features)
        preds = torch.sigmoid(outputs).cpu().numpy()
        
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.flatten())
        
        if "business_type" in batch:
            all_bts.extend(batch["business_type"].cpu().numpy())
        
        # 收集门控统计
        if has_hierarchical and hasattr(model, 'get_gate_stats'):
            gate_info = model.get_gate_stats(features)
            for feat, stats in gate_info.items():
                gate_stats[feat].append(stats["mean_gate"])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
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
            
            # 检查是否有两个类别
            unique_labels = np.unique(bt_labels)
            if len(unique_labels) < 2:
                # 只有一个类别，跳过 AUC 计算
                continue
            
            bt_pcoc = float(bt_preds.mean() / bt_labels.mean()) if bt_labels.mean() > 0 else float("nan")
            bt_metrics[f"bt_{int(bt_id)}"] = {
                "count": int(mask.sum()),
                "auc": float(roc_auc_score(bt_labels, bt_preds)),
                "pcoc": bt_pcoc
            }
        metrics["bt_grouped"] = bt_metrics
    
    # 平均门控权重
    if gate_stats:
        avg_gates = {feat: float(np.mean(gates)) for feat, gates in gate_stats.items() if gates}
        metrics["gate_stats"] = avg_gates
    
    return metrics


def print_results(label, metrics):
    print(f"\n{label} Results:")
    print(f"  Test AUC:  {metrics['auc']:.4f}")
    print(f"  Test PCOC: {metrics['pcoc']:.4f}")
    print(f"  Test LogLoss: {metrics['logloss']:.4f}")
    
    if "bt_grouped" in metrics:
        print(f"\n  By BT:")
        print(f"  {'BT':<8} {'Count':<10} {'AUC':<8} {'PCOC':<8}")
        print(f"  {'-'*36}")
        for bt_id, m in sorted(metrics["bt_grouped"].items()):
            print(f"  {bt_id:<8} {m['count']:<10,} {m['auc']:<8.4f} {m['pcoc']:<8.4f}")
    
    if "gate_stats" in metrics:
        print(f"\n  Gate Statistics (mean=fine 粒度权重):")
        for feat, gate_mean in metrics["gate_stats"].items():
            interpretation = "偏向细粒度" if gate_mean > 0.6 else ("偏向粗粒度" if gate_mean < 0.4 else "均衡")
            print(f"    {feat}: {gate_mean:.3f} ({interpretation})")


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("exp003: Hierarchical Embeddings")
    print("=" * 60)
    
    print("\n[1/3] Loading data...")
    loader = IVRDataLoader(label_col=LABEL_COL)
    feature_config = loader.get_feature_config()
    
    # 获取层次化特征的 vocab sizes
    hier_vocab = loader.get_hierarchy_vocab_sizes(HIERARCHICAL_PAIRS)
    print(f"\nHierarchical feature vocab sizes:")
    for fine, (fine_v, coarse_v) in hier_vocab.items():
        coarse = HIERARCHICAL_PAIRS[fine]
        print(f"  {fine} ({fine_v:,}) → {coarse} ({coarse_v:,})")
    
    # 加载数据集
    train_dataset, test_dataset = loader.load_datasets(
        hierarchical_features=HIERARCHICAL_PAIRS
    )
    
    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nData loaded:")
    print(f"  - Train samples: {len(train_dataset):,}")
    print(f"  - Test samples:  {len(test_dataset):,}")
    print(f"  - Features: {len(feature_config)}")
    
    print(f"\n[2/3] Training models...")
    
    # ===== 模型 1: Baseline (普通 WideDeep) =====
    print("\n" + "="*40)
    print("Baseline: Standard WideDeep")
    print("="*40)
    
    torch.manual_seed(42)
    baseline_model = HierarchicalWideDeep(
        feature_config=feature_config,
        hierarchical_pairs={},  # 不用层次化
        embedding_size=64,
        dnn_hidden_units=[1024, 512, 256, 128],
        dropout=0.3
    )
    
    baseline_trainer = DynamicTrainer(baseline_model, device=DEVICE, lr=LR)
    train_metrics = baseline_trainer.train_epoch(train_loader, epoch=1, total_epochs=EPOCHS)
    print(f"Train loss: {train_metrics['train_loss']:.4f}")
    
    baseline_metrics = evaluate_with_gates(baseline_model, test_loader, baseline_trainer.device, has_hierarchical=False)
    print_results("Baseline", baseline_metrics)
    
    # ===== 模型 2: Hierarchical Embedding =====
    print("\n" + "="*40)
    print("Hierarchical Embedding")
    print("="*40)
    
    torch.manual_seed(42)
    hier_model = HierarchicalWideDeep(
        feature_config=feature_config,
        hierarchical_pairs=HIERARCHICAL_PAIRS,
        embedding_size=64,
        dnn_hidden_units=[1024, 512, 256, 128],
        dropout=0.3
    )
    
    hier_trainer = DynamicTrainer(hier_model, device=DEVICE, lr=LR)
    train_metrics = hier_trainer.train_epoch(train_loader, epoch=1, total_epochs=EPOCHS)
    print(f"Train loss: {train_metrics['train_loss']:.4f}")
    
    hier_metrics = evaluate_with_gates(hier_model, test_loader, hier_trainer.device, has_hierarchical=True)
    print_results("Hierarchical", hier_metrics)
    
    # ============================================================
    # 汇总对比
    # ============================================================
    print("\n" + "=" * 60)
    print("Summary: Baseline vs Hierarchical")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'Baseline':<12} {'Hierarchical':<12} {'Delta':<10}")
    print("-" * 54)
    print(f"{'Overall AUC':<20} {baseline_metrics['auc']:<12.4f} {hier_metrics['auc']:<12.4f} {hier_metrics['auc'] - baseline_metrics['auc']:+.4f}")
    print(f"{'Overall PCOC':<20} {baseline_metrics['pcoc']:<12.4f} {hier_metrics['pcoc']:<12.4f} {hier_metrics['pcoc'] - baseline_metrics['pcoc']:+.4f}")
    
    if "bt_grouped" in baseline_metrics and "bt_grouped" in hier_metrics:
        print(f"\n{'BT':<8} {'Baseline AUC':<12} {'Hierarchical AUC':<16} {'Delta':<10}")
        print("-" * 46)
        for bt_id in sorted(baseline_metrics["bt_grouped"].keys()):
            if bt_id in hier_metrics["bt_grouped"]:
                base_auc = baseline_metrics["bt_grouped"][bt_id]["auc"]
                hier_auc = hier_metrics["bt_grouped"][bt_id]["auc"]
                delta = hier_auc - base_auc
                marker = "🔺" if delta > 0.001 else ("🔻" if delta < -0.001 else "")
                print(f"{bt_id:<8} {base_auc:<12.4f} {hier_auc:<16.4f} {delta:+.4f} {marker}")
    
    # 保存结果
    results = {
        "experiment": "exp003_hierarchical",
        "baseline": baseline_metrics,
        "hierarchical": hier_metrics,
        "hierarchical_pairs": HIERARCHICAL_PAIRS,
        "config": {
            "label_col": LABEL_COL,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "embedding_size": 64,
            "dnn_hidden_units": [1024, 512, 256, 128]
        }
    }
    
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 保存模型
    torch.save({
        "model_state_dict": hier_model.state_dict(),
        "metrics": hier_metrics,
        "config": {"hierarchical_pairs": HIERARCHICAL_PAIRS}
    }, str(OUTPUT_DIR / "model_hierarchical.pt"))
    
    print(f"\n✅ Results saved to {OUTPUT_DIR}")
