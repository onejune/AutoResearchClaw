"""
exp005: Domain Adversarial Neural Network (DANN)

使用域对抗训练对齐训练集和 OOD 域的特征分布，
重点解决 bt_16 分布漂移问题。

λ 消融：[0.0, 0.1, 1.0, 10.0]
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.loader import IVRDataLoader, IVRCTCVRDataset, create_dataloader
from methods.dann import DANNModel, DANNTrainer

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp005_dann"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 配置
# ============================================================
TRAIN_BTS = [0, 1, 2, 3, 4, 5, 6, 11]
TEST_BTS = [7, 8, 10, 13, 16]
LABEL_COL = "click_label"
BATCH_SIZE = 512
LR = 5e-5
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOMAIN_WEIGHTS = [0.0, 0.1, 1.0, 10.0]


# ============================================================
# 评估函数
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, device, prefix="test"):
    model.eval()
    all_preds, all_labels, all_bts = [], [], []

    for batch_features, batch_labels in dataloader:
        batch_features = {k: v.to(device) for k, v in batch_features.items()}
        outputs = model(batch_features, return_domain=False)
        preds = torch.sigmoid(outputs["logits"]).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.numpy())
        if "business_type" in batch_features:
            all_bts.extend(batch_features["business_type"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pcoc = float(all_preds.mean() / all_labels.mean()) if all_labels.mean() > 0 else float("nan")
    metrics = {
        f"{prefix}_auc": float(roc_auc_score(all_labels, all_preds)),
        f"{prefix}_pcoc": pcoc,
        f"{prefix}_logloss": float(log_loss(all_labels, all_preds)),
    }

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
            bt_metrics[f"bt_{int(bt_id)}"] = {
                "count": int(mask.sum()),
                "auc": float(roc_auc_score(bt_labels, bt_preds)),
                "pcoc": bt_pcoc,
                "positive_rate": float(bt_labels.mean())
            }
        metrics["bt_grouped"] = bt_metrics

    return metrics


def print_results(label, metrics):
    print(f"\n  {label} Results:")
    print(f"    Test AUC:  {metrics['test_auc']:.4f}")
    print(f"    Test PCOC: {metrics['test_pcoc']:.4f}")
    print(f"    Test LogLoss: {metrics['test_logloss']:.4f}")
    if "bt_grouped" in metrics:
        print(f"\n    By BT:")
        print(f"    {'BT':<8} {'Count':<10} {'AUC':<8} {'PCOC':<8}")
        print(f"    {'-'*36}")
        for bt_id, m in sorted(metrics["bt_grouped"].items()):
            print(f"    {bt_id:<8} {m['count']:<10,} {m['auc']:<8.4f} {m['pcoc']:<8.4f}")


# ============================================================
# 主流程
# ============================================================
print("=" * 60)
print("exp005: Domain Adversarial Neural Network (DANN)")
print("=" * 60)

print("\n[1/2] Preparing data...")
loader = IVRDataLoader()
feature_cols = loader.feature_cols
feature_config = loader.get_feature_config()

train_df, test_df = loader.create_cross_domain_split(
    train_bts=TRAIN_BTS,
    test_bts=TEST_BTS
)

# 添加 business_type 到特征列表（用于分组，但不参与模型训练）
# 注意：feature_cols 里可能已经包含 business_type，需要去重
cols_with_bt = list(dict.fromkeys(feature_cols + [LABEL_COL, "business_type"]))

# 保存临时文件
train_path = "/tmp/exp005_train.parquet"
test_path = "/tmp/exp005_test.parquet"
train_df[cols_with_bt].to_parquet(train_path, index=False)
test_df[cols_with_bt].to_parquet(test_path, index=False)

# 数据集
train_dataset = IVRCTCVRDataset(train_path, feature_cols, LABEL_COL)
test_dataset = IVRCTCVRDataset(test_path, feature_cols, LABEL_COL)

# 源域 loader（有标签，shuffle）
source_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# 目标域 loader（无标签，不需要 shuffle）
target_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # shuffle 以便采样
test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"  Source (train): {len(train_dataset):,} samples")
print(f"  Target (test):  {len(test_dataset):,} samples")

# ============================================================
# λ 消融实验
# ============================================================
print(f"\n[2/2] Running DANN with λ = {DOMAIN_WEIGHTS}")

all_results = {}

for domain_weight in DOMAIN_WEIGHTS:
    print(f"\n{'='*60}")
    print(f"λ = {domain_weight}")
    print("=" * 60)
    
    # 初始化模型
    torch.manual_seed(42)
    model = DANNModel(
        feature_config=feature_config,
        embedding_size=8,
        dnn_hidden_units=[1024, 512, 256, 128],
        domain_hidden_units=[256, 128, 1],
        dropout=0.3
    )
    
    # 训练
    trainer = DANNTrainer(model=model, device=DEVICE, lr=LR, domain_weight=domain_weight)
    
    train_metrics = trainer.train_epoch(
        source_loader=source_loader,
        target_loader=target_loader,
        epoch=1,
        total_epochs=1,
        lambda_schedule="linear"
    )
    
    print(f"  Train loss: {train_metrics['train_loss']:.4f} (label: {train_metrics['label_loss']:.4f}, domain: {train_metrics['domain_loss']:.4f})")
    
    # 评估
    metrics = evaluate(model, test_loader, DEVICE, prefix="test")
    print_results(f"λ={domain_weight}", metrics)
    
    # 保存
    all_results[f"lambda_{domain_weight}"] = {
        "domain_weight": domain_weight,
        "test_auc": metrics["test_auc"],
        "test_pcoc": metrics["test_pcoc"],
        "test_logloss": metrics["test_logloss"],
        "bt_grouped": metrics.get("bt_grouped", {}),
        "train_metrics": train_metrics
    }
    
    torch.save({
        "domain_weight": domain_weight,
        "model_state_dict": model.state_dict(),
        **metrics
    }, str(OUTPUT_DIR / f"model_lambda_{domain_weight}.pt"))

# ============================================================
# 汇总对比
# ============================================================
print("\n" + "=" * 60)
print("exp005 Summary: DANN Domain Weight Ablation")
print("=" * 60)
print(f"\n{'λ':<8} {'AUC':<10} {'PCOC':<10} {'LogLoss':<10}")
print("-" * 38)
for name, r in all_results.items():
    print(f"{r['domain_weight']:<8} {r['test_auc']:<10.4f} {r['test_pcoc']:<10.4f} {r['test_logloss']:<10.4f}")

# bt_16 对比
print(f"\nbt_16 AUC 对比:")
for name, r in all_results.items():
    if "bt_grouped" in r and "bt_16" in r["bt_grouped"]:
        bt16 = r["bt_grouped"]["bt_16"]
        print(f"  λ={r['domain_weight']}: AUC={bt16['auc']:.4f}, PCOC={bt16['pcoc']:.4f}")

# 保存汇总
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Results saved to {OUTPUT_DIR}")
