"""
exp004: 特征子集选择实验

对比不同特征子集方案下的 OOD 性能：
- A: 全量特征 (125)  — 对照组（复现 exp001）
- B: 去掉虚假相关特征 (122)  — os, duf_inner_dev_pkg_pur_11_30d, duf_inner_dev_pkg_pur_61_180d
- C: 只保留因果稳定特征 (17)  — exp002 识别的因果候选
- D: 因果特征 + EmbNorm Top20 (~40)  — 折中方案
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.loader import IVRDataLoader, IVRCTCVRDataset, create_dataloader
from models.baseline import WideDeepBaseline
from train.trainer import Trainer

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp004_feature_subset"
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

# 虚假相关特征（Phase 1 识别，OOD 排名漂移最大）
SPURIOUS_FEATURES = {
    "os",                         # ID:90 → OOD:125，完全失效
    "duf_inner_dev_pkg_pur_11_30d",  # ID:50 → OOD:102，-52 漂移
    "duf_inner_dev_pkg_pur_61_180d", # ID:52 → OOD:101，-49 漂移
}

# 因果稳定特征（Phase 1 识别，ID+OOD 排名均靠前）
CAUSAL_FEATURES = [
    "is_interstitial_ad",
    "business_type",
    "demand_pkgname",
    "connectiontype",
    "duf_inner_dev_pkg_imp_31_60d",
    "duf_inner_dev_pkg_re_bucket_15d",
    "duf_inner_dev_pkg_imp_bucket_15d",
    "offerid",
    "duf_inner_dev_pkg_imp_bucket_3d",
    "duf_inner_dev_pkg_imp_bucket_7d",
    "duf_inner_dev_pkg_re_bucket_7d",
    "duf_inner_dev_pkg_re_bucket_3d",
    "duf_inner_dev_pkg_imp_4_10d",
    "duf_inner_dev_pkg_open_60d",
    "huf_deviceid_demand_pkgname_re_24h",
    "duf_inner_dev_pkg_open_4_10d",
    "adx",
]

# EmbNorm Top20（来自 Phase 1，按 embedding norm 排序）
EMBNORM_TOP20 = [
    "duf_inner_dev_pkg_open_90d",
    "is_reward_ad",
    "duf_outer_dev_pkg_open_61_180d",
    "duf_inner_dev_pkg_open_31_60d",
    "duf_outer_dev_pkg_open_31_60d",
    "duf_inner_dev_pkg_open_11_30d",
    "duf_outer_dev_pkg_open_4_10d",
    "duf_inner_dev_pkg_open_4_10d",
    "duf_outer_dev_pkg_open_11_30d",
    "duf_inner_dev_pkg_open_60d",
    "duf_outer_dev_pkg_open_4_10d",
    "duf_inner_dev_pkg_pur_61_180d",
    "duf_inner_dev_pkg_pur_11_30d",
    "duf_inner_dev_pkg_pur_4_10d",
    "duf_outer_dev_pkg_pur_61_180d",
    "duf_outer_dev_pkg_pur_11_30d",
    "duf_outer_dev_pkg_pur_4_10d",
    "duf_inner_dev_pkg_imp_bucket_15d",
    "duf_inner_dev_pkg_re_bucket_15d",
    "demand_pkgname",
]


# ============================================================
# 评估函数
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, device, prefix="test"):
    model.eval()
    all_preds, all_labels, all_bts = [], [], []

    for batch_features, batch_labels in dataloader:
        batch_features = {k: v.to(device) for k, v in batch_features.items()}
        logits = model(batch_features)
        preds = torch.sigmoid(logits).cpu().numpy()
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


def run_experiment(scheme_name, feature_cols, feature_config, train_df, test_df):
    """运行单个特征子集实验"""
    print(f"\n{'='*60}")
    print(f"Scheme {scheme_name}: {len(feature_cols)} features")
    print("="*60)

    # 保存临时数据
    train_path = f"/tmp/exp004_train_{scheme_name}.parquet"
    test_path = f"/tmp/exp004_test_{scheme_name}.parquet"

    # 只保留需要的列
    cols_needed = feature_cols + [LABEL_COL, "business_type"]
    cols_needed = list(dict.fromkeys(cols_needed))  # 去重保序

    train_df[cols_needed].to_parquet(train_path, index=False)
    test_df[cols_needed].to_parquet(test_path, index=False)

    # 数据集
    train_dataset = IVRCTCVRDataset(train_path, feature_cols, LABEL_COL)
    test_dataset = IVRCTCVRDataset(test_path, feature_cols, LABEL_COL)
    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化模型
    torch.manual_seed(42)
    model = WideDeepBaseline(
        feature_config=feature_config,
        embedding_size=8,
        dnn_hidden_units=[1024, 512, 256, 128],
        dropout=0.3
    )

    # 训练
    trainer = Trainer(model=model, device=DEVICE, lr=LR)
    train_metrics = trainer.train_epoch(train_loader, epoch=1)
    print(f"  Train loss: {train_metrics['train_loss']:.4f}")

    # 评估
    device = torch.device(DEVICE)
    metrics = evaluate(model, test_loader, device, prefix="test")
    print_results(scheme_name, metrics)

    # 保存模型
    torch.save({
        "scheme": scheme_name,
        "n_features": len(feature_cols),
        "model_state_dict": model.state_dict(),
        **metrics
    }, str(OUTPUT_DIR / f"model_{scheme_name}.pt"))

    return metrics


# ============================================================
# 主流程
# ============================================================
print("=" * 60)
print("exp004: Feature Subset Selection")
print("=" * 60)

print("\n[1/2] Preparing data...")
data_loader = IVRDataLoader()
all_feature_cols = data_loader.feature_cols  # 125 个（已排除 deviceid）
all_feature_config = data_loader.get_feature_config()

train_df, test_df = data_loader.create_cross_domain_split(
    train_bts=TRAIN_BTS,
    test_bts=TEST_BTS
)
print(f"  Train: {len(train_df):,} samples (BT {TRAIN_BTS})")
print(f"  Test:  {len(test_df):,} samples (BT {TEST_BTS})")

# ============================================================
# 定义各方案的特征集
# ============================================================

# 方案 A：全量特征（对照组）
scheme_A_features = all_feature_cols

# 方案 B：去掉虚假相关特征
scheme_B_features = [f for f in all_feature_cols if f not in SPURIOUS_FEATURES]

# 方案 C：只保留因果稳定特征（过滤掉数据中不存在的）
scheme_C_features = [f for f in CAUSAL_FEATURES if f in all_feature_cols]

# 方案 D：因果特征 + EmbNorm Top20（去重）
scheme_D_raw = CAUSAL_FEATURES + [f for f in EMBNORM_TOP20 if f not in CAUSAL_FEATURES]
scheme_D_features = [f for f in scheme_D_raw if f in all_feature_cols]

schemes = {
    "A_full": scheme_A_features,
    "B_no_spurious": scheme_B_features,
    "C_causal_only": scheme_C_features,
    "D_causal_plus_top20": scheme_D_features,
}

print(f"\n[2/2] Running experiments...")
for name, feats in schemes.items():
    print(f"  {name}: {len(feats)} features")

# ============================================================
# 逐方案训练
# ============================================================
all_results = {}

for scheme_name, feature_cols in schemes.items():
    # 对应的 feature_config（只包含该方案的特征）
    feature_config = {f: all_feature_config[f] for f in feature_cols}

    metrics = run_experiment(scheme_name, feature_cols, feature_config, train_df, test_df)
    all_results[scheme_name] = {
        "n_features": len(feature_cols),
        "test_auc": metrics["test_auc"],
        "test_pcoc": metrics["test_pcoc"],
        "test_logloss": metrics["test_logloss"],
        "bt_grouped": metrics.get("bt_grouped", {})
    }

# ============================================================
# 汇总对比
# ============================================================
print("\n" + "=" * 60)
print("exp004 Summary: Feature Subset Ablation")
print("=" * 60)
print(f"\n{'方案':<25} {'特征数':<8} {'AUC':<10} {'PCOC':<10} {'LogLoss':<10}")
print("-" * 63)
for name, r in all_results.items():
    print(f"{name:<25} {r['n_features']:<8} {r['test_auc']:<10.4f} {r['test_pcoc']:<10.4f} {r['test_logloss']:<10.4f}")

# PCOC 方差对比
print(f"\nPCOC variance by BT (越小越稳定):")
for name, r in all_results.items():
    if r["bt_grouped"]:
        pcocs = [m["pcoc"] for m in r["bt_grouped"].values()]
        print(f"  {name}: std={np.std(pcocs):.4f}, min={min(pcocs):.4f}, max={max(pcocs):.4f}")

# 保存汇总
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Results saved to {OUTPUT_DIR}")
