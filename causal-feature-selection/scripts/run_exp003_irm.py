"""
exp003: IRM 多环境训练

对比不同 λ 下 IRM vs ERM baseline 的 OOD 性能
- λ=0:   纯 ERM（对照组，应与 exp001 接近）
- λ=1:   轻度 IRM
- λ=10:  中度 IRM
- λ=100: 强 IRM
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
from methods.irm import IRMTrainer
from train.trainer import Trainer

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp003_irm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 配置
# ============================================================
ENVIRONMENTS = {
    "env_0": [0, 1],
    "env_1": [2, 3, 4],
    "env_2": [5, 6, 11],
}
TEST_BTS = [7, 8, 10, 13, 16]
LABEL_COL = "click_label"
BATCH_SIZE = 512
LR = 5e-5
EPOCHS = 1
DEVICE = "cuda"
PENALTY_ANNEAL_ITERS = 200  # 前 200 步 warm-up

LAMBDAS = [0.0, 1.0, 10.0, 100.0]  # 消融实验


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


# ============================================================
# 数据准备
# ============================================================
print("=" * 60)
print("exp003: IRM Multi-Environment Training")
print("=" * 60)

print("\n[1/3] Preparing data...")
data_loader = IVRDataLoader()
feature_config = data_loader.get_feature_config()
feature_names = data_loader.feature_cols

# 加载全量训练数据
all_train_bts = [bt for bts in ENVIRONMENTS.values() for bt in bts]
train_df, test_df = data_loader.create_cross_domain_split(
    train_bts=all_train_bts,
    test_bts=TEST_BTS
)

# 按环境划分
env_loaders = []
for env_name, env_bts in ENVIRONMENTS.items():
    env_df = train_df[train_df["business_type"].isin(env_bts)]
    env_path = f"/tmp/irm_{env_name}.parquet"
    env_df.to_parquet(env_path, index=False)
    env_dataset = IVRCTCVRDataset(env_path, feature_names, LABEL_COL)
    env_loader = create_dataloader(env_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    env_loaders.append(env_loader)
    print(f"  {env_name} (BT {env_bts}): {len(env_df):,} samples")

# 测试集
test_path = "/tmp/irm_test.parquet"
test_df.to_parquet(test_path, index=False)
test_dataset = IVRCTCVRDataset(test_path, feature_names, LABEL_COL)
test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"  test  (BT {TEST_BTS}): {len(test_df):,} samples")

# ============================================================
# 消融实验：不同 λ
# ============================================================
all_results = {}

for lam in LAMBDAS:
    print(f"\n{'='*60}")
    print(f"[2/3] Training IRM λ={lam}")
    print("="*60)

    # 初始化模型（每次重新初始化，保证公平对比）
    torch.manual_seed(42)
    model = WideDeepBaseline(
        feature_config=feature_config,
        embedding_size=8,
        dnn_hidden_units=[1024, 512, 256, 128],
        dropout=0.3
    )

    if lam == 0.0:
        # λ=0 用普通 ERM Trainer（对照组）
        trainer = Trainer(model=model, device=DEVICE, lr=LR)
        # 合并所有环境数据训练
        all_train_path = "/tmp/irm_all_train.parquet"
        train_df.to_parquet(all_train_path, index=False)
        all_train_dataset = IVRCTCVRDataset(all_train_path, feature_names, LABEL_COL)
        all_train_loader = create_dataloader(all_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        train_metrics = trainer.train_epoch(all_train_loader, epoch=1)
        print(f"  ERM loss: {train_metrics['train_loss']:.4f}")
    else:
        # IRM 训练
        irm_trainer = IRMTrainer(
            model=model,
            device=DEVICE,
            lr=LR,
            penalty_weight=lam,
            penalty_anneal_iters=PENALTY_ANNEAL_ITERS
        )
        train_metrics = irm_trainer.train_epoch(env_loaders, epoch=1)
        print(f"  ERM loss: {train_metrics['train_erm_loss']:.4f}, IRM penalty: {train_metrics['train_irm_penalty']:.6f}")

    # 评估
    print(f"\n[3/3] Evaluating λ={lam}...")
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    metrics = evaluate(model, test_loader, device, prefix="test")

    print(f"\n  λ={lam} Results:")
    print(f"    Test AUC:  {metrics['test_auc']:.4f}")
    print(f"    Test PCOC: {metrics['test_pcoc']:.4f}")
    print(f"    Test LogLoss: {metrics['test_logloss']:.4f}")

    if "bt_grouped" in metrics:
        print(f"\n    By BT:")
        print(f"    {'BT':<8} {'Count':<10} {'AUC':<8} {'PCOC':<8}")
        print(f"    {'-'*36}")
        for bt_id, m in sorted(metrics["bt_grouped"].items()):
            print(f"    {bt_id:<8} {m['count']:<10,} {m['auc']:<8.4f} {m['pcoc']:<8.4f}")

    # 保存模型
    model_path = OUTPUT_DIR / f"model_lambda{lam}.pt"
    torch.save({"lambda": lam, "model_state_dict": model.state_dict(), **metrics}, str(model_path))

    all_results[f"lambda_{lam}"] = {
        "lambda": lam,
        "test_auc": metrics["test_auc"],
        "test_pcoc": metrics["test_pcoc"],
        "test_logloss": metrics["test_logloss"],
        "bt_grouped": metrics.get("bt_grouped", {})
    }

# ============================================================
# 汇总对比
# ============================================================
print("\n" + "=" * 60)
print("exp003 Summary: IRM λ Ablation")
print("=" * 60)
print(f"\n{'λ':<10} {'AUC':<10} {'PCOC':<10} {'LogLoss':<10}")
print("-" * 40)
for key, r in all_results.items():
    print(f"{r['lambda']:<10} {r['test_auc']:<10.4f} {r['test_pcoc']:<10.4f} {r['test_logloss']:<10.4f}")

# PCOC 方差对比（越小越好）
print(f"\nPCOC variance by BT (越小越稳定):")
for key, r in all_results.items():
    if r["bt_grouped"]:
        pcocs = [m["pcoc"] for m in r["bt_grouped"].values()]
        print(f"  λ={r['lambda']}: std={np.std(pcocs):.4f}, min={min(pcocs):.4f}, max={max(pcocs):.4f}")

# 保存汇总
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Results saved to {OUTPUT_DIR}")
