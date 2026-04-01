"""
Phase 0 Baseline 训练脚本
- 全量数据，不采样
- 训练集: BT 0,1,2,3,4,5,6,11 (top 8, ~93%)
- 测试集: BT 7,8,10,13,16 (长尾)
- 模型: WideDeep (embedding=8, dnn=[1024,512,256,128], dropout=0.3)
- 评估: AUC + PCOC (整体 + 分 BT)
"""

import sys
import json
from pathlib import Path

# 路径
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.loader import IVRDataLoader, IVRCTCVRDataset, create_dataloader
from models.baseline import WideDeepBaseline
from train.trainer import Trainer

import pandas as pd
import torch

# ============================================================
# 配置
# ============================================================
TRAIN_BTS = [0, 1, 2, 3, 4, 5, 6, 11]
TEST_BTS  = [7, 8, 10, 13, 16]
LABEL_COL = "click_label"
BATCH_SIZE = 512
LR = 5e-5
EMBEDDING_SIZE = 8
DNN_HIDDEN_UNITS = [1024, 512, 256, 128]
DROPOUT = 0.3
EPOCHS = 1
DEVICE = "cuda"
SAVE_PATH = str(PROJECT_ROOT / "results" / "baseline" / "model.pt")
LOG_PATH  = str(PROJECT_ROOT / "results" / "baseline" / "metrics.json")

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("Phase 0 Baseline - Full Data Training")
print("=" * 60)

print("\n[1/4] Loading data...")
data_loader = IVRDataLoader()

train_df, test_df = data_loader.create_cross_domain_split(
    train_bts=TRAIN_BTS,
    test_bts=TEST_BTS
)

print(f"\nTrain: {len(train_df):,} samples")
print(f"Test:  {len(test_df):,} samples")
print(f"Train positive rate: {train_df[LABEL_COL].mean():.4f}")
print(f"Test  positive rate: {test_df[LABEL_COL].mean():.4f}")

# 保存为 parquet 供 Dataset 使用
train_path = "/tmp/baseline_train_full.parquet"
test_path  = "/tmp/baseline_test_full.parquet"
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)

train_dataset = IVRCTCVRDataset(train_path, data_loader.feature_cols, LABEL_COL)
test_dataset  = IVRCTCVRDataset(test_path,  data_loader.feature_cols, LABEL_COL)

train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_loader  = create_dataloader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# 2. 构建模型
# ============================================================
print("\n[2/4] Building model...")
feature_config = data_loader.get_feature_config()

model = WideDeepBaseline(
    feature_config=feature_config,
    embedding_size=EMBEDDING_SIZE,
    dnn_hidden_units=DNN_HIDDEN_UNITS,
    dropout=DROPOUT
)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# ============================================================
# 3. 训练
# ============================================================
print("\n[3/4] Training (1 epoch, full data)...")
trainer = Trainer(model=model, device=DEVICE, lr=LR)

history = trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=EPOCHS,
    early_stopping_patience=3,
    save_path=SAVE_PATH
)

# ============================================================
# 4. 最终评估
# ============================================================
print("\n[4/4] Final evaluation...")
final_metrics = trainer.evaluate(test_loader, prefix="test")

print("\n" + "=" * 60)
print("Final Results")
print("=" * 60)
print(f"  Test AUC:  {final_metrics['test_auc']:.4f}")
print(f"  Test PCOC: {final_metrics['test_pcoc']:.4f}")
print(f"  Test LogLoss: {final_metrics['test_logloss']:.4f}")

if "bt_grouped" in final_metrics:
    print(f"\nBy Business Type:")
    print(f"  {'BT':<8} {'Count':<10} {'AUC':<8} {'PCOC':<8} {'PosRate':<8}")
    print(f"  {'-'*44}")
    for bt_id, m in sorted(final_metrics["bt_grouped"].items()):
        print(f"  {bt_id:<8} {m['count']:<10,} {m['auc']:<8.4f} {m['pcoc']:<8.4f} {m['positive_rate']:<8.4f}")

# 保存结果
results = {
    "config": {
        "train_bts": TRAIN_BTS,
        "test_bts": TEST_BTS,
        "label_col": LABEL_COL,
        "embedding_size": EMBEDDING_SIZE,
        "dnn_hidden_units": DNN_HIDDEN_UNITS,
        "dropout": DROPOUT,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    },
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "test_auc": final_metrics["test_auc"],
    "test_pcoc": final_metrics["test_pcoc"],
    "test_logloss": final_metrics["test_logloss"],
    "bt_grouped": final_metrics.get("bt_grouped", {})
}

with open(LOG_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {LOG_PATH}")
print(f"✅ Model saved to {SAVE_PATH}")
