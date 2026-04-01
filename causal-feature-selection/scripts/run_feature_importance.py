"""
Phase 1: 特征重要性分析脚本

1. 加载 Phase 0 训练好的 baseline 模型
2. 用三种方法计算特征重要性:
   - Embedding Norm (零成本，直接读权重)
   - Gradient-based (一次反向传播)
   - Permutation Importance (最可靠，但最慢)
3. 对比 in-domain vs OOD 场景下的重要性差异
4. 输出: results/phase1_feature_importance/
"""

import sys
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.loader import IVRDataLoader, IVRCTCVRDataset, create_dataloader
from models.baseline import WideDeepBaseline
from methods.feature_importance import (
    EmbeddingNormImportance,
    GradientImportance,
    PermutationImportance,
    compare_importance_methods
)

import pandas as pd
import numpy as np

# ============================================================
# 配置
# ============================================================
CHECKPOINT_PATH = str(PROJECT_ROOT / "results" / "baseline" / "model.pt")
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase1_feature_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BTS = [0, 1, 2, 3, 4, 5, 6, 11]
TEST_BTS  = [7, 8, 10, 13, 16]
LABEL_COL = "click_label"
DEVICE = "cuda"
BATCH_SIZE = 512

# ============================================================
# 1. 加载数据 & 模型
# ============================================================
print("=" * 60)
print("Phase 1: Feature Importance Analysis")
print("=" * 60)

print("\n[1/5] Loading data & model...")
data_loader = IVRDataLoader()
feature_config = data_loader.get_feature_config()
feature_names = data_loader.feature_cols

# 加载模型
model = WideDeepBaseline(
    feature_config=feature_config,
    embedding_size=8,
    dnn_hidden_units=[1024, 512, 256, 128],
    dropout=0.3
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print(f"✅ Loaded checkpoint: epoch={checkpoint['epoch']}, val_auc={checkpoint['val_auc']:.4f}")

# 加载 in-domain 和 OOD 数据 (各取 5 万样本，PI 计算速度)
train_df, test_df = data_loader.create_cross_domain_split(
    train_bts=TRAIN_BTS, test_bts=TEST_BTS
)

# in-domain: 从训练 BT 中取样
indomain_df = train_df.sample(n=50000, random_state=42)
indomain_df.to_parquet("/tmp/indomain_sample.parquet", index=False)
indomain_dataset = IVRCTCVRDataset("/tmp/indomain_sample.parquet", feature_names, LABEL_COL)
indomain_loader = create_dataloader(indomain_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# OOD: 全量测试集
test_df.to_parquet("/tmp/ood_sample.parquet", index=False)
ood_dataset = IVRCTCVRDataset("/tmp/ood_sample.parquet", feature_names, LABEL_COL)
ood_loader = create_dataloader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"In-domain: {len(indomain_df):,} samples")
print(f"OOD: {len(test_df):,} samples")

# ============================================================
# 2. Embedding Norm Importance (零成本)
# ============================================================
print("\n[2/5] Embedding Norm Importance...")
en_analyzer = EmbeddingNormImportance(model)
en_df = en_analyzer.compute(feature_names)
en_df.to_csv(OUTPUT_DIR / "embedding_norm_importance.csv", index=False)

print(f"\nTop 20 by Embedding Norm:")
print(en_df.head(20)[["feature", "mean_norm", "max_norm"]].to_string(index=False))

# ============================================================
# 3. Gradient Importance (in-domain)
# ============================================================
print("\n[3/5] Gradient Importance (in-domain)...")
gi_analyzer = GradientImportance(model, device=DEVICE)
gi_indomain_df = gi_analyzer.compute(indomain_loader, feature_names, n_batches=100)
gi_indomain_df.to_csv(OUTPUT_DIR / "gradient_importance_indomain.csv", index=False)

print(f"\nTop 20 by Gradient (in-domain):")
print(gi_indomain_df.head(20)[["feature", "mean_importance"]].to_string(index=False))

# ============================================================
# 4. Gradient Importance (OOD)
# ============================================================
print("\n[4/5] Gradient Importance (OOD)...")
gi_ood_df = gi_analyzer.compute(ood_loader, feature_names, n_batches=100)
gi_ood_df.to_csv(OUTPUT_DIR / "gradient_importance_ood.csv", index=False)

print(f"\nTop 20 by Gradient (OOD):")
print(gi_ood_df.head(20)[["feature", "mean_importance"]].to_string(index=False))

# ============================================================
# 5. 对比分析: in-domain vs OOD 重要性差异
# ============================================================
print("\n[5/5] Comparing in-domain vs OOD importance...")

# 合并两个 GI 结果
gi_compare = gi_indomain_df[["feature", "mean_importance"]].rename(
    columns={"mean_importance": "gi_indomain"}
).merge(
    gi_ood_df[["feature", "mean_importance"]].rename(
        columns={"mean_importance": "gi_ood"}
    ),
    on="feature"
)

# 计算排名
gi_compare["rank_indomain"] = gi_compare["gi_indomain"].rank(ascending=False).astype(int)
gi_compare["rank_ood"] = gi_compare["gi_ood"].rank(ascending=False).astype(int)
gi_compare["rank_shift"] = gi_compare["rank_indomain"] - gi_compare["rank_ood"]  # 正 = OOD 时重要性下降

# 排名变化最大的特征 (可能是虚假相关特征)
gi_compare_sorted = gi_compare.sort_values("rank_shift", key=abs, ascending=False)
gi_compare_sorted.to_csv(OUTPUT_DIR / "importance_shift_indomain_vs_ood.csv", index=False)

print(f"\n特征重要性排名变化最大 (可能是虚假相关特征):")
print(f"{'Feature':<45} {'Rank(ID)':<10} {'Rank(OOD)':<10} {'Shift':<8}")
print("-" * 73)
for _, row in gi_compare_sorted.head(20).iterrows():
    shift_str = f"+{int(row['rank_shift'])}" if row['rank_shift'] > 0 else str(int(row['rank_shift']))
    print(f"{row['feature']:<45} {int(row['rank_indomain']):<10} {int(row['rank_ood']):<10} {shift_str:<8}")

# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("Phase 1 Summary")
print("=" * 60)

# 找出 in-domain 重要但 OOD 不重要的特征 (虚假相关候选)
spurious_candidates = gi_compare[
    (gi_compare["rank_indomain"] <= 20) &  # in-domain top 20
    (gi_compare["rank_ood"] > 50)           # OOD rank 50 以外
].sort_values("rank_indomain")

print(f"\n⚠️  虚假相关特征候选 (in-domain 重要但 OOD 不重要):")
if len(spurious_candidates) > 0:
    for _, row in spurious_candidates.iterrows():
        print(f"  {row['feature']}: rank_id={int(row['rank_indomain'])}, rank_ood={int(row['rank_ood'])}")
else:
    print("  (无明显虚假相关特征)")

# 找出两者都重要的特征 (可能是因果特征)
causal_candidates = gi_compare[
    (gi_compare["rank_indomain"] <= 20) &
    (gi_compare["rank_ood"] <= 20)
].sort_values("rank_indomain")

print(f"\n✅ 因果特征候选 (in-domain 和 OOD 都重要):")
for _, row in causal_candidates.iterrows():
    print(f"  {row['feature']}: rank_id={int(row['rank_indomain'])}, rank_ood={int(row['rank_ood'])}")

# 保存汇总
summary = {
    "spurious_candidates": spurious_candidates["feature"].tolist(),
    "causal_candidates": causal_candidates["feature"].tolist(),
    "top20_indomain": gi_indomain_df.head(20)["feature"].tolist(),
    "top20_ood": gi_ood_df.head(20)["feature"].tolist(),
    "top20_emb_norm": en_df.head(20)["feature"].tolist()
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to {OUTPUT_DIR}")
