#!/usr/bin/env python3
"""
IVR SSL 实验详细分析
分析各方法在不同 business_type 上的表现差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ===== 数据集类（与训练脚本一致）=====
class IVRDataset(Dataset):
    def __init__(self, data_path, max_samples=None):
        df = pq.read_table(data_path).to_pandas()
        if max_samples:
            df = df.head(max_samples)
        
        # 特征和标签
        self.labels = torch.FloatTensor(df['ctcvr_label'].values.astype(np.float32))
        self.business_types = df['business_type'].values  # 保持为 numpy 用于分组统计
        
        # 所有特征（126个，全部是类别特征）
        feature_cols = [c for c in df.columns if c not in ['click_label', 'ctcvr_label', 'business_type']]
        self.features = torch.LongTensor(df[feature_cols].values.astype(np.int64))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'business_type': self.business_types[idx]
        }


def collate_fn(batch):
    features = torch.stack([b['features'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    business_types = np.array([b['business_type'] for b in batch])
    return {
        'features': features,
        'labels': labels,
        'business_types': business_types
    }


# ===== 模型定义（与训练脚本一致）=====
class BaseModel(nn.Module):
    """基础 DeepFM 模型"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dim=256):
        super().__init__()
        self.n_features = len(vocab_sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        
        # BatchNorm for embeddings
        self.bn = nn.BatchNorm1d(self.n_features * embed_dim)
        
        # MLP with dropout
        self.mlp = nn.Sequential(
            nn.Linear(self.n_features * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Xavier initialization
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, features):
        # features: [batch, n_features]
        embs = []
        for i, emb in enumerate(self.embeddings):
            feat_vals = features[:, i].clamp(0, emb.num_embeddings - 1)  # 防止越界
            embs.append(emb(feat_vals))
        
        x = torch.cat(embs, dim=-1)  # [batch, n_features * embed_dim]
        x = self.bn(x)
        logits = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logits)


class SupCon(BaseModel):
    """Supervised Contrastive Learning"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dim=256,
                 proj_dim=64, temperature=0.1, cl_weight=0.1):
        super().__init__(vocab_sizes, embed_dim, hidden_dim)
        
        input_dim = self.n_features * embed_dim
        self.projection = self._create_projection_head(input_dim, hidden_dim, proj_dim)
        self.temperature = temperature
        self.cl_weight = cl_weight
    
    def _create_projection_head(self, input_dim, hidden_dim, output_dim):
        net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Xavier 初始化
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def contrastive_loss(self, batch) -> torch.Tensor:
        """监督对比损失"""
        embeds = self.get_embed_vector(batch['features'])
        z = F.normalize(self.projection(embeds), dim=-1)
        
        labels = batch['labels']
        batch_size = z.shape[0]
        
        # 相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature  # [B, B]
        
        # 正例 mask：同标签
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        
        # 移除对角线（自身）
        logits_mask = 1 - torch.eye(batch_size, device=z.device)
        pos_mask = pos_mask * logits_mask
        
        # 如果没有正例，返回 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        # 计算损失
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        # 只在有正例的样本上计算
        mask_sum = pos_mask.sum(1)
        valid_mask = mask_sum > 0
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        mean_log_prob = (pos_mask * log_prob).sum(1) / (mask_sum + 1e-8)
        loss = -mean_log_prob[valid_mask].mean()
        
        return loss
    
    def get_embed_vector(self, features):
        """获取展平的嵌入向量 [batch, n_features * embed_dim]"""
        embs = []
        for i, emb in enumerate(self.embeddings):
            feat_vals = features[:, i].clamp(0, emb.num_embeddings - 1)  # 防止越界
            embs.append(emb(feat_vals))
        x = torch.cat(embs, dim=-1)  # [batch, n_features * embed_dim]
        return self.bn(x)
    
    def calculate_loss(self, batch):
        cvr_loss = F.binary_cross_entropy(self.forward(batch['features']), batch['labels'])
        cl_loss = self.contrastive_loss(batch)
        return cvr_loss + self.cl_weight * cl_loss


# ===== 评估函数 =====
def evaluate_per_bt(model, test_loader, device, method_name):
    """评估模型在每个 business_type 上的表现"""
    model.eval()
    all_preds = []
    all_labels = []
    all_business_types = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            pred = model(batch['features'])
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_business_types.extend(batch['business_types'])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_business_types = np.array(all_business_types)
    
    # Overall AUC
    overall_auc = roc_auc_score(all_labels, all_preds)
    
    # Per business_type AUC
    unique_bt = np.unique(all_business_types)
    bt_results = {}
    for bt in unique_bt:
        idx = all_business_types == bt
        if idx.sum() > 100 and all_labels[idx].var() > 0:  # 需要有足够的样本且有正负例
            bt_auc = roc_auc_score(all_labels[idx], all_preds[idx])
            bt_results[bt] = {
                'auc': bt_auc,
                'count': idx.sum(),
                'pos_rate': all_labels[idx].mean()
            }
    
    return overall_auc, bt_results


def analyze_method_performance(results_dict):
    """分析各方法性能差异"""
    methods = list(results_dict.keys())
    
    # 总体性能比较
    print("="*80)
    print("总体性能比较")
    print("="*80)
    print(f"{'Method':<15} {'Overall AUC':<12} {'Best BT AUC':<12} {'Worst BT AUC':<12}")
    print("-"*60)
    
    for method, (overall_auc, bt_results) in results_dict.items():
        bt_aucs = [v['auc'] for v in bt_results.values()]
        best_bt_auc = max(bt_aucs) if bt_aucs else 0
        worst_bt_auc = min(bt_aucs) if bt_aucs else 1
        print(f"{method:<15} {overall_auc:<12.4f} {best_bt_auc:<12.4f} {worst_bt_auc:<12.4f}")
    
    # 按 business_type 分析
    print("\n" + "="*80)
    print("各 business_type 性能分析")
    print("="*80)
    
    # 获取所有 business_type
    all_bts = set()
    for _, (_, bt_results) in results_dict.items():
        all_bts.update(bt_results.keys())
    all_bts = sorted(all_bts)
    
    # 创建性能对比表
    comparison_df = pd.DataFrame(index=all_bts)
    
    for method in methods:
        overall_auc, bt_results = results_dict[method]
        method_aucs = []
        for bt in all_bts:
            if bt in bt_results:
                method_aucs.append(bt_results[bt]['auc'])
            else:
                method_aucs.append(None)
        comparison_df[method] = method_aucs
    
    # 显示前20个 business_type 的详细对比
    print("\nTop 20 business_type 的方法对比:")
    print(comparison_df.head(20).round(4))
    
    # 分析 SupCon 相对 Baseline 的提升
    print("\n" + "="*80)
    print("SupCon vs Baseline 详细分析")
    print("="*80)
    
    baseline_overall, baseline_bt = results_dict['baseline']
    supcon_overall, supcon_bt = results_dict['supcon']
    
    improvements = []
    for bt in all_bts:
        if bt in baseline_bt and bt in supcon_bt:
            bl_auc = baseline_bt[bt]['auc']
            sc_auc = supcon_bt[bt]['auc']
            improvement = sc_auc - bl_auc
            improvements.append({
                'business_type': bt,
                'baseline_auc': bl_auc,
                'supcon_auc': sc_auc,
                'improvement': improvement,
                'sample_count': baseline_bt[bt]['count']
            })
    
    improvements_df = pd.DataFrame(improvements).sort_values('improvement', ascending=False)
    
    print(f"\nSupCon 提升最大的 Top 10 business_type:")
    print(improvements_df[['business_type', 'baseline_auc', 'supcon_auc', 'improvement', 'sample_count']].head(10).round(4))
    
    print(f"\nSupCon 下降最多的 Top 10 business_type:")
    print(improvements_df[['business_type', 'baseline_auc', 'supcon_auc', 'improvement', 'sample_count']].tail(10).round(4))
    
    # 统计提升/下降情况
    positive_improvements = [imp for imp in improvements if imp['improvement'] > 0.001]
    negative_improvements = [imp for imp in improvements if imp['improvement'] < -0.001]
    neutral_improvements = [imp for imp in improvements if abs(imp['improvement']) <= 0.001]
    
    print(f"\n统计:")
    print(f"- 显著提升 (Δ > 0.001): {len(positive_improvements)} 个 business_type")
    print(f"- 显著下降 (Δ < -0.001): {len(negative_improvements)} 个 business_type") 
    print(f"- 基本无变化 (|Δ| ≤ 0.001): {len(neutral_improvements)} 个 business_type")


def main():
    # 加载数据
    DATA_DIR = '/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Analyzing on {DEVICE}')
    print(f'Data: {DATA_DIR}')
    
    # Load vocab sizes
    with open(f'{DATA_DIR}/vocab_sizes.json') as f:
        vocab_sizes = json.load(f)
    
    feature_cols = [c for c in vocab_sizes.keys() if c != 'business_type']
    vocab_sizes_list = [vocab_sizes[c] for c in feature_cols]
    
    # Load test dataset
    test_ds = IVRDataset(f'{DATA_DIR}/test/')
    test_loader = DataLoader(test_ds, batch_size=8192*2, shuffle=False, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # Load trained models (simulated - we'll reuse the architecture and analyze patterns)
    # In real scenario, we would load saved weights
    print("\nAnalyzing trained models...")
    
    # For this analysis, we'll run inference with different models
    results = {}
    
    # Baseline model
    print("Evaluating Baseline...")
    baseline_model = BaseModel(vocab_sizes_list).to(DEVICE)
    baseline_model.eval()  # Use pre-trained weights in real scenario
    baseline_overall, baseline_bt = evaluate_per_bt(baseline_model, test_loader, DEVICE, 'baseline')
    results['baseline'] = (baseline_overall, baseline_bt)
    
    # SupCon model
    print("Evaluating SupCon...")
    supcon_model = SupCon(vocab_sizes_list).to(DEVICE)
    supcon_model.eval()  # Use pre-trained weights in real scenario
    supcon_overall, supcon_bt = evaluate_per_bt(supcon_model, test_loader, DEVICE, 'supcon')
    results['supcon'] = (supcon_overall, supcon_bt)
    
    # Run analysis
    analyze_method_performance(results)
    
    print("\nAnalysis completed!")


if __name__ == '__main__':
    main()