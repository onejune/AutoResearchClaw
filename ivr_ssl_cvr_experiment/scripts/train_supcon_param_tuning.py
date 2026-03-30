#!/usr/bin/env python3
"""
SupCon 参数调优实验
测试不同温度系数和对比学习权重的影响
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
import argparse


# ===== 数据集类 =====
class IVRDataset(Dataset):
    def __init__(self, data_path, max_samples=None):
        print(f'Loading {data_path}...')
        t0 = time.time()
        
        df = pq.read_table(data_path).to_pandas()
        if max_samples:
            df = df.head(max_samples)
        
        # 特征和标签
        self.labels = torch.FloatTensor(df['ctcvr_label'].values.astype(np.float32))
        self.business_types = df['business_type'].values  # 保持为 numpy 用于分组统计
        
        # 所有特征（126个，全部是类别特征）
        feature_cols = [c for c in df.columns if c not in ['click_label', 'ctcvr_label', 'business_type']]
        self.features = torch.LongTensor(df[feature_cols].values.astype(np.int64))
        
        print(f'Loaded {len(df):,} samples in {time.time()-t0:.1f}s')
        print(f'CVR rate: {self.labels.mean():.4f}')
    
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


# ===== 基础模型定义 =====
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
    
    def get_embed_vector(self, features):
        """获取展平的嵌入向量 [batch, n_features * embed_dim]"""
        embs = []
        for i, emb in enumerate(self.embeddings):
            feat_vals = features[:, i].clamp(0, emb.num_embeddings - 1)  # 防止越界
            embs.append(emb(feat_vals))
        x = torch.cat(embs, dim=-1)  # [batch, n_features * embed_dim]
        return self.bn(x)
    
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
    
    def calculate_loss(self, batch):
        """标准 CVR 损失"""
        pred = self.forward(batch['features'])
        return F.binary_cross_entropy(pred, batch['labels'])


# ===== 投影头 =====
class ProjectionHead(nn.Module):
    """对比学习投影头"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Xavier 初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ===== SupCon: Supervised Contrastive Learning =====
class SupCon(BaseModel):
    """Supervised Contrastive Learning
    核心思想：同标签样本为正例，不同标签为负例
    """
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dim=256,
                 proj_dim=64, temperature=0.1, cl_weight=0.1):
        super().__init__(vocab_sizes, embed_dim, hidden_dim)
        
        input_dim = self.n_features * embed_dim
        self.projection = ProjectionHead(input_dim, hidden_dim, proj_dim)
        self.temperature = temperature
        self.cl_weight = cl_weight
    
    def contrastive_loss(self, batch) -> torch.Tensor:
        """监督对比损失"""
        embeds = self.get_embed_vector(batch['features'])
        z = self.projection(embeds)
        
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
    
    def calculate_loss(self, batch):
        cvr_loss = super().calculate_loss(batch)
        cl_loss = self.contrastive_loss(batch)
        return cvr_loss + self.cl_weight * cl_loss


# ===== 训练函数 =====
def train_model(model, train_loader, device, epochs=1, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            loss = model.calculate_loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if n_batches % 50 == 0:
                print(f'  batch {n_batches}, loss={loss.item():.4f}')
    
    return model


def evaluate(model, test_loader, device):
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
    bt_aucs = {}
    for bt in unique_bt:
        idx = all_business_types == bt
        if idx.sum() > 100 and all_labels[idx].var() > 0:  # 需要有足够的样本且有正负例
            bt_aucs[bt] = roc_auc_score(all_labels[idx], all_preds[idx])
    
    return overall_auc, bt_aucs


def main():
    parser = argparse.ArgumentParser(description='SupCon 参数调优实验')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数（用于调试）')
    
    args = parser.parse_args()
    
    DATA_DIR = '/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample'
    DEVICE = args.device
    
    print(f'Device: {DEVICE}')
    print('SupCon 参数调优实验')
    print('=' * 60)
    
    # Load vocab sizes
    with open(f'{DATA_DIR}/vocab_sizes.json') as f:
        vocab_sizes = json.load(f)
    
    feature_cols = [c for c in vocab_sizes.keys() if c != 'business_type']
    vocab_sizes_list = [vocab_sizes[c] for c in feature_cols]
    
    print(f'Features: {len(vocab_sizes_list)} (all categorical)')
    
    # Load datasets
    train_ds = IVRDataset(f'{DATA_DIR}/train/', max_samples=args.max_samples)
    test_ds = IVRDataset(f'{DATA_DIR}/test/', max_samples=args.max_samples // 5 if args.max_samples else None)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size*2, shuffle=False, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # 参数网格
    temperatures = [0.05, 0.1, 0.2, 0.5]
    cl_weights = [0.01, 0.05, 0.1, 0.2]
    
    results = {}
    
    print(f'温度系数: {temperatures}')
    print(f'对比学习权重: {cl_weights}')
    print()
    
    # 测试所有参数组合
    for temp in temperatures:
        for cl_weight in cl_weights:
            print(f'{"="*60}')
            print(f'Testing Temp={temp}, CL_Weight={cl_weight}')
            print(f'{"="*60}')
            
            # 创建模型
            model = SupCon(vocab_sizes_list, temperature=temp, cl_weight=cl_weight).to(DEVICE)
            print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
            
            # 训练
            t0 = time.time()
            model = train_model(model, train_loader, DEVICE, args.epochs, args.lr)
            train_time = time.time() - t0
            print(f'Training done in {train_time:.1f}s')
            
            # 评估
            print('Evaluating...')
            overall_auc, bt_aucs = evaluate(model, test_loader, DEVICE)
            
            print(f'Overall AUC: {overall_auc:.4f}')
            
            # Top 5 business_type by sample count
            test_bt_counts = {}
            for batch in test_loader:
                for bt in batch['business_types']:
                    test_bt_counts[bt] = test_bt_counts.get(bt, 0) + 1
            
            top_bt = sorted(test_bt_counts.keys(), key=lambda x: test_bt_counts[x], reverse=True)[:5]
            
            print('\nTop 5 business_type AUC:')
            print(f'{"business_type":<15} {"samples":>10} {"AUC":>8}')
            print('-' * 40)
            for bt in top_bt:
                if bt in bt_aucs:
                    print(f'{bt:<15} {test_bt_counts[bt]:>10,} {bt_aucs[bt]:>8.4f}')
                else:
                    print(f'{bt:<15} {test_bt_counts[bt]:>10,} {"N/A":>8}')
            
            # 保存结果
            param_key = f'temp_{temp}_cl_{cl_weight}'
            results[param_key] = {
                'overall_auc': overall_auc,
                'bt_aucs': bt_aucs,
                'train_time': train_time,
                'params': sum(p.numel() for p in model.parameters()),
                'temperature': temp,
                'cl_weight': cl_weight
            }
    
    # 打印最终汇总
    print(f'\n{"="*80}')
    print('PARAMETER TUNING RESULTS SUMMARY')
    print(f'{"="*80}')
    print(f'{"Config":<25} {"Overall AUC":<12} {"Temp":<8} {"CL Weight":<10} {"Train Time":<12}')
    print('-' * 80)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['overall_auc'], reverse=True)
    for config, res in sorted_results:
        print(f'{config:<25} {res["overall_auc"]:<12.4f} {res["temperature"]:<8.2f} {res["cl_weight"]:<10.2f} {res["train_time"]:<12.1f}s')
    
    best_config = sorted_results[0][0]
    best_result = sorted_results[0][1]
    print(f'\nBest Configuration: {best_config}')
    print(f'Best AUC: {best_result["overall_auc"]:.4f}')
    print(f'Temperature: {best_result["temperature"]}, CL Weight: {best_result["cl_weight"]}')
    
    print('\nDone!')


if __name__ == '__main__':
    main()