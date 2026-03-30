#!/usr/bin/env python3
"""
MoCo (Momentum Contrast) 对比学习方法
使用动量编码器和队列机制进行对比学习
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


# ===== MoCo 编码器 =====
class EncoderQ(BaseModel):
    """查询编码器 Q"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dim=256, proj_dim=128):
        super().__init__(vocab_sizes, embed_dim, hidden_dim)
        
        # 额外的投影层用于对比学习
        self.proj_head = nn.Sequential(
            nn.Linear(self.n_features * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Xavier 初始化
        for layer in self.proj_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def encode_features(self, features):
        """编码特征为投影空间向量"""
        embeds = self.get_embed_vector(features)  # [B, D]
        proj = self.proj_head(embeds)  # [B, proj_dim]
        return F.normalize(proj, dim=1)  # [B, proj_dim]


class EncoderK(BaseModel):
    """键编码器 K (动量更新)"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dim=256, proj_dim=128):
        super().__init__(vocab_sizes, embed_dim, hidden_dim)
        
        # 额外的投影层用于对比学习
        self.proj_head = nn.Sequential(
            nn.Linear(self.n_features * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Xavier 初始化
        for layer in self.proj_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def encode_features(self, features):
        """编码特征为投影空间向量"""
        embeds = self.get_embed_vector(features)  # [B, D]
        proj = self.proj_head(embeds)  # [B, proj_dim]
        return F.normalize(proj, dim=1)  # [B, proj_dim]


# ===== MoCo 模型 =====
class MoCoSSL(nn.Module):
    """MoCo 对比学习模型
    核心思想：使用动量编码器和队列维护大量负样本
    """
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dim=256, 
                 proj_dim=128, queue_size=65536, momentum=0.999, temperature=0.2, cl_weight=0.1):
        super().__init__()
        
        # 查询编码器和键编码器
        self.encoder_q = EncoderQ(vocab_sizes, embed_dim, hidden_dim, proj_dim)
        self.encoder_k = EncoderK(vocab_sizes, embed_dim, hidden_dim, proj_dim)
        
        # 动量参数
        self.momentum = momentum
        
        # 温度参数
        self.temperature = temperature
        self.cl_weight = cl_weight
        
        # 初始化键编码器参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # 队列
        self.register_buffer("queue", torch.randn(queue_size, proj_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新键编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """入队出队操作"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # 如果队列不够大，截断keys
        if ptr + batch_size > self.queue_size:
            # 分段入队
            num1 = self.queue_size - ptr
            self.queue[ptr:] = keys[:num1]
            num2 = batch_size - num1
            self.queue[:num2] = keys[num1:]
            ptr = num2
        else:
            # 正常入队
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size  # move pointer
        
        self.queue_ptr[0] = ptr
    
    def contrastive_loss(self, batch):
        """MoCo 对比损失"""
        features = batch['features']
        
        # 生成正样本对（相同输入的不同增强）
        # 在这里，我们使用相同的输入两次，但通过编码器的不同状态产生差异
        q = self.encoder_q.encode_features(features)  # queries: [B, proj_dim]
        
        # 使用动量编码器生成键
        with torch.no_grad():  # no gradient to keys
            k = self.encoder_k.encode_features(features)  # keys: [B, proj_dim]
        
        # 计算相似度
        # positive pairs: [B, B]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B, 1]
        
        # negative pairs: [B, queue_size]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach().T])  # [B, queue_size]
        
        # logits: [B, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # apply temperature
        logits /= self.temperature
        
        # labels: positive key indicators [B]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        # compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def forward(self, batch):
        # CVR prediction
        pred = self.encoder_q(batch['features'])
        
        # 对比学习
        cl_loss = self.contrastive_loss(batch)
        
        # 标准 BCE loss
        cvr_loss = F.binary_cross_entropy(pred, batch['labels'])
        
        # 总损失
        total_loss = cvr_loss + self.cl_weight * cl_loss
        
        return total_loss, cvr_loss, cl_loss


# ===== 训练函数 =====
def train_model(model, train_loader, device, epochs=1, lr=1e-3):
    model.train()
    
    # 分别优化查询编码器和分类头
    params = list(model.encoder_q.parameters()) + list(model.encoder_q.mlp.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        total_loss = 0
        cvr_loss_total = 0
        cl_loss_total = 0
        n_batches = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            loss, cvr_loss, cl_loss = model(batch)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 动量更新键编码器
            model._momentum_update_key_encoder()
            
            # 入队出队操作
            with torch.no_grad():
                k = model.encoder_k.encode_features(batch['features'])
                model._dequeue_and_enqueue(k)
            
            total_loss += loss.item()
            cvr_loss_total += cvr_loss.item()
            cl_loss_total += cl_loss.item()
            n_batches += 1
            
            if n_batches % 50 == 0:
                print(f'  batch {n_batches}, total_loss={loss.item():.4f}, cvr_loss={cvr_loss.item():.4f}, cl_loss={cl_loss.item():.4f}')
    
    return model


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_business_types = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            pred = model.encoder_q(batch['features'])
            
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
    DATA_DIR = '/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 4096
    EPOCHS = 1
    
    print(f'Device: {DEVICE}')
    print('MoCo SSL 实验')
    print('=' * 60)
    
    # Load vocab sizes
    with open(f'{DATA_DIR}/vocab_sizes.json') as f:
        vocab_sizes = json.load(f)
    
    feature_cols = [c for c in vocab_sizes.keys() if c != 'business_type']
    vocab_sizes_list = [vocab_sizes[c] for c in feature_cols]
    
    print(f'Features: {len(vocab_sizes_list)} (all categorical)')
    
    # Load datasets
    train_ds = IVRDataset(f'{DATA_DIR}/train/')
    test_ds = IVRDataset(f'{DATA_DIR}/test/')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # 创建 MoCo 模型
    print(f'Creating MoCo SSL model...')
    model = MoCoSSL(vocab_sizes_list, 
                   proj_dim=128, 
                   queue_size=16384,  # 减小队列大小以适应内存
                   momentum=0.999, 
                   temperature=0.2, 
                   cl_weight=0.1).to(DEVICE)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # 训练
    print(f'Training...')
    t0 = time.time()
    model = train_model(model, train_loader, DEVICE, EPOCHS)
    train_time = time.time() - t0
    print(f'Training done in {train_time:.1f}s')
    
    # 评估
    print(f'Evaluating...')
    overall_auc, bt_aucs = evaluate(model, test_loader, DEVICE)
    
    print(f'Overall AUC: {overall_auc:.4f}')
    
    # Top 10 business_type by sample count
    test_bt_counts = {}
    for batch in test_loader:
        for bt in batch['business_types']:
            test_bt_counts[bt] = test_bt_counts.get(bt, 0) + 1
    
    top_bt = sorted(test_bt_counts.keys(), key=lambda x: test_bt_counts[x], reverse=True)[:10]
    
    print('\nTop 10 business_type AUC:')
    print(f'{"business_type":<15} {"samples":>10} {"AUC":>8}')
    print('-' * 40)
    for bt in top_bt:
        if bt in bt_aucs:
            print(f'{bt:<15} {test_bt_counts[bt]:>10,} {bt_aucs[bt]:>8.4f}')
        else:
            print(f'{bt:<15} {test_bt_counts[bt]:>10,} {"N/A":>8}')
    
    print(f'\nMoCo SSL Result: AUC = {overall_auc:.4f}')
    print(f'Training time: {train_time:.1f}s')
    print('Done!')


if __name__ == '__main__':
    main()