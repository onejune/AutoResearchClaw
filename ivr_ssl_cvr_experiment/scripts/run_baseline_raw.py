#!/usr/bin/env python3
"""
data_raw baseline：直接用原始数据，不做任何 hash
特征分为两类：
- 字符串特征（hash）：23 个，vocab 基于实际 unique
- 数值特征（float）：101 个，duf_*/huf_* 类，直接归一化后输入

模型结构：EmbeddingLayer（类别特征） + 数值特征 concat + MLP
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
DATA_DIR = f'{PROJECT_DIR}/data_raw'
RESULTS_DIR = f'{PROJECT_DIR}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_sizes, embed_dim):
        super().__init__()
        self.embs = nn.ModuleDict()
        for k, v in vocab_sizes.items():
            self.embs[k] = nn.Embedding(v, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x_dict):
        embs = []
        for k, emb in self.embs.items():
            embs.append(emb(x_dict[k].clamp(0, emb.num_embeddings - 1)))
        return torch.stack(embs, dim=1)  # [B, n_str_feat, embed_dim]


class BaselineCVR(nn.Module):
    """字符串特征 embedding + 数值特征 + MLP"""
    def __init__(self, str_features, str_vocab_sizes, num_features,
                 embed_dim=32, hidden_dims=[256, 128]):
        super().__init__()
        self.str_features = str_features
        self.num_features = num_features

        self.embedding = EmbeddingLayer(str_vocab_sizes, embed_dim)
        self.num_bn = nn.BatchNorm1d(len(num_features))

        total_dim = embed_dim * len(str_features) + len(num_features)
        layers = []
        dims = [total_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_dict):
        # 类别特征
        str_emb = self.embedding(x_dict)  # [B, n_str, embed_dim]
        str_flat = str_emb.view(str_emb.size(0), -1)  # [B, n_str*embed_dim]
        # 数值特征
        num_vals = torch.stack([x_dict[f] for f in self.num_features], dim=-1)  # [B, n_num]
        num_norm = self.num_bn(num_vals)
        # concat
        x = torch.cat([str_flat, num_norm], dim=-1)
        out = self.net(x)
        return torch.sigmoid(self.head(out)).squeeze(-1)


class RawDataset(Dataset):
    def __init__(self, df, str_features, num_features):
        self.str_data = {f: torch.LongTensor(df[f].values) for f in str_features}
        self.num_data = {f: torch.FloatTensor(df[f].fillna(0).values) for f in num_features}
        self.label = torch.FloatTensor(df['label'].values)
        self.bt_id = torch.LongTensor(df['business_type_id'].values)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        features = {}
        for f in self.str_data:
            features[f] = self.str_data[f][idx]
        for f in self.num_data:
            features[f] = self.num_data[f][idx]
        return features, self.label[idx], self.bt_id[idx]


def collate_fn(batch):
    features = {k: torch.stack([item[0][k] for item in batch]) for k in batch[0][0].keys()}
    labels = torch.stack([item[1] for item in batch])
    bt_ids = torch.stack([item[2] for item in batch])
    return features, labels, bt_ids


def evaluate(model, data_loader, device):
    model.eval()
    preds, labels_list, bt_ids_list = [], [], []
    with torch.no_grad():
        for features, labels, bt_ids in data_loader:
            features = {k: v.to(device) for k, v in features.items()}
            p = model(features).cpu().numpy()
            preds.extend(p.tolist())
            labels_list.extend(labels.numpy().tolist())
            bt_ids_list.extend(bt_ids.numpy().tolist())
    preds = np.array(preds)
    labels = np.array(labels_list)
    bt_ids = np.array(bt_ids_list)
    return roc_auc_score(labels, preds), preds, labels, bt_ids


def main():
    print("加载数据...")
    train_df = pq.read_table(f'{DATA_DIR}/train.parquet').to_pandas()
    test_df = pq.read_table(f'{DATA_DIR}/test.parquet').to_pandas()

    with open(f'{DATA_DIR}/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    str_features = meta['str_features']
    num_features = meta['num_features']
    str_vocab_sizes = meta['vocab_sizes']
    bt_encoder = meta['bt_encoder']

    print(f"\n训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    print(f"字符串特征: {len(str_features)}")
    print(f"数值特征: {len(num_features)}")
    print(f"训练集正率: {train_df['label'].mean():.4%}")
    print(f"Business Type: {list(bt_encoder.classes_)}")

    # DataLoader
    train_ds = RawDataset(train_df, str_features, num_features)
    test_ds = RawDataset(test_df, str_features, num_features)
    BS = 8192
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 模型
    model = BaselineCVR(str_features, str_vocab_sizes, num_features,
                        embed_dim=32, hidden_dims=[512, 256, 128]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练 1 epoch
    print("\n训练 1 epoch...")
    model.train()
    total_loss, n_batches = 0, 0
    for bi, (features, labels, bt_ids) in enumerate(train_dl):
        features = {k: v.to(DEVICE) for k, v in features.items()}
        labels = labels.to(DEVICE)
        pred = model(features)
        loss = F.binary_cross_entropy(pred, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n_batches += 1
        if bi % 200 == 0:
            print(f"  batch {bi}/{len(train_dl)}: loss={loss.item():.4f}")

    print(f"\n训练完成, avg_loss={total_loss/n_batches:.4f}, total_batches={n_batches}")

    # 评估
    print("\n评估...")
    overall_auc, preds, labels, bt_ids = evaluate(model, test_dl, DEVICE)
    print(f"\n{'='*60}")
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"{'='*60}")

    # per-BT AUC
    print("\nper-BT AUC:")
    bt_results = {}
    for bt_id in np.unique(bt_ids):
        mask = bt_ids == bt_id
        if labels[mask].sum() < 5:
            continue
        name = bt_encoder.inverse_transform([bt_id])[0]
        auc = roc_auc_score(labels[mask], preds[mask])
        n_pos = labels[mask].sum()
        n_total = mask.sum()
        bt_results[name] = {'auc': round(auc, 4), 'n': int(n_total), 'n_pos': int(n_pos),
                            'pos_rate': round(n_pos/n_total, 4)}
        print(f"  [{name}] AUC={auc:.4f}  n={n_total:,}  pos={n_pos:,}  rate={n_pos/n_total:.2%}")

    # 保存
    result = {
        'model': 'baseline',
        'overall_auc': round(overall_auc, 4),
        'per_bt': bt_results,
        'avg_train_loss': round(total_loss/n_batches, 4),
        'config': {'embed_dim': 32, 'hidden_dims': [512, 256, 128], 'lr': 1e-3, 'batch_size': BS}
    }
    with open(f'{RESULTS_DIR}/data_raw_baseline.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # MD 报告
    bt_order = list(bt_encoder.classes_)
    out_md = f"""# data_raw Baseline 结果

## Overall AUC: {overall_auc:.4f}

## per-BT AUC

| BT | AUC | 样本数 | 正样本 | 正率 |
|----|-----|--------|--------|------|
"""
    for name in bt_order:
        if name not in bt_results:
            continue
        r = bt_results[name]
        out_md += f"| {name} | {r['auc']:.4f} | {r['n']:,} | {r['n_pos']:,} | {r['pos_rate']:.2%} |\n"

    out_md += f"""
## 配置
- embed_dim: 32
- hidden_dims: [512, 256, 128]
- 字符串特征: {len(str_features)} 个 (embedding)
- 数值特征: {len(num_features)} 个 (BatchNorm后直接输入)
- lr: 1e-3, batch_size: {BS}, epochs: 1
- 参数量: {sum(p.numel() for p in model.parameters()):,}

## 数据
- 训练集: {len(train_df):,} 行, 正率 {train_df['label'].mean():.2%}
- 测试集: {len(test_df):,} 行, 正率 {test_df['label'].mean():.2%}
"""
    with open(f'{RESULTS_DIR}/data_raw_baseline.md', 'w') as f:
        f.write(out_md)

    print(f"\n结果已保存: {RESULTS_DIR}/data_raw_baseline.json")


if __name__ == '__main__':
    main()