#!/usr/bin/env python3
"""
data_raw Baseline（全部特征当类别特征）
- 124 个特征全部用 Embedding
- vocab 基于实际 unique，max 限制到 50000
- embed_dim=32, hidden=[512,256,128], 1 epoch
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
        return torch.stack(embs, dim=1)  # [B, n_feat, embed_dim]


class BaselineCVR(nn.Module):
    """全部特征用 embedding"""
    def __init__(self, vocab_sizes, embed_dim=32, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        total_dim = embed_dim * len(vocab_sizes)
        layers = []
        dims = [total_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_dict):
        embed = self.embedding(x_dict)  # [B, n_feat, embed_dim]
        x = embed.view(embed.size(0), -1)  # [B, n_feat * embed_dim]
        out = self.net(x)
        return torch.sigmoid(self.head(out)).squeeze(-1)


class RawDataset(Dataset):
    def __init__(self, df, feat_names):
        self.feat_data = {f: torch.LongTensor(df[f].values.astype(np.int64)) for f in feat_names}
        self.label = torch.FloatTensor(df['label'].values.astype(np.float32))
        self.bt_id = torch.LongTensor(df['business_type_id'].values.astype(np.int64))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {f: self.feat_data[f][idx] for f in self.feat_data}, self.label[idx], self.bt_id[idx]


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

    feat_names = meta['features']
    bt_encoder = meta['bt_encoder']

    # vocab：基于 meta.vocab_sizes，但全部用（之前只用了 str_features）
    vocab_sizes = meta['vocab_sizes']

    print(f"\n训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
    print(f"特征数: {len(feat_names)}")
    print(f"训练集正率: {train_df['label'].mean():.4%}")
    print(f"Business Type: {list(bt_encoder.classes_)}")

    # DataLoader
    train_ds = RawDataset(train_df, feat_names)
    test_ds = RawDataset(test_df, feat_names)
    BS = 8192
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 模型
    model = BaselineCVR(vocab_sizes, embed_dim=32, hidden_dims=[512, 256, 128]).to(DEVICE)
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
    for bt_id in sorted(np.unique(bt_ids)):
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
        'model': 'baseline_all_cat',
        'overall_auc': round(overall_auc, 4),
        'per_bt': bt_results,
        'avg_train_loss': round(total_loss/n_batches, 4),
        'config': {'embed_dim': 32, 'hidden_dims': [512, 256, 128],
                   'n_features': len(feat_names), 'lr': 1e-3, 'batch_size': BS}
    }
    with open(f'{RESULTS_DIR}/data_raw_baseline_all_cat.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    out_md = f"""# data_raw Baseline（全类别）结果

## Overall AUC: {overall_auc:.4f}

## per-BT AUC

| BT | AUC | 样本数 | 正样本 | 正率 |
|----|-----|--------|--------|------|
"""
    for name, r in sorted(bt_results.items(), key=lambda x: -x[1]['auc']):
        out_md += f"| {name} | {r['auc']:.4f} | {r['n']:,} | {r['n_pos']:,} | {r['pos_rate']:.2%} |\n"

    out_md += f"""
## 配置
- embed_dim: 32
- hidden_dims: [512, 256, 128]
- 全部特征当类别: {len(feat_names)} 个
- lr: 1e-3, batch_size: {BS}, epochs: 1
- 参数量: {sum(p.numel() for p in model.parameters()):,}
"""
    with open(f'{RESULTS_DIR}/data_raw_baseline_all_cat.md', 'w') as f:
        f.write(out_md)

    print(f"\n结果已保存: {RESULTS_DIR}/data_raw_baseline_all_cat.json")


if __name__ == '__main__':
    main()