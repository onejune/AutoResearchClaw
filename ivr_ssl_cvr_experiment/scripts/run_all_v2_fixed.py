#!/usr/bin/env python3
"""
运行所有对比学习实验 v2 - 使用 data_v2 数据集，支持用户对比学习
修复版：直接运行而非 subprocess，避免 stdout 丢失
"""
import os
import sys
import json
from datetime import datetime

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
RESULTS_DIR = f'{PROJECT_DIR}/results'
DATA_DIR = f'{PROJECT_DIR}/data_v2'
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)

import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from models import build_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_dir: str = DATA_DIR
    model_name: str = 'baseline'
    embed_dim: int = 32
    hidden_dims: List[int] = None
    proj_dim: int = 64
    temperature: float = 0.1
    batch_size: int = 4096
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    contrastive_weight: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class IVRDatasetV2(Dataset):
    def __init__(self, df: pd.DataFrame, features: List[str]):
        self.features = features
        self.data = {}
        for feat in features:
            if feat in df.columns:
                self.data[feat] = torch.tensor(df[feat].values.astype(np.int64), dtype=torch.long)
        self.purchase = torch.tensor(df['purchase'].values.astype(np.float32), dtype=torch.float32)
        self.business_type_id = torch.tensor(df['business_type_id'].values.astype(np.int64), dtype=torch.long)
        if 'user_id' in df.columns:
            self.user_id = torch.tensor(df['user_id'].values.astype(np.int64), dtype=torch.long)
        else:
            self.user_id = None
    
    def __len__(self):
        return len(self.purchase)
    
    def __getitem__(self, idx):
        features = {feat: self.data[feat][idx] for feat in self.data}
        item = {
            'features': features,
            'purchase': self.purchase[idx],
            'business_type_id': self.business_type_id[idx]
        }
        if self.user_id is not None:
            item['user_id'] = self.user_id[idx]
        return item


def compute_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_pred)


def run_experiment(model_name, cl_weight, train_df, val_df, features, vocab_sizes):
    print(f"\n{'='*60}")
    print(f"Running {model_name} with cl_weight={cl_weight}")
    print(f"{'='*60}")
    
    config = Config()
    config.model_name = model_name
    config.contrastive_weight = cl_weight
    
    train_dataset = IVRDatasetV2(train_df, features)
    val_dataset = IVRDatasetV2(val_df, features)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    model = build_model(model_name, vocab_sizes, config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Train
    model.train()
    for batch in tqdm(train_loader, desc=f'Training {model_name}'):
        features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
        purchase = batch['purchase'].to(config.device)
        bt_id = batch['business_type_id'].to(config.device)
        user_id = batch.get('user_id')
        if user_id is not None:
            user_id = user_id.to(config.device)
        
        outputs = model(features_batch)
        bce_loss = F.binary_cross_entropy(outputs['purchase'], purchase)
        
        cl_loss = torch.tensor(0.0, device=config.device)
        if model_name == 'augment_contrastive' and 'z1' in outputs:
            cl_loss = model.contrastive_loss(outputs['z1'], outputs['z2'])
        elif model_name == 'user_contrastive' and 'z' in outputs and user_id is not None:
            cl_loss = model.contrastive_loss(outputs['z'], user_id)
        elif model_name == 'bt_contrastive' and 'z' in outputs:
            cl_loss = model.contrastive_loss(outputs['z'], bt_id)
        elif model_name == 'contrastive' and 'z' in outputs:
            cl_loss = model.contrastive_loss(outputs['z'], purchase.long())
        
        total_loss = bce_loss + cl_weight * cl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Evaluating {model_name}'):
            features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
            outputs = model(features_batch)
            all_preds.extend(outputs['purchase'].cpu().numpy().tolist())
            all_labels.extend(batch['purchase'].numpy().tolist())
    
    auc = compute_auc(np.array(all_labels), np.array(all_preds))
    print(f">>> {model_name} (cw={cl_weight}) AUC: {auc:.4f}")
    return {'model': model_name, 'cl_weight': cl_weight, 'auc': auc}


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*60)
    print("IVR SSL CVR 完整实验 v2（含用户对比学习）")
    print("="*60)
    
    # Load data once
    print("Loading data...")
    train_df = pd.read_pickle(os.path.join(DATA_DIR, 'train.pkl'))
    val_df = pd.read_pickle(os.path.join(DATA_DIR, 'val.pkl'))
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    features = meta['features']
    vocab_sizes = meta['vocab_sizes']
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Features: {len(features)}")
    
    experiments = [
        ('baseline', 0.1),
        ('bt_contrastive', 0.05),
        ('bt_contrastive', 0.1),
        ('bt_contrastive', 0.2),
        ('user_contrastive', 0.1),
        ('augment_contrastive', 0.1),
    ]
    
    ALL_RESULTS = []
    for i, (model, cw) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {model} cl_weight={cw}")
        r = run_experiment(model, cw, train_df, val_df, features, vocab_sizes)
        ALL_RESULTS.append(r)
    
    # Summary
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    print(f"{'Model':<25} {'cl_weight':<12} {'AUC':<10}")
    print("-"*47)
    for r in ALL_RESULTS:
        auc_str = f"{r['auc']:.4f}" if r['auc'] else 'N/A'
        print(f"{r['model']:<25} {r['cl_weight']:<12} {auc_str:<10}")
    
    # Save
    result_file = f'{RESULTS_DIR}/all_experiments_v2_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n结果已保存到: {result_file}")
    
    # Best
    valid_results = [r for r in ALL_RESULTS if r['auc'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['auc'])
        baseline_auc = next((r['auc'] for r in ALL_RESULTS if r['model'] == 'baseline'), None)
        print(f"\n🏆 最佳: {best['model']} (cl_weight={best['cl_weight']}) AUC={best['auc']:.4f}")
        if baseline_auc:
            gain = (best['auc'] - baseline_auc) * 1000
            print(f"   相比 Baseline 提升: {gain:.2f} 千分点")
