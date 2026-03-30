#!/usr/bin/env python3
"""
对比学习 CVR 训练脚本 v2 - 支持用户对比学习
"""

import os
import pickle
import logging
import argparse
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models import build_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # 数据
    data_dir: str = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr/data_v2/'
    
    # 模型
    model_name: str = 'baseline'  # baseline / contrastive / bt_contrastive / user_contrastive / augment_contrastive
    embed_dim: int = 32
    hidden_dims: List[int] = None
    proj_dim: int = 64
    temperature: float = 0.1
    
    # 训练
    batch_size: int = 4096
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    contrastive_weight: float = 0.1
    
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class IVRDatasetV2(Dataset):
    """IVR 数据集 v2 - 支持 user_id"""
    def __init__(self, df: pd.DataFrame, features: List[str]):
        self.features = features
        self.data = {}
        
        for feat in features:
            if feat in df.columns:
                self.data[feat] = torch.tensor(df[feat].values.astype(np.int64), dtype=torch.long)
        
        self.purchase = torch.tensor(df['purchase'].values.astype(np.float32), dtype=torch.float32)
        self.business_type_id = torch.tensor(df['business_type_id'].values.astype(np.int64), dtype=torch.long)
        
        # user_id 用于用户对比学习
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


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_pred)


def compute_pcoc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.mean() == 0:
        return 0.0
    return y_pred.mean() / y_true.mean()


class TrainerV2:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_losses = defaultdict(float)
        num_batches = 0
        
        for batch in tqdm(train_loader, desc='Training', leave=False):
            features = {k: v.to(self.device) for k, v in batch['features'].items()}
            purchase = batch['purchase'].to(self.device)
            bt_id = batch['business_type_id'].to(self.device)
            user_id = batch.get('user_id')
            if user_id is not None:
                user_id = user_id.to(self.device)
            
            outputs = self.model(features)
            
            # BCE loss
            bce_loss = F.binary_cross_entropy(outputs['purchase'], purchase)
            
            # 对比损失
            cl_loss = torch.tensor(0.0, device=self.device)
            
            if self.config.model_name == 'augment_contrastive' and 'z1' in outputs:
                cl_loss = self.model.contrastive_loss(outputs['z1'], outputs['z2'])
            elif self.config.model_name == 'user_contrastive' and 'z' in outputs and user_id is not None:
                cl_loss = self.model.contrastive_loss(outputs['z'], user_id)
            elif self.config.model_name == 'bt_contrastive' and 'z' in outputs:
                cl_loss = self.model.contrastive_loss(outputs['z'], bt_id)
            elif self.config.model_name == 'contrastive' and 'z' in outputs:
                cl_loss = self.model.contrastive_loss(outputs['z'], purchase.long())
            
            total_loss = bce_loss + self.config.contrastive_weight * cl_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_losses['bce'] += bce_loss.item()
            total_losses['cl'] += cl_loss.item()
            total_losses['total'] += total_loss.item()
            num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_bt_ids = []
        
        for batch in tqdm(val_loader, desc='Evaluating', leave=False):
            features = {k: v.to(self.device) for k, v in batch['features'].items()}
            
            outputs = self.model(features)
            
            all_preds.extend(outputs['purchase'].cpu().numpy().tolist())
            all_labels.extend(batch['purchase'].numpy().tolist())
            all_bt_ids.extend(batch['business_type_id'].numpy().tolist())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_bt_ids = np.array(all_bt_ids)
        
        metrics = {
            'auc': compute_auc(all_labels, all_preds),
            'pcoc': compute_pcoc(all_labels, all_preds)
        }
        
        # 分 business_type 评估
        unique_bts = np.unique(all_bt_ids)
        bt_metrics = {}
        for bt in unique_bts:
            mask = all_bt_ids == bt
            if mask.sum() >= 100:
                bt_metrics[int(bt)] = {
                    'auc': compute_auc(all_labels[mask], all_preds[mask]),
                    'pcoc': compute_pcoc(all_labels[mask], all_preds[mask]),
                    'count': int(mask.sum())
                }
        metrics['by_bt'] = bt_metrics
        
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        best_auc = 0
        best_metrics = None
        
        for epoch in range(self.config.epochs):
            logger.info(f'Epoch {epoch+1}/{self.config.epochs}')
            
            train_losses = self.train_epoch(train_loader)
            logger.info(f'  Train Loss: {train_losses["total"]:.4f} (BCE: {train_losses["bce"]:.4f}, CL: {train_losses["cl"]:.4f})')
            
            metrics = self.evaluate(val_loader)
            logger.info(f'  Val AUC: {metrics["auc"]:.4f}, PCOC: {metrics["pcoc"]:.2f}')
            
            for bt, bt_m in metrics['by_bt'].items():
                logger.info(f'    BT {bt}: AUC={bt_m["auc"]:.4f}, PCOC={bt_m["pcoc"]:.2f}, n={bt_m["count"]}')
            
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_metrics = metrics
        
        return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', 
                        choices=['baseline', 'contrastive', 'bt_contrastive', 'user_contrastive', 'augment_contrastive'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--cl_weight', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr/data_v2/')
    args = parser.parse_args()
    
    config = Config()
    config.model_name = args.model
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.contrastive_weight = args.cl_weight
    config.data_dir = args.data_dir
    
    # 加载数据
    logger.info('Loading data...')
    train_df = pd.read_pickle(os.path.join(config.data_dir, 'train.pkl'))
    val_df = pd.read_pickle(os.path.join(config.data_dir, 'val.pkl'))
    
    with open(os.path.join(config.data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    features = meta['features']
    vocab_sizes = meta['vocab_sizes']
    
    logger.info(f'Train: {len(train_df):,}, Val: {len(val_df):,}')
    logger.info(f'Features: {len(features)}, Vocab sizes: {len(vocab_sizes)}')
    logger.info(f'Has user_id: {"user_id" in train_df.columns}')
    
    # 创建数据集
    train_dataset = IVRDatasetV2(train_df, features)
    val_dataset = IVRDatasetV2(val_df, features)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    # 构建模型
    logger.info(f'Building model: {config.model_name}')
    model = build_model(config.model_name, vocab_sizes, config)
    
    # 训练
    logger.info(f'Training with cl_weight={config.contrastive_weight}')
    trainer = TrainerV2(model, config)
    metrics = trainer.fit(train_loader, val_loader)
    
    # 输出结果
    logger.info('='*50)
    logger.info(f'Model: {config.model_name}')
    logger.info(f'Final AUC: {metrics["auc"]:.4f}')
    logger.info(f'Final PCOC: {metrics["pcoc"]:.2f}')
    logger.info('='*50)
    
    return metrics


if __name__ == '__main__':
    main()
