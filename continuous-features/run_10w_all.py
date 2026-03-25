"""
10万样本，12种编码器全量对比
"""
import sys, os
sys.path.insert(0, '/mnt/workspace/open_research/autoresearch/exp_continuous_features')
os.chdir('/mnt/workspace/open_research/autoresearch/exp_continuous_features')

import numpy as np
import torch
from config import Config
from data import get_dataloaders
from feature_encoders import build_encoder, BucketEncoder, MinMaxEncoder, StandardScalerEncoder
from models import DeepFM
from trainer import Trainer
import random, time

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run(enc_name, config, train_loader, test_loader, device):
    set_seed(config.seed)
    config.encoder = enc_name
    enc = build_encoder(config)
    if isinstance(enc, (BucketEncoder, MinMaxEncoder, StandardScalerEncoder)):
        ds = train_loader.dataset
        n = min(50_000, len(ds))
        arr = np.array([ds[i][0].numpy() for i in range(n)])
        enc.fit(arr)
    model = DeepFM(
        feature_encoder=enc,
        cat_vocab_sizes=[config.cat_vocab_size]*config.n_categorical,
        embedding_dim=config.embedding_dim,
        n_continuous=config.n_continuous,
        n_categorical=config.n_categorical,
    )
    trainer = Trainer(model, config, device)
    auc, elapsed = trainer.fit(train_loader, test_loader)
    params_k = count_params(model) / 1000
    print(f"RESULT {enc_name:16s} AUC={auc:.4f}  params={params_k:.1f}K  time={elapsed:.0f}s", flush=True)
    return auc, params_k, elapsed

config = Config()
config.sample_size = 1_000_000
config.dataset = 'criteo_std'
config.epochs = 1
config.patience = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}", flush=True)

train_loader, test_loader, _ = get_dataloaders(config)
print(f"n_continuous={config.n_continuous}, n_categorical={config.n_categorical}", flush=True)
print(f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}", flush=True)

encoders = [
    'none', 'scalar', 'bucket', 'autodis', 'numeric',
    'fttransformer', 'periodic',
    'field', 'dlrm', 'minmax', 'standard', 'log',
]

results = {}
for enc_name in encoders:
    print(f"\n>>> 开始 {enc_name} ...", flush=True)
    try:
        auc, params_k, elapsed = run(enc_name, config, train_loader, test_loader, device)
        results[enc_name] = (auc, params_k, elapsed)
    except Exception as e:
        print(f"ERROR {enc_name}: {e}", flush=True)
        results[enc_name] = (float('nan'), 0, 0)

print("\n" + "="*65)
print(f"{'方法':<22} {'AUC':>8} {'参数量':>10} {'时间':>8}")
print("-"*65)
for enc_name in encoders:
    auc, params_k, elapsed = results[enc_name]
    print(f"{enc_name:<22} {auc:>8.4f} {params_k:>8.1f}K {elapsed:>6.0f}s")
print("="*65)
print("ALL DONE")
