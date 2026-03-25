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
        n = min(100_000, len(ds))
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
    print(f"{enc_name:16s} AUC={auc:.4f}  params={count_params(model)//1000}K  time={elapsed:.0f}s")
    return auc

config = Config()
config.sample_size = 1_000_000
config.dataset = 'criteo_std'
config.epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")

train_loader, test_loader, _ = get_dataloaders(config)
print(f"n_continuous={config.n_continuous}, n_categorical={config.n_categorical}")

for enc in ['numeric', 'periodic', 'fttransformer']:
    run(enc, config, train_loader, test_loader, device)

print("GROUP3 DONE")
