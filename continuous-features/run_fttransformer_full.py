"""临时脚本：全量 Criteo 标准版，只跑 FTTransformerEncoder"""
import os
import sys
import time
import random
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data import get_dataloaders
from feature_encoders import build_encoder
from models import DeepFM
from trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

config = Config()
config.sample_size = 0  # 全量（0 表示不采样）
config.dataset = "criteo_std"
config.encoder = "fttransformer"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}", flush=True)
print(f"数据集: {config.dataset}, sample_size={config.sample_size} (全量)", flush=True)

print("加载数据...", flush=True)
train_loader, test_loader, dataset = get_dataloaders(config)
print(f"连续特征: {config.n_continuous}, 类别特征: {config.n_categorical}", flush=True)
print(f"训练集: {len(train_loader.dataset):,}, 测试集: {len(test_loader.dataset):,}", flush=True)

set_seed(config.seed)

encoder = build_encoder(config)
cat_vocab_sizes = [config.cat_vocab_size] * config.n_categorical
model = DeepFM(
    feature_encoder=encoder,
    cat_vocab_sizes=cat_vocab_sizes,
    embedding_dim=config.embedding_dim,
    n_continuous=config.n_continuous,
    n_categorical=config.n_categorical,
)

n_params = count_parameters(model)
print(f"参数量: {n_params:,} ({n_params/1000:.1f} K)", flush=True)

trainer = Trainer(model, config, device)
best_auc, total_time = trainer.fit(train_loader, test_loader)

result_line = f"FTTransformer AUC={best_auc:.4f}  params={n_params/1000:.0f}K  time={total_time:.0f}s"
print(f"\n{result_line}", flush=True)

# 写入结果文件以防万一
with open("/tmp/fttransformer_result.txt", "w") as f:
    f.write(result_line + "\n")
print("结果已写入 /tmp/fttransformer_result.txt", flush=True)
