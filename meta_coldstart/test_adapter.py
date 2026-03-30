#!/usr/bin/env python3
"""
测试修复后的 Adapter 实现
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import PurchaseModel
from src.config import Config
from src.adapter import AdapterModel

# 创建测试模型
cfg = Config()
vocab_sizes = {'feat1': 100, 'feat2': 200}  # 示例词汇表大小
original_model = PurchaseModel(vocab_sizes, cfg.embedding_dim, cfg.mlp_dims, cfg.dropout)

print("Original model:")
print(f"Total params: {sum(p.numel() for p in original_model.parameters())}")
print(f"Trainable params: {sum(p.numel() for p in original_model.parameters() if p.requires_grad)}")

# 创建 Adapter 模型
adapter_model = AdapterModel(original_model, adapter_bottleneck=32)

print("\nAdapter model:")
print(f"Total params: {sum(p.numel() for p in adapter_model.parameters())}")
print(f"Trainable params: {sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)}")

# 测试前向传播
x = torch.randint(0, 10, (2, 2))  # 2个样本，2个特征
try:
    output = adapter_model(x)
    print(f"\nForward pass successful! Output shape: {output.shape}")
    print("Adapter implementation is working correctly.")
except Exception as e:
    print(f"\nError in forward pass: {e}")