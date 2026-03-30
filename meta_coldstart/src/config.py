"""实验配置"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── 数据 ──────────────────────────────────────────────
    train_path: str = "../exp_multitask/data/train.pkl"
    val_path:   str = "../exp_multitask/data/val.pkl"
    meta_path:  str = "../exp_multitask/data/meta.pkl"

    # Task 构建
    min_pos_samples: int = 5     # campaignset 最少正样本数
    min_samples:     int = 50    # campaignset 最少总样本数
    meta_train_ratio: float = 0.8
    seed: int = 42

    # K-shot
    k_shot: int = 100            # support set 大小

    # ── 模型 ──────────────────────────────────────────────
    embedding_dim: int = 16
    mlp_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1

    # ── 训练（全局/预训练阶段）────────────────────────────
    batch_size:    int = 2048
    epochs:        int = 1
    lr:            float = 1e-3
    weight_decay:  float = 1e-5

    # ── Meta-Learning ─────────────────────────────────────
    meta_lr:       float = 1e-4   # outer loop lr (降低以稳定训练)
    inner_lr:      float = 1e-2   # inner loop lr
    inner_steps:   int = 3        # inner loop 梯度步数 (减少以避免过适配)
    meta_batch_size: int = 4      # 每次 meta-update 采样的 task 数 (减少内存压力)
    meta_epochs:   int = 30

    # ── 输出 ──────────────────────────────────────────────
    output_dir: str = "results/"
    device: str = "cuda"
