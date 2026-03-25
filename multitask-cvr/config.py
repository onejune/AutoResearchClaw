"""
config.py - 实验配置
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    # ── 数据 ──────────────────────────────────────────────
    data_dir: str = ""              # Ali-CCP 数据目录，空字符串 = 使用合成数据
    sample_size: Optional[int] = 500_000   # 采样数，None=全量
    batch_size: int = 4096

    # ── 模型 ──────────────────────────────────────────────
    model_name: str = "esmm"        # shared_bottom / esmm / mmoe / escm2
    embedding_dim: int = 16
    mlp_dims: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    n_experts: int = 4              # MMoE 专家数
    escm2_lambda: float = 0.1      # ESCM2 反事实正则化强度

    # ── 训练 ──────────────────────────────────────────────
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42
    early_stopping_patience: int = 3

    # ── 输出 ──────────────────────────────────────────────
    output_dir: str = "results"
