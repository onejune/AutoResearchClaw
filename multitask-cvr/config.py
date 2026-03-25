"""
config.py - 实验配置
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    # ── 数据 ──────────────────────────────────────────────
    dataset: str = "synthetic"      # 数据集名称：synthetic / ali_ccp / taobao
    data_dir: str = ""              # 数据目录（ali_ccp 必填）
    data_path: str = ""             # 单文件路径（taobao 必填）
    sample_size: Optional[int] = None      # 采样数，None=全量（分层采样时忽略）
    batch_size: int = 4096

    # ── Ali-CCP 分层采样 ───────────────────────────────────
    ali_ccp_neg_sample_rate: float = 0.05    # click=0 样本采样率
    ali_ccp_click_sample_rate: float = 0.5   # click=1, buy=0 样本采样率

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

    # ── Taobao 时序切分 ────────────────────────────────────
    taobao_temporal: bool = False   # True=时序切分（无穿越），False=随机切分（默认）

    # ── 输出 ──────────────────────────────────────────────
    output_dir: str = "results"
