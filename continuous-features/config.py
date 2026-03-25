"""实验配置"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # 数据集选择："criteo_conv" 或 "criteo_std"
    dataset: str = "criteo_std"

    # 数据路径（空则使用各数据集默认路径）
    data_path: str = ""

    sample_size: int = 1_000_000
    batch_size: int = 4096
    embedding_dim: int = 16
    epochs: int = 3
    lr: float = 1e-3
    seed: int = 42
    encoder: str = "autodis"  # none/scalar/bucket/autodis/numeric/fttransformer/periodic
    output_dir: str = "results"

    # BucketEncoder 参数
    n_buckets: int = 10

    # AutoDisEncoder 参数
    n_meta_embeddings: int = 16

    # FTTransformerEncoder 参数
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2

    # PeriodicEncoder 参数
    n_frequencies: int = 16

    # Early stopping
    patience: int = 2

    # 类别特征词表大小
    cat_vocab_size: int = 10000

    # 由数据集决定，加载后自动填充（不要手动设置）
    n_continuous: int = 0
    n_categorical: int = 0

    # 数据集划分
    train_ratio: float = 0.8
