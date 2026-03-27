"""
ChorusCVR 配置文件
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

# 数据路径
DATA_ROOT = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp"

@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = DATA_ROOT
    train_file: str = "ali_ccp_train.csv"
    test_file: str = "ali_ccp_test.csv"
    
    # 特征配置 (Ali-CCP 数据集) - 根据实际数据文件
    sparse_features: List[str] = field(default_factory=lambda: [
        '101', '121', '122', '124', '125', '126', '127', '128', '129',
        '205', '206', '207', '210', '216', '508', '509', '702', '853',
        '301'  # 类目特征
    ])
    dense_features: List[str] = field(default_factory=lambda: [
        '109_14', '110_14', '127_14', '150_14',  # 整数型统计特征
        'D109_14', 'D110_14', 'D127_14', 'D150_14',  # 归一化特征
        'D508', 'D509', 'D702', 'D853'  # 其他归一化特征
    ])
    
    # 标签
    click_label: str = "click"
    conversion_label: str = "purchase"
    
    # 数据处理
    max_samples: Optional[int] = None  # None 表示全量
    test_ratio: float = 0.2
    val_ratio: float = 0.1


@dataclass
class ModelConfig:
    """模型配置"""
    # Embedding
    embedding_dim: int = 16
    
    # 共享底层网络
    shared_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # Tower 网络
    tower_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    
    # Dropout
    dropout_rate: float = 0.1
    
    # 激活函数
    activation: str = "relu"


@dataclass
class TrainConfig:
    """训练配置"""
    # 基础参数
    batch_size: int = 4096
    epochs: int = 1  # CTR/CVR 只跑 1 epoch
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # 损失权重
    loss_weights: dict = field(default_factory=lambda: {
        'ctcvr': 1.0,
        'cvr_ipw': 1.0,
        'ctuncvr': 1.0,
        'uncvr_ipw': 1.0,
        'align_ipw': 1.0,
    })
    
    # IPW 裁剪 (防止除以过小的值)
    ipw_clip_min: float = 0.01
    ipw_clip_max: float = 1.0
    
    # 早停
    early_stop_patience: int = 3
    
    # 设备
    device: str = "cuda"
    
    # 日志
    log_interval: int = 100
    
    # 随机种子
    seed: int = 42


@dataclass
class ExperimentConfig:
    """实验总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # 实验名称
    exp_name: str = "chorus_cvr"
    output_dir: str = "./results"


# 默认配置
DEFAULT_CONFIG = ExperimentConfig()
