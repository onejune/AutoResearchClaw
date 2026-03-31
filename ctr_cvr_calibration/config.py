"""
CTR/CVR 预估校准研究 - 配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据配置
class DataConfig:
    # 数据路径
    data_dir = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/'
    train_file = 'train/'
    test_file = 'test/'
    vocab_file = 'vocab_sizes.json'
    meta_file = 'meta.json'
    encoder_file = 'encoders.pkl'
    
    # 特征配置
    num_features = 126
    label_cols = ['click_label', 'ctcvr_label']
    
    # 数据划分
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

# 模型配置
class ModelConfig:
    # Embedding
    embed_dim = 16
    max_vocab_size = 100000
    
    # MLP
    hidden_dims = [256, 128, 64]
    dropout = 0.2
    batch_norm = True

# 训练配置
class TrainConfig:
    # 基础配置
    epochs = 3  # 减少到 3 个 epoch 进行快速测试
    batch_size = 1024
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    # 优化器
    optimizer = 'adam'
    
    # 早停
    early_stopping = True
    patience = 3
    
    # 设备
    num_workers = 4
    
    # 日志
    log_interval = 200  # 减少日志频率

# 校准配置
class CalibrationConfig:
    # Temperature Scaling
    temperature = 1.0
    
    # Histogram Binning
    num_bins = 20
    
    # Focal Loss
    focal_gamma = 2.0
    focal_alpha = 0.25
    
    # Label Smoothing
    label_smoothing = 0.1
    
    # 评估
    ece_bins = 20

# 实验配置
class ExperimentConfig:
    # 随机种子
    seed = 42
    
    # 结果保存
    results_dir = PROJECT_ROOT / 'results'
    logs_dir = PROJECT_ROOT / 'logs'

# 指标配置
class MetricConfig:
    # 评估指标
    metrics = ['auc', 'logloss', 'ece', 'mce', 'pcoc']

# 路径工具函数
def get_data_path(file_type='train'):
    """获取数据路径"""
    if file_type == 'train':
        return os.path.join(DataConfig.data_dir, DataConfig.train_file)
    elif file_type == 'test':
        return os.path.join(DataConfig.data_dir, DataConfig.test_file)
    elif file_type == 'vocab':
        return os.path.join(DataConfig.data_dir, DataConfig.vocab_file)
    elif file_type == 'meta':
        return os.path.join(DataConfig.data_dir, DataConfig.meta_file)
    else:
        raise ValueError(f"Unknown file type: {file_type}")
