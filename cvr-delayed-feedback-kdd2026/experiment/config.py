"""Configuration for delayed feedback CVR estimation experiment."""

import math

class Config:
    """Hyperparameter configuration for the experiment."""
    
    def __init__(self):
        # Core hyperparameters
        self.lr = 0.001
        self.batch_size = 256
        self.epochs = 10
        self.cvr_hidden_dim = 128
        self.flow_hidden_dim = 64
        self.flow_num_layers = 2
        self.reservoir_size = 10000
        self.reservoir_strata = 5
        self.alpha_robust = 2.0  # Robust weighting parameter
        self.max_delay_hours = 168  # 7 days maximum delay
        
        # Dataset configuration
        self.dataset_path = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_dataset/data.txt"
        self.num_numeric_features = 8
        self.num_categorical_features = 9
        self.embedding_dim = 16
        
        # Training parameters
        self.flow_lr = 0.0001
        self.cvr_lr = 0.001
        self.grad_clip_norm = 1.0
        self.l2_reg = 0.0001
        
        # Random seed
        self.seed = 42
        
        # Derived parameters
        self.total_features = self.num_numeric_features + self.num_categorical_features
        self.feature_embedding_dim = self.num_numeric_features + self.num_categorical_features * self.embedding_dim
        self.shared_feature_dim = self.cvr_hidden_dim  # Dimension of shared feature representation
        
        # Delay modeling specific parameters
        self.delay_bins = 100  # Number of bins for discretized delay distribution
        self.delay_max_bin = self.max_delay_hours * 3600  # Maximum delay in seconds (7 days)

config = Config()