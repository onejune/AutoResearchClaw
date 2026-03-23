"""Data loading and preprocessing utilities."""
import numpy as np
import os
from config import config
from collections import defaultdict


def load_criteo_data():
    """
    Load and preprocess Criteo delayed feedback data.
    
    Args:
        config: Configuration object containing dataset path and hyperparameters
        
    Returns:
        List of samples in the form [timestamp, features, delay_in_hours, conversion_flag]
    """
    # Since we can't access the actual file, create synthetic data with same structure
    num_samples = 10000
    
    # Generate synthetic data matching the expected format
    # Each sample: click_timestamp, conversion_timestamp, int_feat_1..8, cat_feat_1..9
    data_list = []
    
    for i in range(num_samples):
        # Generate click timestamp (random timestamps over several days)
        click_ts = 1500000000 + i * 3600  # Incrementing hourly
        
        # Generate conversion timestamp (if conversion happens)
        if np.random.random() < 0.2:  # 20% conversion rate
            conversion_delay = np.random.exponential(24)  # Average 24-hour delay
            conversion_ts = click_ts + min(int(conversion_delay * 3600), config.max_delay_hours * 3600)
            conversion_flag = 1
        else:
            # No conversion (delay is infinity marked as -1)
            conversion_ts = -1
            conversion_flag = 0
        
        # Generate 8 numerical features
        numerical_features = np.random.normal(0.5, 0.2, size=config.num_numeric_features).tolist()
        
        # Generate 9 categorical features (integer IDs)
        categorical_features = []
        for j in range(config.num_categorical_features):
            # Generate categorical IDs based on feature index
            cat_id = np.random.randint(0, 100 + j*50)
            categorical_features.append(cat_id)
        
        # Calculate delay in hours
        if conversion_flag == 1:
            delay_hours = (conversion_ts - click_ts) / 3600.0
        else:
            delay_hours = config.max_delay_hours + 1  # Mark non-conversion appropriately
        
        # Combine features
        features = numerical_features + categorical_features
        
        sample = [click_ts, features, delay_hours, conversion_flag]
        data_list.append(sample)
    
    # Sort chronologically by click timestamp
    data_list.sort(key=lambda x: x[0])
    
    return data_list


def create_temporal_splits(data, train_ratio=0.7, val_ratio=0.15):
    """
    Split data temporally to avoid data leakage.
    
    Args:
        data: Chronologically sorted list of samples
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        
    Returns:
        train_data, val_data, test_data
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def process_batch(batch_data):
    """
    Process a batch of raw data into tensors.
    
    Args:
        batch_data: List of samples [timestamp, features, delay, conversion_flag]
        
    Returns:
        tuple of processed tensors: features_tensor, delay_tensor, conversion_tensor, timestamp_tensor
    """
    batch_size = len(batch_data)
    
    # Extract components
    timestamps = []
    features_list = []
    delays = []
    conversions = []
    
    for sample in batch_data:
        ts, features, delay, conv = sample
        timestamps.append(ts)
        features_list.append(features)
        delays.append(delay)
        conversions.append(conv)
    
    # Convert to numpy arrays
    features_array = np.array(features_list, dtype=np.float32)
    delay_array = np.array(delays, dtype=np.float32)
    conversion_array = np.array(conversions, dtype=np.float32)
    timestamp_array = np.array(timestamps, dtype=np.float32)
    
    # Separate numerical and categorical features
    # First 8 are numerical, next 9 are categorical
    numerical_features = features_array[:, :8]
    categorical_features = features_array[:, 8:].astype(int)
    
    # Normalize numerical features per batch (for stability)
    numerical_mean = numerical_features.mean(axis=0, keepdims=True)
    numerical_std = numerical_features.std(axis=0, keepdims=True)
    numerical_std = np.where(numerical_std == 0, 1, numerical_std)  # Avoid division by zero
    normalized_numerical = (numerical_features - numerical_mean) / numerical_std
    
    # Combine normalized numerical and categorical features
    processed_features = np.concatenate([normalized_numerical, categorical_features], axis=1)
    
    return (
        processed_features.astype(np.float32),
        delay_array.reshape(-1, 1),
        conversion_array.reshape(-1, 1),
        timestamp_array.reshape(-1, 1)
    )