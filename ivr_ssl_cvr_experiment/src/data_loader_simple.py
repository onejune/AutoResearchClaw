"""
IVR Dataset Loader for Contrastive Learning - Simplified Version
Uses features from combine_schema for consistency with multitask project
"""
import os
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from collections import Counter


class IVRDataset(Dataset):
    """
    IVR Dataset for CVR prediction with contrastive learning capabilities
    Uses features from combine_schema to ensure consistency with multitask project
    """
    def __init__(self, data_path, date_range=None, sample_ratio=1.0, neg_sample_ratio=None):
        """
        Initialize IVR dataset
        
        Args:
            data_path: Path to IVR parquet files
            date_range: List of dates to include, e.g. ['2025-11-01', '2025-11-02']
            sample_ratio: Ratio of data to sample (for faster experimentation)
            neg_sample_ratio: Negative sampling ratio (None for no sampling)
        """
        self.data_path = data_path
        self.date_range = date_range
        self.sample_ratio = sample_ratio
        self.neg_sample_ratio = neg_sample_ratio
        self.data = []
        
        # Load features from combine_schema
        self.features = self._load_combine_schema()
        
        # Load a small sample for testing
        self._load_sample_data()
    
    def _load_combine_schema(self):
        """Load features from combine_schema file"""
        schema_file = "/mnt/workspace/open_research/autoresearch/multitask/combine_schema"
        try:
            with open(schema_file, 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(features)} features from combine_schema")
            return features
        except FileNotFoundError:
            print(f"Warning: combine_schema file not found at {schema_file}")
            # Return a minimal set of expected features
            return ['business_type', 'expid', 'region', 'imptype', 'objective_type', 'traffic_type', 'click', 'purchase', 'atc', 'cv']
    
    def _load_sample_data(self):
        """Load a small sample of data to understand structure"""
        # Get available dates
        if self.date_range is None:
            parquet_dir = os.path.join(self.data_path, 'parquet')
            available_dates = sorted([d for d in os.listdir(parquet_dir) 
                                     if os.path.isdir(os.path.join(parquet_dir, d))])
            self.date_range = available_dates[:1]  # Use just the first date for initial testing
        else:
            available_dates = self.date_range
        
        print(f"Loading data from {len(available_dates)} dates...")
        
        # Load a single small file to understand structure
        for date in available_dates:
            date_path = os.path.join(self.data_path, 'parquet', date)
            if os.path.exists(date_path):
                parquet_files = [f for f in os.listdir(date_path) if f.endswith('.parquet')]
                
                # Load just the first file with a small number of rows
                if parquet_files:
                    sample_file = parquet_files[0]
                    file_path = os.path.join(date_path, sample_file)
                    
                    print(f"Loading sample from {date}: {sample_file}")
                    
                    # Load just a few rows to understand structure
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    
                    # Sample data if needed
                    if self.sample_ratio < 1.0:
                        n_samples = min(int(len(df) * self.sample_ratio), 100)  # Max 100 rows for memory efficiency
                        df = df.sample(n=n_samples, random_state=42)
                    
                    # Parse mul_labels to extract click, purchase, atc
                    df = self._parse_mul_labels(df)
                    
                    # Filter business types as in multitask project
                    df = self._filter_business_types(df)
                    
                    self.df = df
                    print(f"Loaded sample: {len(self.df)} rows, {len(self.df.columns)} columns")
                    break
    
    def _parse_mul_labels(self, df):
        """Parse mul_labels column to extract click, purchase, atc labels"""
        if 'mul_labels' not in df.columns:
            # If no mul_labels column, create default values
            df['click'] = 0
            df['purchase'] = 0
            df['atc'] = 0
            return df
        
        def parse_single_label(mul_labels_val):
            # Handle various cases for mul_labels
            if mul_labels_val is None or (isinstance(mul_labels_val, (list, tuple)) and len(mul_labels_val) == 0):
                return {'click': 0, 'purchase': 0, 'atc': 0}
            
            try:
                # mul_labels is a list of tuples like [('atc', -1), ('open', 1), ('purchase', 0)]
                if isinstance(mul_labels_val, list):
                    label_dict = {k: v for k, v in mul_labels_val}
                    
                    # Convert to binary labels (1 for positive, 0 for negative/missing)
                    return {
                        'click': 1 if label_dict.get('click', 0) > 0 else 0,
                        'purchase': 1 if label_dict.get('purchase', 0) > 0 else 0,
                        'atc': 1 if label_dict.get('atc', 0) > 0 else 0
                    }
                else:
                    return {'click': 0, 'purchase': 0, 'atc': 0}
            except Exception as e:
                print(f"Error parsing mul_labels: {e}")
                return {'click': 0, 'purchase': 0, 'atc': 0}
        
        # Apply parsing
        try:
            parsed_data = df['mul_labels'].apply(parse_single_label)
            df['click'] = parsed_data.apply(lambda x: x['click'])
            df['purchase'] = parsed_data.apply(lambda x: x['purchase'])
            df['atc'] = parsed_data.apply(lambda x: x['atc'])
        except Exception as e:
            print(f"Error applying mul_labels parsing: {e}")
            # Create default columns
            df['click'] = 0
            df['purchase'] = 0
            df['atc'] = 0
        
        return df
    
    def _filter_business_types(self, df):
        """Filter for target business types as in multitask project"""
        if 'business_type' not in df.columns:
            return df  # If no business_type column, return as is
            
        target_types = [
            'shein',
            'aecps', 'aedsp', 'aerta',
            'shopee_cps',
            'lazada_cps', 'lazada_rta'
        ]
        
        # Create mask for valid business types
        mask = df['business_type'].isin(target_types)
        return df[mask].copy()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create batch with essential schema features and behavioral labels
        batch_item = {}
        
        # Add behavioral labels (these are targets, not in schema but needed)
        batch_item['click'] = int(row.get('click', 0))
        batch_item['purchase'] = int(row.get('purchase', 0))
        batch_item['atc'] = int(row.get('atc', 0))
        batch_item['cv'] = int(row.get('cv', 0)) if 'cv' in row else int(row.get('purchase', 0))  # cv often same as purchase
        
        # Add available categorical features from schema that are actually present in the data
        # Using business_type as the main feature for contrastive learning
        for feat in ['business_type', 'imptype', 'objective_type', 'traffic_type']:
            if feat in row:
                val = row[feat]
                # Convert to integer index (handle missing values)
                if pd.isna(val):
                    batch_item[feat] = 0  # Use 0 for unknown/missing
                else:
                    # Convert to string first to handle different types, then hash to index
                    val_str = str(val)
                    # Simple hashing approach for now - in practice you'd use fitted label encoders
                    batch_item[feat] = abs(hash(val_str)) % 10000  # Modulo to limit vocab size
            else:
                batch_item[feat] = 0  # Missing feature gets index 0
        
        # Convert all values to tensors
        for key, value in batch_item.items():
            if isinstance(value, (int, float)):
                batch_item[key] = torch.tensor(value, dtype=torch.long)
        
        return batch_item


def collate_fn(batch):
    """
    Custom collate function to handle schema-based features
    """
    if not batch:
        return {}
    
    # Get all feature names from the first item
    all_features = set(batch[0].keys())
    
    # Create batched tensors for each feature
    batched = {}
    for feature in all_features:
        values = [item[feature] for item in batch]
        batched[feature] = torch.stack(values)
    
    return batched


def get_ivr_dataloader(data_path, date_range=None, batch_size=32, sample_ratio=1.0, shuffle=True):
    """
    Create dataloader for IVR dataset with schema-compliant features
    """
    dataset = IVRDataset(data_path, date_range=date_range, sample_ratio=sample_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
    return dataloader


if __name__ == "__main__":
    # Test the data loader with a tiny sample
    data_path = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/"
    dataloader = get_ivr_dataloader(
        data_path, 
        date_range=["2025-11-01"], 
        batch_size=2,  # Very small batch for testing
        sample_ratio=0.0001  # Tiny sample for memory efficiency
    )
    
    # Print sample batch
    for batch in dataloader:
        print("Number of features in batch:", len(batch.keys()))
        print("Feature keys:", list(batch.keys()))
        print("Purchase tensor:", batch['purchase'])
        print("Click tensor:", batch['click'])
        print("Business type tensor:", batch['business_type'])
        print("Sample feature tensor shape:", batch[list(batch.keys())[0]].shape)
        break
    print("Data loader test completed successfully!")