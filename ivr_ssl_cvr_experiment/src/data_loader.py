"""
IVR Dataset Loader for Contrastive Learning
"""
import os
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


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
        
        # Load data from parquet files
        self._load_data()
    
    def _load_combine_schema(self):
        """Load features from combine_schema file"""
        schema_file = "/mnt/workspace/open_research/autoresearch/multitask/combine_schema"
        with open(schema_file, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(features)} features from combine_schema")
        return features
    
    def _load_data(self):
        """Load data from parquet files based on date range"""
        if self.date_range is None:
            # Get all available dates
            available_dates = [d for d in os.listdir(os.path.join(self.data_path, 'parquet')) 
                              if os.path.isdir(os.path.join(self.data_path, 'parquet', d))]
        else:
            available_dates = self.date_range
            
        print(f"Loading data from {len(available_dates)} dates...")
        
        for date in available_dates:
            date_path = os.path.join(self.data_path, 'parquet', date)
            if os.path.exists(date_path):
                # Load all parquet files for this date
                for file in os.listdir(date_path):
                    if file.endswith('.parquet'):
                        parquet_file = os.path.join(date_path, file)
                        try:
                            # First, read a small sample to check column existence
                            # This avoids issues with reading all columns at once
                            table = pq.read_table(parquet_file)
                            
                            # Get the actual columns available in this file
                            available_cols = set(table.schema.names)
                            
                            # Filter our desired columns to only those that exist
                            desired_cols = set(self.features + ['mul_labels', 'revenue', 'business_type'])
                            cols_to_read = list(available_cols.intersection(desired_cols))
                            
                            # Read only the available columns
                            table_subset = pq.read_table(parquet_file, columns=cols_to_read)
                            
                            # Convert to pandas for easier processing
                            df = table_subset.to_pandas()
                            
                            # Parse mul_labels to extract click, purchase, atc
                            df = self._parse_mul_labels(df)
                            
                            # Filter business types as in multitask project
                            df = self._filter_business_types(df)
                            
                            # Sample data if needed
                            if self.sample_ratio < 1.0:
                                n_samples = int(len(df) * self.sample_ratio)
                                df = df.sample(n=n_samples, random_state=42)
                                
                            # Add to our dataset
                            self.data.append(df)
                        except Exception as e:
                            print(f"Warning: Could not load {parquet_file}: {e}")
                            continue
        
        # Concatenate all dataframes
        if self.data:
            self.df = pd.concat(self.data, ignore_index=True)
            print(f"Loaded {len(self.df)} samples")
            
            # Apply negative sampling if specified
            if self.neg_sample_ratio and self.neg_sample_ratio < 1.0:
                self.df = self._negative_sampling(self.df, self.neg_sample_ratio)
                print(f"After negative sampling: {len(self.df)} samples")
        else:
            raise ValueError("No data found in specified date range")
    
    def _parse_mul_labels(self, df):
        """Parse mul_labels column to extract click, purchase, atc labels"""
        def parse_single_label(mul_labels_str):
            if pd.isna(mul_labels_str):
                return {'click': 0, 'purchase': 0, 'atc': 0}
            
            try:
                # Format: [('atc', -1), ('open', 0), ('purchase', 0), ('content_view', -1)]
                labels_list = eval(mul_labels_str) if isinstance(mul_labels_str, str) else mul_labels_str
                label_dict = {k: v for k, v in labels_list}
                
                # Convert to binary labels (1 for positive, 0 for negative/missing)
                return {
                    'click': 1 if label_dict.get('click', 0) > 0 else 0,
                    'purchase': 1 if label_dict.get('purchase', 0) > 0 else 0,
                    'atc': 1 if label_dict.get('atc', 0) > 0 else 0
                }
            except:
                return {'click': 0, 'purchase': 0, 'atc': 0}
        
        # Apply parsing
        parsed_data = df['mul_labels'].apply(parse_single_label)
        df['click'] = parsed_data.apply(lambda x: x['click'])
        df['purchase'] = parsed_data.apply(lambda x: x['purchase'])
        df['atc'] = parsed_data.apply(lambda x: x['atc'])
        
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
    
    def _negative_sampling(self, df, neg_sample_ratio):
        """Apply negative sampling to balance the dataset"""
        pos_mask = df['purchase'] == 1
        neg_mask = df['purchase'] == 0
        
        pos_df = df[pos_mask]
        neg_df = df[neg_mask].sample(frac=neg_sample_ratio, random_state=42)
        
        return pd.concat([pos_df, neg_df], ignore_index=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create batch with all schema features
        batch_item = {}
        
        # Process all schema features and convert to indices
        for feat in self.features:
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
        
        # Add behavioral labels (these are not in schema but are targets)
        batch_item['click'] = int(row.get('click', 0))
        batch_item['purchase'] = int(row.get('purchase', 0))
        batch_item['atc'] = int(row.get('atc', 0))
        
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


def get_ivr_dataloader(data_path, date_range=None, batch_size=1024, sample_ratio=1.0, shuffle=True):
    """
    Create dataloader for IVR dataset with schema-compliant features
    """
    dataset = IVRDataset(data_path, date_range=date_range, sample_ratio=sample_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
    return dataloader


if __name__ == "__main__":
    # Test the data loader
    data_path = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/"
    dataloader = get_ivr_dataloader(
        data_path, 
        date_range=["2025-11-01"], 
        batch_size=32,
        sample_ratio=0.001  # Only use 0.1% for testing to avoid memory issues
    )
    
    # Print sample batch
    for batch in dataloader:
        print("Number of features:", len(batch.keys()))
        print("Sample feature keys:", list(batch.keys())[:10])  # Show first 10 features
        print("Click tensor shape:", batch['click'].shape)
        print("Purchase tensor shape:", batch['purchase'].shape)
        print("Sample feature tensor shape:", batch[list(batch.keys())[0]].shape)
        break