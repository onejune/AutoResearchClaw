"""
IVR Dataset Analysis for Contrastive Learning Design
"""
import os
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import json


def analyze_ivr_dataset(data_path, sample_dates=5):
    """
    Analyze IVR dataset to understand its characteristics for contrastive learning
    """
    print("Analyzing IVR dataset...")
    
    # Get available dates
    parquet_dir = os.path.join(data_path, 'parquet')
    available_dates = sorted([d for d in os.listdir(parquet_dir) 
                             if os.path.isdir(os.path.join(parquet_dir, d))])
    
    print(f"Found {len(available_dates)} available dates. Analyzing first {sample_dates} dates.")
    
    # Sample some data for analysis
    all_data = []
    for date in available_dates[:sample_dates]:
        date_path = os.path.join(parquet_dir, date)
        parquet_files = [f for f in os.listdir(date_path) if f.endswith('.parquet')]
        
        # Take first parquet file for each date
        if parquet_files:
            sample_file = parquet_files[0]
            file_path = os.path.join(date_path, sample_file)
            
            print(f"Loading {date}: {sample_file}")
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            # Sample a subset for analysis
            sample_size = min(10000, len(df))  # Limit to 10k rows per date for memory efficiency
            df_sample = df.sample(n=sample_size, random_state=42)
            df_sample['date'] = date  # Add date column for tracking
            all_data.append(df_sample)
    
    # Combine all sampled data
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Total analyzed samples: {len(full_df)}")
    
    # Basic statistics
    print("\n" + "="*50)
    print("BASIC DATASET STATISTICS")
    print("="*50)
    print(f"Total samples: {len(full_df):,}")
    print(f"Date range: {full_df['date'].min()} to {full_df['date'].max()}")
    
    # Look for key conversion-related columns
    conversion_cols = [col for col in full_df.columns if 
                      any(keyword in col.lower() for keyword in 
                          ['click', 'purchase', 'buy', 'cv', 'conversion', 'atc', 'cart', 'revenue', 'gmv'])]
    print(f"\nPotential conversion-related columns: {conversion_cols[:10]}")  # Show first 10
    
    # Analyze business types
    if 'business_type' in full_df.columns:
        business_types = full_df['business_type'].value_counts()
        print(f"\nBusiness Types Distribution:")
        print(business_types.head(10))
    
    # Analyze key conversion metrics if available
    key_cols = ['click', 'purchase', 'atc', 'cv']
    available_key_cols = [col for col in key_cols if col in full_df.columns]
    
    print(f"\nConversion Metrics Analysis:")
    for col in available_key_cols:
        if col in full_df.columns:
            series = full_df[col]
            if series.dtype in ['int64', 'float64', 'int32', 'float32']:
                try:
                    pos_count = (series > 0).sum()
                    total_count = len(series)
                    ratio = pos_count / total_count if total_count > 0 else 0
                    print(f"- {col}: {pos_count:,} positive / {total_count:,} total ({ratio:.4f})")
                except Exception as e:
                    print(f"- {col}: Error analyzing - {str(e)}")
    
    # Identify categorical columns that could be used for contrastive learning
    categorical_candidates = []
    for col in full_df.columns:
        if full_df[col].dtype == 'object' or full_df[col].dtype.name == 'category':
            try:
                unique_vals = full_df[col].nunique()
                if 2 <= unique_vals <= 100:  # Good range for contrastive learning categories
                    categorical_candidates.append((col, unique_vals))
            except TypeError:
                # Handle unhashable types like lists
                print(f"Skipping column {col} due to unhashable type")
                continue
    
    print(f"\nCategorical columns for contrastive learning (2-100 unique values):")
    for col, n_unique in sorted(categorical_candidates, key=lambda x: x[1])[:15]:  # Top 15
        print(f"- {col}: {n_unique} unique values")
    
    # Analyze numerical features for potential embedding
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'date']  # Exclude added date column
    
    print(f"\nTop 20 numerical features (excluding date column):")
    for col in numeric_cols[:20]:
        non_null_count = full_df[col].notna().sum()
        print(f"- {col}: {non_null_count:,} non-null values")
    
    # Prepare recommendations for contrastive learning
    print(f"\n" + "="*50)
    print("CONTRASTIVE LEARNING RECOMMENDATIONS")
    print("="*50)
    
    recommendations = {
        "positive_pairs_strategy": [],
        "negative_pairs_strategy": [],
        "feature_augmentation": [],
        "temporal_aspects": []
    }
    
    # Positive pairs - based on user behavior patterns
    if 'click' in full_df.columns and 'purchase' in full_df.columns:
        click_purchase_corr = full_df[['click', 'purchase']].corr().iloc[0,1]
        recommendations["positive_pairs_strategy"].append(
            f"User behavior sequences: Users with similar click→purchase patterns (correlation: {click_purchase_corr:.3f})"
        )
    
    # Business type contrastive learning
    if 'business_type' in full_df.columns:
        recommendations["positive_pairs_strategy"].append(
            "Cross-business contrastive: Similar items across different business types (shein/aecps/aedsp/aerta/shopee/lazada)"
        )
        recommendations["negative_pairs_strategy"].append(
            "Within-business contrastive: Different items within same business type"
        )
    
    # Temporal aspects
    recommendations["temporal_aspects"].append(
        "Time-based contrastive: Same user behavior across different time periods (daily/weekly patterns)"
    )
    
    # Feature-based augmentation
    recommendations["feature_augmentation"].append(
        "Feature dropout: Randomly mask sparse features to create augmented views"
    )
    recommendations["feature_augmentation"].append(
        "Gaussian noise: Add small noise to dense numerical features"
    )
    
    # Print recommendations
    for category, items in recommendations.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"  • {item}")
    
    # Save analysis results
    analysis_results = {
        "basic_stats": {
            "total_samples": len(full_df),
            "date_range": [full_df['date'].min(), full_df['date'].max()],
            "columns_count": len(full_df.columns)
        },
        "conversion_metrics": {},
        "categorical_features": [col for col, _ in categorical_candidates],
        "recommendations": recommendations,
        "analysis_timestamp": pd.Timestamp.now().isoformat()
    }
    
    for col in available_key_cols:
        if col in full_df.columns:
            series = full_df[col]
            if series.dtype in ['int64', 'float64', 'int32', 'float32']:
                try:
                    pos_count = (series > 0).sum()
                    total_count = len(series)
                    ratio = pos_count / total_count if total_count > 0 else 0
                    analysis_results["conversion_metrics"][col] = {
                        "positive_count": int(pos_count),
                        "total_count": int(total_count),
                        "positive_ratio": float(ratio)
                    }
                except Exception as e:
                    print(f"Error analyzing {col}: {str(e)}")
                    continue
    
    # Save to results directory
    results_dir = "/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr/results"
    os.makedirs(results_dir, exist_ok=True)
    
    analysis_file = os.path.join(results_dir, f"ivr_dataset_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\nDetailed analysis saved to: {analysis_file}")
    
    return analysis_results


def main():
    data_path = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/"
    
    print("IVR Dataset Analysis for Contrastive Learning")
    print("=" * 60)
    
    analysis_results = analyze_ivr_dataset(data_path)
    
    print(f"\nAnalysis completed! Results saved to the results directory.")
    

if __name__ == "__main__":
    main()