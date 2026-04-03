#!/usr/bin/env python3
"""
Experiment 002: ZILN (Zero-Inflated Log-Normal) for LTV Prediction

Paper: Zhang et al. "Predicting Player Lifetime Value with Zero-Inflated Log-Normal." Google, 2019.
Link: https://arxiv.org/pdf/1912.07753
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    print("="*80)
    print("Experiment 002: ZILN Model")
    print("="*80)
    
    # Check if training data exists
    data_path = PROJECT_ROOT / 'data' / 'train_data.parquet'
    if not data_path.exists():
        print(f"\nError: Training data not found at {data_path}")
        print("Please run exp001_baseline.py first to generate training data.")
        return
    
    # Run ZILN experiment
    from models.ziln import run_ziln_experiment
    
    metrics = run_ziln_experiment(str(data_path))
    
    # Update experiment report
    update_experiment_report(metrics)
    
    print("\n" + "="*80)
    print("Experiment 002 Completed!")
    print("="*80)


def update_experiment_report(metrics):
    """Update experiment report with ZILN results"""
    report_path = PROJECT_ROOT / 'experiment_report.md'
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Find and update the ZILN section
        old_text = "| exp002 | ZILN | - | - | - | 📋 待启动 |"
        new_text = f"| exp002 | ZILN | {metrics.get('auc', 0):.4f} | {metrics.get('pcoc_10', 0):.4f} | {metrics.get('rmse', 0):.4f} | ✅ 完成 |"
        
        content = content.replace(old_text, new_text)
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        print(f"Updated experiment report")


if __name__ == "__main__":
    main()
