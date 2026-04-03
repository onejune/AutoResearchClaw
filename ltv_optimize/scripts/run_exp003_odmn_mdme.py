#!/usr/bin/env python3
"""Experiment 003: ODMN + MDME"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    print("="*80)
    print("Experiment 003: ODMN + MDME")
    print("="*80)
    
    data_path = PROJECT_ROOT / 'data' / 'train_data.parquet'
    if not data_path.exists():
        print(f"Error: Training data not found. Run exp001 first.")
        return
    
    from models.odmn_mdme import run_odmn_mdme_experiment
    metrics = run_odmn_mdme_experiment(str(data_path))
    
    print("\nExperiment 003 Completed!")

if __name__ == "__main__":
    main()
