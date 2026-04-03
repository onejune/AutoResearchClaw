#!/usr/bin/env python3
"""Experiment 004: ExpLTV"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if __name__ == "__main__":
    from models.expltv import run_expltv_experiment
    run_expltv_experiment('/mnt/workspace/open_research/autoresearch/ltv_optimize/data/train_data.parquet')
