#!/usr/bin/env python3
"""
Run All LTV Experiments Sequentially

This script runs all experiments in order and generates a comprehensive report.
"""

import subprocess
import sys
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

def run_experiment(exp_num, exp_name):
    """Run a single experiment"""
    print("\n" + "="*80)
    print(f"Running {exp_name}...")
    print("="*80)
    
    script = PROJECT_ROOT / 'scripts' / f'run_exp{exp_num:03d}_{exp_name}.py'
    if not script.exists():
        print(f"Script not found: {script}")
        return None
    
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print(f"✅ {exp_name} completed successfully")
            # Load results
            result_file = PROJECT_ROOT / 'results' / f'exp{exp_num:03d}_{exp_name}' / 'results.json'
            if result_file.exists():
                with open(result_file) as f:
                    return json.load(f)
        else:
            print(f"❌ {exp_name} failed:")
            print(result.stderr)
        
        return None
    except Exception as e:
        print(f"❌ {exp_name} error: {e}")
        return None


def main():
    print("="*80)
    print("LTV Optimization - Running All Experiments")
    print("="*80)
    
    experiments = [
        (1, "baseline"),
        (2, "ziln"),
        (3, "odmn_mdme"),
        (4, "expltv"),
        (5, "cmltv"),
    ]
    
    results = []
    for exp_num, exp_name in experiments:
        result = run_experiment(exp_num, exp_name)
        if result:
            results.append(result)
    
    # Generate summary report
    if results:
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        summary_df = pd.DataFrame(results)
        print(summary_df[['model', 'auc', 'pcoc_10', 'pcoc_20', 'pcoc_30', 'rmse', 'mae']].to_string(index=False))
        
        # Save summary
        summary_file = PROJECT_ROOT / 'results' / 'summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}")
        
        # Update experiment report
        update_report(summary_df)
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)


def update_report(summary_df):
    """Update experiment_report.md with results"""
    report_path = PROJECT_ROOT / 'experiment_report.md'
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Generate table
    table_lines = ["\n### 基线模型结果 (exp001)\n"]
    table_lines.append("| Model | AUC | PCOC@10 | PCOC@20 | PCOC@30 | RMSE | MAE |\n")
    table_lines.append("|-------|-----|---------|---------|---------|------|-----|\n")
    
    for _, row in summary_df.iterrows():
        table_lines.append(
            f"| {row['model']} | {row.get('auc', 0):.4f} | {row.get('pcoc_10', 0):.4f} | "
            f"{row.get('pcoc_20', 0):.4f} | {row.get('pcoc_30', 0):.4f} | "
            f"{row.get('rmse', 0):.4f} | {row.get('mae', 0):.4f} |\n"
        )
    
    # Insert into report
    if "| Model | AUC | PCOC@10 |" not in content:
        insert_pos = content.find("## 实验进度")
        if insert_pos != -1:
            content = content[:insert_pos] + "".join(table_lines) + "\n" + content[insert_pos:]
    
    with open(report_path, 'w') as f:
        f.write(content)
    
    print(f"Updated experiment report")


if __name__ == "__main__":
    main()
