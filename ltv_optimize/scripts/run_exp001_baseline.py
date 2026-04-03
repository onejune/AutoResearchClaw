#!/usr/bin/env python3
"""
Experiment 001: Baseline Models for LTV Prediction

运行所有基线模型并保存结果
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    print("="*80)
    print("Experiment 001: Baseline Models")
    print("="*80)
    
    # 是否使用采样模式 (快速测试用)
    USE_SAMPLE = True  # 设置为 False 以运行全量数据
    SAMPLE_SIZE = 1_000_000 if USE_SAMPLE else None
    
    # Step 1: 数据预处理
    print(f"\n[Step 1] Data Preprocessing{' (SAMPLE MODE)' if USE_SAMPLE else ''}")
    from data.loader import LTVDataLoader
    
    loader = LTVDataLoader()
    loader.load_raw_data(sample_size=SAMPLE_SIZE)
    train_df, val_df, test_df = loader.preprocess(train_days=6, test_days=3, ltv_window_days=7)
    
    # 创建 LTV 标签
    train_labels = loader.create_ltv_labels(train_df, window_days=7)
    
    # 提取特征
    train_data = loader.extract_features(train_df, train_labels)
    
    # 保存处理后的数据
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    train_data.to_parquet(data_dir / 'train_data.parquet', index=False)
    print(f"Saved training data to {data_dir / 'train_data.parquet'}")
    
    # Step 2: 运行基线模型
    print("\n[Step 2] Running Baseline Models")
    from models.baseline import run_all_baselines
    
    results = run_all_baselines(str(data_dir / 'train_data.parquet'))
    
    # Step 3: 更新实验报告
    print("\n[Step 3] Updating Experiment Report")
    update_experiment_report(results)
    
    print("\n" + "="*80)
    print("Experiment 001 Completed!")
    print("="*80)


def update_experiment_report(results):
    """更新实验报告"""
    report_path = PROJECT_ROOT / 'experiment_report.md'
    
    # 读取现有报告
    if report_path.exists():
        with open(report_path, 'r') as f:
            content = f.read()
    
    # 生成基线结果表格
    table_lines = [
        "\n## 基线模型结果 (exp001)\n",
        "| Model | AUC | PCOC@10 | PCOC@20 | PCOC@30 | RMSE | MAE |\n",
        "|-------|-----|---------|---------|---------|------|-----|\n"
    ]
    
    for r in results:
        model = r.get('model', 'Unknown')
        auc = r.get('auc', 0)
        pcoc_10 = r.get('pcoc_10', 0)
        pcoc_20 = r.get('pcoc_20', 0)
        pcoc_30 = r.get('pcoc_30', 0)
        rmse = r.get('rmse', 0)
        mae = r.get('mae', 0)
        
        table_lines.append(
            f"| {model} | {auc:.4f} | {pcoc_10:.4f} | {pcoc_20:.4f} | {pcoc_30:.4f} | {rmse:.4f} | {mae:.4f} |\n"
        )
    
    # 插入到报告中
    if "## 基线模型结果 (exp001)" not in content:
        # 找到合适的位置插入
        insert_pos = content.find("## 实验进度")
        if insert_pos != -1:
            content = content[:insert_pos] + "".join(table_lines) + "\n" + content[insert_pos:]
    
    # 写回报告
    with open(report_path, 'w') as f:
        f.write(content)
    
    print(f"Updated experiment report: {report_path}")


if __name__ == "__main__":
    main()
