#!/usr/bin/env python3
"""
IVR SSL 实验结果汇总与分析
基于之前运行的完整实验结果进行分析
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_experiment_results():
    """分析之前实验的详细结果"""
    print("IVR SSL 对比学习实验详细分析")
    print("="*80)
    
    # 模拟实验结果（基于之前的运行结果）
    results = {
        'baseline': {
            'overall_auc': 0.8484,
            'params': 40633905,
            'train_time': 41.9,
            'bt_aucs': {
                1: 0.7824,  # 最多样本
                11: 0.7165,
                4: 0.8388,
                6: 0.8174,
                2: 0.8029,
                3: 0.7301,
                8: 0.8421,
                0: 0.7872,
                5: 0.7775,
                10: 0.7857
            }
        },
        'simgcl': {
            'overall_auc': 0.8476,
            'params': 41162609,
            'train_time': 71.3,
            'bt_aucs': {
                1: 0.7819,
                11: 0.7269,
                4: 0.8419,
                6: 0.8033,
                2: 0.8021,
                3: 0.7353,
                8: 0.8406,
                0: 0.7756,
                5: 0.7778,
                10: 0.7849
            }
        },
        'supcon': {
            'overall_auc': 0.8490,
            'params': 41162609,
            'train_time': 71.0,
            'bt_aucs': {
                1: 0.7811,
                11: 0.7182,
                4: 0.8385,
                6: 0.8032,
                2: 0.8029,
                3: 0.7326,
                8: 0.8369,
                0: 0.7806,
                5: 0.7788,
                10: 0.7830
            }
        },
        'domain_cl': {
            'overall_auc': 0.8471,
            'params': 41162609,
            'train_time': 71.1,
            'bt_aucs': {
                1: 0.7824,
                11: 0.7270,
                4: 0.8419,
                6: 0.8011,
                2: 0.8005,
                3: 0.7276,
                8: 0.8257,
                0: 0.7770,
                5: 0.7782,
                10: 0.7765
            }
        },
        'feature_mask': {
            'overall_auc': 0.8483,
            'params': 41162609,
            'train_time': 70.0,
            'bt_aucs': {
                1: 0.7818,
                11: 0.7152,
                4: 0.8407,
                6: 0.7962,
                2: 0.8043,
                3: 0.7327,
                8: 0.8438,
                0: 0.7785,
                5: 0.7739,
                10: 0.7815
            }
        },
        'directau': {
            'overall_auc': 0.8467,
            'params': 41162609,
            'train_time': 65.5,
            'bt_aucs': {
                1: 0.7808,
                11: 0.7206,
                4: 0.8402,
                6: 0.8045,
                2: 0.8002,
                3: 0.7259,
                8: 0.8345,
                0: 0.7673,
                5: 0.7752,
                10: 0.7799
            }
        }
    }
    
    # 总体性能比较
    print("总体性能比较")
    print("="*80)
    print(f"{'Method':<15} {'Overall AUC':<12} {'Params(M)':<10} {'Train Time(s)':<15} {'vs Baseline':<12}")
    print("-"*80)
    
    baseline_auc = results['baseline']['overall_auc']
    for method, data in results.items():
        diff = (data['overall_auc'] - baseline_auc) * 1000  # 转换为千分点
        sign = "+" if diff >= 0 else ""
        print(f"{method:<15} {data['overall_auc']:<12.4f} {data['params']/1e6:<9.1f}  {data['train_time']:<14.1f} {sign}{diff:<11.2f}‰")
    
    # 按 business_type 分析
    print(f"\n各 business_type 性能分析")
    print("="*80)
    
    # 创建对比表格
    bt_names = [1, 11, 4, 6, 2, 3, 8, 0, 5, 10]  # Top 10 business types
    bt_names_str = ['BT_1', 'BT_11', 'BT_4', 'BT_6', 'BT_2', 'BT_3', 'BT_8', 'BT_0', 'BT_5', 'BT_10']
    
    # 创建 DataFrame
    df_data = []
    for bt_name, bt_code in zip(bt_names_str, bt_names):
        row = {'business_type': bt_name}
        for method in results.keys():
            if bt_code in results[method]['bt_aucs']:
                row[method] = results[method]['bt_aucs'][bt_code]
            else:
                row[method] = 0.0  # 或者 None
        df_data.append(row)
    
    comparison_df = pd.DataFrame(df_data)
    print("\nTop 10 business_type 的方法对比 (AUC):")
    print(comparison_df.round(4))
    
    # 分析 SupCon 相对 Baseline 的表现
    print(f"\nSupCon vs Baseline 详细分析")
    print("="*80)
    
    baseline_bt = results['baseline']['bt_aucs']
    supcon_bt = results['supcon']['bt_aucs']
    
    improvements = []
    for bt in bt_names:
        if bt in baseline_bt and bt in supcon_bt:
            bl_auc = baseline_bt[bt]
            sc_auc = supcon_bt[bt]
            improvement = (sc_auc - bl_auc) * 1000  # 千分点
            improvements.append({
                'business_type': f'BT_{bt}',
                'baseline_auc': bl_auc,
                'supcon_auc': sc_auc,
                'improvement_point': improvement
            })
    
    improvements_df = pd.DataFrame(improvements).sort_values('improvement_point', ascending=False)
    
    print(f"\nSupCon 相对 Baseline 提升 Top 5 business_type:")
    print(improvements_df[['business_type', 'baseline_auc', 'supcon_auc', 'improvement_point']].head().round(2))
    
    print(f"\nSupCon 相对 Baseline 下降 Top 5 business_type:")
    print(improvements_df[['business_type', 'baseline_auc', 'supcon_auc', 'improvement_point']].tail().round(2))
    
    # 统计分析
    positive_changes = [imp for imp in improvements_df['improvement_point'] if imp > 0.1]
    negative_changes = [imp for imp in improvements_df['improvement_point'] if imp < -0.1]
    neutral_changes = [imp for imp in improvements_df['improvement_point'] if abs(imp) <= 0.1]
    
    print(f"\n变化统计 (>0.1‰, <-0.1‰, ±≤0.1‰): {len(positive_changes)}, {len(negative_changes)}, {len(neutral_changes)}")
    
    # 找出 SupCon 表现最好的 business_type
    print(f"\nSupCon 表现分析")
    print("-"*40)
    best_supcon_bt = max(supcon_bt.items(), key=lambda x: x[1])
    worst_supcon_bt = min(supcon_bt.items(), key=lambda x: x[1])
    print(f"SupCon 最佳 BT_{best_supcon_bt[0]}: AUC={best_supcon_bt[1]:.4f}")
    print(f"SupCon 最差 BT_{worst_supcon_bt[0]}: AUC={worst_supcon_bt[1]:.4f}")
    
    # 方法排名分析
    print(f"\n方法综合分析")
    print("-"*40)
    methods_ranking = sorted([(m, d['overall_auc']) for m, d in results.items()], 
                            key=lambda x: x[1], reverse=True)
    print("方法排名 (Overall AUC):")
    for i, (method, auc) in enumerate(methods_ranking, 1):
        print(f"{i}. {method}: {auc:.4f}")
    
    # 计算各方法的稳定性（标准差）
    print(f"\n方法稳定性分析 (各BT AUC标准差):")
    for method, data in results.items():
        bt_aucs = list(data['bt_aucs'].values())
        stability = np.std(bt_aucs)
        print(f"{method}: {stability:.4f}")
    
    # 结论
    print(f"\n实验结论")
    print("="*80)
    print("1. SupCon 表现最佳 (AUC=0.8490)，相比 Baseline 提升 0.06‰")
    print("2. FeatureMask 排名第二 (AUC=0.8483)，相比 Baseline 降低 0.10‰")
    print("3. SimGCL 排名第三 (AUC=0.8476)，相比 Baseline 降低 0.80‰")
    print("4. DomainCL 和 DirectAU 表现较差")
    print("5. 所有方法在不同 business_type 上表现相对稳定")
    print("6. 对比学习在当前数据集上增益有限，可能需要更强的数据增强策略")
    
    return results


def create_visualizations(results):
    """创建可视化图表"""
    # 这里可以添加绘图代码
    pass


if __name__ == '__main__':
    results = analyze_experiment_results()