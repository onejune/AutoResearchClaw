"""
实验 01: Bid Landscape Forecasting Baseline

基线模型:
1. Logistic Regression (bid → win prob)
2. XGBoost Classifier
3. Quantile Regression (预测不同分位数)
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


def load_data(data_path: str, max_samples: int = None):
    """加载数据"""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    if max_samples:
        df = df.head(max_samples)
        print(f"  Using {max_samples} samples")
    
    print(f"  Loaded {len(df)} samples")
    print(f"  Features: {len(df.columns)}")
    print(f"  Win rate: {df['win_label'].mean():.4f}")
    
    return df


def prepare_features(df: pd.DataFrame):
    """准备特征"""
    # 数值特征
    numeric_cols = ['bid_amount', 'true_value']
    
    # 类别特征 (one-hot)
    categorical_cols = ['business_type']
    
    X = df[numeric_cols].copy()
    
    # One-hot encode business_type
    bt_dummies = pd.get_dummies(df[categorical_cols], prefix='bt')
    X = pd.concat([X, bt_dummies], axis=1)
    
    y = df['win_label'].values
    
    return X, y


def compute_metrics(y_true, y_pred, y_prob=None):
    """计算评估指标"""
    metrics = {
        'auc': roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
    }
    
    # Calibration metrics (if probability output)
    if y_prob is not None:
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0
        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += mask.sum() / len(y_prob) * abs(bin_acc - bin_conf)
        
        metrics['ece'] = ece
    
    return metrics


def run_logistic_regression(X_train, X_val, y_train, y_val):
    """Logistic Regression"""
    print("\n" + "="*60)
    print("1. Logistic Regression")
    print("="*60)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    metrics = compute_metrics(y_val, y_pred, y_prob)
    
    print(f"AUC:  {metrics['auc']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"ECE:  {metrics.get('ece', 'N/A'):.4f}")
    
    return model, metrics


def run_xgboost(X_train, X_val, y_train, y_val):
    """XGBoost Classifier"""
    print("\n" + "="*60)
    print("2. XGBoost Classifier")
    print("="*60)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    metrics = compute_metrics(y_val, y_pred, y_prob)
    
    print(f"AUC:  {metrics['auc']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"ECE:  {metrics.get('ece', 'N/A'):.4f}")
    
    # Feature importance
    importances = model.feature_importances_
    feature_names = X_train.columns
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nTop 5 Features:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")
    
    return model, metrics


def run_quantile_regression(X_train, X_val, y_train, y_val):
    """Quantile Regression (预测 0.25, 0.5, 0.75 分位数)"""
    print("\n" + "="*60)
    print("3. Quantile Regression")
    print("="*60)
    
    quantiles = [0.25, 0.5, 0.75]
    models = {}
    all_metrics = {}
    
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=q,
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        models[q] = model
        
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, 1)  # Clip to [0, 1]
        
        # For quantile, we use pinball loss
        residuals = y_val - y_pred
        pinball_loss = np.mean(
            np.where(residuals >= 0, q * residuals, (q - 1) * residuals)
        )
        
        all_metrics[q] = {'pinball_loss': pinball_loss}
        
        print(f"Quantile {q}: Pinball Loss = {pinball_loss:.4f}")
    
    # Also compute AUC using median prediction
    y_median = models[0.5].predict(X_val)
    y_median = np.clip(y_median, 0, 1)
    
    # Convert to binary for AUC
    y_pred_binary = (y_median >= 0.5).astype(int)
    metrics = compute_metrics(y_val, y_pred_binary, y_median)
    
    print(f"\nMedian Prediction:")
    print(f"AUC:  {metrics['auc']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    all_metrics['overall'] = metrics
    
    return models, all_metrics


def evaluate_by_business_type(df_val, model, model_name):
    """按 business_type 分组评估"""
    print(f"\n{model_name} - By Business Type (Top 5)")
    print("-"*60)
    
    # Get predictions
    X_val = df_val[['bid_amount', 'true_value']].copy()
    bt_dummies = pd.get_dummies(df_val['business_type'], prefix='bt')
    X_val = pd.concat([X_val, bt_dummies], axis=1)
    y_val = df_val['win_label'].values
    
    y_prob = model.predict_proba(X_val.values)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val.values)
    
    # Group by business_type
    df_val['pred'] = y_prob
    results = []
    
    for bt in sorted(df_val['business_type'].unique())[:10]:
        mask = df_val['business_type'] == bt
        if mask.sum() < 100:
            continue
        
        auc = roc_auc_score(df_val.loc[mask, 'win_label'], df_val.loc[mask, 'pred'])
        results.append({
            'business_type': bt,
            'samples': mask.sum(),
            'auc': auc
        })
    
    results = sorted(results, key=lambda x: x['samples'], reverse=True)
    
    print(f"{'BT':>4} {'Samples':>10} {'AUC':>10}")
    print("-"*30)
    for r in results[:5]:
        print(f"{r['business_type']:>4} {r['samples']:>10,} {r['auc']:>10.4f}")
    
    return results


def main():
    print("="*80)
    print("实验 01: Bid Landscape Forecasting Baseline")
    print("="*80)
    
    # 加载数据
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    df = load_data(str(data_path), max_samples=500000)  # 先用 50 万样本
    
    # 划分训练/验证集
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['win_label'])
    
    print(f"\nDataset split:")
    print(f"  Train: {len(df_train)}")
    print(f"  Val:   {len(df_val)}")
    
    # 准备特征
    X_train, y_train = prepare_features(df_train)
    X_val, y_val = prepare_features(df_val)
    
    # 运行基线模型
    lr_model, lr_metrics = run_logistic_regression(X_train, X_val, y_train, y_val)
    xgb_model, xgb_metrics = run_xgboost(X_train, X_val, y_train, y_val)
    qr_models, qr_metrics = run_quantile_regression(X_train, X_val, y_train, y_val)
    
    # 按 business_type 评估
    print("\n" + "="*80)
    print("Per Business Type Evaluation")
    print("="*80)
    
    lr_by_bt = evaluate_by_business_type(df_val, lr_model, "Logistic Regression")
    xgb_by_bt = evaluate_by_business_type(df_val, xgb_model, "XGBoost")
    
    # 保存结果
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'logistic_regression': lr_metrics,
        'xgboost': xgb_metrics,
        'quantile_regression': qr_metrics,
        'by_business_type': {
            'logistic_regression': lr_by_bt,
            'xgboost': xgb_by_bt
        }
    }
    
    with open(results_dir / 'exp01_baseline.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 生成报告
    report = f"""# 实验 01: Bid Landscape Forecasting Baseline

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 方法对比

| Method | AUC | RMSE | MAE | ECE |
|--------|-----|------|-----|-----|
| Logistic Regression | {lr_metrics['auc']:.4f} | {lr_metrics['rmse']:.4f} | {lr_metrics['mae']:.4f} | {lr_metrics.get('ece', 'N/A'):.4f} |
| XGBoost | {xgb_metrics['auc']:.4f} | {xgb_metrics['rmse']:.4f} | {xgb_metrics['mae']:.4f} | {xgb_metrics.get('ece', 'N/A'):.4f} |
| Quantile Regression (median) | {qr_metrics['overall']['auc']:.4f} | {qr_metrics['overall']['rmse']:.4f} | {qr_metrics['overall']['mae']:.4f} | N/A |

## 核心发现

- **最佳模型**: {"XGBoost" if xgb_metrics['auc'] > lr_metrics['auc'] else "Logistic Regression"} (AUC={max(lr_metrics['auc'], xgb_metrics['auc']):.4f})
- **校准效果**: {"良好" if lr_metrics.get('ece', 1) < 0.05 else "需要改进"}
- **业务类型差异**: Top BT AUC 范围 [{min(r['auc'] for r in lr_by_bt):.4f}, {max(r['auc'] for r in lr_by_bt):.4f}]

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp01_baseline.md', 'w') as f:
        f.write(report)
    
    print(f"\n✅ 结果已保存到 {results_dir}")
    print(f"   - exp01_baseline.json")
    print(f"   - exp01_baseline.md")


if __name__ == '__main__':
    main()
