#!/usr/bin/env python3
"""
Baseline Models for LTV Prediction

实现四个基线模型:
1. Linear Regression
2. XGBoost
3. Two-stage Model (Classification + Regression)
4. Simple DNN
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import json
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """基线模型集合"""
    
    def __init__(self):
        self.models = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """准备训练数据"""
        # 特征列（排除 user_id 和 ltv_value）
        feature_cols = [col for col in df.columns if col not in ['user_id', 'ltv_value']]
        
        X = df[feature_cols].values.astype(np.float32)
        y = df['ltv_value'].values.astype(np.float32)
        
        return X, y, feature_cols
    
    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """1. Linear Regression Baseline"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 预测（截断负值）
        y_pred = np.maximum(0, model.predict(X_val))
        
        metrics = self.evaluate(y_val, y_pred, model_type='linear')
        metrics['model'] = 'LinearRegression'
        
        self.models['linear'] = model
        return metrics
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """2. XGBoost Baseline"""
        try:
            from xgboost import XGBRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # 预测（截断负值）
            y_pred = np.maximum(0, model.predict(X_val))
            
            metrics = self.evaluate(y_val, y_pred, model_type='xgboost')
            metrics['model'] = 'XGBoost'
            
            self.models['xgboost'] = model
            return metrics
        except ImportError:
            print("XGBoost not installed, skipping...")
            return None
    
    def train_two_stage(self, X_train, y_train, X_val, y_val):
        """3. Two-stage Model (Classification + Regression)"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
        
        # Stage 1: 分类 - 是否购买
        y_binary = (y_train > 0).astype(int)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_binary)
        
        # Stage 2: 回归 - 只训练购买用户
        payer_mask = y_train > 0
        if payer_mask.sum() > 0:
            reg = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            reg.fit(X_train[payer_mask], y_train[payer_mask])
        else:
            reg = None
        
        # 预测
        prob_buy = clf.predict_proba(X_val)[:, 1]
        
        if reg is not None:
            y_pred_amount = reg.predict(X_val)
            y_pred = prob_buy * np.maximum(0, y_pred_amount)
        else:
            y_pred = np.zeros(len(X_val))
        
        metrics = self.evaluate(y_val, y_pred, prob_pred=prob_buy, model_type='two_stage')
        metrics['model'] = 'TwoStage'
        
        self.models['two_stage'] = (clf, reg)
        return metrics
    
    def train_dnn(self, X_train, y_train, X_val, y_val):
        """4. Simple DNN Baseline"""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import StandardScaler
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # 创建 DataLoader
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_scaled),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_scaled),
                torch.FloatTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            
            # 定义模型
            class SimpleDNN(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_dim, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Softplus()  # 保证输出非负
                    )
                
                def forward(self, x):
                    return self.network(x).squeeze()
            
            model = SimpleDNN(X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # 训练 (1 epoch，遵循 research_notes.md 规范)
            model.train()
            for epoch in range(1):  # 只跑 1 epoch
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # 评估
            model.eval()
            with torch.no_grad():
                val_X = torch.FloatTensor(X_val_scaled)
                y_pred = model(val_X).numpy()
            
            metrics = self.evaluate(y_val, y_pred, model_type='dnn')
            metrics['model'] = 'SimpleDNN'
            metrics['epochs'] = 1
            
            self.models['dnn'] = (model, scaler)
            return metrics
            
        except ImportError:
            print("PyTorch not installed, skipping DNN...")
            return None
    
    def evaluate(self, 
                 y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 prob_pred: np.ndarray = None,
                 model_type: str = 'regression') -> Dict[str, float]:
        """
        评估模型性能
        
        指标包括:
        - AUC (如果有概率预测)
        - PCOC@K (K=10%, 20%, 30%)
        - RMSE, MAE
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
        
        metrics = {}
        
        # AUC (付费预测)
        if prob_pred is not None:
            y_binary = (y_true > 0).astype(int)
            if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
                metrics['auc'] = roc_auc_score(y_binary, prob_pred)
        
        # PCOC@K (Percentage of Cumulative Ordered Coverage)
        for k in [0.1, 0.2, 0.3]:
            top_k_idx = np.argsort(y_pred)[-int(len(y_pred) * k):]
            top_k_actual_ltv = y_true[top_k_idx].sum()
            total_ltv = y_true.sum()
            metrics[f'pcoc_{int(k*100)}'] = top_k_actual_ltv / total_ltv if total_ltv > 0 else 0
        
        # Regression metrics
        metrics['rmse'] = mean_squared_error(y_true, y_pred, squared=True) ** 0.5
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # 只在付费用户上计算回归指标
        payer_mask = y_true > 0
        if payer_mask.sum() > 0:
            metrics['rmse_payers'] = mean_squared_error(y_true[payer_mask], y_pred[payer_mask], squared=True) ** 0.5
            metrics['mae_payers'] = mean_absolute_error(y_true[payer_mask], y_pred[payer_mask])
        
        return metrics


def run_all_baselines(data_path: str = '/mnt/workspace/open_research/autoresearch/ltv_optimize/data/train_data.parquet'):
    """运行所有基线模型"""
    print("="*80)
    print("Running Baseline Models for LTV Prediction")
    print("="*80)
    
    # 加载数据
    print("\nLoading data...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples")
    
    # 划分训练集和验证集 (80/20)
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    
    # 初始化模型
    baselines = BaselineModels()
    
    # 准备数据
    X_train, y_train, feature_cols = baselines.prepare_data(df_train)
    X_val, y_val, _ = baselines.prepare_data(df_val)
    
    results = []
    
    # 运行各个基线
    print("\n" + "="*80)
    print("1. Linear Regression")
    print("="*80)
    result = baselines.train_linear_regression(X_train, y_train, X_val, y_val)
    if result:
        results.append(result)
        print(f"AUC: {result.get('auc', 'N/A'):.4f}")
        print(f"PCOC@10: {result.get('pcoc_10', 'N/A'):.4f}")
        print(f"RMSE: {result.get('rmse', 'N/A'):.4f}")
    
    print("\n" + "="*80)
    print("2. XGBoost")
    print("="*80)
    result = baselines.train_xgboost(X_train, y_train, X_val, y_val)
    if result:
        results.append(result)
        print(f"AUC: {result.get('auc', 'N/A'):.4f}")
        print(f"PCOC@10: {result.get('pcoc_10', 'N/A'):.4f}")
        print(f"RMSE: {result.get('rmse', 'N/A'):.4f}")
    
    print("\n" + "="*80)
    print("3. Two-stage Model")
    print("="*80)
    result = baselines.train_two_stage(X_train, y_train, X_val, y_val)
    if result:
        results.append(result)
        print(f"AUC: {result.get('auc', 'N/A'):.4f}")
        print(f"PCOC@10: {result.get('pcoc_10', 'N/A'):.4f}")
        print(f"RMSE: {result.get('rmse', 'N/A'):.4f}")
    
    print("\n" + "="*80)
    print("4. Simple DNN")
    print("="*80)
    result = baselines.train_dnn(X_train, y_train, X_val, y_val)
    if result:
        results.append(result)
        print(f"AUC: {result.get('auc', 'N/A'):.4f}")
        print(f"PCOC@10: {result.get('pcoc_10', 'N/A'):.4f}")
        print(f"RMSE: {result.get('rmse', 'N/A'):.4f}")
    
    # 保存结果
    output_dir = Path('/mnt/workspace/open_research/autoresearch/ltv_optimize/results/exp001_baseline')
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")
    
    # 打印对比表格
    print("\n" + "="*80)
    print("Baseline Comparison")
    print("="*80)
    print(f"{'Model':<20} {'AUC':>10} {'PCOC@10':>10} {'RMSE':>10}")
    print("-"*50)
    for r in results:
        auc = r.get('auc', 0)
        pcoc = r.get('pcoc_10', 0)
        rmse = r.get('rmse', 0)
        print(f"{r['model']:<20} {auc:>10.4f} {pcoc:>10.4f} {rmse:>10.4f}")
    
    return results


if __name__ == "__main__":
    run_all_baselines()
