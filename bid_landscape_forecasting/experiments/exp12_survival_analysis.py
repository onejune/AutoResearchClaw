"""
实验 12: Survival Analysis for Bid Landscape Forecasting

核心思想:
- 将 RTB 竞价建模为生存分析问题
- "生存时间" T = 最高竞争出价 (highest competing bid)
- "事件发生" = 赢标 (观察到完整 T)
- "右删失" = 输标 (只知道 T >= 我的出价)

实现方法:
1. Cox Proportional Hazards Model - 经典比例风险模型
2. Deep Cox - 神经网络 + Cox Loss
3. DeepHit - 端到端深度生存模型

数据说明:
- 赢标 (win=1): event=1, time=market_price (观察到完整信息)
- 输标 (win=0): event=0, time=bid (删失，真实 market_price >= bid)
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def compute_ece(y_true, y_prob, n_bins=10):
    """计算 Expected Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)


# ============================================================================
# 模型定义
# ============================================================================

class CoxProportionalHazards(nn.Module):
    """
    Cox 比例风险模型
    
    风险函数：h(t|x) = h0(t) * exp(x^T * beta)
    
    在竞价场景中:
    - x: 特征向量 (包括 bid 和上下文特征)
    - h0(t): 基线风险函数 (非参数，通过偏似然估计消除)
    - 输出：风险评分 r(x) = x^T * beta
    
    Cox Partial Likelihood (处理删失):
    L = prod_{i: event_i=1} [exp(r(x_i)) / sum_{j in R(t_i)} exp(r(x_j))]
    
    其中 R(t_i) 是在时间 t_i 仍处于风险中的样本集合
    """
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze(-1)
    
    def cox_loss(self, risk_scores, times, events):
        """
        Cox Partial Likelihood Loss
        
        Args:
            risk_scores: [N] 风险评分 r(x)
            times: [N] 观测时间 (market_price 或 bid)
            events: [N] 事件指示 (1=赢标/事件发生，0=删失)
        
        Returns:
            loss: Cox partial likelihood 的负对数
        """
        # 按时间排序
        sorted_idx = torch.argsort(times, descending=True)
        risk_sorted = risk_scores[sorted_idx]
        events_sorted = events[sorted_idx]
        
        # 计算分母：累积和 sum_{j in R(t_i)} exp(r(x_j))
        exp_risk = torch.exp(risk_sorted)
        cumulative_sum = torch.cumsum(exp_risk, dim=0)
        
        # 计算分子：event_i * r(x_i)
        log_numerators = risk_sorted * events_sorted
        
        # 计算分母的对数：log(sum_{j in R(t_i)} exp(r(x_j)))
        log_denominators = torch.log(cumulative_sum + 1e-10)
        
        # Cox partial log-likelihood
        log_likelihood = torch.sum(log_numerators - log_denominators)
        
        # 返回负对数似然 (最小化)
        return -log_likelihood / (events.sum() + 1e-10)


class DeepCoxModel(nn.Module):
    """
    Deep Cox 模型：用神经网络替代线性风险函数
    
    保持 Cox 的部分似然框架，但用深度学习捕捉非线性关系
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.15)
            ])
            prev_dim = h
        
        self.backbone = nn.Sequential(*layers)
        self.risk_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        h = self.backbone(x)
        return self.risk_head(h).squeeze(-1)
    
    def cox_loss(self, risk_scores, times, events):
        """与 CoxProportionalHazards 相同的 loss"""
        sorted_idx = torch.argsort(times, descending=True)
        risk_sorted = risk_scores[sorted_idx]
        events_sorted = events[sorted_idx]
        
        exp_risk = torch.exp(risk_sorted)
        cumulative_sum = torch.cumsum(exp_risk, dim=0)
        
        log_numerators = risk_sorted * events_sorted
        log_denominators = torch.log(cumulative_sum + 1e-10)
        
        log_likelihood = torch.sum(log_numerators - log_denominators)
        return -log_likelihood / (events.sum() + 1e-10)


class DeepHitNetwork(nn.Module):
    """
    DeepHit: 端到端深度生存模型
    
    核心思想:
    - 直接预测离散时间的生存概率 S(t|x)
    - 使用 cause-specific loss + ranking loss
    
    在竞价场景中 (单事件类型):
    - 将出价空间离散化为 K 个区间
    - 预测每个区间的概率质量函数 f(t_k|x)
    - 生存函数 S(t_k|x) = sum_{j>k} f(t_j|x)
    """
    
    def __init__(self, input_dim, num_time_bins=50, hidden_dims=[128, 64]):
        super().__init__()
        self.num_time_bins = num_time_bins
        
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h
        
        self.backbone = nn.Sequential(*layers)
        
        # 输出：每个时间 bin 的概率质量
        self.pmf_head = nn.Sequential(
            nn.Linear(prev_dim, num_time_bins),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        h = self.backbone(x)
        pmf = self.pmf_head(h)  # [N, K]
        return pmf
    
    def survival_function(self, pmf):
        """从 PMF 计算生存函数 S(t) = P(T > t)"""
        # S(t_k) = sum_{j=k+1}^{K} f(t_j)
        return torch.cumsum(torch.flip(pmf, [-1]), dim=-1).flip([-1])
    
    def deephit_loss(self, pmf, times, events, alpha=0.5, sigma=0.1):
        """
        DeepHit Loss = alpha * log_likelihood + (1-alpha) * ranking_loss
        
        Args:
            pmf: [N, K] 预测的概率质量函数
            times: [N] 观测时间
            events: [N] 事件指示
            alpha: likelihood 和 ranking loss 的权重
            sigma: ranking loss 的平滑参数
        """
        N, K = pmf.shape
        
        # 1. Log-likelihood loss (考虑删失)
        # 找到每个样本对应的时间 bin
        device = pmf.device
        time_bins = torch.linspace(0, 1, K+1).to(device)  # 归一化的时间边界
        
        # 简化的 likelihood：使用交叉熵
        eps = 1e-10
        log_likelihood = 0.0
        
        for i in range(N):
            if events[i] == 1:
                # 事件发生：最大化该时间点的概率
                # 找到对应的 bin
                t_norm = times[i] / times.max()
                bin_idx = min(int(t_norm * K), K-1)
                log_likelihood += torch.log(pmf[i, bin_idx] + eps)
            else:
                # 删失：最大化生存到该时间的概率
                t_norm = times[i] / times.max()
                bin_idx = min(int(t_norm * K), K-1)
                surv_prob = torch.sum(pmf[i, bin_idx+1:]) + eps
                log_likelihood += torch.log(surv_prob)
        
        log_likelihood /= N
        
        # 2. Ranking loss (可选，帮助区分不同风险的样本)
        ranking_loss = 0.0
        count = 0
        for i in range(N):
            for j in range(i+1, N):
                if events[i] == 1 and times[i] < times[j]:
                    # 样本 i 先发生事件，应该有更高的风险
                    risk_i = torch.sum(pmf[i] * torch.arange(K).to(device))
                    risk_j = torch.sum(pmf[j] * torch.arange(K).to(device))
                    diff = risk_j - risk_i
                    ranking_loss += torch.log(1 + torch.exp(diff / sigma))
                    count += 1
        
        if count > 0:
            ranking_loss /= count
        else:
            ranking_loss = 0.0
        
        # 总损失
        total_loss = -alpha * log_likelihood + (1 - alpha) * ranking_loss
        return total_loss


# ============================================================================
# 主实验流程
# ============================================================================

def load_data():
    """加载合成数据"""
    data_dir = project_root / 'data'
    
    # 尝试加载小数据集
    data_path = data_dir / 'bid_landscape_train_small.parquet'
    if not data_path.exists():
        data_path = data_dir / 'bid_landscape_train.parquet'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    return df


def prepare_survival_data(df):
    """
    准备生存分析格式的数据
    
    转换:
    - event: win=1 → event=1 (观察到完整信息)
    - event: win=0 → event=0 (删失)
    - time: 赢标时用 true_value/market_price, 输标时用 bid (作为下界)
    
    数据列名映射:
    - win_label 或 win → event (1=赢标，0=输标)
    - true_value 或 market_price → 真实市场价
    - bid_amount 或 bid → 出价
    """
    # 列名映射 (兼容不同数据集)
    win_col = 'win_label' if 'win_label' in df.columns else ('win' if 'win' in df.columns else None)
    value_col = 'true_value' if 'true_value' in df.columns else ('market_price' if 'market_price' in df.columns else None)
    bid_col = 'bid_amount' if 'bid_amount' in df.columns else ('bid' if 'bid' in df.columns else None)
    
    if win_col is None:
        raise ValueError("Cannot find win column (win_label or win)")
    
    # 特征列 (排除目标列和标识列)
    exclude_cols = ['click_label', 'ctcvr_label', 'win', 'win_label', 'market_price', 'bid', 'bid_amount', 'true_value', 'win_prob']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 如果没有足够的特征，使用数值列
    if len(feature_cols) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude_cols][:20]
    
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    
    # 生存分析格式
    # event: 1=赢标 (事件发生), 0=输标 (删失)
    events = df[win_col].values.astype(np.float32)
    
    # time: 赢标用 true_value/market_price, 输标用 bid (作为下界)
    if value_col and bid_col:
        times = np.where(events == 1, 
                        df[value_col].values,
                        df[bid_col].values)
    elif value_col:
        # 只有 true_value，输标时也用 true_value (因为这是合成数据)
        times = df[value_col].values
    elif bid_col:
        times = df[bid_col].values
    else:
        times = np.ones(len(df))
    
    times = times.astype(np.float32)
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 归一化时间到 [0, 1]
    times = (times - times.min()) / (times.max() - times.min() + 1e-10)
    
    return X, times, events, feature_cols, scaler


def train_cox(X_train, times_train, events_train, X_val, times_val, events_val, config):
    """训练 Cox 模型"""
    print("\n" + "="*60)
    print("Training Cox Proportional Hazards Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 转换为 tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    times_train_t = torch.FloatTensor(times_train).to(device)
    events_train_t = torch.FloatTensor(events_train).to(device)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    times_val_t = torch.FloatTensor(times_val).to(device)
    events_val_t = torch.FloatTensor(events_val).to(device)
    
    # 创建模型
    input_dim = X_train.shape[1]
    model = CoxProportionalHazards(input_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.01))
    
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 50)
    
    dataset = TensorDataset(X_train_t, times_train_t, events_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_times, batch_events in loader:
            optimizer.zero_grad()
            
            risk_scores = model(batch_X)
            loss = model.cox_loss(risk_scores, batch_times, batch_events)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_risk = model(X_val_t)
            
            # 将风险分数转换为 win 概率 (用于 AUC 计算)
            # 高风险 = 高概率赢标
            val_probs = torch.sigmoid(val_risk).cpu().numpy()
            
            val_auc = roc_auc_score(events_val, val_probs)
            
            # 计算 calibration
            val_ece = compute_ece(events_val, val_probs)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Val AUC: {val_auc:.4f} | Val ECE: {val_ece:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    results = {
        'model': 'Cox Proportional Hazards',
        'best_val_auc': best_val_auc,
        'config': config
    }
    
    print(f"\nBest Val AUC: {best_val_auc:.4f}")
    return model, results


def train_deep_cox(X_train, times_train, events_train, X_val, times_val, events_val, config):
    """训练 Deep Cox 模型"""
    print("\n" + "="*60)
    print("Training Deep Cox Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    times_train_t = torch.FloatTensor(times_train).to(device)
    events_train_t = torch.FloatTensor(events_train).to(device)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    times_val_t = torch.FloatTensor(times_val).to(device)
    events_val_t = torch.FloatTensor(events_val).to(device)
    
    input_dim = X_train.shape[1]
    hidden_dims = config.get('hidden_dims', [64, 32])
    model = DeepCoxModel(input_dim, hidden_dims).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 50)
    
    dataset = TensorDataset(X_train_t, times_train_t, events_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_times, batch_events in loader:
            optimizer.zero_grad()
            
            risk_scores = model(batch_X)
            loss = model.cox_loss(risk_scores, batch_times, batch_events)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_risk = model(X_val_t)
            val_probs = torch.sigmoid(val_risk).cpu().numpy()
            
            val_auc = roc_auc_score(events_val, val_probs)
            val_ece = compute_ece(events_val, val_probs)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Val AUC: {val_auc:.4f} | Val ECE: {val_ece:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    results = {
        'model': 'Deep Cox',
        'best_val_auc': best_val_auc,
        'config': config
    }
    
    print(f"\nBest Val AUC: {best_val_auc:.4f}")
    return model, results


def train_deephit(X_train, times_train, events_train, X_val, times_val, events_val, config):
    """训练 DeepHit 模型"""
    print("\n" + "="*60)
    print("Training DeepHit Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    times_train_t = torch.FloatTensor(times_train).to(device)
    events_train_t = torch.FloatTensor(events_train).to(device)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    input_dim = X_train.shape[1]
    num_time_bins = config.get('num_time_bins', 50)
    hidden_dims = config.get('hidden_dims', [128, 64])
    
    model = DeepHitNetwork(input_dim, num_time_bins, hidden_dims).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 50)
    alpha = config.get('alpha', 0.7)  # likelihood vs ranking weight
    
    dataset = TensorDataset(X_train_t, times_train_t, events_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_times, batch_events in loader:
            optimizer.zero_grad()
            
            pmf = model(batch_X)
            loss = model.deephit_loss(pmf, batch_times, batch_events, alpha=alpha)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_pmf = model(X_val_t)
            val_surv = model.survival_function(val_pmf)
            
            # 从生存函数提取 win 概率
            # 简单方法：使用 S(0.5) 作为阈值
            threshold_idx = num_time_bins // 2
            val_probs = val_surv[:, threshold_idx].cpu().numpy()
            val_probs = np.clip(val_probs, 0, 1)
            
            val_auc = roc_auc_score(events_val, val_probs)
            val_ece = compute_ece(events_val, val_probs)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Val AUC: {val_auc:.4f} | Val ECE: {val_ece:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    results = {
        'model': 'DeepHit',
        'best_val_auc': best_val_auc,
        'config': config
    }
    
    print(f"\nBest Val AUC: {best_val_auc:.4f}")
    return model, results


def main():
    print("="*70)
    print("Experiment 12: Survival Analysis for Bid Landscape Forecasting")
    print("="*70)
    
    # 配置 (使用较小的设置快速测试)
    config = {
        'lr': 0.001,
        'batch_size': 512,
        'epochs': 20,  # 减少 epoch 加快测试
        'hidden_dims': [64, 32],
        'num_time_bins': 30,  # 减少时间 bin
        'alpha': 0.7,
    }
    
    # 加载数据
    df = load_data()
    print(f"Loaded {len(df)} samples")
    
    # 使用小数据集快速测试
    if len(df) > 50000:
        print("Using subset for quick test...")
        df = df.sample(n=50000, random_state=42)
        print(f"Sampled {len(df)} samples")
    
    # 准备生存分析格式的数据
    X, times, events, feature_cols, scaler = prepare_survival_data(df)
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Event rate (win rate): {events.mean():.4f}")
    print(f"Censoring rate: {1 - events.mean():.4f}")
    
    # 划分训练/验证集
    X_train, X_val, times_train, times_val, events_train, events_val = \
        train_test_split(X, times, events, test_size=0.2, random_state=42, stratify=events)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # 训练三个模型
    all_results = []
    
    # 1. Cox PH (快速版本)
    cox_model, cox_results = train_cox(
        X_train, times_train, events_train,
        X_val, times_val, events_val,
        {**config, 'lr': 0.01, 'epochs': 15}
    )
    all_results.append(cox_results)
    
    # 2. Deep Cox
    deep_cox_model, deep_cox_results = train_deep_cox(
        X_train, times_train, events_train,
        X_val, times_val, events_val,
        {**config, 'epochs': 20}
    )
    all_results.append(deep_cox_results)
    
    # 3. DeepHit
    deephit_model, deephit_results = train_deephit(
        X_train, times_train, events_train,
        X_val, times_val, events_val,
        {**config, 'epochs': 20}
    )
    all_results.append(deephit_results)
    
    # 汇总结果
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    for r in all_results:
        print(f"{r['model']:20s} | Val AUC: {r['best_val_auc']:.4f}")
    
    # 保存结果
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'exp12_survival_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'experiments': all_results,
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'win_rate': float(events.mean()),
                'num_features': len(feature_cols),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # 生成 Markdown 报告
    md_report = f"""# Experiment 12: Survival Analysis for Bid Landscape

## Objective
Model bid landscape forecasting as a survival analysis problem to handle right-censored data in RTB.

## Methods
1. **Cox Proportional Hazards**: Classic survival model with linear risk function
2. **Deep Cox**: Neural network + Cox partial likelihood
3. **DeepHit**: End-to-end deep survival model with discrete time

## Results

| Model | Validation AUC |
|-------|---------------|
"""
    
    for r in all_results:
        md_report += f"| {r['model']} | {r['best_val_auc']:.4f} |\n"
    
    md_report += f"""
## Dataset Info
- Total samples: {len(df)}
- Train/Val split: {len(X_train)}/{len(X_val)}
- Win rate: {events.mean():.4f}
- Censoring rate: {1-events.mean():.4f}
- Features: {len(feature_cols)}

## Key Insights
- Survival analysis naturally handles right-censored data
- Cox models provide interpretable risk ratios
- DeepHit can capture complex non-linear patterns
"""
    
    md_file = results_dir / 'exp12_survival_analysis.md'
    with open(md_file, 'w') as f:
        f.write(md_report)
    
    print(f"Report saved to {md_file}")
    print("\n✅ Experiment 12 completed!")


if __name__ == '__main__':
    main()
