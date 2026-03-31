"""
生成 Synthetic Bid Landscape 数据

从 CTR 数据推导 bid landscape:
1. 假设每个样本有一个"真实价值" v
2. 生成多个 bid levels
3. 计算 win probability: P(win|b) = sigmoid(k * (b - v))
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def generate_bid_landscape(
    ctr_data_path: str,
    output_path: str,
    n_bid_levels: int = 5,
    noise_scale: float = 0.3,
    seed: int = 42
):
    """
    从 CTR 数据生成 bid landscape
    
    Args:
        ctr_data_path: IVR CTR 数据路径
        output_path: 输出路径
        n_bid_levels: 每个样本生成的 bid 数量
        noise_scale: 噪声尺度
        seed: 随机种子
    """
    np.random.seed(seed)
    
    print(f"加载数据：{ctr_data_path}")
    df = pd.read_parquet(ctr_data_path)
    print(f"原始样本数：{len(df)}")
    
    # 步骤 1: 估计"真实价值" v
    # 假设 CTR ≈ P(win) × constant, 所以 v ∝ CTR
    ctr_col = 'click_label' if 'click_label' in df.columns else 'label'
    
    # 使用 Beta distribution 拟合 CTR
    # v ~ Beta(α * CTR + 1, β * (1-CTR) + 1)
    alpha_scale, beta_scale = 10, 10
    ctr_values = df[ctr_col].values
    
    # 生成真实价值 v (添加噪声)
    alphas = alpha_scale * ctr_values + 1
    betas = beta_scale * (1 - ctr_values) + 1
    true_values = np.array([
        np.random.beta(a, b) for a, b in zip(alphas, betas)
    ])
    
    print(f"真实价值统计：mean={true_values.mean():.4f}, std={true_values.std():.4f}")
    
    # 步骤 2: 生成 bid levels
    print(f"\n生成 {n_bid_levels} 个 bid levels...")
    
    all_samples = []
    
    for idx in tqdm(range(len(df)), desc="Generating bid data"):
        row = df.iloc[idx]
        v = true_values[idx]
        
        # 生成围绕真实价值的 bids
        # bid ∈ {0.5v, 0.7v, v, 1.3v, 1.5v} + noise
        bid_ratios = np.linspace(0.5, 1.5, n_bid_levels)
        bids = v * bid_ratios + np.random.normal(0, noise_scale * v, n_bid_levels)
        bids = np.clip(bids, 0.01, None)  # bid > 0
        
        # 步骤 3: 计算 win probability
        # P(win|b) = sigmoid(k * (b - v))
        k = 5.0  # steepness parameter
        win_probs = 1 / (1 + np.exp(-k * (bids - v)))
        
        # 添加标签噪声
        win_labels = (np.random.uniform(size=n_bid_levels) < win_probs).astype(int)
        
        # 保存样本
        for i in range(n_bid_levels):
            sample = row.to_dict()
            sample.update({
                'true_value': v,
                'bid_amount': bids[i],
                'win_prob': win_probs[i],
                'win_label': win_labels[i]
            })
            all_samples.append(sample)
    
    # 转换为 DataFrame
    result_df = pd.DataFrame(all_samples)
    
    print(f"\n生成的 bid landscape 数据:")
    print(f"  总样本数：{len(result_df)}")
    print(f"  每原始样本平均：{len(result_df) / len(df):.1f} 个 bids")
    print(f"  Win rate: {result_df['win_label'].mean():.4f}")
    print(f"  Bid range: [{result_df['bid_amount'].min():.4f}, {result_df['bid_amount'].max():.4f}]")
    
    # 保存
    result_df.to_parquet(output_path, index=False)
    print(f"\n✅ 已保存到：{output_path}")
    
    return result_df


def analyze_bid_distribution(df: pd.DataFrame):
    """分析 bid 分布"""
    print("\n" + "="*60)
    print("Bid Landscape 数据分析")
    print("="*60)
    
    # 基本统计
    print(f"\n样本数：{len(df):,}")
    print(f"特征数：{len(df.columns)}")
    
    # Bid 分布
    print(f"\nBid Amount 统计:")
    print(f"  Mean: {df['bid_amount'].mean():.4f}")
    print(f"  Std: {df['bid_amount'].std():.4f}")
    print(f"  Min: {df['bid_amount'].min():.4f}")
    print(f"  Max: {df['bid_amount'].max():.4f}")
    print(f"  Median: {df['bid_amount'].median():.4f}")
    
    # Win rate
    print(f"\nWin Rate: {df['win_label'].mean():.4f} ({df['win_label'].sum():,} wins)")
    
    # Bid vs Win correlation
    corr = df['bid_amount'].corr(df['win_label'])
    print(f"\nBid-Win Correlation: {corr:.4f}")
    
    # 分 bin 统计
    print(f"\nBid Distribution (by decile):")
    for i in range(10):
        mask = df['bid_amount'].between(
            df['bid_amount'].quantile(i/10),
            df['bid_amount'].quantile((i+1)/10)
        )
        win_rate = df.loc[mask, 'win_label'].mean()
        count = mask.sum()
        print(f"  Bin {i+1}: {count:>8,} samples, win_rate={win_rate:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                       default='/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/train_0.parquet')
    parser.add_argument('--output', type=str,
                       default='/mnt/workspace/open_research/autoresearch/bid_landscape_forecasting/data/bid_landscape_train.parquet')
    parser.add_argument('--n-bids', type=int, default=5)
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 生成数据
    df = generate_bid_landscape(
        args.input,
        args.output,
        n_bid_levels=args.n_bids,
        noise_scale=args.noise,
        seed=args.seed
    )
    
    # 分析
    analyze_bid_distribution(df)
