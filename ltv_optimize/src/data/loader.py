#!/usr/bin/env python3
"""
LTV Optimization - Data Loading and Preprocessing

根据 research_notes.md 规范：
- LTV 项目可自主选择数据集（不强制使用 IVR）
- 本实现使用 Taobao UserBehavior Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import json


class LTVDataLoader:
    """Taobao UserBehavior 数据加载器"""
    
    def __init__(self, data_path: str = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/taobao/UserBehavior.csv'):
        self.data_path = Path(data_path)
        self.raw_data = None
        
    def load_raw_data(self, sample_size: int = None) -> pd.DataFrame:
        """加载原始数据
        
        Args:
            sample_size: 如果指定，则只加载前 sample_size 条记录用于快速测试
        """
        if self.raw_data is not None:
            return self.raw_data
            
        print(f"Loading data from {self.data_path}...", end=" ")
        
        if sample_size:
            # 只加载前 N 行用于快速测试
            self.raw_data = pd.read_csv(
                self.data_path,
                header=None,
                names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
                nrows=sample_size
            )
            print(f"Loaded SAMPLE of {len(self.raw_data):,} records (first {sample_size})")
        else:
            # 全量加载 - 使用 chunked 读取节省内存
            chunks = []
            chunk_size = 10_000_000  # 每次读取 1000 万行
            
            for chunk in pd.read_csv(
                self.data_path,
                header=None,
                names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
                chunksize=chunk_size
            ):
                chunks.append(chunk)
                print(f"Loaded {len(chunks)*chunk_size:,} records...", end="\r")
            
            self.raw_data = pd.concat(chunks, ignore_index=True)
            print(f"\nLoaded {len(self.raw_data):,} records total")
        
        return self.raw_data
    
    def preprocess(self, 
                   train_days: int = 6,
                   test_days: int = 3,
                   ltv_window_days: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        数据预处理和时间窗口划分
        
        Args:
            train_days: 训练集天数
            test_days: 测试集天数
            ltv_window_days: LTV 预测窗口（未来多少天的购买次数）
            
        Returns:
            train_df, val_df, test_df
        """
        df = self.load_raw_data()
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['timestamp'].dt.date
        
        # 按时间排序
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # 获取时间范围
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        print(f"Data range: {min_date} to {max_date}")
        print(f"Total days: {(max_date - min_date).days}")
        
        # 划分时间窗口 (简化版，实际需要更精细的处理)
        # 这里使用前 70% 作为训练，15% 验证，15% 测试
        all_dates = sorted(df['date'].unique())
        split_point_1 = int(len(all_dates) * 0.7)
        split_point_2 = int(len(all_dates) * 0.85)
        
        train_end_date = all_dates[split_point_1]
        val_end_date = all_dates[split_point_2]
        
        train_df = df[df['date'] <= train_end_date].copy()
        val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)].copy()
        test_df = df[df['date'] > val_end_date].copy()
        
        print(f"Train: {len(train_df):,} records, {train_df['user_id'].nunique():,} users")
        print(f"Val: {len(val_df):,} records, {val_df['user_id'].nunique():,} users")
        print(f"Test: {len(test_df):,} records, {test_df['user_id'].nunique():,} users")
        
        return train_df, val_df, test_df
    
    def create_ltv_labels(self, 
                         df: pd.DataFrame, 
                         window_days: int = 7,
                         cutoff_date = None) -> pd.DataFrame:
        """
        为每个用户创建 LTV 标签（未来 window_days 内的购买次数）
        
        Args:
            df: 用户行为数据
            window_days: LTV 预测窗口天数
            cutoff_date: 观察点日期，如果为 None 则使用数据集的 70% 时间点
        """
        from datetime import timedelta
        
        print("Calculating LTV labels...")
        
        # 如果未指定 cutoff_date，使用数据集时间范围的 70% 处
        if cutoff_date is None:
            all_dates = sorted(df['date'].unique())
            cutoff_idx = int(len(all_dates) * 0.7)
            cutoff_date = all_dates[cutoff_idx]
            print(f"Using automatic cutoff date: {cutoff_date}")
        else:
            print(f"Using specified cutoff date: {cutoff_date}")
        
        ltv_end_date = cutoff_date + timedelta(days=window_days)
        print(f"LTV window: {cutoff_date} to {ltv_end_date}")
        
        # 按用户分组计算 LTV
        def calculate_user_ltv(user_df):
            """计算单个用户的 LTV"""
            user_id = user_df['user_id'].iloc[0]
            
            # 观察点之后的购买行为
            future_buys = user_df[
                (user_df['behavior_type'] == 'buy') &
                (user_df['date'] > cutoff_date) &
                (user_df['date'] <= ltv_end_date)
            ]
            
            return pd.Series({
                'user_id': user_id,
                'ltv_value': len(future_buys)
            })
        
        # 只处理在 cutoff_date 之前有行为的用户
        historical_users = df[df['date'] <= cutoff_date]['user_id'].unique()
        historical_df = df[df['user_id'].isin(historical_users)]
        
        # 对用户应用 LTV 计算
        ltv_labels = historical_df.groupby('user_id').apply(calculate_user_ltv).reset_index(drop=True)
        
        print(f"LTV distribution:")
        print(f"  Total users: {len(ltv_labels):,}")
        print(f"  Zero-inflation: {(ltv_labels['ltv_value'] == 0).sum() / len(ltv_labels) * 100:.2f}%")
        print(f"  Non-zero users: {(ltv_labels['ltv_value'] > 0).sum():,}")
        print(f"  Mean LTV: {ltv_labels['ltv_value'].mean():.2f}")
        print(f"  Max LTV: {ltv_labels['ltv_value'].max()}")
        print(f"  LTV value range: {ltv_labels['ltv_value'].min()} - {ltv_labels['ltv_value'].max()}")
        
        return ltv_labels[['user_id', 'ltv_value']]
    
    def extract_features(self, 
                        df: pd.DataFrame,
                        ltv_labels: pd.DataFrame) -> pd.DataFrame:
        """
        提取用户特征
        
        返回的特征包括:
        - 用户历史行为统计 (pv/cart/fav/buy 次数)
        - 行为比率特征
        - 活跃度特征
        """
        # 用户级行为统计
        user_stats = df.groupby('user_id').agg({
            'item_id': 'count',  # 总交互数
            'behavior_type': lambda x: (x == 'pv').sum(),  # pv 次数
        }).reset_index()
        
        # 各种行为次数
        for behavior in ['pv', 'cart', 'fav', 'buy']:
            user_stats[f'{behavior}_count'] = (df[df['behavior_type'] == behavior]['user_id'].value_counts()).reindex(
                user_stats['user_id']
            ).fillna(0).values
        
        # 唯一物品数
        unique_items = df.groupby('user_id')['item_id'].nunique().reset_index()
        unique_items.columns = ['user_id', 'unique_items']
        user_stats = user_stats.merge(unique_items, on='user_id', how='left')
        
        # 合并 LTV 标签
        user_stats = user_stats.merge(ltv_labels, on='user_id', how='left')
        user_stats['ltv_value'] = user_stats['ltv_value'].fillna(0).astype(int)
        
        # 衍生特征
        user_stats['cart_rate'] = user_stats['cart_count'] / (user_stats['pv_count'] + 1)
        user_stats['buy_rate'] = user_stats['buy_count'] / (user_stats['cart_count'] + 1)
        user_stats['engagement_score'] = (
            user_stats['pv_count'] * 0.1 + 
            user_stats['cart_count'] * 0.3 + 
            user_stats['fav_count'] * 0.2 + 
            user_stats['buy_count'] * 0.4
        )
        
        print(f"Extracted {len(user_stats)} user profiles with {len(user_stats.columns)-2} features")
        
        return user_stats


def main():
    """测试数据加载和预处理流程"""
    print("="*80)
    print("LTV Data Pipeline Test")
    print("="*80)
    
    # 初始化数据加载器
    loader = LTVDataLoader()
    
    # 预处理
    train_df, val_df, test_df = loader.preprocess(train_days=6, test_days=3, ltv_window_days=7)
    
    # 创建 LTV 标签
    print("\nCreating LTV labels...")
    train_labels = loader.create_ltv_labels(train_df, window_days=7)
    
    # 提取特征
    print("\nExtracting features...")
    train_features = loader.extract_features(train_df, train_labels)
    
    # 保存处理后的数据
    output_dir = Path('/mnt/workspace/open_research/autoresearch/ltv_optimize/data')
    output_dir.mkdir(exist_ok=True)
    
    train_features.to_parquet(output_dir / 'train_data.parquet', index=False)
    print(f"\nSaved training data to {output_dir / 'train_data.parquet'}")
    
    # 打印特征列表
    print("\nFeature list:")
    feature_cols = [col for col in train_features.columns if col not in ['user_id', 'ltv_value']]
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nTotal features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
