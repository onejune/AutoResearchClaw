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
        
    def load_raw_data(self) -> pd.DataFrame:
        """加载原始数据"""
        if self.raw_data is not None:
            return self.raw_data
            
        print(f"Loading data from {self.data_path}...")
        # CSV 没有 header，需要指定列名
        self.raw_data = pd.read_csv(
            self.data_path,
            header=None,
            names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
        )
        print(f"Loaded {len(self.raw_data):,} records")
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
                         window_days: int = 7) -> pd.DataFrame:
        """
        为每个用户创建 LTV 标签（未来 window_days 内的购买次数）
        """
        from datetime import timedelta
        
        # 计算每个用户的最后行为日期
        user_last_date = df.groupby('user_id')['date'].max().reset_index()
        user_last_date.columns = ['user_id', 'last_behavior_date']
        
        # 先合并到完整数据集
        df_with_last = df.merge(user_last_date, on='user_id', how='left')
        
        # 计算 LTV 窗口结束日期
        df_with_last['ltv_end_date'] = df_with_last['last_behavior_date'] + timedelta(days=window_days)
        
        # 筛选购买行为
        buys = df_with_last[df_with_last['behavior_type'] == 'buy'][['user_id', 'date', 'last_behavior_date', 'ltv_end_date']].copy()
        
        # 标记在窗口内的购买
        buys['in_window'] = (buys['date'] > buys['last_behavior_date']) & (buys['date'] <= buys['ltv_end_date'])
        
        # 计算每个用户在窗口内的购买次数
        ltv_labels = buys[buys['in_window']].groupby('user_id').size().reset_index(name='ltv_value')
        
        # 合并所有用户（包括 LTV=0 的）
        all_users = pd.DataFrame({'user_id': df['user_id'].unique()})
        ltv_labels = all_users.merge(ltv_labels, on='user_id', how='left')
        ltv_labels['ltv_value'] = ltv_labels['ltv_value'].fillna(0).astype(int)
        
        print(f"LTV distribution:")
        print(f"  Zero-inflation: {(ltv_labels['ltv_value'] == 0).sum() / len(ltv_labels) * 100:.2f}%")
        print(f"  Mean LTV: {ltv_labels['ltv_value'].mean():.2f}")
        print(f"  Max LTV: {ltv_labels['ltv_value'].max()}")
        
        return ltv_labels
    
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
