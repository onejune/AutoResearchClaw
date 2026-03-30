#!/usr/bin/env python3
"""
IVR 数据预处理脚本 - 用于对比学习 CVR 预估
参考 multitask 项目的数据处理方式
"""

import os
import pickle
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 配置
DATA_DIR = '/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet/'
OUTPUT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr/data/'
SCHEMA_FILE = '/mnt/workspace/open_research/autoresearch/multitask/combine_schema'

# 目标 business_type
TARGET_BUSINESS_TYPES = [
    'shein',
    'aecps', 'aedsp', 'aerta',
    'shopee_cps',
    'lazada_cps', 'lazada_rta'
]

MAX_FILES_PER_DAY = 20
VOCAB_SIZE = 100000  # hash 编码的 vocab 大小


def load_combine_schema(schema_file: str) -> List[str]:
    """加载 combine_schema 特征列表"""
    with open(schema_file, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    logger.info(f'Loaded {len(features)} features from combine_schema')
    return features


def parse_mul_labels(mul_labels_val):
    """解析 mul_labels 列"""
    if mul_labels_val is None or (isinstance(mul_labels_val, (list, tuple)) and len(mul_labels_val) == 0):
        return {}
    try:
        if isinstance(mul_labels_val, list):
            return {k: v for k, v in mul_labels_val}
        return {}
    except:
        return {}


def load_parquet_files(dates: List[str], features: List[str], max_files: int = None) -> pd.DataFrame:
    """加载 parquet 文件"""
    dfs = []
    # 避免重复列
    extra_cols = ['business_type', 'mul_labels', 'revenue']
    need_cols = list(set(features + extra_cols))
    
    for date in tqdm(dates, desc='Loading dates'):
        date_dir = os.path.join(DATA_DIR, date)
        if not os.path.exists(date_dir):
            logger.warning(f'Date dir not found: {date_dir}')
            continue
        
        parquet_files = sorted([f for f in os.listdir(date_dir) if f.endswith('.parquet')])
        if max_files:
            parquet_files = parquet_files[:max_files]
        
        for pf in parquet_files:
            try:
                df = pd.read_parquet(os.path.join(date_dir, pf), columns=need_cols)
                dfs.append(df)
            except Exception as e:
                logger.warning(f'Failed to load {pf}: {e}')
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def filter_business_type(df: pd.DataFrame, target_types: List[str]) -> pd.DataFrame:
    """过滤目标 business_type"""
    mask = df['business_type'].isin(target_types).values
    return df.loc[mask].reset_index(drop=True)


def encode_features_hash(df: pd.DataFrame, features: List[str], vocab_size: int = VOCAB_SIZE) -> pd.DataFrame:
    """特征编码 - 使用 hash 方式，速度快"""
    df = df.copy()  # 避免修改原始 df 影响外部变量
    for feat in tqdm(features, desc='Encoding features'):
        if feat not in df.columns:
            df[feat] = 0
            continue
        # 向量化 hash
        df[feat] = pd.util.hash_pandas_object(df[feat].fillna('').astype(str), index=False) % vocab_size
    return df


def process_data(train_dates: List[str], val_date: str, features: List[str], 
                 neg_sample_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """完整数据处理流程"""
    
    # 加载数据
    logger.info('Loading training data...')
    train_df = load_parquet_files(train_dates, features, MAX_FILES_PER_DAY)
    logger.info(f'Train raw: {len(train_df):,}')
    
    logger.info('Loading validation data...')
    val_df = load_parquet_files([val_date], features, MAX_FILES_PER_DAY)
    logger.info(f'Val raw: {len(val_df):,}')
    
    # 过滤 business_type
    train_df = filter_business_type(train_df, TARGET_BUSINESS_TYPES)
    val_df = filter_business_type(val_df, TARGET_BUSINESS_TYPES)
    logger.info(f'After filter - Train: {len(train_df):,}, Val: {len(val_df):,}')
    
    # 解析 mul_labels
    logger.info('Parsing mul_labels...')
    for df in [train_df, val_df]:
        parsed = df['mul_labels'].apply(parse_mul_labels)
        df['atc'] = parsed.apply(lambda x: x.get('atc', -1))
        df['open'] = parsed.apply(lambda x: x.get('open', -1))
        df['purchase'] = parsed.apply(lambda x: x.get('purchase', 0))
        df['content_view'] = parsed.apply(lambda x: x.get('content_view', -1))
    
    # 负采样
    logger.info(f'Negative sampling {neg_sample_ratio*100}%...')
    pos_mask = train_df['purchase'] == 1
    neg_mask = train_df['purchase'] == 0
    pos_df = train_df[pos_mask]
    neg_df = train_df[neg_mask].sample(frac=neg_sample_ratio, random_state=42)
    train_df = pd.concat([pos_df, neg_df], ignore_index=True)
    logger.info(f'After neg sampling - Train: {len(train_df):,}')
    
    # business_type 编码（用于对比学习）- 在特征编码前保存原始 business_type
    train_bt = train_df['business_type'].values.tolist()  # 转为 list 避免引用问题
    val_bt = val_df['business_type'].values.tolist()
    
    # 特征编码（使用 hash）
    logger.info('Encoding features...')
    train_df = encode_features_hash(train_df, features)
    val_df = encode_features_hash(val_df, features)
    
    # business_type id 编码
    bt_encoder = LabelEncoder()
    all_bt = list(set(train_bt + val_bt + ['__UNKNOWN__']))
    bt_encoder.fit(all_bt)
    train_df['business_type_id'] = [
        bt_encoder.transform([x])[0] if x in bt_encoder.classes_ else bt_encoder.transform(['__UNKNOWN__'])[0]
        for x in train_bt
    ]
    val_df['business_type_id'] = [
        bt_encoder.transform([x])[0] if x in bt_encoder.classes_ else bt_encoder.transform(['__UNKNOWN__'])[0]
        for x in val_bt
    ]
    
    # vocab_sizes 统一用 hash 的 vocab_size
    vocab_sizes = {feat: VOCAB_SIZE for feat in features}
    vocab_sizes['business_type_id'] = len(bt_encoder.classes_)
    
    meta = {
        'bt_encoder': bt_encoder,
        'vocab_sizes': vocab_sizes,
        'features': features
    }
    
    # 统计
    logger.info('=== 数据统计 ===')
    for name, df, bt in [('Train', train_df, train_bt), ('Val', val_df, val_bt)]:
        logger.info(f'\n{name}: {len(df):,}')
        for col in ['atc', 'open', 'purchase', 'content_view']:
            logger.info(f'  {col}: {df[col].value_counts().to_dict()}')
        logger.info(f'  revenue>0: {(df["revenue"] > 0).sum():,}')
        bt_counts = pd.Series(bt).value_counts().head(10).to_dict()
        logger.info(f'  business_type: {bt_counts}')
    
    return train_df, val_df, meta


def main():
    features = load_combine_schema(SCHEMA_FILE)
    
    train_df, val_df, meta = process_data(
        train_dates=['2026-01-01', '2026-01-02', '2026-01-03'],
        val_date='2026-01-04',
        features=features,
        neg_sample_ratio=0.1
    )
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_pickle(os.path.join(OUTPUT_DIR, 'train.pkl'))
    val_df.to_pickle(os.path.join(OUTPUT_DIR, 'val.pkl'))
    
    with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    logger.info(f'Saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
