#!/usr/bin/env python3
"""
IVR 数据预处理脚本 v2 - 增加 deviceid 用于用户对比学习
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
OUTPUT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr/data_v2/'
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
USER_VOCAB_SIZE = 500000  # 用户 ID 的 vocab 大小（更大以减少碰撞）


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


def load_parquet_files(dates: List[str], features: List[str], max_files: int = None,
                       extra_cols: List[str] = None) -> pd.DataFrame:
    """加载 parquet 文件"""
    dfs = []
    # 基础额外列
    default_extra = ['business_type', 'mul_labels', 'revenue']
    if extra_cols:
        default_extra = list(set(default_extra + extra_cols))
    need_cols = list(set(features + default_extra))
    
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
    df = df.copy()
    for feat in tqdm(features, desc='Encoding features'):
        if feat not in df.columns:
            df[feat] = 0
            continue
        df[feat] = pd.util.hash_pandas_object(df[feat].fillna('').astype(str), index=False) % vocab_size
    return df


def process_data(train_dates: List[str], val_date: str, features: List[str], 
                 neg_sample_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """完整数据处理流程 - v2 增加 deviceid"""
    
    # 额外加载 deviceid 用于用户对比学习
    extra_cols = ['deviceid']
    
    # 加载数据
    logger.info('Loading training data...')
    train_df = load_parquet_files(train_dates, features, MAX_FILES_PER_DAY, extra_cols)
    logger.info(f'Train raw: {len(train_df):,}')
    
    logger.info('Loading validation data...')
    val_df = load_parquet_files([val_date], features, MAX_FILES_PER_DAY, extra_cols)
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
    
    # 保存原始值用于编码
    train_bt = train_df['business_type'].values.tolist()
    val_bt = val_df['business_type'].values.tolist()
    train_deviceid = train_df['deviceid'].values.tolist() if 'deviceid' in train_df.columns else [None] * len(train_df)
    val_deviceid = val_df['deviceid'].values.tolist() if 'deviceid' in val_df.columns else [None] * len(val_df)
    
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
    
    # deviceid hash 编码（用于用户对比学习）
    logger.info('Encoding deviceid...')
    train_df['user_id'] = pd.util.hash_pandas_object(
        pd.Series(train_deviceid).fillna('').astype(str), index=False
    ) % USER_VOCAB_SIZE
    val_df['user_id'] = pd.util.hash_pandas_object(
        pd.Series(val_deviceid).fillna('').astype(str), index=False
    ) % USER_VOCAB_SIZE
    
    # 统计用户分布
    train_user_counts = train_df['user_id'].value_counts()
    logger.info(f'Train unique users (hashed): {len(train_user_counts):,}')
    logger.info(f'Train users with >=2 samples: {(train_user_counts >= 2).sum():,}')
    logger.info(f'Train users with >=5 samples: {(train_user_counts >= 5).sum():,}')
    
    # vocab_sizes
    vocab_sizes = {feat: VOCAB_SIZE for feat in features}
    vocab_sizes['business_type_id'] = len(bt_encoder.classes_)
    vocab_sizes['user_id'] = USER_VOCAB_SIZE
    
    meta = {
        'bt_encoder': bt_encoder,
        'vocab_sizes': vocab_sizes,
        'features': features,
        'version': 'v2',
        'has_user_id': True,
        'user_vocab_size': USER_VOCAB_SIZE,
        'train_dates': train_dates,
        'val_date': val_date,
        'neg_sample_ratio': neg_sample_ratio,
        'target_business_types': TARGET_BUSINESS_TYPES
    }
    
    # 统计
    logger.info('=== 数据统计 ===')
    for name, df, bt in [('Train', train_df, train_bt), ('Val', val_df, val_bt)]:
        logger.info(f'\n{name}: {len(df):,}')
        for col in ['atc', 'open', 'purchase', 'content_view']:
            if col in df.columns:
                logger.info(f'  {col}: {df[col].value_counts().to_dict()}')
        if 'revenue' in df.columns:
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
    
    # 删除 mul_labels 列（不需要保存，已解析为单独的标签列）
    for df in [train_df, val_df]:
        if 'mul_labels' in df.columns:
            df.drop(columns=['mul_labels'], inplace=True)
    
    # 保存为 pickle（快速加载）
    train_df.to_pickle(os.path.join(OUTPUT_DIR, 'train.pkl'))
    val_df.to_pickle(os.path.join(OUTPUT_DIR, 'val.pkl'))
    
    with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    # 保存为 parquet（更通用，可跨语言）
    train_df.to_parquet(os.path.join(OUTPUT_DIR, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(OUTPUT_DIR, 'val.parquet'), index=False)
    
    logger.info(f'Saved to {OUTPUT_DIR}')
    logger.info(f'  train.pkl: {os.path.getsize(os.path.join(OUTPUT_DIR, "train.pkl")) / 1e6:.1f} MB')
    logger.info(f'  val.pkl: {os.path.getsize(os.path.join(OUTPUT_DIR, "val.pkl")) / 1e6:.1f} MB')
    logger.info(f'  train.parquet: {os.path.getsize(os.path.join(OUTPUT_DIR, "train.parquet")) / 1e6:.1f} MB')
    logger.info(f'  val.parquet: {os.path.getsize(os.path.join(OUTPUT_DIR, "val.parquet")) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
