#!/usr/bin/env python3
"""
重新预处理 ivr_sample_v16_ctcvr，所有 124 个特征全部当类别特征（hash编码）
不做任何采样
"""
import os, sys, pickle, json, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr'
OUT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr/data_all_cat'
os.makedirs(OUT_DIR, exist_ok=True)

# 跳过列
SKIP_COLS = {'ctcvr_label', 'click_label', 'deviceid', 'business_type'}

# 所有特征列
ALL_COLS = [
    'adid', 'adsize', 'adx', 'bundle', 'campaignid', 'campaignsetid',
    'carrier', 'city', 'connectiontype', 'country', 'country-hour',
    'demand_pkgname', 'devicetype',
    'duf_inner_dev_pkg_atc_11_30d', 'duf_inner_dev_pkg_atc_180d', 'duf_inner_dev_pkg_atc_30d',
    'duf_inner_dev_pkg_atc_31_60d', 'duf_inner_dev_pkg_atc_4_10d', 'duf_inner_dev_pkg_atc_60d',
    'duf_inner_dev_pkg_atc_61_180d', 'duf_inner_dev_pkg_atc_90d',
    'duf_inner_dev_pkg_cv_15d', 'duf_inner_dev_pkg_cv_30d', 'duf_inner_dev_pkg_cv_3d',
    'duf_inner_dev_pkg_cv_7d',
    'duf_inner_dev_pkg_imp_11_30d', 'duf_inner_dev_pkg_imp_15d', 'duf_inner_dev_pkg_imp_30d',
    'duf_inner_dev_pkg_imp_31_60d', 'duf_inner_dev_pkg_imp_3d', 'duf_inner_dev_pkg_imp_4_10d',
    'duf_inner_dev_pkg_imp_7d',
    'duf_inner_dev_pkg_imp_bucket_15d', 'duf_inner_dev_pkg_imp_bucket_3d', 'duf_inner_dev_pkg_imp_bucket_7d',
    'duf_inner_dev_pkg_open_11_30d', 'duf_inner_dev_pkg_open_180d', 'duf_inner_dev_pkg_open_31_60d',
    'duf_inner_dev_pkg_open_4_10d', 'duf_inner_dev_pkg_open_60d', 'duf_inner_dev_pkg_open_61_180d',
    'duf_inner_dev_pkg_open_90d',
    'duf_inner_dev_pkg_pur_11_30d', 'duf_inner_dev_pkg_pur_15d', 'duf_inner_dev_pkg_pur_180d',
    'duf_inner_dev_pkg_pur_30d', 'duf_inner_dev_pkg_pur_31_60d', 'duf_inner_dev_pkg_pur_3d',
    'duf_inner_dev_pkg_pur_4_10d', 'duf_inner_dev_pkg_pur_60d', 'duf_inner_dev_pkg_pur_61_180d',
    'duf_inner_dev_pkg_pur_7d', 'duf_inner_dev_pkg_pur_90d',
    'duf_inner_dev_pkg_pur_bucket_15d', 'duf_inner_dev_pkg_pur_bucket_3d', 'duf_inner_dev_pkg_pur_bucket_7d',
    'duf_inner_dev_pkg_re_15d', 'duf_inner_dev_pkg_re_30d', 'duf_inner_dev_pkg_re_3d',
    'duf_inner_dev_pkg_re_7d',
    'duf_inner_dev_pkg_re_bucket_15d', 'duf_inner_dev_pkg_re_bucket_3d', 'duf_inner_dev_pkg_re_bucket_7d',
    'duf_outer_dev_pkg_atc_11_30d', 'duf_outer_dev_pkg_atc_15d', 'duf_outer_dev_pkg_atc_180d',
    'duf_outer_dev_pkg_atc_1d', 'duf_outer_dev_pkg_atc_30d', 'duf_outer_dev_pkg_atc_31_60d',
    'duf_outer_dev_pkg_atc_3d', 'duf_outer_dev_pkg_atc_4_10d', 'duf_outer_dev_pkg_atc_60d',
    'duf_outer_dev_pkg_atc_61_180d', 'duf_outer_dev_pkg_atc_7d', 'duf_outer_dev_pkg_atc_90d',
    'duf_outer_dev_pkg_cv_15d', 'duf_outer_dev_pkg_cv_1d', 'duf_outer_dev_pkg_cv_30d',
    'duf_outer_dev_pkg_cv_3d', 'duf_outer_dev_pkg_cv_60d', 'duf_outer_dev_pkg_cv_7d',
    'duf_outer_dev_pkg_cv_90d',
    'duf_outer_dev_pkg_open_11_30d', 'duf_outer_dev_pkg_open_180d', 'duf_outer_dev_pkg_open_31_60d',
    'duf_outer_dev_pkg_open_4_10d', 'duf_outer_dev_pkg_open_61_180d',
    'duf_outer_dev_pkg_pur_11_30d', 'duf_outer_dev_pkg_pur_15d', 'duf_outer_dev_pkg_pur_180d',
    'duf_outer_dev_pkg_pur_1d', 'duf_outer_dev_pkg_pur_30d', 'duf_outer_dev_pkg_pur_31_60d',
    'duf_outer_dev_pkg_pur_3d', 'duf_outer_dev_pkg_pur_4_10d', 'duf_outer_dev_pkg_pur_60d',
    'duf_outer_dev_pkg_pur_61_180d', 'duf_outer_dev_pkg_pur_7d', 'duf_outer_dev_pkg_pur_90d',
    'duf_outer_dev_pkg_view_11_30d', 'duf_outer_dev_pkg_view_180d', 'duf_outer_dev_pkg_view_31_60d',
    'duf_outer_dev_pkg_view_4_10d', 'duf_outer_dev_pkg_view_61_180d',
    'huf_deviceid_demand_pkgname_gmv_24h', 'huf_deviceid_demand_pkgname_imp_12h',
    'huf_deviceid_demand_pkgname_imp_1h', 'huf_deviceid_demand_pkgname_imp_24h',
    'huf_deviceid_demand_pkgname_imp_3h', 'huf_deviceid_demand_pkgname_pur_24h',
    'huf_deviceid_demand_pkgname_re_12h', 'huf_deviceid_demand_pkgname_re_1h',
    'huf_deviceid_demand_pkgname_re_24h', 'huf_deviceid_demand_pkgname_re_3h',
    'imptype', 'is_interstitial_ad', 'is_reward_ad', 'language', 'make', 'model',
    'os', 'osv', 'offerid', 'subcategory_id',
]


def hash_val(v, vocab_size):
    """任意值 -> [0, vocab_size)"""
    if pd.isna(v):
        return 0
    return hash(str(v)) % vocab_size


def main():
    print("="*60)
    print("预处理 ivr_sample_v16_ctcvr（全类别特征）")
    print("="*60)

    train_files = sorted(glob.glob(f'{SRC_DIR}/train/*.parquet'))
    test_files = sorted(glob.glob(f'{SRC_DIR}/test/*.parquet'))
    print(f"训练集: {len(train_files)} files")
    print(f"测试集: {len(test_files)} files")

    # 1. 扫描全部数据，确定 vocab_sizes 和 bt 类别
    print("\n[1/4] 扫描数据确定 vocab_sizes...")
    vocab_sizes = {}
    bt_values = set()

    for feat in ALL_COLS:
        uniq = set()
        for fp in train_files[:3]:
            sub = pq.read_table(fp, columns=[feat]).to_pandas()[feat]
            uniq.update(sub.fillna('__NULL__').astype(str).unique())
        vocab_sizes[feat] = min(len(uniq) + 1, 50000)

    for fp in train_files + test_files:
        df = pq.read_table(fp, columns=['business_type']).to_pandas()
        bt_values.update(df['business_type'].fillna('__UNKNOWN__').unique())

    print(f"特征数: {len(ALL_COLS)}")
    print(f"vocab_sizes: min={min(vocab_sizes.values())}, max={max(vocab_sizes.values())}")
    print(f"business_type: {len(bt_values)} classes")

    bt_encoder = LabelEncoder()
    bt_encoder.fit(list(bt_values))

    # 2. 处理训练集（向量化）
    print("\n[2/4] 处理训练集...")
    train_dfs = []
    for fi, fp in enumerate(tqdm(train_files, desc='Train')):
        df = pq.read_table(fp).to_pandas()
        label = df['ctcvr_label'].astype(np.int64).values
        bt_id = bt_encoder.transform(df['business_type'].fillna('__UNKNOWN__'))

        out = {'label': label, 'business_type_id': bt_id}
        for feat in ALL_COLS:
            vs = vocab_sizes[feat]
            raw = df[feat].fillna('__NULL__').astype(str)
            out[feat] = raw.apply(lambda x: hash_val(x, vs)).astype(np.int64)
        
        train_dfs.append(pd.DataFrame(out))

    train_df = pd.concat(train_dfs, ignore_index=True)
    del train_dfs
    print(f"训练集: {len(train_df):,} 行, 正率 {train_df['label'].mean():.4%}")

    # 3. 处理测试集
    print("\n[3/4] 处理测试集...")
    test_dfs = []
    for fp in tqdm(test_files, desc='Test'):
        df = pq.read_table(fp).to_pandas()
        label = df['ctcvr_label'].astype(np.int64).values
        bt_id = bt_encoder.transform(df['business_type'].fillna('__UNKNOWN__'))

        out = {'label': label, 'business_type_id': bt_id}
        for feat in ALL_COLS:
            vs = vocab_sizes[feat]
            raw = df[feat].fillna('__NULL__').astype(str)
            out[feat] = raw.apply(lambda x: hash_val(x, vs)).astype(np.int64)
        
        test_dfs.append(pd.DataFrame(out))

    test_df = pd.concat(test_dfs, ignore_index=True)
    del test_dfs
    print(f"测试集: {len(test_df):,} 行, 正率 {test_df['label'].mean():.4%}")

    # 4. 写入 parquet
    print("\n[4/4] 写入 parquet...")
    feat_cols = ALL_COLS
    train_df[feat_cols] = train_df[feat_cols].astype(np.int64)
    test_df[feat_cols] = test_df[feat_cols].astype(np.int64)
    train_df.to_parquet(f'{OUT_DIR}/train.parquet', index=False, compression='snappy')
    test_df.to_parquet(f'{OUT_DIR}/test.parquet', index=False, compression='snappy')

    meta = {
        'features': feat_cols,
        'vocab_sizes': vocab_sizes,
        'bt_encoder': bt_encoder,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'pos_rate_train': float(train_df['label'].mean()),
        'pos_rate_test': float(test_df['label'].mean()),
    }
    with open(f'{OUT_DIR}/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    readme = f"""# ivr_sample_v16_ctcvr（全类别）预处理数据

## 数据规模
- 训练集: {len(train_df):,} 行, 正率 {train_df['label'].mean():.4%}
- 测试集: {len(test_df):,} 行, 正率 {test_df['label'].mean():.4%}

## 预处理方式
- 所有 {len(feat_cols)} 个特征全部当类别特征
- hash(str(v)) % vocab_size -> [0, vocab_size)
- vocab_size 基于实测 unique 值（max cap=50000）
- 无降采样，无过滤

## Business Type
{bt_encoder.classes_}
"""
    with open(f'{OUT_DIR}/README.md', 'w') as f:
        f.write(readme)

    print(f"\n完成!")
    print(f"  训练集: {OUT_DIR}/train.parquet ({os.path.getsize(f'{OUT_DIR}/train.parquet')/1e9:.1f}GB)")
    print(f"  测试集: {OUT_DIR}/test.parquet ({os.path.getsize(f'{OUT_DIR}/test.parquet')/1e9:.1f}GB)")
    print(f"  vocab_sizes: {vocab_sizes}")


if __name__ == '__main__':
    main()