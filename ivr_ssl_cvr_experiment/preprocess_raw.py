#!/usr/bin/env python3
"""
重新预处理 ivr_sample_v16_ctcvr 原始数据（不做任何采样）
- 直接从原始 parquet 读，不经过中间格式
- 正确处理字符串特征（hash -> 0~vocab_size）
- 标签 = ctcvr_label
- 划分 train/test（按时间顺序 8:2）
"""
import os, sys, pickle, json, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr'
OUT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr/data_raw'
os.makedirs(OUT_DIR, exist_ok=True)

# 特征列定义（去掉 label、click_label、deviceid、business_type）
SKIP_COLS = {'ctcvr_label', 'click_label', 'deviceid', 'business_type', 'user_id'}

# 字符串类特征（需要 hash 编码）
STR_FEATURES = [
    'adid', 'adsize', 'adx', 'bundle', 'campaignid', 'campaignsetid',
    'carrier', 'city', 'connectiontype', 'country', 'country-hour',
    'demand_pkgname', 'devicetype', 'language', 'make', 'model',
    'os', 'osv', 'offerid', 'imptype', 'subcategory_id',
    'is_interstitial_ad', 'is_reward_ad',
]

# 数值类特征（duf_*/huf_*，直接转为 float）
NUM_FEATURES = [
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
]


def hash_str(s, vocab_size):
    """字符串 -> [0, vocab_size) 的整数"""
    if pd.isna(s) or str(s).strip() == '' or str(s) == '-':
        return 0
    return hash(str(s)) % vocab_size


def parse_num(val):
    """数值解析：空/非数字 -> 0"""
    if pd.isna(val):
        return 0.0
    try:
        return float(val)
    except:
        return 0.0


def preprocess_chunk(df, str_encoders, str_vocab_sizes, num_features):
    """预处理一个 chunk"""
    # 标签
    label = df['ctcvr_label'].astype(np.int64).values
    
    # business_type 编码
    bt_id = str_encoders['bt'].transform(df['business_type'].fillna('__UNKNOWN__'))
    
    # deviceid -> user_id 编码（用 hash，LabelEncoder 内存开销太大）
    deviceids = df['deviceid'].fillna('__NULL__').values
    known_uids = np.array([hash(str(x)) % 200000 for x in deviceids], dtype=np.int64)
    
    # 字符串特征
    str_out = {}
    for feat in STR_FEATURES:
        if feat not in df.columns:
            continue
        vs = str_vocab_sizes.get(feat, 10000)
        col = df[feat].fillna('__NULL__')
        out = col.apply(lambda x: hash_str(x, vs)).astype(np.int64).values
        str_out[feat] = out
    
    # 数值特征
    num_out = {}
    for feat in num_features:
        if feat not in df.columns:
            continue
        num_out[feat] = df[feat].apply(parse_num).astype(np.float32).values
    
    return label, bt_id, known_uids, str_out, num_out


def main():
    print("="*60)
    print("重新预处理 ivr_sample_v16_ctcvr（无采样）")
    print("="*60)
    
    # 读取所有 parquet 文件
    print("\n[1/5] 读取所有 parquet 文件...")
    train_files = sorted(glob.glob(f'{SRC_DIR}/train/*.parquet'))
    test_files = sorted(glob.glob(f'{SRC_DIR}/test/*.parquet'))
    
    print(f"训练集: {len(train_files)} files")
    print(f"测试集: {len(test_files)} files")
    
    # 读取训练数据确定编码器
    print("\n[2/5] 建立编码器...")
    
    # 先扫描训练集，建立 vocabulary
    str_vocab_sizes = {}
    bt_values = set()
    user_values = set()
    
    # 确定有哪些特征
    sample_df = pq.read_table(train_files[0]).to_pandas()
    all_cols = set(sample_df.columns)
    
    # 收集 str feature 的 unique 值数量（采样）
    print("采样统计字符串特征 vocab 大小...")
    for feat in STR_FEATURES:
        if feat not in all_cols:
            continue
        # 采样 100k 行估计 vocab
        uniq = set()
        for f in train_files[:3]:  # 前3个文件
            sub = pq.read_table(f, columns=[feat]).to_pandas()[feat]
            uniq.update(sub.fillna('__NULL__').unique())
        str_vocab_sizes[feat] = min(len(uniq) + 1, 50000)
        if len(uniq) > 0:
            print(f"  {feat}: vocab={str_vocab_sizes[feat]} (实测 {len(uniq)})")
    
    # 收集 business_type 和 user（包含测试集，防止未知值）
    print("收集全部 business_type 和 deviceid...")
    for f in train_files + test_files:
        df = pq.read_table(f, columns=['business_type', 'deviceid']).to_pandas()
        bt_values.update(df['business_type'].fillna('__UNKNOWN__').unique())
        user_values.update(df['deviceid'].fillna('__UNKNOWN__').unique())
    
    print(f"\nbusiness_type classes: {len(bt_values)}")
    print(f"unique deviceid: {len(user_values):,}")
    
    # 建立编码器
    bt_encoder = LabelEncoder()
    bt_encoder.fit(list(bt_values))
    
    user_encoder = LabelEncoder()
    user_encoder.fit(list(user_values) + ['__UNKNOWN__'])
    
    str_encoders = {'bt': bt_encoder}
    
    # 处理所有数据
    print("\n[3/5] 处理训练集...")
    train_labels = []
    train_bt_ids = []
    train_uids = []
    train_str_data = {f: [] for f in str_vocab_sizes}
    train_num_data = {f: [] for f in NUM_FEATURES if f in all_cols}
    
    for fi, fpath in enumerate(tqdm(train_files, desc='Train')):
        df = pq.read_table(fpath).to_pandas()
        label, bt_id, uid, str_out, num_out = preprocess_chunk(
            df, str_encoders, str_vocab_sizes, [f for f in NUM_FEATURES if f in all_cols]
        )
        train_labels.append(label)
        train_bt_ids.append(bt_id)
        train_uids.append(uid)
        for feat in str_out:
            train_str_data[feat].append(str_out[feat])
        for feat in num_out:
            train_num_data[feat].append(num_out[feat])
    
    train_labels = np.concatenate(train_labels)
    train_bt_ids = np.concatenate(train_bt_ids)
    train_uids = np.concatenate(train_uids)
    for feat in train_str_data:
        train_str_data[feat] = np.concatenate(train_str_data[feat])
    for feat in train_num_data:
        train_num_data[feat] = np.concatenate(train_num_data[feat])
    
    print(f"\n训练集: {len(train_labels):,} 行, 正样本 {train_labels.sum():,} ({train_labels.mean():.2%})")
    
    # 处理测试集
    print("\n[4/5] 处理测试集...")
    test_labels = []
    test_bt_ids = []
    test_uids = []
    test_str_data = {f: [] for f in str_vocab_sizes}
    test_num_data = {f: [] for f in NUM_FEATURES if f in all_cols}
    
    for fpath in tqdm(test_files, desc='Test'):
        df = pq.read_table(fpath).to_pandas()
        label, bt_id, uid, str_out, num_out = preprocess_chunk(
            df, str_encoders, str_vocab_sizes, [f for f in NUM_FEATURES if f in all_cols]
        )
        test_labels.append(label)
        test_bt_ids.append(bt_id)
        test_uids.append(uid)
        for feat in str_out:
            test_str_data[feat].append(str_out[feat])
        for feat in num_out:
            test_num_data[feat].append(num_out[feat])
    
    test_labels = np.concatenate(test_labels)
    test_bt_ids = np.concatenate(test_bt_ids)
    test_uids = np.concatenate(test_uids)
    for feat in test_str_data:
        test_str_data[feat] = np.concatenate(test_str_data[feat])
    for feat in test_num_data:
        test_num_data[feat] = np.concatenate(test_num_data[feat])
    
    print(f"\n测试集: {len(test_labels):,} 行, 正样本 {test_labels.sum():,} ({test_labels.mean():.2%})")
    
    # 写入 parquet
    print("\n[5/5] 写入 parquet...")
    
    # 特征列顺序
    all_features = list(str_vocab_sizes.keys()) + list(train_num_data.keys())
    
    # 训练集
    train_df = pd.DataFrame({'label': train_labels, 'business_type_id': train_bt_ids, 'user_id': train_uids})
    for feat in all_features:
        if feat in str_vocab_sizes:
            train_df[feat] = train_str_data[feat]
        else:
            train_df[feat] = train_num_data[feat]
    
    # 测试集
    test_df = pd.DataFrame({'label': test_labels, 'business_type_id': test_bt_ids, 'user_id': test_uids})
    for feat in all_features:
        if feat in str_vocab_sizes:
            test_df[feat] = test_str_data[feat]
        else:
            test_df[feat] = test_num_data[feat]
    
    train_df.to_parquet(f'{OUT_DIR}/train.parquet', index=False, compression='snappy')
    test_df.to_parquet(f'{OUT_DIR}/test.parquet', index=False, compression='snappy')
    
    # 写入 meta
    meta = {
        'features': all_features,
        'str_features': list(str_vocab_sizes.keys()),
        'num_features': list(train_num_data.keys()),
        'vocab_sizes': {**str_vocab_sizes},
        'bt_encoder': bt_encoder,
        'n_train': len(train_labels),
        'n_test': len(test_labels),
        'pos_rate_train': float(train_labels.mean()),
        'pos_rate_test': float(test_labels.mean()),
    }
    
    with open(f'{OUT_DIR}/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    # README
    readme = f"""# ivr_sample_v16_ctcvr 预处理数据

## 数据规模
- 训练集: {len(train_labels):,} 行
- 测试集: {len(test_labels):,} 行

## 标签
- 列: `label` (ctcvr_label)
- 训练集正率: {train_labels.mean():.4%}
- 测试集正率: {test_labels.mean():.4%}

## 特征
- 字符串特征（hash编码）: {len(str_vocab_sizes)} 个, vocab_sizes 见 meta
- 数值特征: {len(train_num_data)} 个

## Business Type
{bt_encoder.classes_}

## 预处理方式
- 字符串特征: hash(str) % vocab_size -> [0, vocab_size)
- 数值特征: 直接转为 float，空值填 0
- user_id: deviceid 经 LabelEncoder 编码
- 无降采样，无过滤
"""
    with open(f'{OUT_DIR}/README.md', 'w') as f:
        f.write(readme)
    
    print(f"\n完成！")
    print(f"  训练集: {OUT_DIR}/train.parquet ({os.path.getsize(f'{OUT_DIR}/train.parquet')/1e9:.1f}GB)")
    print(f"  测试集: {OUT_DIR}/test.parquet ({os.path.getsize(f'{OUT_DIR}/test.parquet')/1e9:.1f}GB)")
    print(f"  特征数: {len(all_features)} (str={len(str_vocab_sizes)}, num={len(train_num_data)})")
    print(f"  vocab_sizes: {str_vocab_sizes}")


if __name__ == '__main__':
    main()