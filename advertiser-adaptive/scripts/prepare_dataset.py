#!/usr/bin/env python3
"""
prepare_dataset.py
数据预处理脚本：读取 parquet → 采样 → 保存到 dataset/ 目录，供后续训练复用。

用法：
    python scripts/prepare_dataset.py \
        --start_date 2025-01-01 \
        --end_date 2025-01-07 \
        --val_date 2025-01-08 \
        --output_dir ./dataset \
        --schema_path ./combine_schema \
        --data_path /mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.spark_loader import load_data_spark
from src.data.dataset import IVRDataset, load_schema
from src.utils.logger import get_logger

logger = get_logger("prepare_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="IVR 数据预处理")
    parser.add_argument("--start_date", required=True, help="训练开始日期 YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="训练结束日期 YYYY-MM-DD")
    parser.add_argument("--val_date", required=True, help="验证日期 YYYY-MM-DD")
    parser.add_argument("--output_dir", default="./dataset", help="输出目录")
    parser.add_argument("--schema_path", default="./combine_schema", help="特征配置文件路径")
    parser.add_argument("--data_path",
                        default="/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet",
                        help="parquet 数据根目录")
    parser.add_argument("--force", action="store_true", help="强制重新生成（忽略缓存）")
    return parser.parse_args()


def main():
    args = parse_args()
    feature_cols = load_schema(args.schema_path)
    logger.info(f"特征数量：{len(feature_cols)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 训练集 ────────────────────────────────────────────────────────────
    train_dir = os.path.join(args.output_dir, f"train_{args.start_date}_{args.end_date}")
    train_path = os.path.join(train_dir, "data.parquet")

    if os.path.exists(train_path) and not args.force:
        logger.info(f"训练集缓存已存在，跳过：{train_path}")
    else:
        logger.info(f"开始处理训练集：{args.start_date} ~ {args.end_date}")
        df_train = load_data_spark(
            base_path=args.data_path,
            feature_cols=feature_cols,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        os.makedirs(train_dir, exist_ok=True)
        IVRDataset.save_parquet(df_train, train_path)
        logger.info(f"训练集已保存：{train_path}，共 {len(df_train)} 行")

    # ── 验证集 ────────────────────────────────────────────────────────────
    val_dir = os.path.join(args.output_dir, f"val_{args.val_date}")
    val_path = os.path.join(val_dir, "data.parquet")

    if os.path.exists(val_path) and not args.force:
        logger.info(f"验证集缓存已存在，跳过：{val_path}")
    else:
        logger.info(f"开始处理验证集：{args.val_date}")
        df_val = load_data_spark(
            base_path=args.data_path,
            feature_cols=feature_cols,
            start_date=args.val_date,
            end_date=args.val_date,
        )
        os.makedirs(val_dir, exist_ok=True)
        IVRDataset.save_parquet(df_val, val_path)
        logger.info(f"验证集已保存：{val_path}，共 {len(df_val)} 行")

    logger.info("数据预处理完成！")
    logger.info(f"  训练集：{train_path}")
    logger.info(f"  验证集：{val_path}")


if __name__ == "__main__":
    main()
