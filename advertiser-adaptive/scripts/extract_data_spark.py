#!/usr/bin/env python3
"""
extract_data_spark.py
用 PySpark 直接处理数据并写入 parquet，避免内存问题。

用法：
    python scripts/extract_data_spark.py \
        --start_date 2026-03-01 --end_date 2026-03-14 \
        --output ./dataset/train/data.parquet
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ADVERTISER_WHITELIST = ("shein", "ae", "shopee", "lazada")


def date_range(start, end):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    cur = s
    while cur <= e:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_path", default="/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet")
    args = parser.parse_args()

    # 加载特征列表
    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "combine_schema")
    with open(schema_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Spark 初始化
    spark = SparkSession.builder \
        .appName("extract_data") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "400") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print(f"[INFO] Spark appId: {spark.sparkContext.applicationId}")

    # 构建数据路径列表
    paths = []
    for d in date_range(args.start_date, args.end_date):
        part_path = os.path.join(args.base_path, f"part={d}")
        plain_path = os.path.join(args.base_path, d)
        if os.path.exists(part_path):
            paths.append(part_path)
        elif os.path.exists(plain_path):
            paths.append(plain_path)
        else:
            print(f"[WARN] 跳过 {d}")

    if not paths:
        raise RuntimeError("没有有效数据路径")

    print(f"[INFO] 读取 {len(paths)} 天数据...")

    # 逐天读取并合并，避免列名冲突
    dfs = []
    for i, p in enumerate(paths):
        print(f"[INFO] 读取 {p}")
        df_day = spark.read.parquet(p)
        # 选需要的列（去重）
        all_cols = list(dict.fromkeys(feature_cols + ["label", "business_type", "objective_type"]))
        avail = set(df_day.columns)
        select_cols = [c for c in all_cols if c in avail]
        df_day = df_day.select(*select_cols)
        dfs.append(df_day)

    print(f"[INFO] 合并 {len(dfs)} 天数据...")
    from functools import reduce
    df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)

    # 类型转换
    df = df.withColumn("label", F.col("label").cast("float"))
    for c in df.columns:
        if c != "label":
            df = df.withColumn(c, F.col(c).cast("string"))

    # 白名单过滤
    cond = F.lit(False)
    for prefix in ADVERTISER_WHITELIST:
        cond = cond | F.lower(F.col("business_type")).startswith(prefix)
    df = df.filter(cond)

    # 采样
    df = df.filter(
        (F.col("label") == 1) |
        (
            (F.col("label") == 0) &
            F.when(
                F.lower(F.col("business_type")).startswith("shein") |
                (F.col("objective_type") == "SALES_WEBSITE"),
                F.rand(seed=42) < 0.01
            ).otherwise(F.rand(seed=42) < 0.1)
        )
    )
    df = df.fillna("none")

    # 添加 domain_indicator
    domain_expr = (
        F.when(F.lower(F.col("business_type")).startswith("shein"), F.lit(0))
        .when(F.lower(F.col("business_type")).startswith("ae"), F.lit(1))
        .when(F.lower(F.col("business_type")).startswith("shopee"), F.lit(2))
        .when(F.lower(F.col("business_type")).startswith("lazada"), F.lit(3))
        .otherwise(F.lit(0))
    )
    df = df.withColumn("domain_indicator", domain_expr)

    # 只保留需要的列
    keep_cols = feature_cols + ["label", "domain_indicator"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.select(*keep_cols)

    # 写入 parquet
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"[INFO] 写入 {args.output}...")
    df.write.mode("overwrite").parquet(args.output)

    # 不 count，直接结束
    print(f"[INFO] 完成！数据已写入 {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
