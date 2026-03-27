#!/usr/bin/env python3
"""
process_one_day.py
处理单天数据，独立 Spark session，避免内存累积。

用法：
    python scripts/process_one_day.py --date 2026-03-01 --output ./dataset/daily/2026-03-01.parquet
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# domain 映射规则
ADVERTISER_WHITELIST_PREFIXES = ("shein", "ae", "shopee", "lazada")
SHEIN_NEG_RATIO = 0.01
OTHER_NEG_RATIO = 0.10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 特征列表
    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "combine_schema")
    with open(schema_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # 数据路径
    base_path = "/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet"
    part_path = os.path.join(base_path, f"part={args.date}")
    plain_path = os.path.join(base_path, args.date)
    data_path = part_path if os.path.exists(part_path) else plain_path

    if not os.path.exists(data_path):
        print(f"[WARN] 数据不存在：{data_path}")
        pd.DataFrame(columns=feature_cols + ["label", "domain_indicator"]).to_parquet(args.output, index=False)
        return

    # Spark 处理
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder \
        .appName(f"process_{args.date}") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    all_cols = feature_cols + ["label", "business_type", "objective_type"]
    all_cols = list(dict.fromkeys(all_cols))

    df = spark.read.parquet(data_path)
    avail = set(df.columns)
    select_cols = [c for c in all_cols if c in avail]
    df = df.select(*select_cols)

    for col in df.columns:
        if col == "label":
            df = df.withColumn(col, F.col(col).cast("float"))
        else:
            df = df.withColumn(col, F.col(col).cast("string"))

    # 白名单过滤
    whitelist_cond = F.lit(False)
    for prefix in ADVERTISER_WHITELIST_PREFIXES:
        whitelist_cond = whitelist_cond | F.lower(F.col("business_type")).startswith(prefix)
    df = df.filter(whitelist_cond)

    # 采样
    df = df.filter(
        (F.col("label") == 1) | (
            (F.col("label") == 0) &
            F.when(
                F.lower(F.col("business_type")).startswith("shein") |
                (F.col("objective_type") == "SALES_WEBSITE"),
                F.rand(seed=42) < SHEIN_NEG_RATIO
            ).otherwise(F.rand(seed=42) < OTHER_NEG_RATIO)
        )
    )
    df = df.fillna("none")

    # 添加 domain_indicator（Spark 端）
    domain_map_expr = (
        F.when(F.lower(F.col("business_type")).startswith("shein"), F.lit(0))
        .when(F.lower(F.col("business_type")).startswith("ae"), F.lit(1))
        .when(F.lower(F.col("business_type")).startswith("shopee"), F.lit(2))
        .when(F.lower(F.col("business_type")).startswith("lazada"), F.lit(3))
        .otherwise(F.lit(0))
    )
    df = df.withColumn("domain_indicator", domain_map_expr)

    # 只保留需要的列
    keep_cols = feature_cols + ["label", "domain_indicator"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.select(*keep_cols)

    # 写入 parquet（不转 pandas，避免 OOM）
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.write.mode("overwrite").parquet(args.output)

    # 统计
    total = df.count()
    pos = df.filter(F.col("label") == 1).count()
    spark.stop()

    print(f"[INFO] {args.date}: {total} 行，正样本 {pos}")


if __name__ == "__main__":
    main()
