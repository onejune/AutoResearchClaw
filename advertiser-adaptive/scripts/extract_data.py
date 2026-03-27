#!/usr/bin/env python3
"""
extract_data.py
复用 rec-autopilot 的基础设施，读取采样数据并保存为 parquet。

用法：
    python scripts/extract_data.py \
        --start_date 2026-03-01 --end_date 2026-03-14 \
        --output ./dataset/train.parquet
"""
import argparse
import os
import sys

sys.path.insert(0, "/mnt/workspace/open_research/rec-autopilot/src")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import metaspore as ms
from pyspark.sql import functions as F
from movas_logger import MovasLogger


# 白名单过滤
ADVERTISER_WHITELIST = ("shein", "ae", "shopee", "lazada")

def filter_advertisers(df):
    """只保留目标广告主"""
    cond = F.lit(False)
    for prefix in ADVERTISER_WHITELIST:
        cond = cond | F.lower(F.col("business_type")).startswith(prefix)
    return df.filter(cond)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 加载配置
    with open("/mnt/workspace/open_research/rec-autopilot/conf/base.yaml") as f:
        cfg = yaml.safe_load(f)

    # 初始化 Spark（复用 rec-autopilot 配置）
    spark_confs = {
        "spark.eventLog.enabled": "false",
        "spark.driver.memory": "20g",
        "spark.executor.memory": "16g",
        "spark.executor.instances": "1",
        "spark.executor.cores": "8",
        "spark.default.parallelism": "16",
        "spark.sql.execution.arrow.pyspark.enabled": "false",
        "spark.network.timeout": "800s",
        "spark.executor.heartbeatInterval": "60s",
    }
    spark = ms.spark.get_session(
        local=cfg["local_spark"],
        app_name="extract_data",
        batch_size=10000,
        worker_count=cfg["worker_count"],
        server_count=cfg["server_count"],
        worker_memory=cfg["worker_memory"],
        server_memory=cfg["server_memory"],
        coordinator_memory=cfg["coordinator_memory"],
        spark_confs=spark_confs
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"[INFO] Spark 初始化完成，appId: {spark.sparkContext.applicationId}")

    # 读取特征列表
    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "combine_schema")
    with open(schema_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # 读取数据
    base_path = cfg["train_path_prefix"]
    dates = _date_range(args.start_date, args.end_date)

    dfs = []
    for date_str in dates:
        part_path = os.path.join(base_path, f"part={date_str}")
        plain_path = os.path.join(base_path, date_str)
        data_path = part_path if os.path.exists(part_path) else plain_path
        if not os.path.exists(data_path):
            print(f"[WARN] 跳过 {date_str}")
            continue

        print(f"[INFO] 读取 {date_str}")
        df = spark.read.parquet(data_path)

        # 选列
        all_cols = feature_cols + ["label", "business_type", "objective_type"]
        avail = set(df.columns)
        select_cols = [c for c in all_cols if c in avail]
        df = df.select(*select_cols)

        # 类型转换
        df = df.withColumn("label", F.col("label").cast("float"))
        for c in df.columns:
            if c != "label":
                df = df.withColumn(c, F.col(c).cast("string"))

        # 白名单过滤
        df = filter_advertisers(df)

        # 采样（复用 rec-autopilot 逻辑）
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
        dfs.append(df)

    if not dfs:
        raise RuntimeError("没有数据")

    # 合并
    from functools import reduce
    merged = reduce(lambda a, b: a.union(b), dfs)

    # 添加 domain_indicator
    from pyspark.sql import types as T
    domain_map_expr = (
        F.when(F.lower(F.col("business_type")).startswith("shein"), F.lit(0))
        .when(F.lower(F.col("business_type")).startswith("ae"), F.lit(1))
        .when(F.lower(F.col("business_type")).startswith("shopee"), F.lit(2))
        .when(F.lower(F.col("business_type")).startswith("lazada"), F.lit(3))
        .otherwise(F.lit(0))
    )
    merged = merged.withColumn("domain_indicator", domain_map_expr)

    # 只保留需要的列
    keep_cols = feature_cols + ["label", "domain_indicator"]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged.select(*keep_cols)

    # 保存
    print(f"[INFO] 保存到 {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.write.mode("overwrite").parquet(args.output)

    # 统计
    total = merged.count()
    pos = merged.filter(F.col("label") == 1).count()
    print(f"[INFO] 完成：{total} 行，正样本 {pos}")

    spark.sparkContext.stop()


def _date_range(start: str, end: str):
    from datetime import datetime, timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    cur = s
    while cur <= e:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


if __name__ == "__main__":
    main()
