"""
spark_loader.py
用 PySpark 读取 ivr_sample_v16 parquet 数据，执行采样，输出 pandas DataFrame。
支持多天合并，结果可缓存到本地 parquet。
"""
import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

# domain 映射规则（前缀匹配，顺序敏感）
DOMAIN_PREFIX_MAP = [
    ("shein", 0),
    ("ae", 1),       # aliexpress
    ("shopee", 2),
    ("lazada", 3),
]
DEFAULT_DOMAIN = 0

# 目标广告主白名单前缀（只保留这些 business_type 的样本）
ADVERTISER_WHITELIST_PREFIXES = ("shein", "ae", "shopee", "lazada")

# 采样率
SHEIN_NEG_RATIO = 0.01
OTHER_NEG_RATIO = 0.10


def _map_domain(business_type: str) -> int:
    """将 business_type 字符串映射为 domain id。"""
    if not business_type or business_type == "none":
        return DEFAULT_DOMAIN
    bt = business_type.lower()
    for prefix, did in DOMAIN_PREFIX_MAP:
        if bt.startswith(prefix):
            return did
    return DEFAULT_DOMAIN


def _date_range(start: str, end: str) -> List[str]:
    """生成 [start, end] 的日期列表，格式 YYYY-MM-DD。"""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    cur = s
    while cur <= e:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def load_data_spark(
    base_path: str,
    feature_cols: List[str],
    start_date: str,
    end_date: str,
    spark_conf: Optional[dict] = None,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    用 PySpark 读取多天 parquet 数据，采样后返回 pandas DataFrame。

    Args:
        base_path: parquet 根目录，子目录格式 part=YYYY-MM-DD 或 YYYY-MM-DD
        feature_cols: 特征列名列表（来自 combine_schema）
        start_date: 训练开始日期 YYYY-MM-DD
        end_date: 训练结束日期 YYYY-MM-DD
        spark_conf: Spark 配置字典（可选）
        label_col: 标签列名

    Returns:
        pandas DataFrame，包含 feature_cols + label + domain_indicator
    """
    try:
        import pyspark
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError:
        raise ImportError("PySpark 未安装，请先安装：pip install pyspark")

    # 初始化 Spark
    builder = SparkSession.builder.appName("advertiser_adaptive_loader")
    if spark_conf:
        master = spark_conf.get("local_spark", "local")
        builder = builder.master(master)
        for k, v in spark_conf.items():
            if k not in ("local_spark", "app_name"):
                builder = builder.config(f"spark.{k}", str(v))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    dates = _date_range(start_date, end_date)
    all_cols = feature_cols + [label_col, "business_type", "objective_type"]
    # 去重
    all_cols = list(dict.fromkeys(all_cols))

    dfs = []
    for date_str in dates:
        part_path = os.path.join(base_path, f"part={date_str}")
        plain_path = os.path.join(base_path, date_str)
        data_path = part_path if os.path.exists(part_path) else plain_path
        if not os.path.exists(data_path):
            print(f"[WARN] 数据路径不存在，跳过：{data_path}")
            continue

        df = spark.read.parquet(data_path)
        # 只选需要的列（容错：列不存在则跳过）
        avail = set(df.columns)
        select_cols = [c for c in all_cols if c in avail]
        df = df.select(*select_cols)

        # 类型转换
        for col in df.columns:
            if col == label_col:
                df = df.withColumn(col, F.col(col).cast("float"))
            else:
                df = df.withColumn(col, F.col(col).cast("string"))

        # ── 广告主白名单过滤（只保留 shein / ae* / shopee* / lazada*）──────
        # 用 startswith 前缀匹配，过滤掉其他广告主
        whitelist_cond = F.lit(False)
        for prefix in ADVERTISER_WHITELIST_PREFIXES:
            whitelist_cond = whitelist_cond | F.lower(F.col("business_type")).startswith(prefix)
        df = df.filter(whitelist_cond)

        # ── 采样：正样本全保留，负样本按广告主差异化采样 ─────────────────
        # shein（或 SALES_WEBSITE 目标）负样本 1%，其他 10%
        df = df.filter(
            (F.col(label_col) == 1) | (
                (F.col(label_col) == 0) &
                F.when(
                    F.lower(F.col("business_type")).startswith("shein") |
                    (F.col("objective_type") == "SALES_WEBSITE"),
                    F.rand(seed=42) < SHEIN_NEG_RATIO
                ).otherwise(F.rand(seed=42) < OTHER_NEG_RATIO)
            )
        )
        df = df.fillna("none")
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"没有找到任何有效数据，路径：{base_path}，日期：{dates}")

    # 合并多天
    from functools import reduce
    merged = reduce(lambda a, b: a.union(b), dfs)

    # 转 pandas
    pdf = merged.toPandas()
    spark.stop()

    # 添加 domain_indicator（pandas 端处理，避免 UDF 开销）
    pdf["domain_indicator"] = pdf["business_type"].apply(
        lambda x: _map_domain(str(x)) if pd.notna(x) else DEFAULT_DOMAIN
    )

    # 只保留最终需要的列
    keep_cols = feature_cols + [label_col, "domain_indicator"]
    keep_cols = [c for c in keep_cols if c in pdf.columns]
    pdf = pdf[keep_cols].fillna("none")

    domain_id_to_name = {0: "shein", 1: "aliexpress", 2: "shopee", 3: "lazada"}
    domain_dist = {domain_id_to_name.get(k, k): v
                   for k, v in pdf["domain_indicator"].value_counts().to_dict().items()}
    print(f"[INFO] 加载完成：{len(pdf)} 行，正样本 {int(pdf[label_col].sum())}，"
          f"domain 分布：{domain_dist}")
    return pdf
