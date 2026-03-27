#!/usr/bin/env python3
"""
prepare_dataset_v2.py
逐天处理数据，避免 Spark OOM：
1. 每天单独采样后保存为 parquet
2. 最后合并所有天的数据

用法：
    python scripts/prepare_dataset_v2.py \
        --start_date 2026-03-01 --end_date 2026-03-14 \
        --val_date 2026-03-15,2026-03-16,2026-03-17
"""
import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta


def date_range(start: str, end: str):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    cur = s
    while cur <= e:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def process_single_day(date_str: str, output_dir: str, force: bool):
    """处理单天数据，保存到 {output_dir}/daily/{date_str}.parquet"""
    daily_dir = os.path.join(output_dir, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    out_path = os.path.join(daily_dir, f"{date_str}.parquet")

    if os.path.exists(out_path) and not force:
        print(f"[SKIP] {date_str} 已存在：{out_path}")
        return out_path

    print(f"[PROC] 处理 {date_str}...")
    # 调用单天处理脚本
    script = os.path.join(os.path.dirname(__file__), "process_one_day.py")
    result = subprocess.run(
        [sys.executable, script, "--date", date_str, "--output", out_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] {date_str} 处理失败：\n{result.stderr}")
        return None
    print(f"[DONE] {date_str} → {out_path}")
    return out_path


def merge_parquets(parquet_paths: list, output_path: str):
    """合并多个 parquet 文件"""
    import pandas as pd
    dfs = []
    for p in parquet_paths:
        if p and os.path.exists(p):
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise RuntimeError("没有有效的 parquet 文件可合并")
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(output_path, index=False)
    print(f"[MERGE] 合并 {len(dfs)} 个文件 → {output_path}，共 {len(merged)} 行")
    return len(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--val_date", required=True)
    parser.add_argument("--output_dir", default="./dataset")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # 处理训练集
    print("=" * 60)
    print("训练集")
    print("=" * 60)
    train_paths = []
    for d in date_range(args.start_date, args.end_date):
        p = process_single_day(d, args.output_dir, args.force)
        if p:
            train_paths.append(p)

    train_out = os.path.join(args.output_dir, f"train_{args.start_date}_{args.end_date}", "data.parquet")
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    merge_parquets(train_paths, train_out)

    # 处理验证集
    print("\n" + "=" * 60)
    print("验证集")
    print("=" * 60)
    val_dates = [d.strip() for d in args.val_date.split(",") if d.strip()]
    val_paths = []
    for d in val_dates:
        p = process_single_day(d, args.output_dir, args.force)
        if p:
            val_paths.append(p)

    val_tag = "_".join(val_dates) if len(val_dates) > 1 else val_dates[0]
    val_out = os.path.join(args.output_dir, f"val_{val_tag}", "data.parquet")
    os.makedirs(os.path.dirname(val_out), exist_ok=True)
    merge_parquets(val_paths, val_out)

    print("\n" + "=" * 60)
    print("完成！")
    print(f"训练集：{train_out}")
    print(f"验证集：{val_out}")


if __name__ == "__main__":
    main()
