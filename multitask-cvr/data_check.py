#!/usr/bin/env python3
"""
Ali-CCP 数据集质量检查脚本
检查内容：
1. skeleton 文件基本信息（总行数、正样本率）
2. common_features join 质量
3. 特征缺失率（join 后）
4. 特征 vocab 统计
"""

import sys
import os
import time
from collections import defaultdict, Counter

# 数据路径
SKELETON_PATH = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/sample_skeleton_train.csv"
COMMON_FEATURES_PATH = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/common_features_train.csv"
OUTPUT_PATH = "/mnt/workspace/open_research/autoresearch/exp_multitask/results/data_quality_ali_ccp.md"

SAMPLE_SIZE = 500_000  # 采样 50 万行

# 需要统计的 feature_id
SKELETON_FEATURE_IDS = ['205', '206', '207', '210', '216', '301']
COMMON_FEATURE_IDS = ['101', '121', '122', '124', '125', '126', '127', '128', '129']
ALL_FEATURE_IDS = SKELETON_FEATURE_IDS + COMMON_FEATURE_IDS

SEP1 = '\x01'  # 特征间分隔符
SEP2 = '\x02'  # feature_id 与 value 分隔符
SEP3 = '\x03'  # value 与 weight 分隔符


def parse_feature_str(feat_str):
    """解析特征串，返回 {feature_id: [value, ...]} 字典（同一 feature_id 可能有多个值）"""
    result = defaultdict(list)
    if not feat_str or feat_str.strip() == '':
        return result
    parts = feat_str.split(SEP1)
    for part in parts:
        if SEP2 in part:
            fid_val = part.split(SEP2)
            fid = fid_val[0].strip()
            val_weight = fid_val[1] if len(fid_val) > 1 else ''
            val = val_weight.split(SEP3)[0].strip() if SEP3 in val_weight else val_weight.strip()
            result[fid].append(val)
    return result


def main():
    lines = []
    print("=" * 60)
    print("Ali-CCP 数据质量检查")
    print("=" * 60)

    # ============================================================
    # 1. skeleton 文件基本信息
    # ============================================================
    print("\n[1/4] 读取 skeleton 文件（前 50 万行）...")
    t0 = time.time()

    sample_rows = []  # (click, buy, user_id, feat_dict)
    total_lines_counted = 0
    click_sum = 0
    buy_sum = 0
    ctcvr_sum = 0

    with open(SKELETON_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_lines_counted += 1
            line = line.rstrip('\n')
            parts = line.split(',', 5)
            if len(parts) < 5:
                continue
            try:
                click = int(parts[1])
                buy = int(parts[2])
            except ValueError:
                continue
            click_sum += click
            buy_sum += buy
            ctcvr_sum += (click * buy)

            if i < SAMPLE_SIZE:
                user_id = parts[3]
                feat_str = parts[5] if len(parts) > 5 else ''
                feat_dict = parse_feature_str(feat_str)
                sample_rows.append((click, buy, user_id, feat_dict))

    elapsed = time.time() - t0
    total_lines = total_lines_counted
    sample_n = len(sample_rows)
    click_rate = click_sum / total_lines if total_lines > 0 else 0
    buy_rate = buy_sum / total_lines if total_lines > 0 else 0
    ctcvr_rate = ctcvr_sum / total_lines if total_lines > 0 else 0

    print(f"  总行数: {total_lines:,}")
    print(f"  采样行数: {sample_n:,}")
    print(f"  click 正样本率: {click_rate:.4%}")
    print(f"  buy 正样本率: {buy_rate:.4%}")
    print(f"  ctcvr 正样本率: {ctcvr_rate:.4%}")
    print(f"  耗时: {elapsed:.1f}s")

    lines.append("# Ali-CCP 数据质量检查报告\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append("---\n\n")

    lines.append("## 1. Skeleton 文件基本信息\n\n")
    lines.append(f"- **数据路径**: `{SKELETON_PATH}`\n")
    lines.append(f"- **总行数**: {total_lines:,}\n")
    lines.append(f"- **采样行数（用于后续分析）**: {sample_n:,}\n")
    lines.append(f"- **click 正样本率**: {click_rate:.4%} ({click_sum:,} / {total_lines:,})\n")
    lines.append(f"- **buy 正样本率**: {buy_rate:.4%} ({buy_sum:,} / {total_lines:,})\n")
    lines.append(f"- **ctcvr 正样本率（click×buy）**: {ctcvr_rate:.4%} ({ctcvr_sum:,} / {total_lines:,})\n\n")

    # ============================================================
    # 2. common_features join 质量
    # ============================================================
    print("\n[2/4] 读取 common_features 并构建索引...")
    t0 = time.time()

    # 获取采样中的 user_id 集合，只加载这些用户的特征
    sample_user_ids = set(row[2] for row in sample_rows)
    print(f"  采样中唯一 user_id 数: {len(sample_user_ids):,}")

    cf_dict = {}  # user_id -> feat_dict
    cf_total_lines = 0
    with open(COMMON_FEATURES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            cf_total_lines += 1
            line = line.rstrip('\n')
            parts = line.split(',', 2)
            if len(parts) < 2:
                continue
            user_id = parts[0].strip()
            if user_id in sample_user_ids:
                feat_str = parts[2] if len(parts) > 2 else ''
                feat_dict = parse_feature_str(feat_str)
                cf_dict[user_id] = feat_dict

    elapsed = time.time() - t0
    hit_user_ids = set(cf_dict.keys())
    hit_count = sum(1 for row in sample_rows if row[2] in hit_user_ids)
    hit_rate = hit_count / sample_n if sample_n > 0 else 0

    print(f"  common_features 总行数: {cf_total_lines:,}")
    print(f"  命中 user_id 数: {len(hit_user_ids):,}")
    print(f"  采样中 join 命中行数: {hit_count:,} / {sample_n:,} ({hit_rate:.2%})")
    print(f"  耗时: {elapsed:.1f}s")

    lines.append("## 2. Common Features Join 质量\n\n")
    lines.append(f"- **common_features 路径**: `{COMMON_FEATURES_PATH}`\n")
    lines.append(f"- **common_features 总行数**: {cf_total_lines:,}\n")
    lines.append(f"- **采样中唯一 user_id 数**: {len(sample_user_ids):,}\n")
    lines.append(f"- **命中 user_id 数（在 common_features 中找到）**: {len(hit_user_ids):,}\n")
    lines.append(f"- **采样中 join 命中行数**: {hit_count:,} / {sample_n:,}\n")
    lines.append(f"- **join 命中率**: {hit_rate:.4%}\n\n")

    # ============================================================
    # 3. 特征缺失率（join 后）
    # ============================================================
    print("\n[3/4] 统计特征缺失率...")
    t0 = time.time()

    # 只统计 join 命中的行
    hit_rows = [(row[0], row[1], row[2], row[3]) for row in sample_rows if row[2] in hit_user_ids]
    hit_n = len(hit_rows)

    # skeleton 特征缺失率：从 skeleton 的特征串中解析
    skel_feat_missing = {fid: 0 for fid in SKELETON_FEATURE_IDS}
    # common_features 特征缺失率：从 common_features 中解析
    cf_feat_missing = {fid: 0 for fid in COMMON_FEATURE_IDS}

    for click, buy, user_id, skel_feat_dict in hit_rows:
        cf_feat_dict = cf_dict.get(user_id, {})
        for fid in SKELETON_FEATURE_IDS:
            if fid not in skel_feat_dict or len(skel_feat_dict[fid]) == 0:
                skel_feat_missing[fid] += 1
        for fid in COMMON_FEATURE_IDS:
            if fid not in cf_feat_dict or len(cf_feat_dict[fid]) == 0:
                cf_feat_missing[fid] += 1

    elapsed = time.time() - t0
    print(f"  join 命中行数: {hit_n:,}")
    print(f"  耗时: {elapsed:.1f}s")

    lines.append("## 3. 特征缺失率（Join 后，基于命中行）\n\n")
    lines.append(f"分析基于 join 命中的 **{hit_n:,}** 行数据。\n\n")
    lines.append("### 3.1 Skeleton 特征（feature_id: 205, 206, 207, 210, 216, 301）\n\n")
    lines.append("| feature_id | 缺失行数 | 缺失率 |\n")
    lines.append("|-----------|---------|-------|\n")
    for fid in SKELETON_FEATURE_IDS:
        miss = skel_feat_missing[fid]
        rate = miss / hit_n if hit_n > 0 else 0
        lines.append(f"| {fid} | {miss:,} | {rate:.4%} |\n")
    lines.append("\n")

    lines.append("### 3.2 Common Features 特征（feature_id: 101, 121, 122, 124, 125, 126, 127, 128, 129）\n\n")
    lines.append("| feature_id | 缺失行数 | 缺失率 |\n")
    lines.append("|-----------|---------|-------|\n")
    for fid in COMMON_FEATURE_IDS:
        miss = cf_feat_missing[fid]
        rate = miss / hit_n if hit_n > 0 else 0
        lines.append(f"| {fid} | {miss:,} | {rate:.4%} |\n")
    lines.append("\n")

    # ============================================================
    # 4. 特征 vocab 统计
    # ============================================================
    print("\n[4/4] 统计特征 vocab...")
    t0 = time.time()

    # 收集各特征的所有 value
    feat_counters = {fid: Counter() for fid in ALL_FEATURE_IDS}

    for click, buy, user_id, skel_feat_dict in hit_rows:
        cf_feat_dict = cf_dict.get(user_id, {})
        for fid in SKELETON_FEATURE_IDS:
            for val in skel_feat_dict.get(fid, []):
                feat_counters[fid][val] += 1
        for fid in COMMON_FEATURE_IDS:
            for val in cf_feat_dict.get(fid, []):
                feat_counters[fid][val] += 1

    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s")

    lines.append("## 4. 特征 Vocab 统计\n\n")
    lines.append(f"基于 join 命中的 **{hit_n:,}** 行数据统计。\n\n")

    lines.append("### 4.1 Skeleton 特征\n\n")
    for fid in SKELETON_FEATURE_IDS:
        counter = feat_counters[fid]
        unique_count = len(counter)
        top5 = counter.most_common(5)
        lines.append(f"#### feature_id = {fid}\n\n")
        lines.append(f"- **unique value 数**: {unique_count:,}\n")
        lines.append(f"- **Top-5 values**:\n\n")
        lines.append("  | value | 出现次数 |\n")
        lines.append("  |-------|--------|\n")
        for val, cnt in top5:
            lines.append(f"  | `{val}` | {cnt:,} |\n")
        lines.append("\n")

    lines.append("### 4.2 Common Features 特征\n\n")
    for fid in COMMON_FEATURE_IDS:
        counter = feat_counters[fid]
        unique_count = len(counter)
        top5 = counter.most_common(5)
        lines.append(f"#### feature_id = {fid}\n\n")
        lines.append(f"- **unique value 数**: {unique_count:,}\n")
        lines.append(f"- **Top-5 values**:\n\n")
        lines.append("  | value | 出现次数 |\n")
        lines.append("  |-------|--------|\n")
        for val, cnt in top5:
            lines.append(f"  | `{val}` | {cnt:,} |\n")
        lines.append("\n")

    # ============================================================
    # 写入输出文件
    # ============================================================
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\n结果已写入: {OUTPUT_PATH}")
    print("\nDATA CHECK DONE")


if __name__ == '__main__':
    main()
