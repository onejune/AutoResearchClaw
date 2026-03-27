#!/usr/bin/env python3
"""
run_exp_002_005.py
串行跑 exp_002~005（SharedBottom / MMoE / PLE / STAR），完成后发飞书消息。
修复了分层采样 bug 后的重跑版本。
"""
import os
import sys
import subprocess
import json
import time
from datetime import datetime

sys.path.insert(0, "/mnt/workspace/open_research/rec-autopilot/src")
from feishu_notifier import FeishuNotifier

import pandas as pd

WORK_DIR = "/mnt/workspace/open_research/autoresearch/exp_advertiser_adaptive"
PYTHON = sys.executable
TRAIN_SCRIPT = os.path.join(WORK_DIR, "scripts/train.py")

EXPERIMENTS = [
    "exp_002_shared_bottom",
    "exp_003_mmoe",
    "exp_004_ple",
    "exp_005_star",
]

def run_exp(exp_name):
    conf_path = os.path.join(WORK_DIR, f"conf/experiments/{exp_name}.yaml")
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动实验: {exp_name}")
    print(f"{'='*60}")
    start = time.time()
    ret = subprocess.run(
        [PYTHON, TRAIN_SCRIPT, "--conf", conf_path, "--exp_name", exp_name],
        cwd=WORK_DIR,
    )
    elapsed = (time.time() - start) / 60
    success = ret.returncode == 0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {exp_name} {'完成' if success else '失败'} ({elapsed:.1f} min)")
    return success, elapsed

def load_metrics(exp_name):
    path = os.path.join(WORK_DIR, f"experiments/{exp_name}/metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def send_feishu_report(results):
    """发送完整对比表到飞书"""
    rows = []
    for exp_name, metrics, elapsed, success in results:
        if not success or metrics is None:
            rows.append({
                "模型": exp_name,
                "Overall AUC": "失败",
                "PCOC": "-",
                "ae AUC": "-",
                "shopee AUC": "-",
                "lazada AUC": "-",
                "shein AUC": "-",
                "耗时(min)": f"{elapsed:.1f}",
            })
            continue
        m = metrics.get("metrics", {})
        overall = m.get("Overall", {})
        rows.append({
            "模型": exp_name.replace("exp_00", "exp_0").replace("_", " "),
            "Overall AUC": f"{overall.get('auc', 0):.4f}",
            "PCOC": f"{overall.get('pcoc', 0):.3f}",
            "ae AUC": f"{m.get('aliexpress', {}).get('auc', 0):.4f}",
            "shopee AUC": f"{m.get('shopee', {}).get('auc', 0):.4f}",
            "lazada AUC": f"{m.get('lazada', {}).get('auc', 0):.4f}",
            "shein AUC": f"{m.get('shein', {}).get('auc', 0):.4f}",
            "耗时(min)": f"{elapsed:.1f}",
        })

    # 加入 baseline 对比
    baseline_metrics = load_metrics("exp_001_baseline")
    if baseline_metrics:
        m = baseline_metrics.get("metrics", {})
        overall = m.get("Overall", {})
        rows.insert(0, {
            "模型": "exp_001 baseline",
            "Overall AUC": f"{overall.get('auc', 0):.4f}",
            "PCOC": f"{overall.get('pcoc', 0):.3f}",
            "ae AUC": f"{m.get('aliexpress', {}).get('auc', 0):.4f}",
            "shopee AUC": f"{m.get('shopee', {}).get('auc', 0):.4f}",
            "lazada AUC": f"{m.get('lazada', {}).get('auc', 0):.4f}",
            "shein AUC": f"{m.get('shein', {}).get('auc', 0):.4f}",
            "耗时(min)": "39.4",
        })

    df = pd.DataFrame(rows)
    text = "✅ DSP 分广告主自适应建模实验完成（exp_002~005，分层采样修复后重跑）\n数据：ivr_sample_v16，训练 2026-03-01~14，验证 2026-03-15~17"
    FeishuNotifier.send_dataframe_html(
        df,
        title="广告主自适应模型对比结果",
        subject="exp_002~005 实验完成",
        text=text,
    )
    print("\n✅ 飞书消息已发送")

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始串行跑 exp_002~005")
    results = []
    for exp_name in EXPERIMENTS:
        success, elapsed = run_exp(exp_name)
        metrics = load_metrics(exp_name) if success else None
        results.append((exp_name, metrics, elapsed, success))

    print("\n\n===== 全部实验完成，发送飞书消息 =====")
    send_feishu_report(results)

if __name__ == "__main__":
    main()
