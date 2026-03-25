"""
训练主循环

用法：
  python train.py --model shared_bottom
  python train.py --model esmm --sample_size 500000
  python train.py --model mmoe --seed 123
"""

import argparse
import sys
import os
import time

# 确保能 import 同目录模块
sys.path.insert(0, os.path.dirname(__file__))

from data import load_dataset
from models import get_model
from evaluate import evaluate, print_results


def parse_args():
    parser = argparse.ArgumentParser(description="多任务学习训练脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="shared_bottom",
        choices=["shared_bottom", "esmm", "mmoe", "escm2"],
        help="模型类型",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2_000_000,
        help="采样行数（默认 200 万）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--n_experts",
        type=int,
        default=4,
        help="MMoE 专家数量",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=0.1,
        help="LR 正则化系数",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="ESCM2 反事实正则化强度 λ",
    )
    return parser.parse_args()


def train_and_evaluate(model_name: str,
                       sample_size: int = 2_000_000,
                       seed: int = 42,
                       **model_kwargs) -> dict:
    """
    完整的训练+评估流程。

    返回 dict: {"ctr_auc": ..., "cvr_auc": ..., "ctcvr_auc": ...}
    """
    t0 = time.time()

    # 1. 加载数据
    print(f"\n{'='*50}")
    print(f"模型: {model_name} | 采样: {sample_size:,} | seed: {seed}")
    print(f"{'='*50}")
    X_train, X_test, y_train, y_test = load_dataset(
        sample_size=sample_size, seed=seed
    )

    # 2. 初始化模型
    model = get_model(model_name, seed=seed, **model_kwargs)
    print(f"\n[train] 模型: {model}")

    # 3. 训练
    t_train = time.time()
    if model_name == "shared_bottom":
        model.fit(X_train, y_train["ctr_label"], y_train["cvr_label"])
    elif model_name == "esmm":
        model.fit(X_train, y_train["ctr_label"], y_train["ctcvr_label"])
    elif model_name == "mmoe":
        model.fit(X_train, y_train["ctr_label"], y_train["cvr_label"])
    elif model_name == "escm2":
        model.fit(X_train, y_train["ctr_label"], y_train["ctcvr_label"])
    print(f"[train] 训练耗时: {time.time() - t_train:.1f}s")

    # 4. 评估
    results = evaluate(model, X_test, y_test)
    print_results(model_name, results)
    print(f"[train] 总耗时: {time.time() - t0:.1f}s")

    return results


def main():
    args = parse_args()

    model_kwargs = {"C": args.C}
    if args.model == "mmoe":
        model_kwargs["n_experts"] = args.n_experts
    if args.model == "escm2":
        model_kwargs["lam"] = args.lam

    results = train_and_evaluate(
        model_name=args.model,
        sample_size=args.sample_size,
        seed=args.seed,
        **model_kwargs,
    )

    # 输出机器可读结果
    print("\n[结果]")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
