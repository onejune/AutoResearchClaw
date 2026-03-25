"""
main.py - 实验入口

依次训练四个多任务模型，最终输出对比表。
"""

import logging
import os
import sys
import time
import random
import numpy as np
import torch

# 确保当前目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config  import Config
from data    import build_dataloaders
from models  import build_model
from trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(model_name: str, config: Config, feature_info: dict,
                   train_loader, val_loader, test_loader):
    """训练单个模型并返回测试集指标"""
    print(f"\n{'='*60}")
    print(f"  训练模型: {model_name.upper()}")
    print(f"{'='*60}")

    cfg = Config(
        data_dir=config.data_dir,
        sample_size=config.sample_size,
        batch_size=config.batch_size,
        model_name=model_name,
        embedding_dim=config.embedding_dim,
        mlp_dims=config.mlp_dims,
        dropout=config.dropout,
        n_experts=config.n_experts,
        escm2_lambda=config.escm2_lambda,
        epochs=config.epochs,
        lr=config.lr,
        seed=config.seed,
        early_stopping_patience=config.early_stopping_patience,
        output_dir=config.output_dir,
    )

    set_seed(cfg.seed)
    model   = build_model(feature_info, cfg)
    trainer = Trainer(model, cfg)

    t0 = time.time()
    trainer.train(train_loader, val_loader)
    elapsed = time.time() - t0

    test_metrics = trainer.evaluate(test_loader)
    return test_metrics, elapsed


def main():
    # ── 配置 ──────────────────────────────────────────────────
    config = Config(
        data_dir    = "",          # 空 = 使用合成数据
        sample_size = 50_000,      # 快速验证用 5 万，完整实验改为 500_000
        batch_size  = 4096,
        embedding_dim = 16,
        mlp_dims    = (128, 64),   # 快速验证用较小网络
        dropout     = 0.1,
        n_experts   = 4,
        escm2_lambda = 0.1,
        epochs      = 5,
        lr          = 1e-3,
        seed        = 42,
        early_stopping_patience = 3,
    )

    print("\n" + "="*60)
    print("  Ali-CCP 多任务学习实验框架")
    print(f"  样本数: {config.sample_size:,} | Batch: {config.batch_size}")
    print("="*60)

    # ── 数据 ──────────────────────────────────────────────────
    logger.info("构建数据集...")
    train_loader, val_loader, test_loader, feature_info = build_dataloaders(config)
    logger.info("特征信息: %s", feature_info)

    # ── 依次训练四个模型 ──────────────────────────────────────
    model_names = ["shared_bottom", "esmm", "mmoe", "escm2"]
    results = {}

    for name in model_names:
        metrics, elapsed = run_experiment(
            name, config, feature_info,
            train_loader, val_loader, test_loader
        )
        results[name] = {"metrics": metrics, "elapsed": elapsed}
        print(f"  测试集结果 → CTR AUC: {metrics['ctr_auc']:.4f} | "
              f"CVR AUC: {metrics['cvr_auc']:.4f} | "
              f"CTCVR AUC: {metrics['ctcvr_auc']:.4f} | "
              f"耗时: {elapsed:.1f}s")

    # ── 输出对比表 ────────────────────────────────────────────
    name_map = {
        "shared_bottom": "Shared-Bottom",
        "esmm":          "ESMM",
        "mmoe":          "MMoE",
        "escm2":         "ESCM2",
    }

    print("\n" + "="*60)
    print("  多任务学习实验结果对比")
    print("="*60)
    header = f"{'模型':<16}{'CTR AUC':>10}{'CVR AUC':>10}{'CTCVR AUC':>12}{'训练时间':>10}"
    print(header)
    print("-"*60)
    for name in model_names:
        m = results[name]["metrics"]
        t = results[name]["elapsed"]
        row = (
            f"{name_map[name]:<16}"
            f"{m['ctr_auc']:>10.4f}"
            f"{m['cvr_auc']:>10.4f}"
            f"{m['ctcvr_auc']:>12.4f}"
            f"{t:>9.1f}s"
        )
        print(row)
    print("="*60)

    # ── AUC 验证 ──────────────────────────────────────────────
    print("\n[验证] 检查 AUC > 0.6 ...")
    all_pass = True
    for name in model_names:
        m = results[name]["metrics"]
        for key in ["ctr_auc", "ctcvr_auc"]:
            val = m[key]
            status = "✓" if val > 0.6 else "✗"
            if val <= 0.6:
                all_pass = False
            print(f"  {status} {name_map[name]} {key}: {val:.4f}")
    if all_pass:
        print("\n✓ 所有关键 AUC 均 > 0.6，实验通过！")
    else:
        print("\n⚠ 部分 AUC 未达 0.6，请检查数据或模型配置。")


if __name__ == "__main__":
    main()
