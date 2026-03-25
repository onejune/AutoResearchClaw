"""
主入口：依次跑所有 7 种连续特征编码器，输出对比表
支持多数据集，通过 config.dataset 切换
"""
import os
import sys
import time
import random
import logging
import numpy as np
import torch

# 将当前目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data import get_dataloaders
from feature_encoders import (
    build_encoder, BucketEncoder, MinMaxEncoder, StandardScalerEncoder
)
from models import DeepFM
from trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(encoder_name: str, config: Config, train_loader, test_loader, device):
    """运行单个编码器实验，返回 (auc, n_params, time_seconds)"""
    set_seed(config.seed)
    config.encoder = encoder_name

    print(f"\n{'─'*50}")
    print(f"▶ 运行编码器：{encoder_name}")
    print(f"{'─'*50}")

    # 构建编码器（n_continuous 已由 get_dataloaders 写入 config）
    encoder = build_encoder(config)

    # 需要 fit 的编码器：BucketEncoder / MinMaxEncoder / StandardScalerEncoder
    if isinstance(encoder, (BucketEncoder, MinMaxEncoder, StandardScalerEncoder)):
        print("  拟合统计参数（分桶边界 / min-max / mean-std）...")
        train_subset = train_loader.dataset
        n_fit = min(100_000, len(train_subset))
        cont_data = []
        for idx in range(n_fit):
            item = train_subset[idx]
            cont_data.append(item[0].numpy())
        cont_arr = np.array(cont_data)
        encoder.fit(cont_arr)

    # 构建 DeepFM 模型
    cat_vocab_sizes = [config.cat_vocab_size] * config.n_categorical
    model = DeepFM(
        feature_encoder=encoder,
        cat_vocab_sizes=cat_vocab_sizes,
        embedding_dim=config.embedding_dim,
        n_continuous=config.n_continuous,
        n_categorical=config.n_categorical,
    )

    n_params = count_parameters(model)
    print(f"  参数量：{n_params:,} ({n_params/1000:.1f} K)")

    # 训练
    trainer = Trainer(model, config, device)
    best_auc, total_time = trainer.fit(train_loader, test_loader)

    print(f"  最佳 AUC：{best_auc:.4f}，总时间：{total_time:.1f}s")
    return best_auc, n_params, total_time


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"数据集：{config.dataset}")

    # 加载数据（只加载一次，同时更新 config.n_continuous / n_categorical）
    print(f"\n加载数据（{config.sample_size:,} 行）...")
    train_loader, test_loader, dataset = get_dataloaders(config)
    print(f"连续特征：{config.n_continuous}，类别特征：{config.n_categorical}")
    print(f"训练集：{len(train_loader.dataset):,}，测试集：{len(test_loader.dataset):,}")

    # 所有编码器
    encoders = [
        "none",
        "scalar",
        "bucket",
        "autodis",
        "numeric",
        "fttransformer",
        "periodic",
        "field",
        "dlrm",
        "minmax",
        "standard",
        "log",
    ]
    encoder_display_names = {
        "none": "NoneEncoder(ablation)",
        "scalar": "ScalarEncoder",
        "bucket": "BucketEncoder",
        "autodis": "AutoDisEncoder",
        "numeric": "NumericEmbedding",
        "fttransformer": "FTTransformer",
        "periodic": "PeriodicEncoder",
        "field": "FieldEmbedding",
        "dlrm": "DLRMEncoder",
        "minmax": "MinMaxNorm",
        "standard": "StandardScaler",
        "log": "LogTransform",
    }

    results = {}
    for enc_name in encoders:
        try:
            auc, n_params, elapsed = run_experiment(
                enc_name, config, train_loader, test_loader, device
            )
            results[enc_name] = {
                "auc": auc,
                "params": n_params,
                "time": elapsed,
                "display": encoder_display_names[enc_name],
            }
        except Exception as e:
            logger.error(f"编码器 {enc_name} 运行失败：{e}", exc_info=True)
            results[enc_name] = {
                "auc": float("nan"),
                "params": 0,
                "time": 0,
                "display": encoder_display_names[enc_name],
                "error": str(e),
            }

    # ── 打印对比表 ──
    print("\n")
    print("=" * 65)
    print(f"连续特征处理方法对比（{config.dataset}，{config.sample_size//10000}万样本）")
    print("=" * 65)
    print(f"{'方法':<22} {'AUC':>8} {'参数量':>10} {'训练时间':>10}")
    print("-" * 65)
    for enc_name in encoders:
        r = results[enc_name]
        if "error" in r:
            print(f"{r['display']:<22} {'ERROR':>8} {'N/A':>10} {'N/A':>10}")
        else:
            params_k = r["params"] / 1000
            print(
                f"{r['display']:<22} {r['auc']:>8.4f} {params_k:>8.1f} K {r['time']:>8.1f}s"
            )
    print("=" * 65)

    # 保存结果到文件
    result_path = os.path.join(config.output_dir, "comparison.txt")
    with open(result_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write(f"连续特征处理方法对比（{config.dataset}，{config.sample_size//10000}万样本）\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'方法':<22} {'AUC':>8} {'参数量':>10} {'训练时间':>10}\n")
        f.write("-" * 65 + "\n")
        for enc_name in encoders:
            r = results[enc_name]
            if "error" in r:
                f.write(f"{r['display']:<22} {'ERROR':>8} {'N/A':>10} {'N/A':>10}\n")
            else:
                params_k = r["params"] / 1000
                f.write(
                    f"{r['display']:<22} {r['auc']:>8.4f} {params_k:>8.1f} K {r['time']:>8.1f}s\n"
                )
        f.write("=" * 65 + "\n")
    print(f"\n结果已保存到：{result_path}")

    return results


if __name__ == "__main__":
    main()
