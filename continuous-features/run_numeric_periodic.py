"""
临时脚本：全量 Criteo 标准版，只跑 NumericEmbedding + PeriodicEncoder
"""
import os
import sys
import time
import random
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data import get_dataloaders
from feature_encoders import build_encoder
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


def run_experiment(encoder_name, config, train_loader, test_loader, device):
    set_seed(config.seed)
    config.encoder = encoder_name
    print(f"\n{'─'*50}")
    print(f"▶ 运行编码器：{encoder_name}")
    print(f"{'─'*50}")

    encoder = build_encoder(config)
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

    trainer = Trainer(model, config, device)
    best_auc, total_time = trainer.fit(train_loader, test_loader)
    print(f"  最佳 AUC：{best_auc:.4f}，总时间：{total_time:.1f}s")
    return best_auc, n_params, total_time


def main():
    config = Config()
    config.dataset = "criteo_std"
    config.sample_size = 0  # 全量

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"数据集：{config.dataset}，全量（sample_size=0）")

    print("\n加载全量数据...")
    train_loader, test_loader, dataset = get_dataloaders(config)
    print(f"连续特征：{config.n_continuous}，类别特征：{config.n_categorical}")
    print(f"训练集：{len(train_loader.dataset):,}，测试集：{len(test_loader.dataset):,}")

    encoders_to_run = ["numeric", "periodic"]
    display_names = {
        "numeric": "NumericEmbedding",
        "periodic": "PeriodicEncoder",
    }

    results = {}
    for enc_name in encoders_to_run:
        try:
            auc, n_params, elapsed = run_experiment(
                enc_name, config, train_loader, test_loader, device
            )
            results[enc_name] = {"auc": auc, "params": n_params, "time": elapsed}
        except Exception as e:
            logger.error(f"编码器 {enc_name} 失败：{e}", exc_info=True)
            results[enc_name] = {"auc": float("nan"), "params": 0, "time": 0, "error": str(e)}

    # 打印最终结果
    print("\n" + "=" * 60)
    print("最终结果（全量 Criteo 标准版）")
    print("=" * 60)
    for enc_name in encoders_to_run:
        r = results[enc_name]
        dname = display_names[enc_name]
        if "error" in r:
            print(f"{dname:<20} ERROR: {r['error']}")
        else:
            params_k = r["params"] / 1000
            print(f"{dname:<20} AUC={r['auc']:.4f}  params={params_k:.1f}K  time={r['time']:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
