"""
临时脚本：只跑 BucketEncoder + AutoDisEncoder，全量 Criteo 标准版
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
from feature_encoders import build_encoder, BucketEncoder
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

    encoder = build_encoder(config)

    if isinstance(encoder, BucketEncoder):
        print("  拟合分桶边界...")
        train_subset = train_loader.dataset
        n_fit = min(100_000, len(train_subset))
        cont_data = []
        for idx in range(n_fit):
            item = train_subset[idx]
            cont_data.append(item[0].numpy())
        cont_arr = np.array(cont_data)
        encoder.fit(cont_arr)

    cat_vocab_sizes = [config.cat_vocab_size] * config.n_categorical
    model = DeepFM(
        feature_encoder=encoder,
        cat_vocab_sizes=cat_vocab_sizes,
        embedding_dim=config.embedding_dim,
        n_continuous=config.n_continuous,
        n_categorical=config.n_categorical,
    )

    n_params = count_parameters(model)
    trainer = Trainer(model, config, device)
    best_auc, total_time = trainer.fit(train_loader, test_loader)

    return best_auc, n_params, total_time


def main():
    config = Config()
    config.dataset = "criteo_std"
    config.sample_size = 0  # 全量（0 表示不采样）

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"数据集：{config.dataset}，sample_size={config.sample_size}（全量）")

    print("\n加载数据（全量）...")
    train_loader, test_loader, dataset = get_dataloaders(config)
    print(f"连续特征：{config.n_continuous}，类别特征：{config.n_categorical}")
    print(f"训练集：{len(train_loader.dataset):,}，测试集：{len(test_loader.dataset):,}")

    encoders_to_run = ["bucket", "autodis"]
    display_names = {
        "bucket": "BucketEncoder",
        "autodis": "AutoDisEncoder",
    }

    results = {}
    for enc_name in encoders_to_run:
        print(f"\n{'─'*50}")
        print(f"▶ 运行编码器：{enc_name}")
        print(f"{'─'*50}")
        try:
            auc, n_params, elapsed = run_experiment(
                enc_name, config, train_loader, test_loader, device
            )
            results[enc_name] = {"auc": auc, "params": n_params, "time": elapsed}
            print(f"  {display_names[enc_name]}: AUC={auc:.4f}, params={n_params/1000:.1f}K, time={elapsed:.1f}s")
        except Exception as e:
            logger.error(f"编码器 {enc_name} 运行失败：{e}", exc_info=True)
            results[enc_name] = {"auc": float("nan"), "params": 0, "time": 0, "error": str(e)}

    print("\n===== 最终结果 =====")
    for enc_name in encoders_to_run:
        r = results[enc_name]
        if "error" in r:
            print(f"{display_names[enc_name]:<15} ERROR: {r['error']}")
        else:
            print(f"{display_names[enc_name]:<15} AUC={r['auc']:.4f}  params={r['params']/1000:.1f}K  time={r['time']:.1f}s")


if __name__ == "__main__":
    main()
