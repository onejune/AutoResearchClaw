#!/usr/bin/env python3
"""
train.py
训练入口：读取 yaml 配置，加载数据，实例化模型，执行训练，保存结果。

用法：
    python scripts/train.py --conf conf/experiments/exp_005_star.yaml
    python scripts/train.py --conf conf/experiments/exp_005_star.yaml --exp_name exp_005_star_v2
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.data.dataset import IVRDataset, load_schema
from src.models import build_model
from src.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger("train")


def load_config(conf_path: str) -> dict:
    """加载并合并 base.yaml + 实验 yaml。"""
    base_path = os.path.join(os.path.dirname(conf_path), "..", "base.yaml")
    base_path = os.path.normpath(base_path)

    cfg = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg = yaml.safe_load(f) or {}

    with open(conf_path) as f:
        exp_cfg = yaml.safe_load(f) or {}

    # 深度合并（实验配置覆盖 base）
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    return deep_merge(cfg, exp_cfg)


def build_dataset_paths(cfg: dict) -> tuple:
    """根据配置推断训练集/验证集 parquet 路径。
    val_date 支持单天或逗号分隔多天（与 prepare_dataset.py 保持一致）。
    """
    dates = cfg.get("dates", {})
    start = dates.get("train_start", "")
    end = dates.get("train_end", "")
    val_raw = str(dates.get("val_date", ""))
    cache_dir = cfg.get("data", {}).get("dataset_cache_dir", "./dataset")

    train_path = os.path.join(cache_dir, f"train_{start}_{end}", "data.parquet")

    val_dates = [d.strip() for d in val_raw.split(",") if d.strip()]
    val_tag = "_".join(val_dates) if len(val_dates) > 1 else val_dates[0] if val_dates else val_raw
    val_path = os.path.join(cache_dir, f"val_{val_tag}", "data.parquet")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="实验配置文件路径")
    parser.add_argument("--exp_name", default=None, help="实验名称（覆盖配置中的 exp_name）")
    args = parser.parse_args()

    cfg = load_config(args.conf)
    exp_name = args.exp_name or cfg.get("exp_name", "exp_unknown")
    logger.info(f"实验：{exp_name}")
    logger.info(f"配置：{cfg}")

    # ── 加载特征 schema ───────────────────────────────────────────────────
    schema_path = cfg.get("data", {}).get("schema_path", "./combine_schema")
    feature_cols = load_schema(schema_path)
    logger.info(f"特征数量：{len(feature_cols)}")

    # ── 加载数据集 ────────────────────────────────────────────────────────
    train_path, val_path = build_dataset_paths(cfg)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"训练集不存在：{train_path}\n"
            f"请先运行：python scripts/prepare_dataset.py --start_date ... --end_date ... --val_date ..."
        )
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"验证集不存在：{val_path}")

    model_cfg = cfg.get("model", {})
    vocab_size = model_cfg.get("vocab_size", 100_000)
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 1024)

    logger.info(f"加载训练集：{train_path}")
    train_ds = IVRDataset.from_parquet(train_path, feature_cols, vocab_size=vocab_size)
    logger.info(f"加载验证集：{val_path}")
    val_ds = IVRDataset.from_parquet(val_path, feature_cols, vocab_size=vocab_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4)

    # ── 构建模型 ──────────────────────────────────────────────────────────
    model_type = cfg.get("model_type", "baseline")
    model_params = cfg.get("model_params", {})
    model = build_model(
        model_type=model_type,
        feature_cols=feature_cols,
        vocab_size=vocab_size,
        embed_dim=model_cfg.get("embed_dim", 8),
        domain_num=model_cfg.get("domain_num", 4),
        **model_params,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型：{model_type}，参数量：{n_params:,}")

    # ── 训练 ──────────────────────────────────────────────────────────────
    exp_dir = cfg.get("output", {}).get("experiments_dir", "./experiments")
    trainer = Trainer(
        model=model,
        device=train_cfg.get("device", "cpu"),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        early_stop_patience=train_cfg.get("early_stop_patience", 3),
        exp_dir=exp_dir,
        exp_name=exp_name,
    )
    best_metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.get("epochs", 5),
    )
    trainer.save_model()

    # 打印最终结果
    from src.utils.metrics import format_metrics_table
    logger.info(f"\n{'='*60}\n最终验证结果（{exp_name}）\n{'='*60}")
    logger.info("\n" + format_metrics_table(best_metrics))


if __name__ == "__main__":
    main()
