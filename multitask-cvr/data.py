"""
data.py - 多任务数据加载入口（路由层）

根据 config.dataset 自动选择对应数据集加载器：
  - "synthetic" → datasets/synthetic.py（合成数据，无需数据文件）
  - "ali_ccp"   → datasets/ali_ccp.py（真实 Ali-CCP 数据）
  - 未来扩展：在 datasets/__init__.py 注册即可

返回：(train_loader, val_loader, test_loader, feature_info)

feature_info 格式：
  {
    "sparse_vocab":  {feat_name: vocab_size},
    "sparse_feats":  [feat_name, ...],
    "dense_feats":   [feat_name, ...],
    "dense_dim":     int,
  }

每个 batch 格式：
  (features_dict, ctr_label, cvr_label, ctcvr_label)
"""
import logging
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from datasets import get_dataset

logger = logging.getLogger(__name__)


def build_dataloaders(
    config,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    构建 train / val / test DataLoader，并返回 feature_info。

    Parameters
    ----------
    config    : Config  包含 dataset / data_dir / sample_size / batch_size / seed
    val_ratio : float   从训练集中划出的验证比例

    Returns
    -------
    train_loader, val_loader, test_loader, feature_info
    """
    dataset_name = getattr(config, "dataset", "synthetic")
    logger.info("数据集: %s", dataset_name)

    dataset = get_dataset(config)
    train_loader, val_loader, test_loader, feature_info = dataset.get_dataloaders(
        config, val_ratio=val_ratio
    )

    logger.info(
        "数据加载完成 | train=%d batches | val=%d batches | test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader, feature_info
