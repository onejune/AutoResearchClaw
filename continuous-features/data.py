"""
数据加载入口（工厂函数）

根据 config.dataset 自动选择对应数据集加载器，
同时将 n_continuous / n_categorical 写回 config。
"""
import logging
from datasets import get_dataset

logger = logging.getLogger(__name__)


def get_dataloaders(config):
    """
    返回 (train_loader, test_loader)
    同时更新 config.n_continuous 和 config.n_categorical
    """
    dataset = get_dataset(config)

    # 将数据集的特征维度写回 config，供 models.py / feature_encoders.py 使用
    config.n_continuous = dataset.n_continuous
    config.n_categorical = dataset.n_categorical

    logger.info(
        f"数据集：{config.dataset}，"
        f"连续特征：{config.n_continuous}，"
        f"类别特征：{config.n_categorical}"
    )

    train_loader, test_loader = dataset.get_dataloaders(config)
    return train_loader, test_loader, dataset
