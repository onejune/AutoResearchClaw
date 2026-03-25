"""数据集工厂模块"""
from .base import BaseMultiTaskDataset
from .synthetic import SyntheticDataset
from .ali_ccp import AliCCPDataset


def get_dataset(config) -> BaseMultiTaskDataset:
    """根据 config.dataset 返回对应数据集实例"""
    name = config.dataset.lower()
    if name == "synthetic":
        return SyntheticDataset()
    elif name == "ali_ccp":
        return AliCCPDataset()
    else:
        raise ValueError(
            f"未知数据集：{name}，可选：synthetic / ali_ccp"
        )
