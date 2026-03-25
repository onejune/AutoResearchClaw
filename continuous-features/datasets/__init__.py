"""数据集工厂模块"""
from .base import BaseDataset
from .criteo_conv import CriteoConvDataset
from .criteo_std import CriteoStdDataset


def get_dataset(config) -> BaseDataset:
    """根据 config.dataset 返回对应数据集实例"""
    name = config.dataset.lower()
    if name == "criteo_conv":
        return CriteoConvDataset()
    elif name == "criteo_std":
        return CriteoStdDataset()
    else:
        raise ValueError(f"未知数据集：{name}，可选：criteo_conv / criteo_std")
