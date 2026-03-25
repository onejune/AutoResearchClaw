"""数据集工厂模块"""
from .base import BaseMultiTaskDataset
from .synthetic import SyntheticDataset
from .ali_ccp import AliCCPDataset
from .taobao import TaobaoDataset


def get_dataset(config) -> BaseMultiTaskDataset:
    """根据 config.dataset 返回对应数据集实例"""
    name = config.dataset.lower()
    if name == "synthetic":
        return SyntheticDataset()
    elif name == "ali_ccp":
        return AliCCPDataset()
    elif name == "taobao":
        return TaobaoDataset(data_path="")   # data_path 由 get_dataloaders 从 config 读取
    else:
        raise ValueError(
            f"未知数据集：{name}，可选：synthetic / ali_ccp / taobao"
        )
