"""
datasets/base.py - 多任务数据集抽象基类

所有数据集必须实现此接口，返回统一格式：
  (features_dict, ctr_label, cvr_label, ctcvr_label)
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from torch.utils.data import DataLoader


class BaseMultiTaskDataset(ABC):
    """
    多任务数据集基类。

    子类需设置类属性：
      sparse_feats : list[str]  稀疏特征名列表
      dense_feats  : list[str]  数值特征名列表
      dense_dim    : int        数值特征维度

    子类需实现：
      get_dataloaders(config) → (train_loader, val_loader, test_loader, feature_info)
    """

    sparse_feats: list = []
    dense_feats:  list = []
    dense_dim:    int  = 0

    @abstractmethod
    def get_dataloaders(self, config) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
        """
        加载数据，返回 (train_loader, val_loader, test_loader, feature_info)。

        feature_info 格式：
          {
            "sparse_vocab":  {feat_name: vocab_size},   # 稀疏特征词表大小
            "sparse_feats":  [feat_name, ...],           # 稀疏特征名有序列表
            "dense_feats":   [feat_name, ...],           # 数值特征名有序列表
            "dense_dim":     int,                        # 数值特征总维度
          }

        每个 batch 格式：
          (features_dict, ctr_label, cvr_label, ctcvr_label)
          features_dict: {feat_name: Tensor(batch,)}  # 稀疏特征 long
                       + {"dense": Tensor(batch, dense_dim)}  # 数值特征 float（若有）
          ctr_label    : Tensor(batch,) float32
          cvr_label    : Tensor(batch,) float32
          ctcvr_label  : Tensor(batch,) float32
        """
        raise NotImplementedError
