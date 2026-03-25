"""数据集基类"""
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """所有数据集加载器的基类"""

    n_continuous: int   # 连续特征数量（子类必须设置）
    n_categorical: int  # 类别特征数量（子类必须设置）

    @abstractmethod
    def get_dataloaders(self, config) -> tuple:
        """
        加载数据并返回 (train_loader, test_loader)

        Args:
            config: Config 对象，包含 data_path, sample_size, batch_size 等

        Returns:
            (train_loader, test_loader)
        """
        raise NotImplementedError
