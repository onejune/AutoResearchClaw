"""数据集加载器 - 支持 IVR CTCVR 数据"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Dict, List


class IVRDataset(Dataset):
    """
    IVR CTCVR 数据集
    
    支持:
    - CSV/Parquet 格式
    - 自动特征选择
    - 正负样本采样
    """
    
    def __init__(self,
                 data_path: str,
                 label_col: str = 'label',
                 feature_cols: Optional[List[str]] = None,
                 exclude_features: Optional[List[str]] = None,
                 sample_ratio: Optional[float] = None,
                 transform: Optional[callable] = None):
        """
        Args:
            data_path: 数据文件路径 (CSV/Parquet)
            label_col: 标签列名
            feature_cols: 特征列名列表，None 则自动选择所有数值列
            sample_ratio: 负采样比例 (如 0.2 表示正：负=1:5)，None 则不采样
            transform: 额外的数据转换函数
        """
        self.data_path = data_path
        self.label_col = label_col
        self.sample_ratio = sample_ratio
        self.transform = transform
        
        # 加载数据
        if data_path.endswith('.parquet'):
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
        
        # 确定特征列
        # 需要排除的列：标签列 + 用户指定排除的特征
        exclude_set = {label_col, 'click_label', 'ctcvr_label'}  # 排除所有可能的标签列
        if exclude_features:
            exclude_set.update(exclude_features)
        
        if feature_cols is not None:
            self.feature_cols = [c for c in feature_cols if c not in exclude_set]
        else:
            # 自动选择数值列（排除标签和指定特征）
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_cols = [c for c in numeric_cols if c not in exclude_set]
        
        # 打印排除信息
        if exclude_features:
            actually_excluded = [f for f in exclude_features if f in self.df.columns]
            if actually_excluded:
                print(f"   🚫 排除特征: {actually_excluded}")
        
        # 正负样本统计
        self.pos_mask = self.df[label_col] == 1
        self.neg_mask = self.df[label_col] == 0
        self.pos_count = self.pos_mask.sum()
        self.neg_count = self.neg_mask.sum()
        
        print(f"📊 数据集加载完成:")
        print(f"   总样本数：{len(self.df):,}")
        print(f"   正样本：{self.pos_count:,} ({100*self.pos_count/len(self.df):.2f}%)")
        print(f"   负样本：{self.neg_count:,} ({100*self.neg_count/len(self.df):.2f}%)")
        print(f"   正负比：1:{self.neg_count/self.pos_count:.1f}")
        print(f"   特征维度：{len(self.feature_cols)}")
        
        # 转换为数组 (类别特征保持 int64)
        self.features = self.df[self.feature_cols].values.astype(np.int64)
        self.labels = self.df[label_col].values.astype(np.float32)
    
    def _apply_negative_sampling(self, ratio: float):
        """负采样"""
        pos_indices = self.df[self.pos_mask].index.tolist()
        neg_indices = self.df[self.neg_mask].index.tolist()
        
        # 计算需要采样的负样本数量
        target_neg_count = int(len(pos_indices) / ratio)
        
        # 随机采样
        np.random.seed(42)
        sampled_neg_indices = np.random.choice(
            neg_indices, 
            size=min(target_neg_count, len(neg_indices)),
            replace=False
        ).tolist()
        
        # 合并
        sampled_indices = pos_indices + sampled_neg_indices
        self.df = self.df.loc[sampled_indices].reset_index(drop=True)
        
        # 更新统计
        self.pos_mask = self.df[self.label_col] == 1
        self.neg_mask = self.df[self.label_col] == 0
        self.pos_count = self.pos_mask.sum()
        self.neg_count = self.neg_mask.sum()
        
        print(f"   🎲 负采样后:")
        print(f"      正样本：{self.pos_count:,}")
        print(f"      负样本：{self.neg_count:,}")
        print(f"      正负比：1:{self.neg_count/self.pos_count:.1f}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform is not None:
            features, label = self.transform(features, label)
        
        return features, label


class ImbalancedDatasetWrapper:
    """
    不平衡数据集包装器
    
    提供便捷的训练/验证集划分和采样功能
    """
    
    def __init__(self,
                 data_dir: str,
                 train_file: Optional[str] = None,
                 val_file: Optional[str] = None,
                 test_file: Optional[str] = None,
                 label_col: str = "label",
                 feature_cols: Optional[List[str]] = None,
                 exclude_features: Optional[List[str]] = None,
                 use_subdirs: bool = True):
        """
        Args:
            data_dir: 数据目录
            train_file: 训练集文件名 (None 则自动查找)
            val_file: 验证集文件名
            test_file: 测试集文件名
            label_col: 标签列名
            feature_cols: 特征列名
            exclude_features: 需要排除的特征列表
            use_subdirs: 是否使用子目录结构 (train/, val/, test/)
        """
        self.data_dir = data_dir
        self.label_col = label_col
        self.feature_cols = feature_cols
        self.exclude_features = exclude_features or []
        self.use_subdirs = use_subdirs
        
        # 自动查找数据文件
        if train_file is None:
            train_file = self._find_data_file("train")
        if val_file is None:
            val_file = self._find_data_file("val") or self._find_data_file("validation")
        if test_file is None:
            test_file = self._find_data_file("test")
        
        # 加载数据集
        self.train_dataset = IVRDataset(
            data_path=os.path.join(data_dir, train_file) if use_subdirs else train_file,
            label_col=label_col,
            feature_cols=feature_cols,
            exclude_features=self.exclude_features,
        )
        
        # 验证集：优先用 val，否则用 test
        if val_file:
            self.val_dataset = IVRDataset(
                data_path=os.path.join(data_dir, val_file) if use_subdirs else val_file,
                label_col=label_col,
                feature_cols=feature_cols,
                exclude_features=self.exclude_features,
            )
        elif test_file:
            # 没有 val 但有 test，用 test 作为验证集
            print("📌 使用 test 集作为验证集")
            self.val_dataset = IVRDataset(
                data_path=os.path.join(data_dir, test_file) if use_subdirs else test_file,
                label_col=label_col,
                feature_cols=feature_cols,
                exclude_features=self.exclude_features,
            )
        else:
            # 都没有，从训练集中划分
            print("⚠️  未找到验证集，将从训练集中划分 10%")
            self.val_dataset = self._split_validation_set()
        
        # 测试集
        if test_file:
            self.test_dataset = IVRDataset(
                data_path=os.path.join(data_dir, test_file) if use_subdirs else test_file,
                label_col=label_col,
                feature_cols=feature_cols,
                exclude_features=self.exclude_features,
            )
        else:
            self.test_dataset = self.val_dataset
    
    def _find_data_file(self, split_name: str) -> str:
        """自动查找数据文件"""
        if self.use_subdirs:
            split_dir = os.path.join(self.data_dir, split_name)
            if os.path.exists(split_dir):
                # 查找 parquet 或 csv 文件 (排除临时文件)
                for f in sorted(os.listdir(split_dir)):
                    if (f.endswith('.parquet') or f.endswith('.csv')) and not f.startswith('_'):
                        return os.path.join(split_name, f)
        else:
            # 直接在根目录查找
            for f in sorted(os.listdir(self.data_dir)):
                if split_name in f and (f.endswith('.parquet') or f.endswith('.csv')) and not f.startswith('_'):
                    return f
        return None
    
    def _split_validation_set(self, ratio: float = 0.1):
        """从训练集中划分验证集"""
        import numpy as np
        n_val = int(len(self.train_dataset) * ratio)
        indices = np.random.choice(len(self.train_dataset), n_val, replace=False)
        # 简单实现：复制 dataset 并手动设置索引
        # 实际使用时应该用 Subset
        from torch.utils.data import Subset
        all_indices = set(range(len(self.train_dataset)))
        train_indices = list(all_indices - set(indices))
        val_subset = Subset(self.train_dataset, indices)
        # 临时替换训练集
        self.train_dataset = Subset(self.train_dataset, train_indices)
        return val_subset
    
    def get_dataloaders(self,
                        batch_size: int = 256,
                        sample_ratio: Optional[float] = None,
                        num_workers: int = 4,
                        pin_memory: bool = True):
        """
        获取数据加载器
        
        Args:
            batch_size: 批次大小
            sample_ratio: 负采样比例
            num_workers: 工作进程数
            pin_memory: 是否使用 pinned memory
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        # 注意：sample_ratio 需要在创建 dataset 时应用
        # 这里只是简单返回 dataloader
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader


def load_ivr_dataset(data_dir: str = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/",
                     label_col: str = "ctcvr_label",  # 实际数据列名
                     sample_ratio: float = 1.0,  # 数据采样比例
                     exclude_features: Optional[List[str]] = None,  # 排除特征
                     **kwargs):
    """
    便捷函数：加载 IVR 数据集
    
    Args:
        data_dir: 数据目录
        label_col: 标签列名 (默认 ctcvr_label)
        sample_ratio: 数据采样比例 (0~1)，用于快速测试
        **kwargs: 传递给 ImbalancedDatasetWrapper 的参数
    
    Returns:
        ImbalancedDatasetWrapper 实例
    """
    wrapper = ImbalancedDatasetWrapper(data_dir=data_dir, label_col=label_col, exclude_features=exclude_features, **kwargs)
    
    # 应用数据采样
    if sample_ratio < 1.0:
        _apply_sampling(wrapper, sample_ratio)
    
    return wrapper


def _apply_sampling(wrapper, ratio: float):
    """对数据集进行随机采样"""
    import numpy as np
    from torch.utils.data import Subset
    
    def sample_dataset(dataset, ratio):
        if isinstance(dataset, Subset):
            # 已经是 Subset，在其基础上再采样
            n_samples = int(len(dataset) * ratio)
            indices = np.random.choice(len(dataset), n_samples, replace=False)
            return Subset(dataset, indices)
        else:
            # 原始 dataset
            n_samples = int(len(dataset) * ratio)
            indices = np.random.choice(len(dataset), n_samples, replace=False)
            return Subset(dataset, indices)
    
    np.random.seed(42)
    wrapper.train_dataset = sample_dataset(wrapper.train_dataset, ratio)
    wrapper.val_dataset = sample_dataset(wrapper.val_dataset, ratio)
    wrapper.test_dataset = sample_dataset(wrapper.test_dataset, ratio)
    
    print(f"   采样后训练集: {len(wrapper.train_dataset):,} 样本")
    print(f"   采样后验证集: {len(wrapper.val_dataset):,} 样本")


# ============ 测试 ============

if __name__ == "__main__":
    print("=== 数据集加载器测试 ===\n")
    
    # 测试数据路径
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/"
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"⚠️  数据目录不存在：{data_dir}")
        print("创建模拟数据进行测试...\n")
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples = 10000
        n_features = 64
        
        # 生成不平衡数据 (1:9)
        df = pd.DataFrame({
            f'feat_{i}': np.random.randn(n_samples) 
            for i in range(n_features)
        })
        df['label'] = (np.random.rand(n_samples) > 0.9).astype(int)  # 10% 正样本
        
        # 保存
        os.makedirs(data_dir, exist_ok=True)
        df_train = df.iloc[:8000]
        df_val = df.iloc[8000:9000]
        df_test = df.iloc[9000:]
        
        df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
        df_val.to_csv(os.path.join(data_dir, "val.csv"), index=False)
        df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
        
        print(f"✅ 模拟数据已创建\n")
    
    # 加载数据集
    wrapper = load_ivr_dataset(data_dir)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = wrapper.get_dataloaders(
        batch_size=256,
        sample_ratio=0.5  # 正：负=1:2
    )
    
    print(f"\n✅ 数据加载器创建成功!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # 测试单个 batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\n   Batch shape: X={batch_x.shape}, Y={batch_y.shape}")
    print(f"   Pos ratio in batch: {batch_y.mean():.2%}")
