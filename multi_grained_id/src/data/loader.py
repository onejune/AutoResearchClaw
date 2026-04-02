"""
数据加载器 - IVR v16 CTCVR Sample (多粒度 ID 建模版本)

重要规范:
- 训练集: /mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/
- 测试集: /mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/
- 所有特征都是类别特征，已编码好，不做额外处理
- 广告 ID 层级（由细到粗）: campaignid → campaignsetid → offerid → demand_pkgname → business_type
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch


# 广告 ID 层级结构（由细到粗）
AD_ID_HIERARCHY = [
    "campaignid",      # 最细粒度：广告计划
    "campaignsetid",   # 广告组
    "offerid",         # offer
    "demand_pkgname",  # 包名
    "business_type"    # 最粗粒度：业务类型
]

# 排除特征
EXCLUDE_FEATURES = ["deviceid"]


class MultiGrainedIVRDataset(Dataset):
    """支持多粒度 ID 的 PyTorch Dataset
    
    注意：所有特征都是类别特征，已经编码好，直接使用
    """
    
    def __init__(
        self,
        parquet_path: str,
        feature_cols: List[str],
        label_col: str = "ctcvr_label",
        hierarchical_features: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            parquet_path: parquet 文件路径（或目录）
            feature_cols: 特征列名列表
            label_col: 标签列名
            hierarchical_features: {细粒度特征：粗粒度特征} 映射
        """
        self.parquet_path = parquet_path
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.hierarchical_features = hierarchical_features or {}
        
        # 加载数据
        path = Path(parquet_path)
        if path.is_dir():
            # 目录：加载所有 parquet 文件（跳过临时文件）
            files = [f for f in path.glob("*.parquet") if not f.name.startswith("_")]
            if not files:
                raise FileNotFoundError(f"No parquet files found in {path}")
            dfs = [pd.read_parquet(f) for f in sorted(files)]
            self.df = pd.concat(dfs, ignore_index=True)
            print(f"  Loaded {len(files)} files, {len(self.df):,} samples")
        else:
            self.df = pd.read_parquet(path)
        
        # 确定实际可用的特征列
        available_cols = [c for c in feature_cols if c in self.df.columns]
        missing_cols = set(feature_cols) - set(available_cols)
        if missing_cols:
            print(f"  ⚠️  Missing features (will skip): {missing_cols}")
        
        # 所有特征都是类别特征，已编码好，直接转 tensor
        self.features = {}
        for col in available_cols:
            vals = self.df[col].values
            # 直接使用，不做任何额外处理
            self.features[col] = torch.from_numpy(vals.astype(np.int64))
        
        # 层次化特征（粗粒度）
        self.coarse_features = {}
        for fine, coarse in self.hierarchical_features.items():
            if coarse in self.df.columns:
                coarse_vals = self.df[coarse].values
                self.coarse_features[fine] = torch.from_numpy(coarse_vals.astype(np.int64))
        
        # 标签
        if label_col not in self.df.columns:
            raise ValueError(f"Label column '{label_col}' not found in data. Available: {list(self.df.columns)}")
        self.labels = torch.from_numpy(self.df[label_col].values.astype(np.float32))
        
        # business_type 用于分组评估
        if "business_type" in self.df.columns:
            self.business_type = torch.from_numpy(self.df["business_type"].values.astype(np.int64))
        else:
            self.business_type = None
        
        print(f"  Features: {len(self.features)}, Label: {label_col} (mean={self.labels.mean():.4f})")
        print(f"  Hierarchical pairs: {list(self.hierarchical_features.keys())}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            "features": {k: v[idx] for k, v in self.features.items()},
            "label": self.labels[idx]
        }
        
        # 添加粗粒度特征
        for fine, coarse_tensor in self.coarse_features.items():
            item["features"][f"{fine}_coarse"] = coarse_tensor[idx]
        
        # 添加 business_type 用于分组评估
        if self.business_type is not None:
            item["business_type"] = self.business_type[idx]
        
        return item


def collate_fn(batch):
    """批次整理函数"""
    features = {}
    for key in batch[0]["features"]:
        features[key] = torch.stack([item["features"][key] for item in batch])
    
    labels = torch.stack([item["label"] for item in batch])
    
    result = {"features": features, "label": labels}
    
    if "business_type" in batch[0]:
        result["business_type"] = torch.stack([item["business_type"] for item in batch])
    
    return result


def create_dataloader(dataset, batch_size=512, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


class IVRDataLoader:
    """IVR v16 CTCVR Sample 数据加载器
    
    使用方式:
        loader = IVRDataLoader()
        train_dataset, test_dataset = loader.load_datasets(
            hierarchical_features={"campaignid": "campaignsetid", ...}
        )
    """
    
    # 默认数据路径
    DEFAULT_TRAIN_PATH = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/"
    DEFAULT_TEST_PATH = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/"
    
    def __init__(
        self,
        train_path: str = None,
        test_path: str = None,
        label_col: str = "ctcvr_label"
    ):
        self.train_path = train_path or self.DEFAULT_TRAIN_PATH
        self.test_path = test_path or self.DEFAULT_TEST_PATH
        self.label_col = label_col
        
        # 加载元数据
        self._load_metadata()
    
    def _load_metadata(self):
        """加载元数据（vocab_sizes, meta.json）"""
        base_path = Path(self.train_path).parent
        
        # vocab_sizes.json
        vocab_path = base_path / "vocab_sizes.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                self.vocab_sizes = json.load(f)
            print(f"Loaded vocab_sizes: {len(self.vocab_sizes)} features")
        else:
            self.vocab_sizes = {}
            print("⚠️  vocab_sizes.json not found, will infer from data")
        
        # meta.json
        meta_path = base_path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.feature_cols = self.meta.get("feature_cols", [])
            print(f"Loaded meta: {len(self.feature_cols)} features")
        else:
            # 从数据推断
            self.meta = {}
            self._infer_feature_cols()
        
        # 排除特征
        self.feature_cols = [c for c in self.feature_cols if c not in EXCLUDE_FEATURES]
    
    def _infer_feature_cols(self):
        """从数据推断特征列"""
        train_dir = Path(self.train_path)
        files = [f for f in train_dir.glob("*.parquet") if not f.name.startswith("_")]
        if files:
            df = pd.read_parquet(files[0], columns=None)
            # 排除标签列和元数据列
            exclude = {"click_label", "ctcvr_label", "timestamp", "date"}
            self.feature_cols = [c for c in df.columns if c not in exclude]
            print(f"Inferred {len(self.feature_cols)} features from data")
    
    def get_feature_config(self) -> Dict[str, int]:
        """获取特征配置 {特征名: vocab_size}"""
        if self.vocab_sizes:
            return {k: v for k, v in self.vocab_sizes.items() if k in self.feature_cols}
        
        # 从数据推断
        config = {}
        train_dir = Path(self.train_path)
        files = [f for f in train_dir.glob("*.parquet") if not f.name.startswith("_")]
        if files:
            df = pd.read_parquet(files[0])
            for col in self.feature_cols:
                if col in df.columns:
                    config[col] = int(df[col].max()) + 1
        return config
    
    def load_datasets(
        self,
        hierarchical_features: Optional[Dict[str, str]] = None
    ) -> Tuple[MultiGrainedIVRDataset, MultiGrainedIVRDataset]:
        """加载训练集和测试集
        
        Args:
            hierarchical_features: 层次化特征对 {细粒度: 粗粒度}
                如: {"campaignid": "campaignsetid", "offerid": "demand_pkgname"}
        
        Returns:
            (train_dataset, test_dataset)
        """
        print(f"\nLoading train data from: {self.train_path}")
        train_dataset = MultiGrainedIVRDataset(
            parquet_path=self.train_path,
            feature_cols=self.feature_cols,
            label_col=self.label_col,
            hierarchical_features=hierarchical_features
        )
        
        print(f"\nLoading test data from: {self.test_path}")
        test_dataset = MultiGrainedIVRDataset(
            parquet_path=self.test_path,
            feature_cols=self.feature_cols,
            label_col=self.label_col,
            hierarchical_features=hierarchical_features
        )
        
        return train_dataset, test_dataset
    
    def get_hierarchy_vocab_sizes(
        self,
        hierarchical_features: Dict[str, str]
    ) -> Dict[str, Tuple[int, int]]:
        """获取层次化特征的 vocab size
        
        Returns:
            {细粒度特征: (细粒度vocab, 粗粒度vocab)}
        """
        config = self.get_feature_config()
        result = {}
        for fine, coarse in hierarchical_features.items():
            fine_vocab = config.get(fine, 0)
            coarse_vocab = config.get(coarse, 0)
            if coarse_vocab == 0:
                # 尝试从数据推断
                train_dir = Path(self.train_path)
                files = [f for f in train_dir.glob("*.parquet") if not f.name.startswith("_")]
                if files:
                    df = pd.read_parquet(files[0], columns=[coarse] if coarse in pd.read_parquet(files[0], columns=None).columns else [])
                    if coarse in df.columns:
                        coarse_vocab = int(df[coarse].max()) + 1
            result[fine] = (fine_vocab, coarse_vocab)
        return result
