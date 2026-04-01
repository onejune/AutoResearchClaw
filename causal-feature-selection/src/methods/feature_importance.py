"""
Phase 1: 特征重要性分析

方法:
1. Permutation Importance (PI) - 打乱某特征，观察 AUC 下降
2. Gradient-based Importance (GI) - 梯度 * embedding 范数
3. SHAP (DeepExplainer) - 基于 Shapley 值的归因

目标: 对比三种方法在 in-domain vs OOD 场景下的重要性排名差异
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class PermutationImportance:
    """
    Permutation Feature Importance
    
    原理: 打乱某个特征的值，观察 AUC 下降幅度
    - AUC 下降越多 → 该特征越重要
    - 优点: 模型无关，简单可靠
    - 缺点: 计算慢 (每个特征都要重新推理)
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.eval()
    
    @torch.no_grad()
    def _predict(self, features: Dict[str, torch.Tensor]) -> np.ndarray:
        """推理，返回概率"""
        features = {k: v.to(self.device) for k, v in features.items()}
        logits = self.model(features)
        return torch.sigmoid(logits).cpu().numpy()
    
    @torch.no_grad()
    def compute(
        self,
        dataloader,
        feature_names: List[str],
        n_repeats: int = 3,
        top_k: int = None
    ) -> pd.DataFrame:
        """
        计算所有特征的 Permutation Importance
        
        Args:
            dataloader: 数据加载器
            feature_names: 要分析的特征列表
            n_repeats: 每个特征重复打乱次数
            top_k: 只计算前 k 个特征 (None = 全部)
            
        Returns:
            DataFrame: feature, mean_importance, std_importance
        """
        # 先获取 baseline AUC
        all_preds, all_labels = [], []
        all_features_cache = []
        
        print("Computing baseline predictions...")
        for batch_features, batch_labels in tqdm(dataloader):
            preds = self._predict(batch_features)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())
            all_features_cache.append({k: v.clone() for k, v in batch_features.items()})
        
        baseline_auc = roc_auc_score(np.array(all_labels), np.array(all_preds))
        print(f"Baseline AUC: {baseline_auc:.4f}")
        
        # 对每个特征计算重要性
        features_to_analyze = feature_names[:top_k] if top_k else feature_names
        results = []
        
        for feat_name in tqdm(features_to_analyze, desc="Computing PI"):
            importances = []
            
            for _ in range(n_repeats):
                permuted_preds = []
                
                for batch_features in all_features_cache:
                    # 打乱该特征
                    permuted_batch = {k: v.clone() for k, v in batch_features.items()}
                    if feat_name in permuted_batch:
                        idx = torch.randperm(len(permuted_batch[feat_name]))
                        permuted_batch[feat_name] = permuted_batch[feat_name][idx]
                    
                    preds = self._predict(permuted_batch)
                    permuted_preds.extend(preds)
                
                permuted_auc = roc_auc_score(np.array(all_labels), np.array(permuted_preds))
                importances.append(baseline_auc - permuted_auc)
            
            results.append({
                "feature": feat_name,
                "mean_importance": np.mean(importances),
                "std_importance": np.std(importances),
                "baseline_auc": baseline_auc
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values("mean_importance", ascending=False).reset_index(drop=True)
        return df


class GradientImportance:
    """
    Gradient-based Feature Importance
    
    原理: 计算 loss 对 embedding 的梯度，取 L2 范数作为重要性
    - 梯度越大 → 该特征对预测影响越大
    - 优点: 计算快 (一次反向传播)
    - 缺点: 局部近似，可能不稳定
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def compute(
        self,
        dataloader,
        feature_names: List[str],
        n_batches: int = 50
    ) -> pd.DataFrame:
        """
        计算梯度重要性
        
        Args:
            dataloader: 数据加载器
            feature_names: 特征列表
            n_batches: 用多少个 batch 计算 (越多越准)
            
        Returns:
            DataFrame: feature, mean_importance
        """
        self.model.train()  # 需要梯度
        criterion = nn.BCEWithLogitsLoss()
        
        # 累积每个特征 embedding 的梯度范数
        grad_norms = {feat: [] for feat in feature_names}
        
        for i, (batch_features, batch_labels) in enumerate(tqdm(dataloader, desc="Computing GI")):
            if i >= n_batches:
                break
            
            batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
            batch_labels = batch_labels.to(self.device)
            
            # 清空梯度
            self.model.zero_grad()
            
            # Forward
            logits = self.model(batch_features)
            loss = criterion(logits, batch_labels)
            
            # Backward
            loss.backward()
            
            # 收集 embedding 梯度
            for feat_name in feature_names:
                if hasattr(self.model, 'embeddings') and feat_name in self.model.embeddings:
                    emb_layer = self.model.embeddings[feat_name]
                    if emb_layer.weight.grad is not None:
                        # 取 L2 范数
                        grad_norm = emb_layer.weight.grad.norm(dim=1).mean().item()
                        grad_norms[feat_name].append(grad_norm)
        
        self.model.eval()
        
        results = []
        for feat_name, norms in grad_norms.items():
            results.append({
                "feature": feat_name,
                "mean_importance": np.mean(norms) if norms else 0.0,
                "std_importance": np.std(norms) if norms else 0.0
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values("mean_importance", ascending=False).reset_index(drop=True)
        return df


class EmbeddingNormImportance:
    """
    Embedding Norm Importance (最轻量的方法)
    
    原理: 训练后 embedding 权重的 L2 范数反映了特征的"学习程度"
    - 范数越大 → 模型越依赖该特征
    - 优点: 零额外计算，训练完直接读取
    - 缺点: 不考虑特征使用频率，高频特征可能虚高
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def compute(self, feature_names: List[str]) -> pd.DataFrame:
        """计算 embedding 范数重要性"""
        results = []
        
        for feat_name in feature_names:
            if hasattr(self.model, 'embeddings') and feat_name in self.model.embeddings:
                emb_weight = self.model.embeddings[feat_name].weight.data
                # 排除 padding_idx=0
                emb_weight = emb_weight[1:]
                
                mean_norm = emb_weight.norm(dim=1).mean().item()
                max_norm = emb_weight.norm(dim=1).max().item()
                std_norm = emb_weight.norm(dim=1).std().item()
                
                results.append({
                    "feature": feat_name,
                    "mean_norm": mean_norm,
                    "max_norm": max_norm,
                    "std_norm": std_norm,
                    "mean_importance": mean_norm  # 统一接口
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values("mean_importance", ascending=False).reset_index(drop=True)
        return df


def compare_importance_methods(
    pi_df: pd.DataFrame,
    gi_df: pd.DataFrame,
    en_df: pd.DataFrame,
    top_k: int = 20
) -> pd.DataFrame:
    """
    对比三种方法的特征排名
    
    Returns:
        DataFrame: 每个特征在三种方法下的排名 + 排名相关性
    """
    # 获取各方法 top-k 特征
    pi_top = pi_df.head(top_k)[["feature", "mean_importance"]].rename(
        columns={"mean_importance": "pi_importance"}
    )
    pi_top["pi_rank"] = range(1, len(pi_top) + 1)
    
    gi_top = gi_df.head(top_k)[["feature", "mean_importance"]].rename(
        columns={"mean_importance": "gi_importance"}
    )
    gi_top["gi_rank"] = range(1, len(gi_top) + 1)
    
    en_top = en_df.head(top_k)[["feature", "mean_importance"]].rename(
        columns={"mean_importance": "en_importance"}
    )
    en_top["en_rank"] = range(1, len(en_top) + 1)
    
    # 合并
    merged = pi_top.merge(gi_top, on="feature", how="outer")
    merged = merged.merge(en_top, on="feature", how="outer")
    merged = merged.fillna({"pi_rank": top_k + 1, "gi_rank": top_k + 1, "en_rank": top_k + 1})
    
    # 计算排名一致性 (Spearman)
    from scipy.stats import spearmanr
    common_feats = merged.dropna()
    
    if len(common_feats) > 3:
        pi_gi_corr, _ = spearmanr(common_feats["pi_rank"], common_feats["gi_rank"])
        pi_en_corr, _ = spearmanr(common_feats["pi_rank"], common_feats["en_rank"])
        gi_en_corr, _ = spearmanr(common_feats["gi_rank"], common_feats["en_rank"])
        
        print(f"\n=== 方法间排名相关性 (Spearman) ===")
        print(f"  PI vs GI: {pi_gi_corr:.3f}")
        print(f"  PI vs EmbNorm: {pi_en_corr:.3f}")
        print(f"  GI vs EmbNorm: {gi_en_corr:.3f}")
    
    return merged.sort_values("pi_rank")
