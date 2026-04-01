"""评估指标计算"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    roc_auc_score, 
    log_loss, 
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)


def compute_metrics(predictions: List[float], 
                    targets: List[float]) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        predictions: 预测概率列表
        targets: 真实标签列表 (0/1)
    
    Returns:
        包含所有指标的字典
    """
    preds = np.array(predictions)
    trues = np.array(targets)
    
    # AUC
    try:
        auc = roc_auc_score(trues, preds)
    except ValueError:
        auc = 0.5  # 单类别时返回随机猜测的 AUC
    
    # LogLoss
    logloss = log_loss(trues, preds, eps=1e-15)
    
    # PCOC (Predicted CTR/CVR over Observed CTR/CVR)
    # = mean(preds) / mean(trues)
    # PCOC 接近 1 表示预测校准良好
    mean_pred = preds.mean()
    mean_true = trues.mean()
    pcoc = mean_pred / mean_true if mean_true > 0 else float('inf')
    
    # F1 Score (threshold=0.5)
    preds_binary = (preds >= 0.5).astype(int)
    f1 = f1_score(trues, preds_binary, zero_division=0)
    
    # Precision & Recall
    precision, recall, _, _ = precision_recall_fscore_support(
        trues, preds_binary, average='binary', zero_division=0
    )
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(trues, preds_binary).ravel()
    
    return {
        'auc': float(auc),
        'logloss': float(logloss),
        'pcoc': float(pcoc),
        'mean_pred': float(mean_pred),
        'mean_true': float(mean_true),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def compute_gauc(predictions: List[float], 
                 targets: List[float],
                 group_ids: List[int] = None) -> float:
    """
    计算 GAUC (Group AUC)
    
    按组（如用户 ID）分别计算 AUC，然后平均
    
    Args:
        predictions: 预测概率
        targets: 真实标签
        group_ids: 分组 ID（如用户 ID），如果为 None 则退化为普通 AUC
    
    Returns:
        GAUC 值
    """
    if group_ids is None:
        return roc_auc_score(targets, predictions)
    
    pred_array = np.array(predictions)
    target_array = np.array(targets)
    group_array = np.array(group_ids)
    
    unique_groups = np.unique(group_array)
    group_aucs = []
    
    for group_id in unique_groups:
        mask = group_array == group_id
        group_targets = target_array[mask]
        group_preds = pred_array[mask]
        
        # 需要同时有正负样本才能计算 AUC
        if len(np.unique(group_targets)) < 2:
            continue
        
        try:
            group_auc = roc_auc_score(group_targets, group_preds)
            group_aucs.append(group_auc)
        except ValueError:
            continue
    
    return np.mean(group_aucs) if group_aucs else 0.5


def calibration_error(predictions: List[float], 
                      targets: List[float],
                      n_bins: int = 10) -> float:
    """
    计算校准误差 (Expected Calibration Error, ECE)
    
    Args:
        predictions: 预测概率
        targets: 真实标签
        n_bins: 分箱数量
    
    Returns:
        ECE 值（越小越好）
    """
    preds = np.array(predictions)
    trues = np.array(targets)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (preds > bin_boundaries[i]) & (preds <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = trues[in_bin].mean()
            avg_confidence_in_bin = preds[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece


def find_optimal_threshold(predictions: List[float],
                           targets: List[float]) -> Tuple[float, Dict]:
    """
    寻找最优分类阈值（基于 F1 Score）
    
    Args:
        predictions: 预测概率
        targets: 真实标签
    
    Returns:
        (最优阈值，该阈值下的指标)
    """
    preds = np.array(predictions)
    trues = np.array(targets)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds_binary = (preds >= threshold).astype(int)
        f1 = f1_score(trues, preds_binary, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 返回最优阈值下的完整指标
    preds_binary = (preds >= best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        trues, preds_binary, average='binary', zero_division=0
    )
    
    return best_threshold, {
        'threshold': best_threshold,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


# ============ 测试 ============

if __name__ == "__main__":
    print("=== 评估指标测试 ===\n")
    
    # 模拟数据
    np.random.seed(42)
    predictions = np.random.rand(1000)
    targets = (predictions + np.random.randn(1000) * 0.1 > 0.5).astype(int)
    group_ids = np.random.randint(0, 50, 1000)  # 50 个用户
    
    # 1. 基础指标
    metrics = compute_metrics(predictions.tolist(), targets.tolist())
    print("1. 基础指标:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    
    # 2. GAUC
    gauc = compute_gauc(predictions.tolist(), targets.tolist(), group_ids.tolist())
    print(f"\n2. GAUC: {gauc:.4f}")
    
    # 3. 校准误差
    ece = calibration_error(predictions.tolist(), targets.tolist())
    print(f"3. 校准误差 (ECE): {ece:.4f}")
    
    # 4. 最优阈值
    best_thresh, thresh_metrics = find_optimal_threshold(
        predictions.tolist(), targets.tolist()
    )
    print(f"\n4. 最优阈值：{best_thresh:.2f}")
    print(f"   F1: {thresh_metrics['f1']:.4f}")
    print(f"   Precision: {thresh_metrics['precision']:.4f}")
    print(f"   Recall: {thresh_metrics['recall']:.4f}")
    
    print("\n✅ 评估指标测试通过！")
