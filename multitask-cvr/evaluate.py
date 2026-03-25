"""
evaluate.py - 评估指标

主要提供 AUC 计算（基于 sklearn）。
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import roc_auc_score


def compute_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    计算 AUC。若只有一类标签（全 0 或全 1），返回 0.5。

    Parameters
    ----------
    labels : 1-D array  真实标签（0/1）
    preds  : 1-D array  预测概率

    Returns
    -------
    float  AUC 值
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, preds))


def compute_metrics(
    all_ctr_labels:   np.ndarray,
    all_cvr_labels:   np.ndarray,
    all_ctcvr_labels: np.ndarray,
    all_ctr_preds:    np.ndarray,
    all_cvr_preds:    np.ndarray,
    all_ctcvr_preds:  np.ndarray,
) -> Dict[str, float]:
    """
    计算三个任务的 AUC。

    CVR AUC 仅在点击样本（ctr_label=1）上计算（更符合实际意义）。
    若点击样本太少（< 10），则在全量上计算。

    Returns
    -------
    dict: {"ctr_auc": float, "cvr_auc": float, "ctcvr_auc": float}
    """
    ctr_auc   = compute_auc(all_ctr_labels,   all_ctr_preds)
    ctcvr_auc = compute_auc(all_ctcvr_labels, all_ctcvr_preds)

    # CVR：优先在点击样本上评估
    click_mask = all_ctr_labels == 1
    if click_mask.sum() >= 10:
        cvr_auc = compute_auc(all_cvr_labels[click_mask], all_cvr_preds[click_mask])
    else:
        cvr_auc = compute_auc(all_cvr_labels, all_cvr_preds)

    return {
        "ctr_auc":   ctr_auc,
        "cvr_auc":   cvr_auc,
        "ctcvr_auc": ctcvr_auc,
    }
