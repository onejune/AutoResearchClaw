"""评估指标：AUC、PCOC、LogLoss，支持分 domain 计算。"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def compute_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算 AUC。正样本数 < 1 时返回 0.0。"""
    if labels.sum() < 1 or (1 - labels).sum() < 1:
        return 0.0
    return float(roc_auc_score(labels, preds))


def compute_pcoc(labels: np.ndarray, preds: np.ndarray) -> float:
    """PCOC = mean(pred) / mean(label)，衡量预估校准度，理想值为 1.0。"""
    mean_label = labels.mean()
    if mean_label == 0:
        return 0.0
    return float(preds.mean() / mean_label)


def compute_logloss(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算 LogLoss，pred clip 到 [1e-7, 1-1e-7] 防止 log(0)。"""
    preds = np.clip(preds, 1e-7, 1 - 1e-7)
    return float(log_loss(labels, preds))


def evaluate_by_domain(
    labels: np.ndarray,
    preds: np.ndarray,
    domains: np.ndarray,
    domain_names: Optional[Dict[int, str]] = None,
    min_pos: int = 50,
) -> Dict[str, dict]:
    """
    分 domain 计算 AUC / PCOC / LogLoss，同时计算整体指标。

    Args:
        labels: 真实标签 [N]
        preds:  预测概率 [N]
        domains: domain id [N]
        domain_names: {domain_id: name}，用于输出可读名称
        min_pos: 正样本数低于此值的 domain 不计入结果

    Returns:
        {domain_key: {"auc": float, "pcoc": float, "logloss": float, "pos": int, "total": int}}
    """
    if domain_names is None:
        domain_names = {}

    results = {}

    # 整体
    results["Overall"] = {
        "auc": compute_auc(labels, preds),
        "pcoc": compute_pcoc(labels, preds),
        "logloss": compute_logloss(labels, preds),
        "pos": int(labels.sum()),
        "total": len(labels),
    }

    # 分 domain
    for d in np.unique(domains):
        mask = domains == d
        d_labels = labels[mask]
        d_preds = preds[mask]
        if d_labels.sum() < min_pos:
            continue
        name = domain_names.get(int(d), f"domain_{d}")
        results[name] = {
            "auc": compute_auc(d_labels, d_preds),
            "pcoc": compute_pcoc(d_labels, d_preds),
            "logloss": compute_logloss(d_labels, d_preds),
            "pos": int(d_labels.sum()),
            "total": int(mask.sum()),
        }

    return results


def format_metrics_table(results: Dict[str, dict]) -> str:
    """将 evaluate_by_domain 结果格式化为可读表格字符串。"""
    header = f"{'Domain':<20} {'AUC':>8} {'PCOC':>8} {'LogLoss':>10} {'Pos':>8} {'Total':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    # Overall 排第一
    for key in ["Overall"] + [k for k in results if k != "Overall"]:
        if key not in results:
            continue
        r = results[key]
        lines.append(
            f"{key:<20} {r['auc']:>8.4f} {r['pcoc']:>8.4f} {r['logloss']:>10.4f} "
            f"{r['pos']:>8d} {r['total']:>10d}"
        )
    return "\n".join(lines)
