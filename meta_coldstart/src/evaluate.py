"""评估工具：AUC / PCOC / K-shot 曲线"""
import logging
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_model(model, loader, device) -> Dict[str, float]:
    """在 loader 上评估模型，返回 AUC 和 PCOC"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            probs = model.predict_proba(X).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return {"auc": float("nan"), "pcoc": float("nan"), "n": len(all_labels)}

    auc  = roc_auc_score(all_labels, all_probs)
    pcoc = all_probs.mean() / (all_labels.mean() + 1e-10)
    return {"auc": auc, "pcoc": pcoc, "n": len(all_labels)}


def evaluate_tasks(model, tasks, features, device, batch_size=2048) -> Dict:
    """在一批 task 的 query set 上评估，返回均值和各 task 明细"""
    results = []
    for task in tasks:
        loader = task.query_loader(features, batch_size=batch_size)
        if loader is None:
            continue  # 跳过空 query set
        r = evaluate_model(model, loader, device)
        r["campaignset_id"] = task.campaignset_id
        r["business_type"]  = task.business_type
        results.append(r)

    valid = [r for r in results if not np.isnan(r["auc"])]
    mean_auc  = float(np.mean([r["auc"]  for r in valid]))
    mean_pcoc = float(np.mean([r["pcoc"] for r in valid]))

    logger.info(f"Tasks evaluated: {len(valid)}/{len(results)} valid | "
                f"Mean AUC={mean_auc:.4f} | Mean PCOC={mean_pcoc:.3f}")
    return {"mean_auc": mean_auc, "mean_pcoc": mean_pcoc, "details": results}


def kshot_curve(method_fn, tasks, features, k_list, device) -> Dict[int, float]:
    """
    K-shot 敏感性曲线
    method_fn(task, k) -> trained model
    返回 {k: mean_auc}
    """
    curve = {}
    for k in k_list:
        aucs = []
        for task in tasks:
            model = method_fn(task, k)
            loader = task.query_loader(features)
            r = evaluate_model(model, loader, device)
            if not np.isnan(r["auc"]):
                aucs.append(r["auc"])
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        curve[k] = mean_auc
        logger.info(f"K={k:5d} | Mean AUC={mean_auc:.4f} ({len(aucs)} tasks)")
    return curve
