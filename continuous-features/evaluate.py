"""评估模块"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(model, data_loader, device):
    """在给定 DataLoader 上计算 AUC"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_cont, x_cat, labels in data_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            preds = model(x_cont, x_cat)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    return auc
