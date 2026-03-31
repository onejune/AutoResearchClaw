"""
校准方法实现

包含:
- Temperature Scaling
- Isotonic Regression
- Histogram Binning
- Platt Scaling
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar


class TemperatureScaling(nn.Module):
    """Temperature Scaling 校准"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.temperature)
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """学习温度参数"""
        def nll_loss(t):
            if t <= 0:
                return float('inf')
            probs = torch.sigmoid(logits / t)
            return nn.functional.binary_cross_entropy(probs, labels).item()
        
        result = minimize_scalar(nll_loss, bounds=(0.01, 10.0), method='bounded')
        self.temperature.data = torch.tensor([result.x])
        print(f"Optimal temperature: {result.x:.4f}")
        return result.x


class IsotonicCalibration:
    """Isotonic Regression 校准"""
    
    def __init__(self):
        self.regressor = None
    
    def fit(self, probs: np.ndarray, labels: np.ndarray):
        self.regressor = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self.regressor.fit(probs, labels)
        print("Isotonic regression fitted")
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self.regressor.transform(probs)


class HistogramBinning:
    """Histogram Binning 校准"""
    
    def __init__(self, num_bins: int = 20):
        self.num_bins = num_bins
        self.bin_accs = None
    
    def fit(self, probs: np.ndarray, labels: np.ndarray):
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        self.bin_accs = np.zeros(self.num_bins)
        
        for i in range(self.num_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                self.bin_accs[i] = labels[in_bin].mean()
            else:
                self.bin_accs[i] = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
        
        print(f"Histogram binning fitted with {self.num_bins} bins")
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        calibrated = np.zeros_like(probs)
        
        for i in range(self.num_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            calibrated[in_bin] = self.bin_accs[i]
        
        calibrated[probs >= 1.0] = self.bin_accs[-1]
        calibrated[probs < 0.0] = self.bin_accs[0]
        
        return calibrated


# 校准评估指标
def compute_ece(preds: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> float:
    """计算 ECE"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = preds[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_mce(preds: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> float:
    """计算 MCE"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    
    for i in range(n_bins):
        in_bin = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = preds[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def compute_pcoc(preds: np.ndarray, labels: np.ndarray) -> float:
    """计算 PCOC"""
    return preds.sum() / (labels.sum() + 1e-10)
