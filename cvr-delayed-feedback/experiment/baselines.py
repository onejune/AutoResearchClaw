import numpy as np
from numpy_models import SimpleMLPBinary

class FNWModel:
    """
    Fake Negative Weighted: Weight unlabeled samples by 1/P(delay > observed).
    """
    def __init__(self, feature_dim, lr=0.001):
        self.net = SimpleMLPBinary(feature_dim, 128, lr=lr)
        self.delay_cdf = lambda x, t: 1.0 - np.exp(-t/20.0)  # Fixed exponential for baseline
        
    def train_step(self, x, y_observed, elapsed_time):
        """
        y_observed: 0 or 1. If 0, might be fake negative.
        elapsed_time: time since impression
        """
        p_cvr = self.net.predict_proba(x)
        
        if y_observed == 1:
            # Confirmed conversion
            loss = -np.log(p_cvr + 1e-8)
            grad = p_cvr - 1
        else:
            # Fake negative weighting
            # P(not converted yet) = 1 - P(converted and observed)
            # Weight = 1 / P(delay > elapsed_time)
            surv_prob = 1.0 - self.delay_cdf(x, elapsed_time)
            weight = 1.0 / (surv_prob + 0.1)  # clipping
            
            # Weighted BCE
            loss = -weight * np.log(1 - p_cvr + 1e-8)
            grad = weight * p_cvr
        
        self.net.backward(grad)
        return loss

class FNCModel:
    """
    Fake Negative Corrected: EM-style approach.
    """
    def __init__(self, feature_dim, lr=0.001):
        self.net = SimpleMLPBinary(feature_dim, 128, lr=lr)
        self.delay_mean = 24.0
        
    def train_step(self, x, y_observed, elapsed_time):
        if y_observed == 1:
            loss = -np.log(self.net.predict_proba(x) + 1e-8)
            grad = self.net.predict_proba(x) - 1
        else:
            # Probability that this is actually a conversion
            p_cvr = self.net.predict_proba(x)
            p_delay_short = np.exp(-elapsed_time / self.delay_mean)
            p_fake_negative = p_cvr * p_delay_short
            p_true_negative = 1 - p_cvr
            
            # Soft label
            weight = p_fake_negative / (p_fake_negative + p_true_negative + 1e-8)
            
            # Loss with soft label
            loss = -(weight * np.log(p_cvr + 1e-8) + (1-weight) * np.log(1 - p_cvr + 1e-8))
            grad = p_cvr - weight
        
        self.net.backward(grad)
        return loss

class StandardIW:
    """
    Standard Importance Weighting using delay CDF.
    """
    def __init__(self, feature_dim, lr=0.001):
        self.net = SimpleMLPBinary(feature_dim, 128, lr=lr)
        self.delay_net = None  # Will use oracle or simple model
        
    def train_step(self, x, y_observed, delay, delay_cdf_func):
        p_cvr = self.net.predict_proba(x)
        
        if y_observed == 1:
            # Importance weight: 1 / P(delay <= observed_delay | convert)
            # For observed conversion, weight by 1/CDF to correct for observation bias
            # Simplified: use inverse propensity
            if delay > 0:
                ipw = 1.0 / (delay_cdf_func(x, delay) + 0.1)
                ipw = min(ipw, 10.0)  # clipping
            else:
                ipw = 1.0
            
            loss = -ipw * np.log(p_cvr + 1e-8)
            grad = ipw * (p_cvr - 1)
        else:
            loss = -np.log(1 - p_cvr + 1e-8)
            grad = p_cvr
        
        self.net.backward(grad)
        return loss