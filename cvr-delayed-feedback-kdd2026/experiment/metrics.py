import numpy as np

def compute_auc(y_true, y_scores):
    """
    Compute Area Under ROC Curve using Mann-Whitney U statistic.
    Returns 1 - AUC for minimization metric.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5  # Random chance if only one class
    
    # Sort by score
    sorted_indices = np.argsort(y_scores)
    y_true_sorted = y_true[sorted_indices]
    
    # Count positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Compute rank sum
    ranks = np.arange(1, len(y_scores) + 1)
    pos_ranks = ranks[y_true_sorted == 1]
    
    # AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    # Return 1 - AUC for minimization
    return 1.0 - auc

def compute_variance_estimates(predictions, window_size=100):
    """
    Compute rolling variance of CVR predictions.
    Returns normalized variance (variance / mean).
    """
    if len(predictions) < window_size:
        return 0.0
    
    variances = []
    for i in range(0, len(predictions) - window_size, window_size):
        window = predictions[i:i+window_size]
        variances.append(np.var(window))
    
    return np.mean(variances) if variances else 0.0

def compute_metrics_over_time(model, X_test, y_test, timestamps_test, window_size=500):
    """
    Compute metrics over temporal windows to detect drift effects.
    """
    n_samples = len(X_test)
    aucs = []
    variances = []
    
    for start in range(0, n_samples - window_size, window_size):
        end = start + window_size
        X_window = X_test[start:end]
        y_window = y_test[start:end]
        
        # Get predictions
        if hasattr(model, 'forward_pass'):
            p, _ = model.forward_pass(X_window)
        else:
            p = model.model.forward(X_window)
        
        p = p.flatten()
        
        # AUC
        auc = compute_auc(y_window, p)
        aucs.append(auc)
        
        # Variance of predictions
        variances.append(np.var(p))
    
    return {
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'mean_variance': np.mean(variances),
        'variance_of_variance': np.var(variances)
    }