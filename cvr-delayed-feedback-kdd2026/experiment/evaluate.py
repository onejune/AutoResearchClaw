"""
Evaluation metrics and training orchestration
"""
import torch
import numpy as np
import time
from collections import defaultdict

def compute_auc_roc(y_true, y_pred):
    """Compute AUC-ROC using numpy"""
    # Sort by prediction
    desc_order = np.argsort(-y_pred)
    y_true_sorted = y_true[desc_order]
    
    # Compute TPR and FPR at different thresholds
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Trapezoidal rule
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add (0,0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Compute AUC using trapezoidal rule
    auc = np.trapezoid(tpr, fpr)
    return auc

def compute_normalized_variance_cvr(predictions_log, window_size=24):
    """
    Compute normalized variance of CVR predictions across rolling time windows.
    predictions_log: list of (batch_idx, cvr_prob)
    window_size: number of batches to form a window (each representing time interval)
    """
    if len(predictions_log) < window_size * 2:
        return 1.0
    
    # Extract probabilities in chronological order of processing
    # For test set - we'll assume sequential processing with no shuffling
    sorted_preds = sorted(predictions_log, key=lambda x: x[0])  # sort by batch index
    probs = np.array([p for _, p in sorted_preds])
    
    # Rolling windows (24 batches = 24 time intervals)
    variances = []
    for i in range(len(probs) - window_size + 1):
        window = probs[i:i+window_size]
        if len(window) > 1:
            var = np.var(window)
            if not np.isnan(var):  # Avoid nan if constant window
                variances.append(var)
    
    if len(variances) < 2:
        return 1.0
    
    # Normalize by first window variance
    baseline_var = variances[0] if variances[0] > 0 else 1e-6
    normalized_vars = [v / baseline_var for v in variances[1:]]
    
    return np.mean(normalized_vars)

def train_model(model, train_loader, val_loader, config):
    """
    Train model for specified epochs.
    Returns dict with success status and metrics.
    """
    device = config.device
    model.to(device)
    model.train()
    
    history = defaultdict(list)
    predictions_log = []
    
    for epoch in range(config.num_epochs):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Check for NaN in batch
            if torch.isnan(batch['x']).any():
                return {'success': False, 'error': 'NaN in input data'}
            
            metrics = model.train_step(batch)
            
            # Check for NaN/divergence
            if np.isnan(metrics['loss']) or metrics['loss'] > 100:
                return {'success': False, 'error': 'NaN or divergence detected'}
            
            epoch_losses.append(metrics['loss'])
            
            # Log predictions for variance computation
            with torch.no_grad():
                x = batch['x'].to(device)
                features = model.encode(x)
                p_f = model.forward_predict(features).cpu().numpy()
                for p in p_f:
                    predictions_log.append((epoch * len(train_loader) + batch_idx, float(p[0])))
            
            # Store latency after warmup in both history and model buffer
            if batch_idx > config.latency_warmup_batches and 'latency_ms' in metrics:
                history['latency_ms'].append(metrics['latency_ms'])
                # Populate model's latency_buffer for evaluation
                if hasattr(model, 'latency_buffer'):
                    model.latency_buffer.append(metrics['latency_ms'])
        
        avg_loss = np.mean(epoch_losses)
        history['epoch_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                features = model.encode(x)
                p_f = model.forward_predict(features)
                val_preds.append(p_f.cpu().numpy())
                val_labels.append(y.cpu().numpy())
        
        val_preds = np.concatenate(val_preds).squeeze()
        val_labels = np.concatenate(val_labels).squeeze()
        val_auc = compute_auc_roc(val_labels, val_preds)
        history['val_auc'].append(val_auc)
        model.train()
    
    final_loss = history['epoch_loss'][-1] if history['epoch_loss'] else float('inf')
    
    return {
        'success': True,
        'final_loss': final_loss,
        'history': dict(history),
        'predictions_log': predictions_log
    }

def evaluate_all_metrics(model, test_loader, config):
    """
    Evaluate model on test set and compute all metrics.
    """
    device = config.device
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_delays = []
    backward_preds = []
    
    # Track predictions log for variance calculation
    predictions_log = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            delays = batch['delay'].to(device)
            
            features = model.encode(x)
            p_f = model.forward_predict(features)
            
            all_preds.append(p_f.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_delays.append(delays.cpu().numpy())
            
            # Log predictions with batch index for test variance calculation
            for p in p_f.cpu().numpy():
                predictions_log.append((batch_idx, float(p[0])))
            
            # Get backward predictions if available
            if hasattr(model, 'backward_head'):
                p_b = torch.sigmoid(model.backward_head(features))
                backward_preds.append(p_b.cpu().numpy())
    
    all_preds = np.concatenate(all_preds).squeeze()
    all_labels = np.concatenate(all_labels).squeeze()
    all_delays = np.concatenate(all_delays).squeeze()
    
    # Compute AUC-ROC
    auc_roc = compute_auc_roc(all_labels, all_preds)
    negative_auc_roc = 1.0 - auc_roc
    
    # Compute normalized variance (using test predictions as time series proxy)
    norm_var = compute_normalized_variance_cvr(predictions_log, window_size=24)
    
    # Compute backward predictor AUC (detecting false negatives: y=1 but p_f < 0.5)
    backward_auc = 0.5
    
    backward_preds_arr = None
    false_negative_labels = None
    if len(backward_preds) > 0 and hasattr(model, 'backward_head'):
        backward_preds_arr = np.concatenate(backward_preds).squeeze()
        # Check if we actually have predictions (not just empty list)
        if len(backward_preds_arr) > 0 and len(all_preds) > 0:
            # False negatives: true label is 1 but predicted probability < 0.5 (missed conversions)
            false_negative_labels = ((all_labels == 1) & (all_preds < 0.5)).astype(float)
            
            # Only compute if we have both positive and negative samples for the backward task
            if false_negative_labels.sum() > 0 and (1 - false_negative_labels).sum() > 0:
                backward_auc = compute_auc_roc(false_negative_labels, backward_preds_arr)
    
    # Latency p95 (from model's stored latencies if available, else estimate)
    latency_p95 = 10.0  # Default estimate in ms
    if hasattr(model, 'latency_buffer') and len(model.latency_buffer) > 0:
        latency_p95 = np.percentile(model.latency_buffer, 95)
    
    return {
        'normalized_variance_cvr': float(norm_var),
        'negative_auc_roc': float(negative_auc_roc),
        'latency_p95_ms': float(latency_p95),
        'backward_predictor_auc': float(1.0 - backward_auc),  # Return 1-AUC for minimization
        'auc_roc': float(auc_roc)
    }