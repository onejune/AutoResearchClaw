"""
Delayed Feedback Model Implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class MAF(nn.Module):
    """
    Masked Autoregressive Flow for 1D conditional density estimation.
    Simple implementation for univariate target with context.
    """
    def __init__(self, num_layers, hidden_dim, target_dim, context_dim):
        super().__init__()
        self.num_layers = num_layers
        self.target_dim = target_dim
        
        # Each layer is an affine coupling: x = x * exp(s) + t
        # where s and t are computed from context and previous dimensions
        self.scale_nets = nn.ModuleList()
        self.shift_nets = nn.ModuleList()
        
        for _ in range(num_layers):
            # Scale network
            scale_net = nn.Sequential(
                nn.Linear(context_dim + target_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, target_dim),
                nn.Tanh()  # Bound scale
            )
            # Shift network
            shift_net = nn.Sequential(
                nn.Linear(context_dim + target_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, target_dim)
            )
            self.scale_nets.append(scale_net)
            self.shift_nets.append(shift_net)
    
    def forward(self, x, context):
        """
        Forward transform: x -> z (base space)
        Returns: z, log_det
        """
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        
        for i in range(self.num_layers):
            inp = torch.cat([z, context], dim=1)
            s = self.scale_nets[i](inp)
            t = self.shift_nets[i](inp)
            
            # Inverse autoregressive transform (for density estimation)
            z = (z - t) * torch.exp(-s)
            log_det += -s.sum(dim=1)
        
        return z, log_det
    
    def inverse(self, z, context):
        """
        Inverse transform: z -> x (data space)
        """
        x = z
        for i in reversed(range(self.num_layers)):
            inp = torch.cat([x, context], dim=1)
            s = self.scale_nets[i](inp)
            t = self.shift_nets[i](inp)
            x = x * torch.exp(s) + t
        return x
    
    def log_prob(self, x, context):
        """
        Compute log probability p(x|context)
        """
        z, log_det = self.forward(x, context)
        # Base distribution: Standard Normal
        log_prob_base = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return log_prob_base + log_det
    
    def cdf(self, x, context):
        """
        Compute CDF P(X <= x | context)
        """
        with torch.no_grad():
            z, _ = self.forward(x, context)
            # Standard normal CDF
            cdf_val = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        return cdf_val

class BaseDelayedFeedbackModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Shared CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, config.feature_dim),
            nn.ReLU()
        )
        
        # Forward CVR predictor
        self.forward_head = nn.Linear(config.feature_dim, 1)
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': config.cvr_learning_rate},
            {'params': self.forward_head.parameters(), 'lr': config.cvr_learning_rate}
        ])
        
        self.latency_buffer = []
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward_predict(self, features):
        return torch.sigmoid(self.forward_head(features))
    
    def train_step(self, batch):
        raise NotImplementedError

class InversionBase(BaseDelayedFeedbackModel):
    """
    Base class for inversion-based methods with flow and backward predictor.
    Extracted to avoid fragile inheritance between stratified and unstratified variants.
    """
    def __init__(self, config):
        super().__init__(config)
        # Backward predictor
        self.backward_head = nn.Linear(config.feature_dim, 1)
        
        # Flow for delay density
        self.flow = MAF(config.flow_layers, config.flow_hidden, 1, config.feature_dim)
        self.flow_optimizer = torch.optim.Adam(
            self.flow.parameters(), lr=config.flow_learning_rate
        )
    
    def update_flow(self, delays, features):
        """Update MAF on observed delays"""
        context = features.detach()
        log_prob = self.flow.log_prob(delays, context)
        loss_flow = -log_prob.mean()
        
        self.flow_optimizer.zero_grad()
        loss_flow.backward()
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.config.gradient_clip_norm)
        self.flow_optimizer.step()
        
        return loss_flow.detach()
    
    def compute_robust_weights(self, p_f, p_b, delays, features):
        """To be overridden by subclasses"""
        raise NotImplementedError
    
    def update_stratified_reservoir(self, x_batch, delays):
        """To be overridden by subclasses"""
        pass
    
    def sample_backward_context(self, batch_size):
        """To be overridden by subclasses"""
        return None, None
    
    def train_step(self, batch):
        import time
        start_time = time.time()
        
        x = batch['x'].to(self.config.device)
        y = batch['y'].to(self.config.device)
        delays = batch['delay'].to(self.config.device)
        
        # Forward pass
        features = self.encode(x)
        p_f = self.forward_predict(features)
        
        # Update reservoirs (subclass specific)
        self.update_stratified_reservoir(x, delays)
        
        # Backward pass
        feat_future = features.detach()
        p_b = p_f.detach().clone()
        
        x_future, d_future = self.sample_backward_context(x.size(0))
        if x_future is not None:
            feat_future = self.encode(x_future).detach()
            p_b = torch.sigmoid(self.backward_head(feat_future))
        
        # Update flow on conversions only
        conv_mask = (y == 1).squeeze()
        loss_flow = torch.tensor(0.0, device=x.device)
        if conv_mask.sum() > 0:
            loss_flow = self.update_flow(delays[conv_mask], features[conv_mask])
        
        # Compute robust weights
        w_final = self.compute_robust_weights(p_f, p_b, delays, features)
        
        # Ensure weights have the correct shape before computing losses
        target_size = None
        repeat_times = None
        if p_b is not None and p_b.shape[0] != p_f.shape[0]:
            # Adjust size by repeating or truncating
            target_size = p_f.shape[0]
            if p_b.shape[0] < target_size:
                # Repeat last elements to match size
                repeat_times = target_size // p_b.shape[0]
                remainder = target_size % p_b.shape[0]
                p_b_extended = p_b.repeat(repeat_times + 1, 1)[:target_size]
                y_future_extended = y[:target_size].repeat(repeat_times + 1, 1)[:target_size]
                w_extended = w_final[:target_size]
            else:
                p_b_extended = p_b[:target_size]
                y_future_extended = y[:target_size]
                w_extended = w_final[:target_size]
        else:
            y_future_extended = y
            w_extended = w_final
        
        # Weighted BCE losses - ensure weight shape matches [batch_size, 1]
        loss_f = F.binary_cross_entropy(p_f, y, weight=w_final, reduction='mean')
        
        # Backward loss - fix: Use actual labels (not random) and adjust shapes if needed
        loss_b = torch.tensor(0.0, device=x.device)
        actual_y_size = None
        y_actual = None
        current_w = None
        if x_future is not None and p_b.numel() > 0:
            # Use actual labels from the sampled future batch (if available)
            # For simplicity, use actual labels but adjust for the sampled batch size
            if feat_future.shape[0] <= y.shape[0]:  # Make sure sizes align
                actual_y_size = feat_future.shape[0]
                y_actual = y[:actual_y_size]
                current_w = w_final[:actual_y_size] if w_final.shape[0] >= actual_y_size else w_final
                loss_b = F.binary_cross_entropy(p_b[:actual_y_size], y_actual, 
                                               weight=current_w, reduction='mean')
        
        total_loss = loss_f + loss_b + self.config.beta_flow * loss_flow
        
        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        
        latency = (time.time() - start_time) * 1000 / x.size(0)
        
        return {
            'loss': total_loss.item(),
            'loss_f': loss_f.item(),
            'loss_b': loss_b.item(),
            'latency_ms': latency,
            'w_mean': w_final.mean().item()
        }

class StratifiedInversionOFDM(InversionBase):
    def __init__(self, config):
        super().__init__(config)
        
        # Stratified reservoirs
        self.reservoirs = [
            deque(maxlen=config.reservoir_size_per_stratum) 
            for _ in range(config.reservoir_strata)
        ]
        
        self.alpha = config.alpha_robust
    
    def update_stratified_reservoir(self, x_batch, delays):
        """Update stratified reservoirs based on delay quantiles"""
        batch_size = x_batch.size(0)
        for i in range(batch_size):
            delay_val = delays[i].item()
            stratum = int(delay_val / (self.config.max_delay_hours / self.config.reservoir_strata))
            stratum = min(stratum, self.config.reservoir_strata - 1)
            self.reservoirs[stratum].append((x_batch[i].detach().cpu(), delay_val))
    
    def sample_backward_context(self, batch_size):
        """Sample from stratified reservoirs with proper handling of total count"""
        samples_x = []
        samples_d = []
        
        # Calculate how many to sample from each stratum
        per_stratum = max(1, batch_size // self.config.reservoir_strata)
        
        for reservoir in self.reservoirs:
            if len(reservoir) > 0:
                # Take up to 'per_stratum' samples from this reservoir
                n_available = len(reservoir)
                n_to_sample = min(per_stratum, n_available)
                idx = np.random.choice(len(reservoir), n_to_sample, replace=False)
                for j in idx:
                    x, d = reservoir[j]
                    samples_x.append(x)
                    samples_d.append(d)
        
        # If we collected too many samples, randomly select 'batch_size' of them
        if len(samples_x) == 0:
            return None, None
        
        selected_indices = None
        if len(samples_x) > batch_size:
            # Randomly sample from collected samples without replacement
            selected_indices = np.random.choice(len(samples_x), batch_size, replace=False)
            samples_x = [samples_x[i] for i in selected_indices]
            samples_d = [samples_d[i] for i in selected_indices]
        
        # If we have fewer samples than desired batch size, repeat some
        while len(samples_x) < batch_size:
            extra_idx = np.random.choice(len(samples_x))
            samples_x.append(samples_x[extra_idx])
            samples_d.append(samples_d[extra_idx])
        
        x_future = torch.stack(samples_x).to(self.config.device)
        d_future = torch.FloatTensor(samples_d).to(self.config.device).unsqueeze(1)
        return x_future, d_future
    
    def compute_robust_weights(self, p_f, p_b, delays, features):
        """Compute distributionally robust weights"""
        with torch.no_grad():
            # Importance weighting: 1/F(t|x)
            cdf = self.flow.cdf(delays, features)
            if cdf is None:
                cdf = torch.zeros_like(delays)
            w_importance = 1.0 / (cdf + 1e-6)
            w_importance = torch.clamp(w_importance, 0, 100)
            
            # Disagreement detection - properly handle different sizes
            if p_b is not None:
                # Make sure p_b and p_f have compatible sizes for subtraction
                p_b_adjusted = p_b
                if p_b.shape[0] != p_f.shape[0]:
                    # Repeat or trim p_b to match p_f size
                    if p_b.shape[0] < p_f.shape[0]:
                        repeat_times = p_f.shape[0] // p_b.shape[0]
                        remainder = p_f.shape[0] % p_b.shape[0]
                        p_b_adjusted = p_b.repeat(repeat_times + 1, 1)[:p_f.shape[0]]
                    else:
                        p_b_adjusted = p_b[:p_f.shape[0]]
                
                D = torch.abs(p_f - p_b_adjusted)
            else:
                D = torch.zeros_like(p_f)
            
            # Robust weighting
            w_final = w_importance * torch.exp(-self.alpha * D)
        
        return w_final

class UnstratifiedNaiveInversion(InversionBase):
    """
    Unstratified naive inversion - inherits from InversionBase instead of 
    StratifiedInversionOFDM to avoid creating unused stratified reservoirs.
    """
    def __init__(self, config):
        super().__init__(config)
        # Replace with uniform reservoir
        self.uniform_reservoir = deque(maxlen=config.reservoir_strata * config.reservoir_size_per_stratum)
        self.alpha = 0.0  # Disable robust weighting
    
    def update_stratified_reservoir(self, x_batch, delays):
        """Uniform FIFO reservoir"""
        for i in range(x_batch.size(0)):
            self.uniform_reservoir.append((x_batch[i].detach().cpu(), delays[i].item()))
    
    def sample_backward_context(self, batch_size):
        """Uniform sampling"""
        if len(self.uniform_reservoir) == 0:
            return None, None
        n = min(batch_size, len(self.uniform_reservoir))
        idx = np.random.choice(len(self.uniform_reservoir), n, replace=False)
        samples = [self.uniform_reservoir[i] for i in idx]
        x_future = torch.stack([s[0] for s in samples]).to(self.config.device)
        d_future = torch.FloatTensor([s[1] for s in samples]).to(self.config.device).unsqueeze(1)
        return x_future, d_future
    
    def compute_robust_weights(self, p_f, p_b, delays, features):
        """Naive: importance weighting only"""
        with torch.no_grad():
            cdf = self.flow.cdf(delays, features)
            if cdf is None:
                cdf = torch.zeros_like(delays)
            w_importance = 1.0 / (cdf + 1e-6)
            w_importance = torch.clamp(w_importance, 0, 100)
        return w_importance

class NoInversionBaseline(BaseDelayedFeedbackModel):
    def __init__(self, config):
        super().__init__(config)
        self.flow = MAF(config.flow_layers, config.flow_hidden, 1, config.feature_dim)
        self.flow_optimizer = torch.optim.Adam(
            self.flow.parameters(), lr=config.flow_learning_rate
        )
    
    def train_step(self, batch):
        import time
        start_time = time.time()
        
        x = batch['x'].to(self.config.device)
        y = batch['y'].to(self.config.device)
        delays = batch['delay'].to(self.config.device)
        
        features = self.encode(x)
        p_f = self.forward_predict(features)
        
        # Update flow
        conv_mask = (y == 1).squeeze()
        loss_flow = torch.tensor(0.0, device=x.device)
        log_prob = None
        if conv_mask.sum() > 0:
            log_prob = self.flow.log_prob(delays[conv_mask], features[conv_mask].detach())
            loss_flow = -log_prob.mean()
            self.flow_optimizer.zero_grad()
            loss_flow.backward()
            self.flow_optimizer.step()
        
        # Importance weighting only
        with torch.no_grad():
            cdf = self.flow.cdf(delays, features)
            if cdf is None:
                cdf = torch.zeros_like(delays)
            w = torch.clamp(1.0 / (cdf + 1e-6), 0, 100)
        
        # FIXED: Apply importance weights to loss
        loss = F.binary_cross_entropy(p_f, y, weight=w, reduction='mean')
        loss = loss + self.config.beta_flow * loss_flow.detach()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        latency = (time.time() - start_time) * 1000 / x.size(0)
        return {'loss': loss.item(), 'latency_ms': latency}

class DANN(BaseDelayedFeedbackModel):
    def __init__(self, config):
        super().__init__(config)
        self.domain_classifier = nn.Sequential(
            nn.Linear(config.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.domain_optimizer = torch.optim.Adam(
            self.domain_classifier.parameters(), lr=config.cvr_learning_rate
        )
        self.alpha_grl = 1.0
    
    def train_step(self, batch):
        import time
        start_time = time.time()
        
        x = batch['x'].to(self.config.device)
        y = batch['y'].to(self.config.device)
        delays = batch['delay'].to(self.config.device)
        
        features = self.encode(x)
        p_f = self.forward_predict(features)
        
        # Domain labels: early vs late based on median delay
        domain_y = (delays > self.config.max_delay_hours / 2).long().squeeze()
        
        # Gradient reversal
        reversed_features = GradientReversalLayer.apply(features, self.alpha_grl)
        domain_pred = self.domain_classifier(reversed_features)
        
        loss_domain = F.cross_entropy(domain_pred, domain_y)
        loss_task = F.binary_cross_entropy(p_f, y)
        
        total_loss = loss_task + 0.1 * loss_domain
        
        self.optimizer.zero_grad()
        self.domain_optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.domain_optimizer.step()
        
        latency = (time.time() - start_time) * 1000 / x.size(0)
        return {'loss': total_loss.item(), 'loss_domain': loss_domain.item(), 'latency_ms': latency}

class CORAL(BaseDelayedFeedbackModel):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer('target_mean', torch.zeros(config.feature_dim))
        self.register_buffer('target_cov', torch.eye(config.feature_dim))
    
    def compute_coral_loss(self, source, target_mean, target_cov):
        """Correlation alignment loss"""
        source_mean = source.mean(0)
        source_centered = source - source_mean
        source_cov = torch.mm(source_centered.t(), source_centered) / source.size(0)
        
        # Frobenius norm squared
        mean_diff = (source_mean - target_mean).pow(2).sum()
        cov_diff = (source_cov - target_cov).pow(2).sum()
        return mean_diff + cov_diff
    
    def train_step(self, batch):
        import time
        start_time = time.time()
        
        x = batch['x'].to(self.config.device)
        y = batch['y'].to(self.config.device)
        
        features = self.encode(x)
        p_f = self.forward_predict(features)
        
        # Update running statistics
        with torch.no_grad():
            batch_mean = features.mean(0)
            features_centered = features - batch_mean
            batch_cov = torch.mm(features_centered.t(), features_centered) / features.size(0)
            momentum = 0.9
            self.target_mean = momentum * self.target_mean + (1 - momentum) * batch_mean
            self.target_cov = momentum * self.target_cov + (1 - momentum) * batch_cov
        
        # CORAL loss
        coral_loss = self.compute_coral_loss(features, self.target_mean, self.target_cov)
        loss_task = F.binary_cross_entropy(p_f, y)
        
        total_loss = loss_task + 0.01 * coral_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        latency = (time.time() - start_time) * 1000 / x.size(0)
        return {'loss': total_loss.item(), 'coral_loss': coral_loss.item(), 'latency_ms': latency}

class StandardImportanceWeighting(BaseDelayedFeedbackModel):
    def __init__(self, config):
        super().__init__(config)
        self.delay_history = deque(maxlen=10000)
        self.register_buffer('empirical_cdf', torch.zeros(100))
    
    def update_empirical_cdf(self, delays):
        """Update empirical CDF histogram"""
        for d in delays.cpu().numpy():
            self.delay_history.append(d)
        
        cdf = None
        if len(self.delay_history) > 100:
            hist, _ = np.histogram(list(self.delay_history), bins=100, 
                                  range=(0, self.config.max_delay_hours))
            cdf = np.cumsum(hist) / len(self.delay_history)
            self.empirical_cdf.copy_(torch.from_numpy(cdf).float())
    
    def train_step(self, batch):
        import time
        start_time = time.time()
        
        x = batch['x'].to(self.config.device)
        y = batch['y'].to(self.config.device)
        delays = batch['delay'].to(self.config.device)
        
        features = self.encode(x)
        p_f = self.forward_predict(features)
        
        # Update empirical distribution
        self.update_empirical_cdf(delays)
        
        # Compute importance weights
        with torch.no_grad():
            bins = (delays / self.config.max_delay_hours * 99).long().clamp(0, 99)
            cdf_vals = self.empirical_cdf[bins].to(x.device) + 1e-6
            w = 1.0 / cdf_vals
            w = torch.clamp(w, 0, 100).unsqueeze(1)
        
        loss = F.binary_cross_entropy(p_f, y, weight=w, reduction='mean')
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        latency = (time.time() - start_time) * 1000 / x.size(0)
        return {'loss': loss.item(), 'w_mean': w.mean().item(), 'latency_ms': latency}

class FixedExponentialEM(BaseDelayedFeedbackModel):
    def __init__(self, config):
        super().__init__(config)
        self.lambda_delay = nn.Parameter(torch.tensor(1.0 / config.max_delay_hours))
        self.em_optimizer = torch.optim.Adam([self.lambda_delay], lr=0.01)
    
    def train_step(self, batch):
        import time
        start_time = time.time()
        
        x = batch['x'].to(self.config.device)
        y = batch['y'].to(self.config.device)
        delays = batch['delay'].to(self.config.device)
        
        features = self.encode(x)
        p_f = self.forward_predict(features)
        
        # E-step: Expected delay for censored samples
        with torch.no_grad():
            expected_delays = delays.clone()
            censored = (y == 0).squeeze()
            if censored.sum() > 0:
                expected_delays[censored] += 1.0 / (self.lambda_delay.detach() + 1e-6)
        
        # M-step: Maximize log-likelihood of exponential
        log_likelihood = torch.log(self.lambda_delay + 1e-6) - self.lambda_delay * expected_delays.mean()
        loss_em = -log_likelihood
        
        self.em_optimizer.zero_grad()
        loss_em.backward()
        self.em_optimizer.step()
        
        # Importance weighting using exponential CDF: F(t) = 1 - exp(-lambda*t)
        with torch.no_grad():
            cdf = 1.0 - torch.exp(-self.lambda_delay * delays)
            w = 1.0 / (cdf + 1e-6)
            w = torch.clamp(w, 0, 100)
        
        loss_task = F.binary_cross_entropy(p_f, y, weight=w, reduction='mean')
        
        self.optimizer.zero_grad()
        loss_task.backward()
        self.optimizer.step()
        
        latency = (time.time() - start_time) * 1000 / x.size(0)
        return {'loss': loss_task.item(), 'lambda': self.lambda_delay.item(), 'latency_ms': latency}