import numpy as np

class SimpleMLPBinary:
    """
    2-layer MLP for binary classification using numpy.
    Outputs logit for BCEWithLogits loss.
    """
    def __init__(self, input_dim, hidden_dim=128, lr=0.001, l2=1e-4):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr = lr
        self.l2 = l2
        self.hidden_cache = None
        self.input_cache = None
        
    def forward(self, x, return_hidden=False):
        """Forward pass, returns logit"""
        self.input_cache = x.copy()
        z1 = np.dot(x, self.W1) + self.b1
        self.hidden_cache = np.maximum(0, z1)  # ReLU
        logit = np.dot(self.hidden_cache, self.W2) + self.b2
        if return_hidden:
            return logit[0], self.hidden_cache
        return logit[0]
    
    def backward(self, grad_logit):
        """
        Backward pass with gradient clipping.
        grad_logit: dL/d(logit)
        """
        # Clip gradient
        grad_logit = np.clip(grad_logit, -10, 10)
        
        # Output layer gradients
        grad_W2 = np.outer(self.hidden_cache, grad_logit) + self.l2 * self.W2
        grad_b2 = grad_logit
        
        # Backprop through ReLU
        grad_hidden = np.dot(self.W2, grad_logit) * (self.hidden_cache > 0)
        grad_W1 = np.outer(self.input_cache, grad_hidden) + self.l2 * self.W1
        grad_b1 = grad_hidden
        
        # Update with clipping
        self.W2 -= self.lr * np.clip(grad_W2, -1, 1)
        self.b2 -= self.lr * np.clip(grad_b2, -1, 1)
        self.W1 -= self.lr * np.clip(grad_W1, -1, 1)
        self.b1 -= self.lr * np.clip(grad_b1, -1, 1)
        
    def predict_proba(self, x):
        """Return P(y=1|x)"""
        logit = self.forward(x)
        return 1.0 / (1.0 + np.exp(-logit))

class WeibullDelayModel:
    """
    Parametric delay distribution using Weibull (flexible distribution approach).
    Network outputs log(k) and log(lambda) parameters.
    """
    def __init__(self, feature_dim, hidden_dim=64, lr=0.0001):
        self.net = SimpleMLPBinary(feature_dim, hidden_dim, lr=lr)
        # Initialize to reasonable values (k=1, lambda=10)
        self.net.b2 = np.array([0.0, np.log(10)], dtype=np.float32)
        self.net.W2 = np.zeros((hidden_dim, 2), dtype=np.float32)
        
    def get_params(self, x):
        """Return Weibull k and lambda"""
        out = self.net.forward(x)
        # Softplus for positivity
        k = np.log(1 + np.exp(out[0])) + 0.5
        lam = np.log(1 + np.exp(out[1])) + 1.0
        return k, lam
    
    def cdf(self, x, t):
        """F(t|x) = 1 - exp(-(t/lambda)^k)"""
        if t <= 0:
            return 0.0
        k, lam = self.get_params(x)
        return 1.0 - np.exp(-((t / lam) ** k))
    
    def pdf(self, x, t):
        """p(t|x)"""
        if t <= 0:
            return 0.0
        k, lam = self.get_params(x)
        return (k / lam) * ((t / lam) ** (k - 1)) * np.exp(-((t / lam) ** k))
    
    def nll_loss(self, x, t, observed):
        """
        Negative log-likelihood for observed delay t.
        observed: 1 if conversion observed, 0 if censored (not yet converted)
        """
        k, lam = self.get_params(x)
        if observed:
            # PDF
            if t <= 0:
                return 10.0  # Large penalty
            log_pdf = (np.log(k) - np.log(lam) + 
                      (k - 1) * (np.log(t) - np.log(lam)) - 
                      (t / lam) ** k)
            return -log_pdf
        else:
            # Survival
            surv = np.exp(-((t / lam) ** k))
            return -np.log(surv + 1e-8)
    
    def train_step(self, x, t, observed):
        """Manual backprop for Weibull NLL"""
        # Numerical gradients (finite differences) for stability
        eps = 1e-4
        base_loss = self.nll_loss(x, t, observed)
        
        # Perturb output neurons
        orig_b2 = self.net.b2.copy()
        
        # Gradient for k (index 0)
        self.net.b2[0] += eps
        loss_k = self.nll_loss(x, t, observed)
        grad_k = (loss_k - base_loss) / eps
        self.net.b2 = orig_b2.copy()
        
        # Gradient for lambda (index 1)
        self.net.b2[1] += eps
        loss_lam = self.nll_loss(x, t, observed)
        grad_lam = (loss_lam - base_loss) / eps
        self.net.b2 = orig_b2.copy()
        
        # Backprop through network
        grad_out = np.array([grad_k, grad_lam], dtype=np.float32)
        
        # Manual backprop for output layer
        h = self.net.hidden_cache
        grad_W2 = np.outer(h, grad_out)
        grad_b2 = grad_out
        
        # Update output layer only (simplified)
        self.net.W2 -= self.net.lr * np.clip(grad_W2, -1, 1)
        self.net.b2 -= self.net.lr * np.clip(grad_b2, -1, 1)
        
        return base_loss

class StratifiedReservoir:
    """
    Stratified reservoir sampling for temporal inversion.
    Maintains separate reservoirs for different CVR strata.
    """
    def __init__(self, size_per_stratum=2000, n_strata=5, feature_dim=25):
        self.size = size_per_stratum
        self.n_strata = n_strata
        self.feature_dim = feature_dim
        self.reservoirs = [[] for _ in range(n_strata)]
        self.counters = [0] * n_strata
        
    def get_stratum(self, cvr_prob):
        """Assign to stratum based on predicted CVR"""
        # 0 to 1 divided into n_strata bins
        idx = int(cvr_prob * self.n_strata)
        return min(idx, self.n_strata - 1)
    
    def add(self, x, timestamp, cvr_prob):
        """Add sample to appropriate stratum reservoir"""
        s = self.get_stratum(cvr_prob)
        self.counters[s] += 1
        
        reservoir = self.reservoirs[s]
        if len(reservoir) < self.size:
            reservoir.append((x.copy(), timestamp))
        else:
            # Reservoir sampling: replace with decreasing probability
            j = np.random.randint(0, self.counters[s])
            if j < self.size:
                reservoir[j] = (x.copy(), timestamp)
    
    def sample_future(self, current_time, n_samples=32):
        """
        Sample from future contexts (for backward predictor).
        Returns samples with timestamp > current_time (simulated).
        In online setting, we use samples with higher timestamps.
        """
        samples = []
        for reservoir in self.reservoirs:
            for x, t in reservoir:
                if t > current_time and len(samples) < n_samples:
                    samples.append(x)
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break
        
        # Pad if needed
        while len(samples) < n_samples and len(samples) > 0:
            samples.append(samples[0])
        return samples[:n_samples] if samples else None