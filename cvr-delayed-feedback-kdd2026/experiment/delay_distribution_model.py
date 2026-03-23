import numpy as np
from math import lgamma, exp, log

class DelayDistributionModel:
    """
    Flexible distribution approach for modeling conversion delays.
    Uses Weibull distribution as the primary model due to its flexibility.
    """
    def __init__(self, init_shape=1.0, init_scale=1.0):
        self.shape = init_shape  # k parameter
        self.scale = init_scale  # lambda parameter
        self.learning_rate = 0.001
        
    def weibull_pdf(self, t, shape, scale):
        """Weibull probability density function."""
        if t <= 0:
            return 0.0
        return (shape/scale) * ((t/scale)**(shape-1)) * exp(-(t/scale)**shape)
    
    def weibull_cdf(self, t, shape, scale):
        """Weibull cumulative distribution function."""
        if t <= 0:
            return 0.0
        return 1 - exp(-(t/scale)**shape)
    
    def weibull_log_likelihood(self, t, shape, scale):
        """Log likelihood for Weibull distribution."""
        if t <= 0:
            return -np.inf
        loglik = log(shape) - shape*log(scale) + (shape-1)*log(t) - (t/scale)**shape
        return loglik
    
    def update_parameters(self, observed_delays):
        """
        Update shape and scale parameters using maximum likelihood estimation.
        """
        # Filter out infinite delays (non-conversions)
        finite_delays = observed_delays[observed_delays != np.inf]
        
        if len(finite_delays) < 2:
            return  # Not enough data to update
        
        # Method of moments estimators for initial guess
        mean_delay = np.mean(finite_delays)
        var_delay = np.var(finite_delays)
        
        # Estimate shape parameter from ratio of moments
        if var_delay > 0:
            c = var_delay / (mean_delay**2)
            # Approximate relationship to estimate shape
            shape_new = (6.0 / (c * np.pi**2))**0.5
            # Ensure positive values
            shape_new = max(min(shape_new, 10.0), 0.1)
        else:
            shape_new = self.shape
            
        # Update scale parameter
        scale_new = mean_delay / self._gamma_func(1 + 1/shape_new)
        scale_new = max(min(scale_new, 1000.0), 0.1)
        
        # Apply gradient update
        self.shape = 0.95 * self.shape + 0.05 * shape_new  # Slow updating
        self.scale = 0.95 * self.scale + 0.05 * scale_new
    
    def _gamma_func(self, x):
        """Approximate gamma function for common values."""
        # Using Stirling's approximation for larger values and exact values for small ones
        if x == 1.0:
            return 1.0
        elif x == 1.5:
            return 0.5 * np.sqrt(np.pi)
        elif x == 2.0:
            return 1.0
        elif x > 2:
            # Stirling approximation
            return np.sqrt(2*np.pi/x) * (x/np.exp(1))**x
        else:
            # For fractional values between 1 and 2, interpolate
            return exp(lgamma(x))
    
    def survival_prob(self, t):
        """Survival probability S(t) = P(delay > t)."""
        return exp(-(t/self.scale)**self.shape) if t > 0 else 1.0

class OnlineLabelCorrection:
    """
    Implements online label correction under delayed feedback.
    Corrects labels based on delay distribution and elapsed time.
    """
    def __init__(self, delay_model):
        self.delay_model = delay_model
        self.correction_threshold = 0.5
    
    def correct_labels(self, observed_conversions, elapsed_times, current_time):
        """
        Correct labels based on delay distribution and elapsed time.
        
        Args:
            observed_conversions: Boolean array of observed conversions
            elapsed_times: Array of time elapsed since click for each sample
            current_time: Current simulation time
        
        Returns:
            corrected_conversions: Array of corrected binary labels
            confidence_scores: Confidence scores for corrections
        """
        corrected_conversions = np.copy(observed_conversions).astype(float)
        confidence_scores = np.ones(len(observed_conversions))
        
        # For non-converted items, calculate probability they will convert later
        non_converted_mask = ~observed_conversions
        if np.any(non_converted_mask):
            survival_probs = []
            for i, elapsed in enumerate(elapsed_times[non_converted_mask]):
                prob_surviving = self.delay_model.survival_prob(elapsed)
                survival_probs.append(prob_surviving)
            
            survival_probs = np.array(survival_probs)
            # Probability that item will eventually convert
            will_convert_probs = 1.0 - survival_probs
            
            # Update labels probabilistically
            idx_non_converted = np.where(non_converted_mask)[0]
            for i, prob in enumerate(will_convert_probs):
                idx = idx_non_converted[i]
                # Probabilistic label correction
                if prob > 0.5:
                    corrected_conversions[idx] = prob
                    confidence_scores[idx] = min(prob, 1.0 - prob) * 2  # Lower confidence for uncertain corrections
        
        return corrected_conversions, confidence_scores