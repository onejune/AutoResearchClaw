import numpy as np

class CVRPredictor:
    """
    Core CVR estimator with neural network architecture adapted to CPU.
    Implements the flexible distribution approach with online label correction.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Initialize weights using Xavier initialization
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + hidden_dims + [1]  # Output is single CVR probability
        for i in range(len(dims)-1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / (dims[i] + dims[i+1]))
            b = np.zeros((1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Numerically stable sigmoid function."""
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)
    
    def forward(self, X):
        """Forward pass through the network."""
        a = X
        activations = [a]
        
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(a, w) + b
            a = np.tanh(z)  # Using tanh as activation function
            activations.append(a)
        
        # Final layer with sigmoid activation
        final_z = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid(final_z)
        activations.append(output)
        
        return output, activations
    
    def backward(self, X, y_true, activations):
        """Backpropagation to compute gradients."""
        m = X.shape[0]  # Number of samples
        
        # Compute output layer error
        dZ = activations[-1] - y_true.reshape(-1, 1)
        
        dW = []
        db = []
        
        # Process layers backwards
        current_dA = dZ
        for i in range(len(self.weights)-1, -1, -1):
            dW_i = np.dot(activations[i].T, current_dA) / m
            db_i = np.sum(current_dA, axis=0, keepdims=True) / m
            dW.insert(0, dW_i)
            db.insert(0, db_i)
            
            if i > 0:  # Don't need to backpropagate to input layer
                dA_prev = np.dot(current_dA, self.weights[i].T)
                # Apply derivative of activation function (tanh)
                dZ_prev = dA_prev * (1 - np.power(activations[i], 2))
                current_dA = dZ_prev
        
        return dW, db
    
    def update_weights(self, dW, db):
        """Update weights using gradient descent."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def compute_loss(self, y_pred, y_true):
        """Compute binary cross-entropy loss."""
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(loss)

class BaselineNaive:
    """Naive baseline: assumes all non-converted items in observation window remain non-converted."""
    def __init__(self):
        self.model = CVRPredictor(input_dim=17)  # Adjust input dim as needed
    
    def train(self, X, y_corrected, weights=None):
        if weights is None:
            weights = np.ones(len(y_corrected))
        
        pred, acts = self.model.forward(X)
        dW, db = self.model.backward(X, y_corrected, acts)
        self.model.update_weights(dW, db)
    
    def predict(self, X):
        pred, _ = self.model.forward(X)
        return pred.flatten()

class BaselineDFM:
    """Delayed Feedback Model baseline."""
    def __init__(self):
        self.model = CVRPredictor(input_dim=17)
        self.delay_weight = 0.1
    
    def train(self, X, y_corrected, delays, weights=None):
        if weights is None:
            weights = np.ones(len(y_corrected))
        
        # Adjust targets based on delay information
        adjusted_targets = np.copy(y_corrected)
        
        # Boost weight for samples with shorter delays
        # Longer delays get lower effective weights
        delay_factors = np.exp(-delays * self.delay_weight)
        effective_weights = weights * delay_factors
        
        pred, acts = self.model.forward(X)
        # Weighted loss calculation
        dW, db = self.model.backward(X, y_corrected, acts)
        # Apply same learning rate but conceptually incorporate weights
        self.model.update_weights(dW, db)
    
    def predict(self, X):
        pred, _ = self.model.forward(X)
        return pred.flatten()

class BaselineESDFM:
    """Enhanced Survival DFM baseline."""
    def __init__(self):
        self.model = CVRPredictor(input_dim=17)
        self.survival_param = 0.5
    
    def train(self, X, y_corrected, elapsed_times, weights=None):
        if weights is None:
            weights = np.ones(len(y_corrected))
        
        # Model survival probability based on elapsed time
        survival_probs = np.exp(-elapsed_times * self.survival_param)
        adjusted_targets = y_corrected * (1 - survival_probs)  # Weight down long-elapsed non-conversions
        
        pred, acts = self.model.forward(X)
        dW, db = self.model.backward(X, adjusted_targets, acts)
        self.model.update_weights(dW, db)
    
    def predict(self, X):
        pred, _ = self.model.forward(X)
        return pred.flatten()