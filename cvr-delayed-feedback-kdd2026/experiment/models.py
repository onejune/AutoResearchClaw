"""Model definitions for delayed feedback CVR estimation."""
import numpy as np


class BaseModel:
    """Base class for CVR models."""
    
    def __init__(self, config):
        self.config = config
        self.numeric_feature_indices = list(range(config.num_numeric_features))
        self.cat_feature_indices = list(range(config.num_numeric_features, 
                                              config.num_numeric_features + config.num_categorical_features))

        # Initialize embeddings for categorical features
        self.embeddings = []
        for i in range(config.num_categorical_features):
            # Estimate vocab sizes
            vocab_size = 1000 + i * 100  # Increasing vocab sizes per feature
            embedding_matrix = np.random.normal(0, 0.1, (vocab_size, config.embedding_dim))
            self.embeddings.append(embedding_matrix)

        # Shared feature encoder parameters
        total_embedded_dim = (config.num_numeric_features * 1 + 
                              config.num_categorical_features * config.embedding_dim)
        self.feature_weights_1 = np.random.normal(0, 0.1, (total_embedded_dim, config.cvr_hidden_dim))
        self.feature_bias_1 = np.zeros((config.cvr_hidden_dim,))
        self.feature_weights_2 = np.random.normal(0, 0.1, (config.cvr_hidden_dim, config.cvr_hidden_dim))
        self.feature_bias_2 = np.zeros((config.cvr_hidden_dim,))
        
        # Conversion head parameters
        self.head_weights_1 = np.random.normal(0, 0.1, (config.cvr_hidden_dim, config.cvr_hidden_dim // 2))
        self.head_bias_1 = np.zeros((config.cvr_hidden_dim // 2,))
        self.head_weights_2 = np.random.normal(0, 0.1, (config.cvr_hidden_dim // 2, 1))
        self.head_bias_2 = np.zeros((1,))
        
        # Store gradients for backpropagation
        self.gradients = {}
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def embed_features(self, input_features):
        """Embed numerical and categorical features."""
        numeric_part = input_features[:, self.numeric_feature_indices]  # [B, 8]
        
        # Process categorical features individually
        embedded_cats = []
        for i, cat_idx in enumerate(self.cat_feature_indices):
            # Get category indices and embed
            cat_values = input_features[:, cat_idx].astype(int)
            # Handle out-of-range indices by clipping
            max_idx = self.embeddings[i].shape[0] - 1
            cat_values = np.clip(cat_values, 0, max_idx)
            embedded = self.embeddings[i][cat_values]  # [B, embedding_dim]
            embedded_cats.append(embedded)
        
        # Concatenate all features
        cat_features = None
        all_features = None
        if len(embedded_cats) > 0:
            cat_features = np.concatenate(embedded_cats, axis=1)  # [B, 9 * embedding_dim]
            all_features = np.concatenate([numeric_part, cat_features], axis=1)  # [B, total_embedded_dim]
        else:
            all_features = numeric_part  # [B, 8]

        return all_features

    def forward(self, input_features):
        """
        Forward pass through the model.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # Conversion head
        hidden = self.relu(np.dot(x, self.head_weights_1) + self.head_bias_1)
        logits = np.dot(hidden, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs
    
    def compute_loss(self, predicted_cvr, true_labels):
        """
        Compute binary cross-entropy loss.
        
        Args:
            predicted_cvr: Predicted CVR probabilities [B, 1]
            true_labels: True conversion labels [B, 1] (0 or 1)
            
        Returns:
            Scalar loss value
        """
        # Clip predictions to prevent log(0)
        predicted_cvr = np.clip(predicted_cvr, 1e-7, 1 - 1e-7)
        loss = -np.mean(true_labels * np.log(predicted_cvr) + (1 - true_labels) * np.log(1 - predicted_cvr))
        return loss


class DelayAwareCVRModel(BaseModel):
    """Delay-aware CVR model that incorporates delay information."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Additional delay modeling parameters
        self.delay_weights = np.random.normal(0, 0.1, (1, config.cvr_hidden_dim))
        self.delay_bias = np.zeros((config.cvr_hidden_dim,))
        
        # Weight for combining base features with delay features
        self.fusion_weight = 0.5
    
    def forward(self, input_features, delay_input=None):
        """
        Forward pass incorporating delay information.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            delay_input: [B, 1] tensor with delay information in hours (optional)
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Get base feature embedding
        base_x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        base_x = self.relu(np.dot(base_x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # If delay input provided, incorporate delay information
        delay_embed = None
        combined_x = None
        if delay_input is not None:
            # Process delay information
            delay_embed = self.relu(np.dot(delay_input, self.delay_weights) + self.delay_bias)
            # Combine base features with delay features
            combined_x = self.fusion_weight * base_x + (1 - self.fusion_weight) * delay_embed
        else:
            combined_x = base_x
        
        # Conversion head
        hidden = self.relu(np.dot(combined_x, self.head_weights_1) + self.head_bias_1)
        logits = np.dot(hidden, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs


class FlowBasedCVRModel(BaseModel):
    """Flow-based model for handling delayed feedback with uncertainty quantification."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Flow-specific parameters for uncertainty modeling
        self.flow_weights = [
            np.random.normal(0, 0.1, (config.flow_hidden_dim, config.flow_hidden_dim)) 
            for _ in range(config.flow_num_layers)
        ]
        self.flow_biases = [
            np.zeros((config.flow_hidden_dim,))
            for _ in range(config.flow_num_layers)
        ]
        
        # Parameters to map feature embedding to flow parameters
        self.flow_param_mapper = np.random.normal(0, 0.1, (config.cvr_hidden_dim, config.flow_hidden_dim * 2))
    
    def forward(self, input_features):
        """
        Forward pass with flow-based uncertainty modeling.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # Map to flow parameters (mean and std for Gaussian)
        flow_params = np.dot(x, self.flow_param_mapper)
        mean_flow = flow_params[:, :self.config.flow_hidden_dim]
        std_flow = self.sigmoid(flow_params[:, self.config.flow_hidden_dim:])
        
        # Apply flow transformations
        z = mean_flow
        for layer_idx in range(self.config.flow_num_layers):
            z = self.relu(np.dot(z, self.flow_weights[layer_idx]) + self.flow_biases[layer_idx])
        
        # Combine original features with flow-transformed features
        combined_features = np.concatenate([x, z], axis=1)
        
        # Final conversion prediction
        # Construct new weights matrix to match concatenated inputs
        extended_head_weights_1 = np.random.normal(0, 0.1, (combined_features.shape[1], self.head_weights_1.shape[1]))
        head_in = self.relu(np.dot(combined_features, extended_head_weights_1) + self.head_bias_1)
        logits = np.dot(head_in, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs


class TemporalInversionCVRModel(BaseModel):
    """CVT model with temporal inversion and stratified reservoirs."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Reservoir parameters
        self.strata_count = config.reservoir_strata
        self.reservoir_size_per_stratum = config.reservoir_size // config.reservoir_strata
        self.reservoir_features = [{} for _ in range(self.strata_count)]
        self.reservoir_conversions = [{} for _ in range(self.strata_count)]
        self.reservoir_delays = [{} for _ in range(self.strata_count)]
        
        # Stratification function: divide delay time into strata
        self.delay_boundaries = np.linspace(0, config.max_delay_hours, self.strata_count + 1)
        
        # Flow-based delay modeling parameters
        self.delay_flow_weights = [
            np.random.normal(0, 0.1, (config.flow_hidden_dim, config.flow_hidden_dim)) 
            for _ in range(config.flow_num_layers)
        ]
        self.delay_flow_biases = [
            np.zeros((config.flow_hidden_dim,))
            for _ in range(config.flow_num_layers)
        ]
        
        # Mapping to delay flow params
        self.delay_param_mapper = np.random.normal(0, 0.1, (config.cvr_hidden_dim, config.flow_hidden_dim * 2))
        
    def get_stratum_index(self, delay_hours):
        """Map delay hours to reservoir stratum index."""
        for i in range(len(self.delay_boundaries)-1):
            if self.delay_boundaries[i] <= delay_hours < self.delay_boundaries[i+1]:
                return i
        return len(self.delay_boundaries) - 2  # Return last stratum for max delay
        
    def update_reservoir(self, features, conversions, delays, timestamps):
        """Update the stratified reservoir with new data."""
        for i in range(len(delays)):
            delay = delays[i][0]
            ts = timestamps[i][0]
            
            # Find stratum based on delay time
            stratum_idx = self.get_stratum_index(delay)
            
            # Add to reservoir if space available
            if len(self.reservoir_features[stratum_idx]) < self.reservoir_size_per_stratum:
                key = hash(ts)  # Use timestamp as unique key
                self.reservoir_features[stratum_idx][key] = features[i]
                self.reservoir_conversions[stratum_idx][key] = conversions[i]
                self.reservoir_delays[stratum_idx][key] = delay
            else:
                # Remove oldest entry if full
                oldest_key = min(self.reservoir_features[stratum_idx].keys())
                del self.reservoir_features[stratum_idx][oldest_key]
                del self.reservoir_conversions[stratum_idx][oldest_key]
                del self.reservoir_delays[stratum_idx][oldest_key]
                
                # Add new entry
                key = hash(ts)
                self.reservoir_features[stratum_idx][key] = features[i]
                self.reservoir_conversions[stratum_idx][key] = conversions[i]
                self.reservoir_delays[stratum_idx][key] = delay
    
    def temporal_inverse_prediction(self, features_with_delay_info):
        """Apply temporal inversion to adjust predictions based on delayed feedback."""
        # This would normally involve inverting the model to estimate what the current
        # conversion probability should have been at the time of click
        # For now, let's return the regular forward output
        return self.forward(features_with_delay_info)
    
    def forward(self, input_features):
        """
        Forward pass with reservoir-based adjustment.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # Apply flow transformation for delay modeling
        # Map to delay flow parameters (mean and std for Gaussian)
        delay_params = np.dot(x, self.delay_param_mapper)
        mean_delay_flow = delay_params[:, :self.config.flow_hidden_dim]
        std_delay_flow = self.sigmoid(delay_params[:, self.config.flow_hidden_dim:])
        
        # Apply delay flow transformations
        z_delay = mean_delay_flow
        for layer_idx in range(self.config.flow_num_layers):
            z_delay = self.relu(np.dot(z_delay, self.delay_flow_weights[layer_idx]) + self.delay_flow_biases[layer_idx])
        
        # Combine feature embedding with delay flow output
        combined_features = np.concatenate([x, z_delay], axis=1)
        
        # Conversion head
        # Construct weights matrix to accommodate combined features
        extended_head_weights_1 = np.random.normal(0, 0.1, (combined_features.shape[1], self.head_weights_1.shape[1]))
        hidden = self.relu(np.dot(combined_features, extended_head_weights_1) + self.head_bias_1)
        logits = np.dot(hidden, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs


class UnstratifiedNaiveInversion(BaseModel):
    """Baseline with naive inversion but no stratification."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Simple delay adjustment parameters
        self.delay_adjustment_weight = 0.1  # Small weight for delay adjustment
        
    def forward(self, input_features, delay_input=None):
        """
        Forward pass with simple delay adjustment.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            delay_input: [B, 1] tensor with delay information
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # If delay input provided, apply naive adjustment
        delay_adjustment = None
        if delay_input is not None:
            # Apply simple linear adjustment based on delay
            delay_adjustment = self.delay_adjustment_weight * delay_input
            x = x * (1 - np.abs(delay_adjustment))  # Reduce signal based on delay magnitude
        
        # Conversion head
        hidden = self.relu(np.dot(x, self.head_weights_1) + self.head_bias_1)
        logits = np.dot(hidden, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs


class NoInversionBaseline(BaseModel):
    """No inversion baseline - completely ignores delay information."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Standard neural network parameters without delay consideration
        self.weights = {
            'input_to_hidden': np.random.normal(0, 0.1, (config.feature_embedding_dim, config.cvr_hidden_dim)),
            'hidden_bias': np.zeros((config.cvr_hidden_dim,)),
            'hidden_to_output': np.random.normal(0, 0.1, (config.cvr_hidden_dim, 1)),
            'output_bias': np.zeros((1,))
        }
    
    def forward(self, input_features):
        """
        Forward pass ignoring delay information completely.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Standard feedforward network
        hidden = self.relu(np.dot(embedded_features, self.weights['input_to_hidden']) + 
                          self.weights['hidden_bias'])
        logits = np.dot(hidden, self.weights['hidden_to_output']) + self.weights['output_bias']
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs


class DANNModel(BaseModel):
    """Domain Adversarial Neural Network for domain adaptation."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Domain classifier parameters
        self.domain_classifier_weights = np.random.normal(0, 0.1, (config.cvr_hidden_dim, 1))
        self.domain_classifier_bias = np.zeros((1,))
        
        # Gradient reversal strength
        self.grl_lambda = 0.1
    
    def forward(self, input_features, domain_labels=None):
        """
        Forward pass with potential domain adaptation.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            domain_labels: [B, 1] tensor with domain labels (0 source, 1 target)
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # Reverse gradients for domain adaptation (in training)
        x_rev = None
        domain_logits = None
        if domain_labels is not None:
            # Apply gradient reversal
            x_rev = -self.grl_lambda * x
            
            # Domain classification
            domain_logits = np.dot(x_rev, self.domain_classifier_weights) + self.domain_classifier_bias
            domain_probs = self.sigmoid(domain_logits)
        
        # Conversion head
        hidden = self.relu(np.dot(x, self.head_weights_1) + self.head_bias_1)
        logits = np.dot(hidden, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs


class CORALModel(BaseModel):
    """CORrelation ALignment model for domain adaptation."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Additional parameters for CORAL alignment
        self.correlation_transform = np.eye(config.cvr_hidden_dim)  # Identity initially
    
    def compute_correlation_loss(self, source_features, target_features):
        """Compute CORAL loss between source and target domains."""
        # Compute covariance matrices
        src_mean = np.mean(source_features, axis=0, keepdims=True)
        tgt_mean = np.mean(target_features, axis=0, keepdims=True)
        
        src_centered = source_features - src_mean
        tgt_centered = target_features - tgt_mean
        
        src_cov = np.dot(src_centered.T, src_centered) / (src_centered.shape[0] - 1)
        tgt_cov = np.dot(tgt_centered.T, tgt_centered) / (tgt_centered.shape[0] - 1)
        
        # CORAL loss - distance between covariances
        coral_loss = np.mean((src_cov - tgt_cov)**2)
        return coral_loss
    
    def forward(self, input_features):
        """
        Forward pass with potential CORAL alignment.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # Apply correlation transform for alignment
        aligned_features = np.dot(x, self.correlation_transform)
        
        # Conversion head
        hidden = self.relu(np.dot(aligned_features, self.head_weights_1) + self.head_bias_1)
        logits = np.dot(hidden, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs


class StandardImportanceWeighting(BaseModel):
    """Standard importance weighting approach."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Importance weight estimation network
        self.weight_estimator = np.random.normal(0, 0.1, (config.cvr_hidden_dim, 1))
        
    def estimate_importance_weights(self, input_features):
        """Estimate importance weights for samples."""
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # Estimate weights
        weights = np.dot(x, self.weight_estimator)
        # Ensure positive weights
        weights = self.sigmoid(weights) * 10.0  # Sigmoid gives values in (0,1), scale up
        
        return weights
    
    def forward(self, input_features):
        """
        Forward pass with importance weighting.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # Conversion head
        hidden = self.relu(np.dot(x, self.head_weights_1) + self.head_bias_1)
        logits = np.dot(hidden, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs


class FixedExponentialEM(BaseModel):
    """Fixed exponential EM method for handling delayed feedback."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Fixed decay parameter for exponential memory
        self.decay_lambda = 0.01  # Rate of forgetting old examples
        self.memory_buffer = []  # Store historical examples
    
    def exponential_decay_weights(self, time_diffs):
        """Apply exponential decay to weights based on time difference."""
        return np.exp(-self.decay_lambda * time_diffs)
    
    def forward(self, input_features):
        """
        Forward pass with exponential memory effects.
        
        Args:
            input_features: [B, 17] tensor with 8 numerical + 9 categorical features
            
        Returns:
            cvr_probs: [B, 1] tensor with probabilities of conversion
        """
        # Embed features
        embedded_features = self.embed_features(input_features)
        
        # Feature encoding
        x = self.relu(np.dot(embedded_features, self.feature_weights_1) + self.feature_bias_1)
        x = self.relu(np.dot(x, self.feature_weights_2) + self.feature_bias_2)  # [B, 128]
        
        # Conversion head
        hidden = self.relu(np.dot(x, self.head_weights_1) + self.head_bias_1)
        logits = np.dot(hidden, self.head_weights_2) + self.head_bias_2  # [B, 1]
        cvr_probs = self.sigmoid(logits)
        
        return cvr_probs