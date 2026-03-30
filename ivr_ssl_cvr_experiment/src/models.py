"""
Contrastive Learning Models for IVR CVR Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IVRFeatureEncoder(nn.Module):
    """
    Feature encoder that handles categorical and numerical features from combine_schema
    """
    def __init__(self, 
                 categorical_dims, 
                 numerical_dim=0, 
                 embedding_dim=64, 
                 hidden_dims=[128, 64], 
                 output_dim=64, 
                 dropout=0.1):
        """
        Initialize encoder with categorical and numerical features
        
        Args:
            categorical_dims: dict mapping feature name to vocab size {'feature_name': vocab_size}
            numerical_dim: dimension of numerical features
            embedding_dim: embedding dimension for each categorical feature
        """
        super().__init__()
        
        self.categorical_dims = categorical_dims
        self.numerical_dim = numerical_dim
        self.embedding_dim = embedding_dim
        
        # Create embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for feat_name, vocab_size in categorical_dims.items():
            self.embeddings[feat_name] = nn.Embedding(vocab_size, embedding_dim)
            total_embedding_dim += embedding_dim
        
        # Total input dimension after embedding
        total_input_dim = total_embedding_dim + numerical_dim
        
        # Create encoder layers
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.projection_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.BatchNorm1d(prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, output_dim)
        )
    
    def forward(self, categorical_features, numerical_features=None):
        """
        Forward pass
        
        Args:
            categorical_features: dict of {feature_name: tensor_of_indices}
            numerical_features: tensor of numerical features (optional)
        """
        # Embed categorical features
        embedded_cats = []
        for feat_name, indices in categorical_features.items():
            embedded = self.embeddings[feat_name](indices)
            embedded_cats.append(embedded)
        
        # Flatten all embedded features
        if embedded_cats:
            cat_features = torch.cat(embedded_cats, dim=-1)
        else:
            cat_features = torch.empty((categorical_features[list(categorical_features.keys())[0]].size(0), 0))
        
        # Concatenate with numerical features
        if numerical_features is not None:
            x = torch.cat([cat_features, numerical_features], dim=-1)
        else:
            x = cat_features
        
        # Pass through encoder
        encoded = self.encoder(x)
        projected = self.projection_head(encoded)
        return F.normalize(projected, dim=-1)


class IVRBaseEncoder(nn.Module):
    """
    Base encoder for IVR features using schema-compliant features
    """
    def __init__(self, schema_features, embedding_dim=64, hidden_dims=[128, 64], output_dim=64, dropout=0.1):
        super().__init__()
        
        # For now, we'll use a simplified approach where we assume all features are categorical
        # with reasonable vocab sizes
        categorical_dims = {feat: 10000 for feat in schema_features}  # Placeholder vocab sizes
        self.feature_encoder = IVRFeatureEncoder(
            categorical_dims=categorical_dims,
            numerical_dim=0,  # We'll handle numerical features separately if needed
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
    
    def forward(self, batch):
        """
        Forward pass expecting a batch with schema features
        
        Args:
            batch: dict containing categorical features from schema
        """
        # Extract categorical features from batch that are defined in categorical_dims
        categorical_features = {}
        for feat_name in self.feature_encoder.categorical_dims.keys():
            if feat_name in batch and batch[feat_name] is not None:
                categorical_features[feat_name] = batch[feat_name]
        
        # If no matching features were found, use the ones that actually exist in the batch
        if not categorical_features:
            # Use common features that are always present in our data
            for feat_name in ['business_type', 'imptype', 'objective_type', 'traffic_type', 'click', 'purchase', 'atc']:
                if feat_name in batch:
                    categorical_features[feat_name] = batch[feat_name]
        
        # For now, pass dummy numerical features (we'll enhance this later)
        return self.feature_encoder(categorical_features, None)


class IVRSimCLREncoder(nn.Module):
    """
    Encoder for SimCLR-style contrastive learning on IVR data
    """
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64], output_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = IVRBaseEncoder(input_dim, hidden_dims, output_dim, dropout)
        
        # Additional augmentation layers for generating positive pairs
        self.augmentation_transforms = nn.ModuleList([
            nn.Linear(output_dim, output_dim),
            nn.Linear(output_dim, output_dim)
        ])
    
    def forward(self, x):
        h = self.encoder(x)
        return h
    
    def get_augmented_views(self, x):
        """
        Generate two augmented views of the same input
        """
        h = self.encoder(x)
        
        # Apply different transformations to generate views
        view1 = F.normalize(self.augmentation_transforms[0](h), dim=-1)
        view2 = F.normalize(self.augmentation_transforms[1](h), dim=-1)
        
        return view1, view2


class IVRSupConEncoder(nn.Module):
    """
    Supervised Contrastive Learning encoder for IVR data
    Uses purchase labels to form positive pairs
    """
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64], output_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = IVRBaseEncoder(input_dim, hidden_dims, output_dim, dropout)
    
    def forward(self, x):
        return self.encoder(x)


class IVRTemporalContrastiveEncoder(nn.Module):
    """
    Temporal contrastive learning encoder for IVR data
    Contrasts across different time periods
    """
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64], output_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = IVRBaseEncoder(input_dim, hidden_dims, output_dim, dropout)
    
    def forward(self, x):
        return self.encoder(x)


class IVRBusinessTypeContrastiveEncoder(nn.Module):
    """
    Contrastive learning encoder that contrasts across business types
    """
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64], output_dim=64, dropout=0.1, n_business_types=10):
        super().__init__()
        self.encoder = IVRBaseEncoder(input_dim, hidden_dims, output_dim, dropout)
        self.business_type_embedding = nn.Embedding(n_business_types, output_dim)
    
    def forward(self, x, business_type_ids):
        h = self.encoder(x)
        business_emb = self.business_type_embedding(business_type_ids)
        combined = h + business_emb
        return F.normalize(combined, dim=-1)


def info_nce_loss(z_i, z_j, temperature=0.1):
    """
    Compute InfoNCE loss for contrastive learning
    
    Args:
        z_i, z_j: representations of positive pairs (same sample, different augmentations)
        temperature: temperature scaling factor
    
    Returns:
        loss: scalar InfoNCE loss
    """
    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate representations
    representations = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    
    # Compute similarity matrix
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), 
        representations.unsqueeze(0), 
        dim=2
    ) / temperature  # [2N, 2N]
    
    # Remove self-similarity
    sim_ij = torch.diag(similarity_matrix[:z_i.size(0)], z_i.size(0))  # z_i to z_j similarities
    sim_ji = torch.diag(similarity_matrix[z_i.size(0):, :z_i.size(0)])  # z_j to z_i similarities
    positives = torch.cat([sim_ij, sim_ji], dim=0)  # [2N]
    
    # Negative mask (exclude self-similarity)
    N = z_i.size(0)
    mask = torch.eye(2*N, device=z_i.device).bool()
    negatives = similarity_matrix.masked_fill(mask, float('-inf'))
    
    # Compute loss
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)  # [2N, 2N+1]
    labels = torch.zeros(2*N, device=z_i.device, dtype=torch.long)  # Positive samples at index 0
    
    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """
    Supervised Contrastive Learning loss
    
    Args:
        features: feature representations [N, D]
        labels: ground truth labels [N]
        temperature: temperature scaling factor
    
    Returns:
        loss: scalar supervised contrastive loss
    """
    device = features.device
    batch_size = features.shape[0]
    
    # Normalize features
    features = F.normalize(features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature  # [N, N]
    
    # Create label mask - positive pairs have same label
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]
    
    # Remove diagonal (self-similarity)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # Compute log probabilities
    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
    
    # Compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
    # Loss is negative mean of log probabilities
    loss = -mean_log_prob_pos.mean()
    
    return loss


def temporal_contrastive_loss(features, time_indices, temperature=0.1):
    """
    Temporal contrastive loss - contrasts across time periods
    Features from same entity at different times should be similar
    Features from different entities should be dissimilar
    """
    # This is a simplified implementation
    # In practice, we'd group by entity and contrast across time
    return info_nce_loss(features, features, temperature)  # Placeholder


class IVRContrastiveModel(nn.Module):
    """
    Complete IVR Contrastive Learning Model supporting multiple approaches
    Uses features from combine_schema for consistency with multitask project
    """
    def __init__(self, 
                 schema_features,
                 embedding_dim=64, 
                 hidden_dims=[128, 64], 
                 output_dim=64, 
                 dropout=0.1, 
                 ssl_method='simclr',
                 n_business_types=10,
                 temperature=0.1):
        super().__init__()
        
        self.ssl_method = ssl_method
        self.temperature = temperature
        self.schema_features = schema_features
        
        # Load schema features and create encoder
        if ssl_method == 'simclr':
            self.encoder = IVRBaseEncoder(schema_features, embedding_dim, hidden_dims, output_dim, dropout)
        elif ssl_method == 'supervised_contrastive':
            self.encoder = IVRBaseEncoder(schema_features, embedding_dim, hidden_dims, output_dim, dropout)
        elif ssl_method == 'temporal_contrastive':
            self.encoder = IVRBaseEncoder(schema_features, embedding_dim, hidden_dims, output_dim, dropout)
        elif ssl_method == 'business_type_contrastive':
            self.encoder = IVRBaseEncoder(schema_features, embedding_dim, hidden_dims, output_dim, dropout)
        else:
            raise ValueError(f"Unknown SSL method: {ssl_method}")
    
    def forward(self, batch, labels=None):
        # For now, just pass the batch through the encoder
        # The encoder expects to handle the batch dictionary with schema features
        return self.encoder(batch)
    
    def compute_loss(self, batch):
        """
        Compute contrastive loss using schema-compliant features
        """
        if self.ssl_method == 'simclr':
            # For SimCLR, we need to create augmented views of the same input
            # This is more complex with categorical features, so we'll use a simplified approach
            features = self.forward(batch)
            # For now, duplicate features to simulate positive pairs (this is a placeholder)
            view1 = features
            view2 = features + 0.1 * torch.randn_like(features)  # Add small noise as augmentation
            loss = info_nce_loss(view1, view2, self.temperature)
        elif self.ssl_method == 'supervised_contrastive':
            features = self.forward(batch)
            # Use purchase labels for supervised contrastive learning
            labels = batch['purchase'].long()
            loss = supervised_contrastive_loss(features, labels, self.temperature)
        elif self.ssl_method == 'business_type_contrastive':
            features = self.forward(batch)
            # For business type contrastive learning, we'll use business_type as grouping
            business_type_ids = batch['business_type']
            loss = supervised_contrastive_loss(features, business_type_ids, self.temperature)
        else:
            raise ValueError(f"Unknown SSL method: {self.ssl_method}")
        
        return loss