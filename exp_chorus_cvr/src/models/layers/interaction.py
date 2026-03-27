"""
Custom layers for ChorusCVR model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ChorusInteractionLayer(nn.Module):
    """Core interaction layer for ChorusCVR model"""
    
    def __init__(self, input_dim: int, interaction_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.interaction_dim = interaction_dim
        self.num_heads = num_heads
        self.head_dim = interaction_dim // num_heads
        
        assert self.head_dim * num_heads == interaction_dim, "interaction_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(input_dim, interaction_dim)
        self.k_proj = nn.Linear(input_dim, interaction_dim)
        self.v_proj = nn.Linear(input_dim, interaction_dim)
        
        # Output projection
        self.out_proj = nn.Linear(interaction_dim, input_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, _ = x.size()
        
        # Residual connection
        residual = x
        
        # Layer norm
        x = self.norm1(x)
        
        # Multi-head attention
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.interaction_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # First residual connection
        x = residual + output
        
        # Second residual connection with feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class FeatureCrossLayer(nn.Module):
    """Feature crossing layer for enhanced feature interactions"""
    
    def __init__(self, input_dim: int, cross_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.cross_dim = cross_dim
        
        # Cross transformation matrix
        self.W_cross = nn.Parameter(torch.randn(input_dim, cross_dim))
        self.b_cross = nn.Parameter(torch.zeros(cross_dim))
        
        # Element-wise transformation
        self.W_element = nn.Parameter(torch.randn(input_dim))
        self.b_element = nn.Parameter(torch.zeros(input_dim))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W_cross)
        nn.init.xavier_uniform_(self.W_element.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements feature crossing: x_{l+1} = x_0 * W_l * x_l + b_l + element_wise_transform(x_l)
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, input_dim)
        """
        batch_size = x.size(0)
        
        # Compute x_0 * W_cross * x_l (feature crossing part)
        x_0 = x  # Original input
        x_l = x  # Current layer input
        
        # Cross term: x_0 * W_cross * x_l
        cross_term = torch.mm(x_0, self.W_cross)  # (batch_size, cross_dim)
        cross_term = torch.mm(cross_term, x_l.transpose(0, 1))  # (batch_size, batch_size) -> (batch_size, input_dim) is wrong
        # Actually we want x_0 * (W_cross * x_l^T) which doesn't make sense dimensionally
        # Let's implement the correct FM-style crossing
        cross_term = torch.einsum('bi,ij,bj->bj', x_0, self.W_cross, x_l)  # (batch_size, input_dim)
        
        # Element-wise transformation
        element_term = x_l * self.W_element + self.b_element
        
        # Combine both terms
        output = cross_term + element_term + self.b_cross[:x.size(1)] if self.b_cross.size(0) >= x.size(1) else cross_term + element_term
        
        return output


class AdaptiveGate(nn.Module):
    """Adaptive gating mechanism for controlling information flow between tasks"""
    
    def __init__(self, input_dim: int, gate_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        
        # Gate network to compute gate values
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Gated output of same shape
        """
        gate_values = self.gate_network(x)  # (batch_size, 1)
        return x * gate_values  # Element-wise multiplication


class ChorusFeatureExtractor(nn.Module):
    """Feature extraction module with multiple interaction layers"""
    
    def __init__(self, input_dim: int, interaction_dim: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            ChorusInteractionLayer(input_dim, interaction_dim, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x