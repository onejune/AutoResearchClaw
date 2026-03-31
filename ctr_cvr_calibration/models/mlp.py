"""
MLP 模型
"""

import torch
import torch.nn as nn
from typing import List, Dict


class MLPCtr(nn.Module):
    """基础 MLP 模型"""
    
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        feature_cols: List[str],
        embed_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        max_vocab_size: int = 100000
    ):
        super().__init__()
        
        self.feature_cols = feature_cols
        self.num_features = len(feature_cols)
        
        # Embedding 层
        self.embeddings = nn.ModuleDict()
        for col in feature_cols:
            vocab_size = min(vocab_sizes.get(col, max_vocab_size), max_vocab_size)
            self.embeddings[col] = nn.Embedding(vocab_size, embed_dim)
        
        # MLP
        input_dim = embed_dim * self.num_features
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features):
        batch_size = features.size(0)
        
        # Embedding
        embeds = []
        for i, col in enumerate(self.feature_cols):
            # 处理越界值：取模
            feat_vals = features[:, i]
            vocab_size = self.embeddings[col].num_embeddings
            feat_vals = feat_vals % vocab_size  # 防止越界
            embed = self.embeddings[col](feat_vals)
            embeds.append(embed)
        
        x = torch.cat(embeds, dim=1)
        logits = self.mlp(x).squeeze(-1)
        
        return logits
    
    def predict(self, features):
        return torch.sigmoid(self.forward(features))


if __name__ == '__main__':
    vocab_sizes = {f'feat_{i}': 1000 for i in range(10)}
    feature_cols = list(vocab_sizes.keys())
    
    model = MLPCtr(vocab_sizes, feature_cols)
    features = torch.randint(0, 1000, (32, 10))
    
    logits = model(features)
    print(f"Logits shape: {logits.shape}")
