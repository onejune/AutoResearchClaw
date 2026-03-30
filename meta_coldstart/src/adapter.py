"""
Adapter Tuning 实现

在主干模型中插入小型适配器层，只训练适配器参数，实现参数高效微调。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .models import PurchaseModel
from .data   import IVRDataset
from typing import List, Dict


class AdapterLayer(nn.Module):
    """单层 Adapter：Down-project → Non-linear → Up-project"""
    def __init__(self, hidden_size: int, bottleneck_size: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.non_linearity = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        output = self.down_proj(x)
        output = self.non_linearity(output)
        output = self.up_proj(output)
        output = self.layer_norm(output + residual)  # 残差连接
        return output


class MLPWithAdapters(nn.Module):
    """带有 Adapter 的 MLP，插在 Linear 层之间"""
    def __init__(self, original_mlp, adapter_bottleneck: int = 64):
        super().__init__()
        self.original_layers = nn.ModuleList(original_mlp.net)
        self.adapters = nn.ModuleList()
        self.adapter_bottleneck = adapter_bottleneck
        
        # 只在 Linear 层之后插入 adapter（跳过 LayerNorm、ReLU、Dropout 等）
        linear_layer_indices = []
        for i, layer in enumerate(original_mlp.net):
            if isinstance(layer, nn.Linear):
                linear_layer_indices.append(i)
        
        # 在每个 Linear 层后插入 adapter（除了最后一个 Linear 层）
        for i in range(len(linear_layer_indices)-1):  # 不在最后一层后添加
            linear_idx = linear_layer_indices[i]
            hidden_size = original_mlp.net[linear_idx].out_features
            self.adapters.append(AdapterLayer(hidden_size, adapter_bottleneck))
        
        # 复制原始 head
        self.head = nn.Linear(original_mlp.head.in_features, original_mlp.head.out_features)
        self.head.load_state_dict(original_mlp.head.state_dict())
        
        # 冻结原始参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 只训练 adapters 和 head
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        adapter_idx = 0
        for i, layer in enumerate(self.original_layers):
            x = layer(x)
            # 检查当前层是否是 Linear 层，并在非最后一层的 Linear 层后添加 adapter
            if isinstance(layer, nn.Linear):
                # 检查是否还有下一个 Linear 层
                next_linear_found = False
                for j in range(i+1, len(self.original_layers)):
                    if isinstance(self.original_layers[j], nn.Linear):
                        next_linear_found = True
                        break
                if next_linear_found and adapter_idx < len(self.adapters):
                    x = self.adapters[adapter_idx](x)
                    adapter_idx += 1
        return self.head(x).squeeze(-1)


class AdapterModel(nn.Module):
    """带有 Adapters 的完整模型"""
    def __init__(self, original_model: PurchaseModel, adapter_bottleneck: int = 64):
        super().__init__()
        self.embedding = original_model.embedding
        self.mlp_with_adapters = MLPWithAdapters(original_model.mlp, adapter_bottleneck)
        
    def forward(self, x):
        emb = self.embedding(x)
        return self.mlp_with_adapters(emb)
    
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))
    
    def loss(self, x, y):
        logits = self.forward(x)
        return F.binary_cross_entropy_with_logits(logits, y)
    
    def num_trainable_params(self):
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_adapter(adapter_model, support_loader, epochs: int = 5, lr: float = 1e-3):
    """训练 adapter 参数"""
    adapter_model.train()
    
    # 只优化 adapter 和 head 参数
    trainable_params = [p for p in adapter_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found!")
    
    optimizer = optim.Adam(trainable_params, lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for X, y in support_loader:
            optimizer.zero_grad()
            loss = adapter_model.loss(X, y)
            loss.backward()
            # 梯度裁剪防止过拟合
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    return adapter_model


def create_adapted_model(pretrained_model, k_shot: int = 100, adapter_bottleneck: int = 64):
    """创建带 adapter 的模型并进行适配"""
    # 创建带 adapter 的模型
    adapter_model = AdapterModel(pretrained_model, adapter_bottleneck)
    
    return adapter_model