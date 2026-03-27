"""
Base model class for ChorusCVR and related models
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Base model class providing common functionality"""
    
    def __init__(self):
        super().__init__()
        self._initialized = False
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning predictions for all tasks"""
        pass
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prediction method with detached outputs"""
        with torch.no_grad():
            outputs = self.forward(x)
            return {k: v.detach() for k, v in outputs.items()}
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for logging/debugging"""
        return {name: param for name, param in self.named_parameters()}
    
    def initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)


class MultiTaskOutput(nn.Module):
    """Wrapper for multi-task outputs"""
    
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
    
    def forward(self, task_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Ensure all required tasks are present
        for task in self.task_names:
            assert task in task_outputs, f"Missing output for task: {task}"
        return task_outputs


class TaskSpecificTower(nn.Module):
    """Task-specific prediction tower"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = "relu"):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        
        # Final layer for binary classification
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        layers.append(nn.Sigmoid())  # Sigmoid for probability output
        
        self.tower = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(x).squeeze(-1)