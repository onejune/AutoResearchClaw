"""
Simple test for IVR SSL components
"""
import torch
import json
from src.models import IVRContrastiveModel

def test_model_creation():
    print("Testing model creation...")
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create model
    model = IVRContrastiveModel(
        input_dim=3,  # Using 3 for click, purchase, atc
        hidden_dims=[config['model']['hidden_dim']] * config['model']['num_layers'],
        output_dim=config['model']['embedding_dim'],
        dropout=config['model']['dropout'],
        ssl_method='simclr',
        temperature=config['training']['temperature']
    )
    
    print("Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass with dummy data
    batch_size = 4
    dummy_features = torch.randn(batch_size, 3)  # 3 features: click, purchase, atc
    
    model.train()
    output = model(dummy_features)
    
    if isinstance(output, tuple):
        print(f"Output shapes: {[o.shape for o in output]}")
    else:
        print(f"Output shape: {output.shape}")
    
    print("Forward pass successful!")
    
    return model

def test_loss_computation():
    print("\nTesting loss computation...")
    
    from src.models import info_nce_loss
    
    # Create dummy embeddings
    z_i = torch.randn(4, 64)  # 4 samples, 64-dim embeddings
    z_j = torch.randn(4, 64)  # 4 samples, 64-dim embeddings
    
    loss = info_nce_loss(z_i, z_j)
    print(f"InfoNCE loss computed: {loss.item():.4f}")
    
    return loss

if __name__ == "__main__":
    print("Testing IVR SSL components...")
    model = test_model_creation()
    loss = test_loss_computation()
    print("\nAll tests passed!")