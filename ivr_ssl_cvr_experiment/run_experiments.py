"""
Experiment Runner for Different Contrastive Learning Approaches on IVR Data
"""
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import json
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np

from src.data_loader import get_ivr_dataloader
from src.models import IVRContrastiveModel


def run_experiment(config_path, ssl_method, experiment_name):
    """
    Run a single experiment with a specific SSL method using schema-compliant features
    """
    print(f"Running experiment: {experiment_name} with {ssl_method}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update SSL method in config
    original_ssl_method = config['training'].get('ssl_method', 'simclr')
    config['training']['ssl_method'] = ssl_method
    
    # Load schema features
    schema_features = []
    schema_file = "/mnt/workspace/open_research/autoresearch/multitask/combine_schema"
    with open(schema_file, 'r') as f:
        schema_features = [line.strip() for line in f if line.strip()]
    print(f"Using {len(schema_features)} features from combine_schema")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = IVRContrastiveModel(
        schema_features=schema_features,
        embedding_dim=config['model']['embedding_dim'],
        hidden_dims=[config['model']['hidden_dim']] * config['model']['num_layers'],
        output_dim=config['model']['embedding_dim'],
        dropout=config['model']['dropout'],
        ssl_method=ssl_method,
        n_business_types=10,
        temperature=config['training']['temperature']
    ).to(device)
    
    # Optimizer
    optimizer = Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create data loader
    print("Creating data loader...")
    dataloader = get_ivr_dataloader(
        data_path=config['dataset']['path'],
        date_range=config['dataset'].get('date_range', ["2025-11-01"]),  # Use first available date for quick test
        batch_size=config['training']['batch_size'],
        sample_ratio=0.001  # Very small sample for initial experiments
    )
    
    # Training loop (just a few batches for proof of concept)
    model.train()
    total_loss = 0
    num_batches = 0
    
    print("Starting experiment...")
    for epoch in range(min(2, config['training']['epochs'])):  # Just 2 epochs for quick test
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # Limit to 10 batches for quick test
                break
                
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass and loss computation
            try:
                loss = model.compute_loss(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                batch_count += 1
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Experiment {experiment_name} completed. Average Loss: {avg_loss:.4f}")
    
    # Save experiment results
    results_dir = os.path.join(os.path.dirname(config_path), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"experiment_{experiment_name}_{timestamp}.json")
    
    result_data = {
        "experiment_name": experiment_name,
        "ssl_method": ssl_method,
        "average_loss": avg_loss,
        "num_batches_processed": num_batches,
        "timestamp": timestamp,
        "config_used": config
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    return avg_loss, result_file


def run_all_experiments(config_path):
    """
    Run all contrastive learning experiments
    """
    print("Starting all IVR contrastive learning experiments...")
    
    # Define experiment configurations
    experiments = [
        {
            "name": "simclr_ivr",
            "method": "simclr",
            "description": "SimCLR-style contrastive learning on IVR data"
        },
        {
            "name": "supervised_contrastive_ivr",
            "method": "supervised_contrastive",
            "description": "Supervised contrastive learning using purchase labels"
        },
        {
            "name": "business_type_contrastive_ivr",
            "method": "business_type_contrastive",
            "description": "Contrastive learning across business types"
        }
    ]
    
    results = []
    
    for exp in experiments:
        try:
            loss, result_file = run_experiment(
                config_path, 
                exp["method"], 
                exp["name"]
            )
            results.append({
                "experiment": exp["name"],
                "method": exp["method"], 
                "loss": loss,
                "result_file": result_file,
                "status": "completed"
            })
        except Exception as e:
            print(f"Experiment {exp['name']} failed: {str(e)}")
            results.append({
                "experiment": exp["name"],
                "method": exp["method"],
                "loss": float('inf'),
                "result_file": None,
                "status": "failed",
                "error": str(e)
            })
    
    # Save overall results
    results_summary_path = os.path.join(
        os.path.dirname(config_path), 
        "results", 
        f"experiment_results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(results_summary_path, 'w') as f:
        json.dump({
            "experiment_run_timestamp": datetime.now().isoformat(),
            "experiments": results
        }, f, indent=2)
    
    print("\nAll experiments completed!")
    print("Results:")
    for result in results:
        print(f"- {result['experiment']}: {result['loss']:.4f} ({result['status']})")
    
    print(f"\nDetailed results saved to: {results_summary_path}")
    
    return results


def main():
    config_path = "/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr/config.json"
    
    print("Starting IVR Contrastive Learning Experiments")
    print("=" * 50)
    
    results = run_all_experiments(config_path)
    
    # Create a simple report
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY REPORT")
    print("=" * 50)
    
    completed_results = [r for r in results if r['status'] == 'completed']
    if completed_results:
        completed_results.sort(key=lambda x: x['loss'])
        print("Ranking by loss (lower is better):")
        for i, result in enumerate(completed_results, 1):
            print(f"{i}. {result['experiment']}: {result['loss']:.4f}")
    else:
        print("No experiments completed successfully.")


if __name__ == "__main__":
    main()