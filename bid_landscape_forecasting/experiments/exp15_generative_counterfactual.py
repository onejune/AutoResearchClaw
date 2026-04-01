"""
实验 15: Generative Counterfactual Imputation with Conditional VAE

核心创新:
- 使用生成式模型 (VAE/GAN) 学习市场出价的完整分布 P(price | context)
- 对输标样本，从条件分布中采样反事实价格
- 优势：不仅预测期望值，还能捕捉多峰分布和不确定性

方法: Conditional Variational Autoencoder (CVAE)

Encoder: q_φ(z | x, price) - 推断潜在变量 z
Decoder: p_θ(price | x, z) - 从 z 重构价格
Condition: x = 上下文特征 (包括 bid)

训练目标: ELBO (Evidence Lower Bound)
L = E_q[log p(price|x,z)] - KL(q(z|x,price) || p(z|x))

反事实推理:
- 对输标样本，从 p(price | x, bid, z~N(0,I)) 采样
- 约束：采样的价格 >= bid (物理可行性)
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# 生成式模型定义
# ============================================================================

class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for Bid Landscape
    
    学习条件分布：P(market_price | context, bid)
    
    架构:
    - Encoder: (x, price) → (μ_z, σ_z)
    - Decoder: (x, z) → price_distribution
    """
    
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: q(z | x, price)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for price
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: p(price | x, z)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # Output: [loc, scale] for distribution
        )
    
    def encode(self, x, price):
        """Encode (x, price) to latent distribution q(z|x,price)"""
        h = self.encoder(torch.cat([x, price.unsqueeze(-1)], dim=-1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z ~ N(mu, sigma^2)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, x, z):
        """Decode (x, z) to price distribution parameters"""
        h = self.decoder(torch.cat([x, z], dim=-1))
        loc = h[:, 0:1].squeeze(-1)
        scale = F.softplus(h[:, 1:2].squeeze(-1)) + 1e-6
        return loc, scale
    
    def forward(self, x, price=None):
        """
        Forward pass
        
        Training: price provided → compute ELBO
        Sampling: price=None → sample from prior p(price|x)
        """
        if price is not None:
            # Training mode
            mu, logvar = self.encode(x, price)
            z = self.reparameterize(mu, logvar)
            loc, scale = self.decode(x, z)
            
            # Compute ELBO
            recon_loss = self.reconstruction_loss(loc, scale, price)
            kl_loss = self.kl_divergence(mu, logvar)
            
            return recon_loss, kl_loss, loc, scale
        else:
            # Sampling mode
            z = torch.randn(x.size(0), self.latent_dim).to(x.device)
            loc, scale = self.decode(x, z)
            return loc, scale
    
    def reconstruction_loss(self, loc, scale, price):
        """Negative log-likelihood under predicted distribution"""
        # Assume Gaussian: -log N(price; loc, scale)
        dist = torch.distributions.Normal(loc, scale)
        nll = -dist.log_prob(price)
        return nll.mean()
    
    def kl_divergence(self, mu, logvar):
        """KL(q(z|x,y) || p(z)) where p(z) = N(0, I)"""
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl / mu.size(0)
    
    def sample_counterfactual(self, x, bids, n_samples=10, temperature=1.0):
        """
        Sample counterfactual market prices for lost auctions
        
        Args:
            x: Context features
            bids: Bid amounts (lower bound for lost auctions)
            n_samples: Number of samples per instance
            temperature: Sampling temperature (>1 for more diversity)
        
        Returns:
            samples: [N, n_samples] sampled prices
        """
        device = next(self.parameters()).device
        N = x.size(0)
        
        # Multiple samples from the distribution
        all_samples = []
        
        for _ in range(n_samples):
            z = torch.randn(N, self.latent_dim).to(device) * temperature
            loc, scale = self.decode(x, z)
            
            # Sample from predicted distribution
            dist = torch.distributions.Normal(loc, scale)
            samples = dist.sample()
            
            # Physical constraint: for lost auctions, price >= bid
            samples = torch.maximum(samples, bids)
            
            all_samples.append(samples.unsqueeze(-1))
        
        return torch.cat(all_samples, dim=-1)  # [N, n_samples]
    
    def predict_distribution(self, x):
        """Predict full distribution parameters (deterministic)"""
        z = torch.zeros(x.size(0), self.latent_dim).to(x.device)
        loc, scale = self.decode(x, z)
        return loc, scale


class BidLandscapeGAN(nn.Module):
    """
    Conditional GAN for Bid Landscape (Alternative to VAE)
    
    Generator: G(x, noise) → fake_price
    Discriminator: D(x, price) → real/fake
    """
    
    def __init__(self, input_dim, hidden_dim=128, noise_dim=32):
        super().__init__()
        self.noise_dim = noise_dim
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(input_dim + noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def generate(self, x, n_samples=1):
        """Generate fake prices"""
        device = next(self.parameters()).device
        noise = torch.randn(x.size(0), self.noise_dim).to(device)
        fake_price = self.generator(torch.cat([x, noise], dim=-1)).squeeze(-1)
        return fake_price
    
    def discriminate(self, x, price):
        """Discriminate real vs fake"""
        return self.discriminator(torch.cat([x, price.unsqueeze(-1)], dim=-1)).squeeze(-1)


# ============================================================================
# 训练函数
# ============================================================================

def train_cvae(X_train, prices_train, bids_train, events_train,
               X_val, prices_val, bids_val, events_val, config):
    """训练 Conditional VAE"""
    print("\n" + "="*60)
    print("Training Conditional VAE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    X_tr = torch.FloatTensor(X_train).to(device)
    p_tr = torch.FloatTensor(prices_train).to(device)
    b_tr = torch.FloatTensor(bids_train).to(device)
    
    X_va = torch.FloatTensor(X_val).to(device)
    p_va = torch.FloatTensor(prices_val).to(device)
    
    model = ConditionalVAE(
        input_dim=X_train.shape[1],
        hidden_dim=config.get('hidden_dim', 128),
        latent_dim=config.get('latent_dim', 32)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 50)
    beta = config.get('beta', 0.5)  # KL weight
    
    dataset = TensorDataset(X_tr, p_tr, b_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_recon = 0
        total_kl = 0
        total_loss = 0
        
        for bx, bp, bb in loader:
            optimizer.zero_grad()
            
            recon_loss, kl_loss, _, _ = model(bx, bp)
            
            # ELBO: minimize -E[log p] + beta * KL
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            _, _, va_loc, va_scale = model(X_va, p_va)
            
            # Reconstruction quality
            va_recon_loss = ((va_loc - p_va)**2).mean().item()
            
            if va_recon_loss < best_val_loss:
                best_val_loss = va_recon_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Recon: {total_recon/len(loader):.4f} | "
                  f"KL: {total_kl/len(loader):.4f} | "
                  f"Val MSE: {va_recon_loss:.4f}")
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    print(f"\nBest Val MSE: {best_val_loss:.4f}")
    
    results = {
        'model': 'Conditional VAE',
        'best_val_mse': best_val_loss,
        'config': config
    }
    
    return model, results


def train_with_generative_imputation(X_train, bids_train, events_train, values_train,
                                      X_val, bids_val, events_val, values_val, config):
    """
    使用生成式模型进行反事实插补的迭代训练
    
    流程:
    1. 训练 CVAE 学习价格分布
    2. 对输标样本，从分布中采样多个反事实价格
    3. 用采样值训练 win prediction 模型
    4. 迭代优化
    """
    print("\n" + "="*60)
    print("Training with Generative Counterfactual Imputation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_iterations = config.get('max_iter', 3)
    n_samples = config.get('n_samples', 5)
    
    current_values = values_train.copy()
    history = []
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration+1}/{max_iterations} ---")
        
        # Step 1: Train CVAE on current imputed values
        cvae_model, cvae_results = train_cvae(
            X_train, current_values, bids_train, events_train,
            X_val, values_val, bids_val, events_val,
            {**config, 'epochs': 30}
        )
        
        # Step 2: Generate counterfactual samples for lost auctions
        X_tr_t = torch.FloatTensor(X_train).to(device)
        b_tr_t = torch.FloatTensor(bids_train).to(device)
        
        cvae_model.eval()
        with torch.no_grad():
            # Sample multiple counterfactuals
            samples = cvae_model.sample_counterfactual(
                X_tr_t, b_tr_t, n_samples=n_samples, temperature=1.0
            )  # [N, n_samples]
            
            # Use mean of samples as updated imputation
            lost_mask = (events_train == 0)
            if lost_mask.sum() > 0:
                new_imputation = samples[lost_mask].mean(dim=1).cpu().numpy()
                current_values[lost_mask] = new_imputation
        
        print(f"Updated imputation mean (lost): {current_values[events_train==0].mean():.4f}")
        
        # Step 3: Train win prediction model with imputed data
        # (简化版：直接用 CVAE 的预测作为 win probability)
        X_va_t = torch.FloatTensor(X_val).to(device)
        b_va_t = torch.FloatTensor(bids_val).to(device)
        
        with torch.no_grad():
            va_loc, va_scale = cvae_model.predict_distribution(X_va_t)
            va_loc = va_loc.cpu().numpy()
            va_scale = va_scale.cpu().numpy()
            
            # P(win) = P(market_price < bid) = CDF(bid)
            from scipy.stats import norm
            z_scores = (bids_val - va_loc) / (va_scale + 1e-6)
            win_probs = norm.cdf(z_scores)
            
            auc = roc_auc_score(events_val, win_probs)
        
        print(f"Win Rate AUC: {auc:.4f}")
        
        history.append({
            'iteration': iteration + 1,
            'val_mse': cvae_results['best_val_mse'],
            'auc': auc,
            'imputed_mean': current_values[events_train==0].mean()
        })
    
    results = {
        'model': 'Generative Counterfactual Imputation',
        'history': history,
        'final_auc': history[-1]['auc'],
        'config': config
    }
    
    print(f"\nFinal AUC: {history[-1]['auc']:.4f}")
    return cvae_model, results


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("="*70)
    print("Experiment 15: Generative Counterfactual Imputation (CVAE)")
    print("="*70)
    
    config = {
        'lr': 0.001,
        'batch_size': 256,
        'epochs': 50,
        'hidden_dim': 128,
        'latent_dim': 32,
        'beta': 0.5,
        'max_iter': 3,
        'n_samples': 5,
    }
    
    # Load data
    data_path = project_root / 'data' / 'bid_landscape_train_small.parquet'
    df = pd.read_parquet(data_path)
    
    if len(df) > 15000:
        df = df.sample(15000, random_state=42)
    
    print(f"Samples: {len(df)}")
    
    # Features
    feature_cols = ['bid_amount', 'business_type', 'deviceid', 'adid']
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    
    # Format
    events = df['win_label'].values.astype(np.float32)
    true_values = df['true_value'].values.astype(np.float32)
    bids = df['bid_amount'].values.astype(np.float32)
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    value_min, value_max = true_values.min(), true_values.max()
    values_norm = (true_values - value_min) / (value_max - value_min + 1e-10)
    bids_norm = (bids - value_min) / (value_max - value_min + 1e-10)
    
    print(f"Win rate: {events.mean():.4f}")
    
    # Split
    X_train, X_val, b_train, b_val, e_train, e_val, v_train, v_val = \
        train_test_split(X, bids_norm, events, values_norm, 
                        test_size=0.2, random_state=42, stratify=events)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Initial imputation for lost auctions
    v_train_init = v_train.copy()
    lost_mask = (e_train == 0)
    v_train_init[lost_mask] = b_train[lost_mask] + 0.05  # Simple offset
    
    # Train with generative imputation
    model, results = train_with_generative_imputation(
        X_train, b_train, e_train, v_train_init,
        X_val, b_val, e_val, v_val,
        config
    )
    
    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'exp15_generative_counterfactual.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Markdown report
    md_report = f"""# Experiment 15: Generative Counterfactual Imputation

## Objective
Use generative models (CVAE) to learn the full distribution of market prices and sample counterfactuals.

## Method

### Conditional VAE Architecture
- **Encoder**: q(z | x, price) → (μ, σ)
- **Decoder**: p(price | x, z) → (loc, scale)
- **Latent dim**: {config['latent_dim']}
- **Hidden dim**: {config['hidden_dim']}

### Training Objective
ELBO = E_q[log p(price|x,z)] - β·KL(q(z|x,price) || N(0,I))

### Counterfactual Sampling
For lost auctions:
1. Sample z ~ N(0, I)
2. Decode: price ~ p(price | x, z)
3. Constraint: price >= bid
4. Aggregate: Use mean of n_samples={config['n_samples']} samples

## Results

### Training History
| Iteration | Val MSE | Win Rate AUC | Imputed Mean (Lost) |
|-----------|---------|--------------|---------------------|
"""
    
    for h in results['history']:
        md_report += f"| {h['iteration']} | {h['val_mse']:.4f} | {h['auc']:.4f} | {h['imputed_mean']:.4f} |\n"
    
    md_report += f"""
### Final Performance
- **Win Rate AUC**: {results['final_auc']:.4f}

## Key Advantages over Point Estimation

1. **Uncertainty Quantification**: Full distribution, not just mean
2. **Multi-modal Modeling**: Can capture complex price distributions
3. **Better Calibration**: Probabilistic predictions are naturally calibrated
4. **Counterfactual Diversity**: Multiple samples capture what-if scenarios

## Comparison with Previous Methods

| Method | AUC | Type |
|--------|-----|------|
| DeepHit (exp12) | 0.8641 | Survival Analysis |
| Counterfactual v2 (exp14) | 0.8591 | Multi-task Point Est. |
| **Generative CVAE (exp15)** | {results['final_auc']:.4f} | **Distribution Learning** |

## Future Directions
- GAN-based approach for sharper samples
- Diffusion models for higher quality generation
- Incorporate temporal dynamics (sequential bidding)
"""
    
    md_file = results_dir / 'exp15_generative_counterfactual.md'
    with open(md_file, 'w') as f:
        f.write(md_report)
    
    print(f"\nResults saved to {output_file}")
    print(f"Report saved to {md_file}")
    print("\n✅ Experiment 15 completed!")


if __name__ == '__main__':
    main()
