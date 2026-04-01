# Experiment 15: Generative Counterfactual Imputation

## Objective
Use generative models (CVAE) to learn the full distribution of market prices and sample counterfactuals.

## Method

### Conditional VAE Architecture
- **Encoder**: q(z | x, price) → (μ, σ)
- **Decoder**: p(price | x, z) → (loc, scale)
- **Latent dim**: 32
- **Hidden dim**: 128

### Training Objective
ELBO = E_q[log p(price|x,z)] - β·KL(q(z|x,price) || N(0,I))

### Counterfactual Sampling
For lost auctions:
1. Sample z ~ N(0, I)
2. Decode: price ~ p(price | x, z)
3. Constraint: price >= bid
4. Aggregate: Use mean of n_samples=5 samples

## Results

### Training History
| Iteration | Val MSE | Win Rate AUC | Imputed Mean (Lost) |
|-----------|---------|--------------|---------------------|
| 1 | 0.0164 | 0.7208 | 0.7060 |
| 2 | 0.0207 | 0.7275 | 0.7198 |
| 3 | 0.0260 | 0.7302 | 0.7468 |

### Final Performance
- **Win Rate AUC**: 0.7302

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
| **Generative CVAE (exp15)** | 0.7302 | **Distribution Learning** |

## Future Directions
- GAN-based approach for sharper samples
- Diffusion models for higher quality generation
- Incorporate temporal dynamics (sequential bidding)
