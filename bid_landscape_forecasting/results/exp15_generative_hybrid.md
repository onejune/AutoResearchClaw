# Experiment 15 (v2): Hybrid Generative-Discriminative Model

## Objective
Combine generative (CVAE) and discriminative (multi-task) approaches for better performance.

## Architecture

```
Input → Shared Backbone
        ├→ Win Head (Sigmoid)     → P(win)         [Discriminative]
        ├→ Price Head (Linear)    → E[price]       [Point Estimate]
        └→ VAE Encoder/Decoder    → Distribution   [Generative]
```

## Loss Function

L = α·L_BCE(win) + β·L_MSE(price) + γ·L_ELBO(vae)

Where:
- α = 0.6 (win prediction)
- β = 0.3 (price prediction)  
- γ = 0.1 (uncertainty quantification)

## Results

| Metric | Value |
|--------|-------|
| **Win Rate AUC** | **0.8601** |
| Configuration | hidden=128, latent=16 |

## Comparison with Other Methods

| Method | Type | AUC |
|--------|------|-----|
| DeepHit (exp12) | Survival Analysis | 0.8641 |
| Counterfactual v2 (exp14) | Multi-task Point Est. | 0.8591 |
| Generative CVAE (exp15-v1) | Pure Generative | 0.7302 |
| **Hybrid (exp15-v2)** | **Gen+Disc** | **0.8601** |

## Key Advantages

1. **Direct Optimization**: Win head directly optimizes AUC
2. **Uncertainty Quantification**: VAE provides calibrated uncertainty
3. **Multi-task Learning**: Shared representation improves generalization
4. **Best of Both Worlds**: Combines generative and discriminative strengths

## Insights

- Pure generative models (v1) struggle with indirect win rate estimation
- Hybrid approach maintains direct supervision while capturing uncertainty
- Latent space regularization prevents overfitting
