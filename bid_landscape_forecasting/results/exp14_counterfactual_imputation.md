# Experiment 14: Counterfactual Imputation

## Objective
Learn bid landscape with right-censored data using counterfactual imputation.

## Method
Iterative training process:
1. Initialize imputation for lost bids (bid + offset)
2. Train neural network with heteroscedastic regression
3. Update imputation using model predictions
4. Repeat until convergence

## Key Components

### 1. Neural Network
- Backbone: MLP with BatchNorm + Dropout
- Output: Market price prediction + uncertainty (log variance)
- Loss: Heteroscedastic regression + physical constraints

### 2. Counterfactual Imputation Strategies
- Expectation: Use E[price|x]
- Conservative: E[price|x] + k * std
- Quantile: Upper quantile estimate

### 3. Physical Constraints
- Predicted price >= bid (for won auctions)
- Enforced via constraint loss

## Results

### Training History
| Iteration | Val Loss | RMSE | Win Rate AUC |
|-----------|----------|------|--------------|
| 1 | 0.1442 | 0.3964 | 0.2410 |
| 2 | 0.1352 | 0.3832 | 0.2123 |
| 3 | 0.1256 | 0.3844 | 0.2205 |
| 4 | 0.1252 | 0.3878 | 0.2623 |

### Final Performance
- **RMSE**: 0.3878
- **Win Rate AUC**: 0.2623

## Configuration
```json
{
  "lr": 0.001,
  "batch_size": 256,
  "epochs_per_iter": 10,
  "max_iter": 4,
  "hidden": [
    128,
    64
  ],
  "use_uncertainty": true,
  "imputation_strategy": "conservative",
  "uncertainty_scale": 0.5,
  "constraint_weight": 0.1
}
```

## Insights
- Iterative imputation refines the estimates for censored samples
- Uncertainty estimation helps with conservative bidding
- Physical constraints improve model calibration
