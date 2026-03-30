# IVR SSL CVR Project - Initial Experiment Report

## Project Overview
This project aims to apply contrastive learning techniques to improve CVR (Conversion Rate) prediction on the IVR dataset, specifically addressing the known negative sampling imbalance problem in IVR data.

## Dataset Analysis Results

### Key Findings
- **Total samples analyzed**: 50,000 (from 5 dates: 2025-11-01 to 2025-11-05)
- **Business types**: 10 major types including shein (16,404 samples), lazada_rta (7,993), aecps (5,293), shopee_cps (5,123)
- **Conversion metrics**:
  - ATC (Add To Cart): 295 positive samples (0.59%)
  - CV (Conversion): 0 positive samples (0.00%)
  - Purchase: Not found in analyzed columns
  - Click: Not found in analyzed columns

### Characteristics for Contrastive Learning
1. **Positive Pairs Strategy**:
   - Cross-business contrastive: Similar items across different business types (shein/aecps/aedsp/aerta/shopee/lazada)

2. **Negative Pairs Strategy**:
   - Within-business contrastive: Different items within same business type

3. **Feature Augmentation Options**:
   - Feature dropout: Randomly mask sparse features to create augmented views
   - Gaussian noise: Add small noise to dense numerical features

4. **Temporal Aspects**:
   - Time-based contrastive: Same user behavior across different time periods (daily/weekly patterns)

## Implemented Components

### 1. Data Loader (`src/data_loader.py`)
- Handles IVR parquet files from multiple dates
- Extracts key features (click, purchase, atc, business_type, numeric features)
- Provides batching capabilities

### 2. Contrastive Learning Models (`src/models.py`)
- **IVRSimCLREncoder**: SimCLR-style contrastive learning
- **IVRSupConEncoder**: Supervised contrastive learning using purchase labels
- **IVRTemporalContrastiveEncoder**: Temporal contrastive learning
- **IVRBusinessTypeContrastiveEncoder**: Contrastive learning across business types
- **Loss Functions**: InfoNCE and Supervised Contrastive Loss implementations

### 3. Core Algorithms
- InfoNCE Loss: Standard contrastive learning objective
- Supervised Contrastive Loss: Uses ground truth labels to form positive pairs
- Feature normalization and projection heads

## Technical Validation

### Component Tests Passed
- ✅ Model creation: 117,568 parameters successfully initialized
- ✅ Forward pass: Generates proper output shapes [4, 64] for both augmented views
- ✅ Loss computation: InfoNCE loss computed successfully (2.9308)

## Recommended Next Steps

### 1. Enhanced Data Preprocessing
- Address the lack of positive CV samples in early analysis
- Implement proper handling of sparse features in IVR dataset
- Create more sophisticated feature engineering pipeline

### 2. Contrastive Learning Strategies
Based on the dataset analysis, prioritize these approaches:

#### A. Business-Type Contrastive Learning
- Use business_type as a key categorical variable
- Contrast similar products across different business types
- Leverage the diversity in business types (shein, lazada, aecps, etc.)

#### B. Temporal Contrastive Learning
- Use the time-series nature of IVR data
- Contrast user behavior across different time periods
- Implement sequence-based contrastive learning

#### C. Feature-Augmented Contrastive Learning
- Apply feature dropout to create augmented views
- Use Gaussian noise on dense numerical features
- Implement more sophisticated augmentation strategies

### 3. Experimental Design
- Start with a balanced subset of business types
- Focus on ATC (Add To Cart) as intermediate signal since CV is extremely rare
- Compare different SSL methods: SimCLR, Supervised Contrastive, Business-Type Contrastive

### 4. Evaluation Framework
- Implement proper CVR prediction evaluation metrics
- Compare against baseline models (traditional ML approaches)
- Track improvements in handling negative sampling imbalance

## Expected Impact
By applying contrastive learning to the IVR dataset, we expect to:
1. Improve representation quality for CVR prediction
2. Better handle the negative sampling imbalance problem
3. Leverage cross-business similarities for improved generalization
4. Achieve better performance on the extremely sparse conversion signal

## Implementation Status
- [x] Project structure and configuration
- [x] Data loader implementation  
- [x] Core contrastive learning models
- [x] Loss functions implementation
- [x] Component validation
- [ ] Full experimental pipeline
- [ ] Results evaluation