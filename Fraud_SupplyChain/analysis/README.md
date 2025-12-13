# Model Comparison - Run Instructions

## What This Does

Compares **4 different feature sets** to evaluate impact of network features:

1. **Transaction Only** (57 features) - Baseline
2. **Transaction + Old Network** (61 features) - 4 network features
3. **Transaction + New Network** (65 features) - 8 network features  
4. **Network Only** (8 features) - Network features alone

## How to Run

```bash
cd Fraud_SupplyChain/analysis
python comprehensive_comparison.py
```

## Output

1. **Console**: Performance metrics for each feature set
2. **results/model_comparison.csv**: Summary table
3. **results/feature_importance.csv**: Feature rankings

## Expected Results

You'll see which feature set gives:
- Highest AUC
- Best Recall  
- Best F1-Score
- Which new features (Eigenvector, Clustering) contribute most

## Why Use This?

To **scientifically prove** that network features improve fraud detection!
