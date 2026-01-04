# Paper Implementation: IJISA-V17-N2-2

**Exact replication of the methodology from:**  
"Unveiling Hidden Patterns: A Deep Learning Framework Utilizing PCA for Fraudulent Scheme Detection in Supply Chain Analytics"

Published in: International Journal of Intelligent Systems and Applications (IJISA), Vol.17, No.2, 2025

---

## 📋 Overview

This implementation replicates the **exact methodology** described in the IJISA-V17-N2-2 paper for fraud detection in supply chain analytics. The goal is to:

1. Understand the paper's approach
2. Compare performance with enhanced methods (SNA + Cost-Sensitive Loss)
3. Validate the paper's reported results

---

## 🏗️ Architecture

### Data Preprocessing
- **Feature Selection**: Remove redundant columns (emails, images, IDs)
- **Categorical Encoding**: Label Encoding for 17 categorical features
- **Numerical Scaling**: StandardScaler for 16 numerical features
- **DateTime Decomposition**: Extract Year, Month, Day from 2 datetime columns
- **PCA**: Reduce from 53 → **22 principal components**
- **SMOTE**: Balance class distribution

### Model Architecture
```
Input: 22 PCA components
↓
Dense(512, relu) + Dropout(0.2)
↓
Dense(128, relu) + Dropout(0.2)
↓
Dense(64, relu) + Dropout(0.2)
↓
Dense(1, sigmoid)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Cross Entropy
- **Batch Size**: 64
- **Epochs**: 50
- **Early Stopping**: Patience=5

### Hyperparameter Tuning
- **Method**: Bayesian Optimization
- **Search Space**: Layer sizes, dropout rates, learning rates, batch sizes
- **Objective**: Maximize F1-Score

---

## 📊 Target Performance (from Paper)

| Metric | Target |
|--------|--------|
| **Accuracy** | 99.42% |
| **Recall** | 94.71% |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place `DataCoSupplyChainDataset.csv` in `data/raw/` directory.

### 3. Run Complete Pipeline

```bash
# Step 1: Preprocess data (PCA + SMOTE)
python preprocessing.py

# Step 2: (Optional) Bayesian Optimization for hyperparameters
python bayesian_tuning.py

# Step 3: Train model
python train.py

# Step 4: Evaluate model
python evaluate.py
```

---

## 📁 Project Structure

```
Paper_Implementation/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # PCA-transformed, SMOTE-balanced data
├── models/
│   └── saved_models/           # Trained models
├── results/
│   ├── figures/                # Confusion matrix, ROC curve
│   └── metrics/                # Performance metrics
├── config.py                   # Paper's exact parameters
├── preprocessing.py            # Data preprocessing pipeline
├── model.py                    # DNN architecture
├── bayesian_tuning.py          # Hyperparameter optimization
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🔬 Methodology Details

### 1. Data Preprocessing (Section 3.1 & 3.2)

**Feature Engineering:**
- Remove non-informative columns (customer emails, product images, etc.)
- Encode 17 categorical features using Label Encoding
- Normalize 16 numerical features using StandardScaler (mean=0, std=1)
- Decompose 2 datetime features into Year, Month, Day

**Dimensionality Reduction:**
- Apply PCA to reduce from 53 features to **22 principal components**
- Retain maximum variance while reducing noise

### 2. Class Imbalance Handling (Section 3.3)

**SMOTE (Synthetic Minority Over-sampling Technique):**
- Oversample minority class (fraud transactions)
- Balance training data for better learning

### 3. Model Architecture (Section 3.4)

**Deep Neural Network:**
- 3 hidden layers: 512 → 128 → 64 neurons
- ReLU activation for hidden layers
- Sigmoid activation for output layer
- Dropout (0.2) for regularization
- Binary Cross Entropy loss

### 4. Hyperparameter Tuning

**Bayesian Optimization:**
- More sample-efficient than Grid Search or Random Search
- Builds probabilistic model of objective function
- Optimizes: layer sizes, dropout, learning rate, batch size

---

## 📈 Results

After running the pipeline, results will be saved to:

- **Training History**: `results/training_history.png`
- **Confusion Matrix**: `results/figures/confusion_matrix.png`
- **ROC Curve**: `results/figures/roc_curve.png`
- **Metrics Report**: `results/metrics/evaluation_metrics.json`
- **Comparison Table**: `results/metrics/metrics_comparison.csv`

---

## 🆚 Comparison with Enhanced Approach

| Feature | Paper Approach | Enhanced Approach |
|---------|---------------|-------------------|
| **Features** | Transaction only (53 → 22 PCA) | Transaction + Network (65 → 45 PCA) |
| **Loss Function** | Binary Cross Entropy | Cost-Sensitive Focal Loss |
| **Architecture** | 512-128-64 | 256-128-64 + BatchNorm |
| **Hyperparameter Tuning** | Bayesian Optimization | Manual + Ensemble |
| **Network Analysis** | ❌ No | ✅ Yes (SNA features) |

---

## 🎯 Key Differences from Paper

### What's the Same:
- ✅ Dataset (DataCo Smart Supply Chain)
- ✅ DNN architecture (512-128-64)
- ✅ PCA (22 components)
- ✅ SMOTE for class imbalance
- ✅ Binary Cross Entropy loss
- ✅ Adam optimizer

### What's Different:
- ❌ No network features (paper uses only transaction features)
- ❌ No cost-sensitive loss (paper uses standard BCE)
- ❌ No BatchNormalization (paper doesn't mention it)

---

## 📚 Paper Reference

```bibtex
@article{ijisa2025fraud,
  title={Unveiling Hidden Patterns: A Deep Learning Framework Utilizing PCA for Fraudulent Scheme Detection in Supply Chain Analytics},
  journal={International Journal of Intelligent Systems and Applications},
  volume={17},
  number={2},
  year={2025}
}
```

---

## 🔧 Configuration

Edit `config.py` to adjust:
- PCA components (default: 22)
- Model architecture (default: 512-128-64)
- Dropout rate (default: 0.2)
- Training epochs (default: 50)
- Batch size (default: 64)
- Learning rate (default: 0.001)

---

## 📝 Notes

> **Purpose**: This implementation is for **comparison and validation** purposes. It replicates the paper's methodology exactly without enhancements.

> **Enhanced Version**: For the enhanced version with SNA features and cost-sensitive loss, see the main `Fraud_SupplyChain` directory.

---

## ✅ Validation Checklist

- [x] PCA reduces to 22 components
- [x] DNN architecture: 512-128-64
- [x] Dropout: 0.2 (uniform)
- [x] Loss: Binary Cross Entropy
- [x] Optimizer: Adam (lr=0.001)
- [x] Batch size: 64
- [x] Epochs: 50
- [x] SMOTE for class balance
- [x] Bayesian Optimization for hyperparameters

---

## 🤝 Contributing

This is a research implementation for comparison purposes. For improvements and enhancements, please refer to the main project directory.

---

## 📧 Contact

For questions about this implementation, please refer to the paper or the main project documentation.
