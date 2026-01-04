"""
Configuration for Paper-Based Fraud Detection (IJISA-V17-N2-2)

This configuration replicates the exact parameters from the paper:
"Unveiling Hidden Patterns: A Deep Learning Framework Utilizing PCA 
for Fraudulent Scheme Detection in Supply Chain Analytics"
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'DataCoSupplyChainDataset.csv'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models' / 'saved_models'
RESULTS_DIR = BASE_DIR / 'results'

# ============================================================================
# Data Preprocessing (Paper Section 3.1 & 3.2)
# ============================================================================

# PCA Configuration
N_COMPONENTS = 22  # Paper: Reduced from 53 to 22 principal components
PCA_RANDOM_STATE = 42

# Feature counts (from paper)
N_CATEGORICAL_FEATURES = 17  # Label encoded
N_NUMERICAL_FEATURES = 16    # StandardScaler normalized
N_DATETIME_FEATURES = 2      # Decomposed to Year, Month, Day

# SMOTE Configuration (Paper Section 3.3)
SMOTE_SAMPLING_STRATEGY = 'auto'  # Balance minority class
SMOTE_RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5

# ============================================================================
# Model Architecture (Paper Section 3.4)
# ============================================================================

# Network Architecture
INPUT_DIM = N_COMPONENTS  # 22 PCA components
LAYER_1_SIZE = 512        # Paper: First hidden layer
LAYER_2_SIZE = 128        # Paper: Second hidden layer  
LAYER_3_SIZE = 64         # Paper: Third hidden layer
OUTPUT_SIZE = 1           # Binary classification

# Activation Functions
HIDDEN_ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'sigmoid'

# Regularization
DROPOUT_RATE = 0.2  # Paper: Uniform dropout rate

# ============================================================================
# Training Configuration (Paper Section 3.4)
# ============================================================================

# Optimizer
OPTIMIZER = 'adam'
LEARNING_RATE = 0.001  # Paper: Adam default learning rate

# Loss Function
LOSS_FUNCTION = 'binary_crossentropy'  # Paper: Standard BCE

# Training Parameters
EPOCHS = 50           # Paper: 50 epochs
BATCH_SIZE = 64       # Paper: Batch size of 64
VALIDATION_SPLIT = 0.2

# Early Stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_MODE = 'min'

# ============================================================================
# Data Split
# ============================================================================

TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY = True  # Maintain class distribution

# ============================================================================
# Bayesian Optimization (Paper Section 3.4)
# ============================================================================

# Search Space for Hyperparameter Tuning
BAYESIAN_OPT_ENABLED = True
BAYESIAN_OPT_N_CALLS = 50  # Number of optimization iterations

SEARCH_SPACE = {
    'layer_1_size': [256, 512, 768, 1024],
    'layer_2_size': [64, 128, 256, 512],
    'layer_3_size': [32, 64, 128, 256],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64, 128],
}

# ============================================================================
# Evaluation Metrics (Paper Section 4)
# ============================================================================

# Target Performance (from paper results)
TARGET_ACCURACY = 0.9942   # 99.42%
TARGET_RECALL = 0.9471     # 94.71% (Fraud Detection Rate)

# Classification Threshold
CLASSIFICATION_THRESHOLD = 0.5  # Paper: Standard threshold

# Metrics to Calculate
METRICS = [
    'accuracy',
    'precision', 
    'recall',
    'f1_score',
    'roc_auc',
    'confusion_matrix'
]

# ============================================================================
# Feature Selection (Paper Section 3.1)
# ============================================================================

# Columns to remove (redundant/non-informative)
COLUMNS_TO_REMOVE = [
    'Customer Email',
    'Customer Fname',
    'Customer Lname', 
    'Customer Password',
    'Customer Street',
    'Customer Zipcode',
    'Product Image',
    'Product Description',
    'Order Zipcode',
    'order date (DateOrders)',
    'shipping date (DateOrders)',
]

# Categorical columns for Label Encoding
CATEGORICAL_COLUMNS = [
    'Type',
    'Delivery Status',
    'Shipping Mode',
    'Customer Segment',
    'Customer City',
    'Customer State',
    'Customer Country',
    'Market',
    'Order City',
    'Order Country',
    'Order Region',
    'Order State',
    'Category Name',
    'Department Name',
    'Product Name',
    'Product Status',
    'Order Status',
]

# Numerical columns for StandardScaler
NUMERICAL_COLUMNS = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Sales per customer',
    'Late_delivery_risk',
    'Category Id',
    'Customer Id',
    'Department Id',
    'Latitude',
    'Longitude',
    'Order Item Cardprod Id',
    'Order Item Discount',
    'Order Item Discount Rate',
    'Order Item Id',
    'Order Item Product Price',
    'Order Item Profit Ratio',
    'Order Item Quantity',
    'Sales',
    'Order Item Total',
    'Order Profit Per Order',
]

# DateTime columns for decomposition
DATETIME_COLUMNS = [
    'order date (DateOrders)',
    'shipping date (DateOrders)',
]

# Target column
TARGET_COLUMN = 'Order Status'
FRAUD_LABEL = 'SUSPECTED_FRAUD'

# ============================================================================
# Logging and Visualization
# ============================================================================

VERBOSE = 1  # Training verbosity
SAVE_PLOTS = True
PLOT_FORMAT = 'png'
PLOT_DPI = 300

# ============================================================================
# Reproducibility
# ============================================================================

import numpy as np
import tensorflow as tf
import random

def set_seed(seed=RANDOM_STATE):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check paths
    if not RAW_DATA_PATH.parent.exists():
        errors.append(f"Data directory does not exist: {RAW_DATA_PATH.parent}")
    
    # Check hyperparameters
    if N_COMPONENTS <= 0:
        errors.append(f"N_COMPONENTS must be positive: {N_COMPONENTS}")
    
    if EPOCHS <= 0:
        errors.append(f"EPOCHS must be positive: {EPOCHS}")
    
    if BATCH_SIZE <= 0:
        errors.append(f"BATCH_SIZE must be positive: {BATCH_SIZE}")
    
    if not (0 < VALIDATION_SPLIT < 1):
        errors.append(f"VALIDATION_SPLIT must be between 0 and 1: {VALIDATION_SPLIT}")
    
    if TRAIN_SIZE + VAL_SIZE + TEST_SIZE != 1.0:
        errors.append(f"Data splits must sum to 1.0: {TRAIN_SIZE + VAL_SIZE + TEST_SIZE}")
    
    # Print errors
    if errors:
        for error in errors:
            print(f"❌ Configuration Error: {error}")
        return False
    
    return True


if __name__ == '__main__':
    print("=" * 80)
    print("PAPER-BASED FRAUD DETECTION CONFIGURATION")
    print("=" * 80)
    
    if validate_config():
        print("✅ Configuration is valid\n")
        print("Key Settings (from IJISA-V17-N2-2 paper):")
        print(f"  📊 PCA Components: {N_COMPONENTS}")
        print(f"  🧠 Architecture: {LAYER_1_SIZE}-{LAYER_2_SIZE}-{LAYER_3_SIZE}")
        print(f"  💧 Dropout: {DROPOUT_RATE}")
        print(f"  📈 Epochs: {EPOCHS}")
        print(f"  📦 Batch Size: {BATCH_SIZE}")
        print(f"  🎯 Loss: {LOSS_FUNCTION}")
        print(f"  🔧 Optimizer: {OPTIMIZER} (lr={LEARNING_RATE})")
        print(f"  🎲 SMOTE: {SMOTE_SAMPLING_STRATEGY}")
        print(f"\n  🎯 Target Performance:")
        print(f"     - Accuracy: {TARGET_ACCURACY:.2%}")
        print(f"     - Recall: {TARGET_RECALL:.2%}")
    else:
        print("❌ Configuration validation failed")
