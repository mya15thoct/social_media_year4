"""
Configuration for Combined Fraud Detection Model (Transaction + Network Features).

This module contains all hyperparameters, paths, and settings for training
the fraud detection model.
"""

import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

CURRENT_DIR = Path(__file__).parent
DATA_PATH = CURRENT_DIR.parent / 'data' / 'processed' / 'combined_features.csv'
MODEL_SAVE_PATH = CURRENT_DIR / 'combined_model.keras'
RESULTS_PATH = CURRENT_DIR / 'results'
SAVED_MODELS_PATH = CURRENT_DIR / 'saved_models'

# ============================================================================
# Model Architecture
# ============================================================================

# PCA dimensionality reduction
N_COMPONENTS = 45  # Increased from 35 to retain more information

# Network layer sizes
LAYER_1_SIZE = 256
LAYER_2_SIZE = 128
LAYER_3_SIZE = 64

# Dropout rates
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.3
DROPOUT_RATE_3 = 0.2

# ============================================================================
# Training Parameters
# ============================================================================

# Training epochs and batch size
EPOCHS = 100  # Increased from 50 (with early stopping)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Early stopping patience
EARLY_STOPPING_PATIENCE = 10

# Learning rate (Adam optimizer default)
LEARNING_RATE = 0.001

# ============================================================================
# Loss Function Configuration
# ============================================================================

# Loss function selection
USE_FOCAL_LOSS = False  # Disable standard focal loss
USE_COST_SENSITIVE = True  # BEST: Enable cost-sensitive focal loss

# Focal loss parameters
FOCAL_GAMMA = 0.8  # BEST: Optimized gamma parameter (focusing parameter)
FOCAL_ALPHA = 0.80  # BEST: Optimized alpha for fraud class (class balance)

# Cost-sensitive parameters
FN_COST = 15.0  # BEST: False Negative costs 15x more than False Positive

# ============================================================================
# Data Balancing (SMOTE)
# ============================================================================

# SMOTE sampling strategy
SAMPLING_STRATEGY = 1.0  # BEST: Fully balanced training (Fraud = 100% of Not Fraud)

# ============================================================================
# Evaluation Parameters
# ============================================================================

# Classification threshold
THRESHOLD = 0.20  # BEST: Aggressive threshold for maximum Recall (73.08%)

# ============================================================================
# Ensemble Configuration
# ============================================================================

# Ensemble training
USE_ENSEMBLE = True  # Train multiple models with different seeds
ENSEMBLE_SEEDS = [42, 123, 456]  # 3 random seeds for ensemble

# ============================================================================
# Random State
# ============================================================================

RANDOM_STATE = 42  # Default random state for reproducibility

# ============================================================================
# Validation
# ============================================================================

def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    
    # Validate paths exist
    if not DATA_PATH.parent.exists():
        errors.append(f"Data directory does not exist: {DATA_PATH.parent}")
    
    # Validate hyperparameters
    if N_COMPONENTS <= 0:
        errors.append(f"N_COMPONENTS must be positive, got: {N_COMPONENTS}")
    
    if EPOCHS <= 0:
        errors.append(f"EPOCHS must be positive, got: {EPOCHS}")
    
    if BATCH_SIZE <= 0:
        errors.append(f"BATCH_SIZE must be positive, got: {BATCH_SIZE}")
    
    if not (0 < VALIDATION_SPLIT < 1):
        errors.append(f"VALIDATION_SPLIT must be between 0 and 1, got: {VALIDATION_SPLIT}")
    
    if not (0 < THRESHOLD < 1):
        errors.append(f"THRESHOLD must be between 0 and 1, got: {THRESHOLD}")
    
    if SAMPLING_STRATEGY <= 0:
        errors.append(f"SAMPLING_STRATEGY must be positive, got: {SAMPLING_STRATEGY}")
    
    # Print errors if any
    if errors:
        for error in errors:
            print(f"Configuration Error: {error}")
        return False
    
    return True


if __name__ == '__main__':
    # Validate configuration when run directly
    if validate_config():
        print("✓ Configuration is valid")
        print(f"\nKey settings:")
        print(f"  Data path: {DATA_PATH}")
        print(f"  Model save path: {MODEL_SAVE_PATH}")
        print(f"  PCA components: {N_COMPONENTS}")
        print(f"  Epochs: {EPOCHS}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Ensemble: {USE_ENSEMBLE} (seeds: {ENSEMBLE_SEEDS})")
        print(f"  Cost-sensitive: {USE_COST_SENSITIVE} (FN cost: {FN_COST}x)")
    else:
        print("✗ Configuration validation failed")
