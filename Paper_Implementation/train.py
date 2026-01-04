"""
Training Script (IJISA-V17-N2-2 Paper Implementation)

Train the DNN model with paper's exact configuration:
- Epochs: 50
- Batch size: 64
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross Entropy
- Early stopping with patience=5
"""

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import logging
import json
from pathlib import Path

import config
from model import build_paper_model, get_callbacks, print_model_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_paper_model(X_train, y_train, use_best_params=False):
    """
    Train model following paper methodology
    
    Args:
        X_train: Training features (PCA-transformed, SMOTE-balanced)
        y_train: Training labels
        use_best_params: Whether to use Bayesian-optimized parameters
    
    Returns:
        Trained model and training history
    """
    
    logger.info("=" * 80)
    logger.info("TRAINING DNN MODEL (IJISA-V17-N2-2)")
    logger.info("=" * 80)
    
    # Set random seed for reproducibility
    config.set_seed()
    
    # Split train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )
    
    logger.info(f"\n📊 Data Split:")
    logger.info(f"  Training: {X_train_split.shape[0]:,} samples")
    logger.info(f"  Validation: {X_val.shape[0]:,} samples")
    logger.info(f"  Features: {X_train_split.shape[1]} (PCA components)")
    
    # Check class distribution
    train_fraud_rate = y_train_split.mean()
    val_fraud_rate = y_val.mean()
    logger.info(f"\n  Training fraud rate: {train_fraud_rate:.2%}")
    logger.info(f"  Validation fraud rate: {val_fraud_rate:.2%}")
    
    # Load best hyperparameters if available
    if use_best_params:
        best_params_path = config.RESULTS_DIR / 'best_hyperparameters.json'
        if best_params_path.exists():
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            
            logger.info(f"\n🔧 Using Bayesian-optimized hyperparameters:")
            logger.info(f"  Layer 1: {best_params['layer_1_size']}")
            logger.info(f"  Layer 2: {best_params['layer_2_size']}")
            logger.info(f"  Layer 3: {best_params['layer_3_size']}")
            logger.info(f"  Dropout: {best_params['dropout_rate']:.3f}")
            logger.info(f"  Learning rate: {best_params['learning_rate']:.6f}")
            logger.info(f"  Batch size: {best_params['batch_size']}")
            
            # Update config
            config.LAYER_1_SIZE = best_params['layer_1_size']
            config.LAYER_2_SIZE = best_params['layer_2_size']
            config.LAYER_3_SIZE = best_params['layer_3_size']
            config.DROPOUT_RATE = best_params['dropout_rate']
            config.LEARNING_RATE = best_params['learning_rate']
            config.BATCH_SIZE = best_params['batch_size']
        else:
            logger.warning(f"Best hyperparameters not found at {best_params_path}")
            logger.info("Using default paper hyperparameters")
    
    # Build model
    model = build_paper_model(input_dim=X_train_split.shape[1])
    print_model_summary(model)
    
    # Prepare callbacks
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_save_path = str(config.MODEL_DIR / 'best_model.keras')
    results_path = str(config.RESULTS_DIR)
    
    callbacks = get_callbacks(model_save_path, results_path)
    
    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.info("=" * 80 + "\n")
    
    history = model.fit(
        X_train_split, y_train_split,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=config.VERBOSE
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    
    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    logger.info(f"\n📊 Final Metrics:")
    logger.info(f"  Training Loss: {final_train_loss:.4f}")
    logger.info(f"  Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"  Validation Loss: {final_val_loss:.4f}")
    logger.info(f"  Validation Accuracy: {final_val_acc:.4f}")
    
    logger.info(f"\n💾 Model saved to: {model_save_path}")
    logger.info(f"📝 Training log saved to: {results_path}/training_log.csv")
    
    return model, history


def plot_training_history(history, save_path: Path):
    """Plot and save training history"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png', dpi=config.PLOT_DPI)
    logger.info(f"📊 Training history plot saved to: {save_path / 'training_history.png'}")
    plt.close()


def main():
    """Main training execution"""
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_train = np.load(config.PROCESSED_DATA_DIR / 'X_train_pca.npy')
    y_train = np.load(config.PROCESSED_DATA_DIR / 'y_train.npy')
    
    logger.info(f"✅ Loaded data:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  Fraud rate: {y_train.mean():.2%}")
    
    # Train model
    model, history = train_paper_model(X_train, y_train, use_best_params=False)
    
    # Plot training history
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, config.RESULTS_DIR)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Run evaluation: python evaluate.py")
    logger.info(f"  2. Compare with paper results:")
    logger.info(f"     - Target Accuracy: {config.TARGET_ACCURACY:.2%}")
    logger.info(f"     - Target Recall: {config.TARGET_RECALL:.2%}")


if __name__ == '__main__':
    main()
