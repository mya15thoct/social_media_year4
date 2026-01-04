"""
Deep Neural Network Model (IJISA-V17-N2-2 Paper Implementation)

This module implements the exact DNN architecture from the paper:
- Input: 22 PCA components
- Hidden Layer 1: 512 neurons, ReLU, Dropout(0.2)
- Hidden Layer 2: 128 neurons, ReLU, Dropout(0.2)
- Hidden Layer 3: 64 neurons, ReLU, Dropout(0.2)
- Output Layer: 1 neuron, Sigmoid

Training configuration:
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross Entropy
- Batch size: 64
- Epochs: 50
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import logging

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_paper_model(input_dim: int = config.N_COMPONENTS) -> keras.Model:
    """
    Build DNN model following IJISA-V17-N2-2 paper architecture
    
    Architecture (Paper Section 3.4):
    - Input: 22 PCA components
    - Dense(512, relu) + Dropout(0.2)
    - Dense(128, relu) + Dropout(0.2)
    - Dense(64, relu) + Dropout(0.2)
    - Dense(1, sigmoid)
    
    Args:
        input_dim: Number of input features (default: 22 PCA components)
    
    Returns:
        Compiled Keras model
    """
    
    logger.info("=" * 80)
    logger.info("BUILDING DNN MODEL (IJISA-V17-N2-2 ARCHITECTURE)")
    logger.info("=" * 80)
    
    model = models.Sequential(name='Paper_DNN_Model')
    
    # Input layer + Hidden layer 1: 512 neurons
    model.add(layers.Dense(
        config.LAYER_1_SIZE,
        activation=config.HIDDEN_ACTIVATION,
        input_dim=input_dim,
        name='dense_1'
    ))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout_1'))
    
    # Hidden layer 2: 128 neurons
    model.add(layers.Dense(
        config.LAYER_2_SIZE,
        activation=config.HIDDEN_ACTIVATION,
        name='dense_2'
    ))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout_2'))
    
    # Hidden layer 3: 64 neurons
    model.add(layers.Dense(
        config.LAYER_3_SIZE,
        activation=config.HIDDEN_ACTIVATION,
        name='dense_3'
    ))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout_3'))
    
    # Output layer: Binary classification
    model.add(layers.Dense(
        config.OUTPUT_SIZE,
        activation=config.OUTPUT_ACTIVATION,
        name='output'
    ))
    
    logger.info("\n📊 Model Architecture:")
    logger.info(f"  Input: {input_dim} features (PCA components)")
    logger.info(f"  Layer 1: {config.LAYER_1_SIZE} neurons ({config.HIDDEN_ACTIVATION}) + Dropout({config.DROPOUT_RATE})")
    logger.info(f"  Layer 2: {config.LAYER_2_SIZE} neurons ({config.HIDDEN_ACTIVATION}) + Dropout({config.DROPOUT_RATE})")
    logger.info(f"  Layer 3: {config.LAYER_3_SIZE} neurons ({config.HIDDEN_ACTIVATION}) + Dropout({config.DROPOUT_RATE})")
    logger.info(f"  Output: {config.OUTPUT_SIZE} neuron ({config.OUTPUT_ACTIVATION})")
    
    # Compile model
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss=config.LOSS_FUNCTION,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    logger.info("\n🔧 Training Configuration:")
    logger.info(f"  Optimizer: {config.OPTIMIZER} (lr={config.LEARNING_RATE})")
    logger.info(f"  Loss: {config.LOSS_FUNCTION}")
    logger.info(f"  Metrics: Accuracy, Precision, Recall, AUC")
    
    logger.info("=" * 80)
    
    return model


def get_callbacks(model_save_path: str, results_path: str) -> list:
    """
    Get training callbacks
    
    Args:
        model_save_path: Path to save best model
        results_path: Path to save training logs
    
    Returns:
        List of Keras callbacks
    """
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=model_save_path,
            monitor=config.EARLY_STOPPING_MONITOR,
            save_best_only=True,
            mode=config.EARLY_STOPPING_MODE,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            mode=config.EARLY_STOPPING_MODE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # CSV logger
        CSVLogger(
            filename=f'{results_path}/training_log.csv',
            append=False
        )
    ]
    
    return callbacks


def print_model_summary(model: keras.Model):
    """Print detailed model summary"""
    
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print("\n" + "=" * 80)
    print(f"Total Parameters: {total_params:,}")
    print("=" * 80)


if __name__ == '__main__':
    # Test model building
    config.set_seed()
    
    model = build_paper_model()
    print_model_summary(model)
    
    print("\n✅ Model built successfully!")
    print(f"\nThis model replicates the IJISA-V17-N2-2 paper architecture:")
    print(f"  - Input: {config.N_COMPONENTS} PCA components")
    print(f"  - Architecture: {config.LAYER_1_SIZE}-{config.LAYER_2_SIZE}-{config.LAYER_3_SIZE}")
    print(f"  - Dropout: {config.DROPOUT_RATE}")
    print(f"  - Loss: {config.LOSS_FUNCTION}")
    print(f"  - Optimizer: {config.OPTIMIZER} (lr={config.LEARNING_RATE})")
