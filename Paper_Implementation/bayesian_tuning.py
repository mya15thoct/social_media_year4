"""
Bayesian Optimization for Hyperparameter Tuning (IJISA-V17-N2-2)

This module implements Bayesian Optimization as described in the paper
for finding optimal hyperparameters. The paper states that Bayesian
Optimization is more sample-efficient than Grid Search or Random Search.

Search space:
- Layer sizes: [256, 512, 768, 1024] for layer 1
- Layer sizes: [64, 128, 256, 512] for layer 2/3
- Dropout rates: [0.1, 0.2, 0.3, 0.4, 0.5]
- Learning rates: [0.0001, 0.001, 0.01]
- Batch sizes: [32, 64, 128]
"""

import numpy as np
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import logging
import json

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define search space
search_space = [
    Integer(256, 1024, name='layer_1_size'),
    Integer(64, 512, name='layer_2_size'),
    Integer(32, 256, name='layer_3_size'),
    Real(0.1, 0.5, name='dropout_rate'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
    Categorical([32, 64, 128], name='batch_size'),
]


class BayesianTuner:
    """Bayesian Optimization for hyperparameter tuning"""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_score = 0.0
        self.best_params = None
        self.iteration = 0
        
    def build_model(self, layer_1_size, layer_2_size, layer_3_size, 
                   dropout_rate, learning_rate):
        """Build model with given hyperparameters"""
        
        model = models.Sequential(name='Tuning_Model')
        
        # Layer 1
        model.add(layers.Dense(
            layer_1_size,
            activation='relu',
            input_dim=config.N_COMPONENTS,
            name='dense_1'
        ))
        model.add(layers.Dropout(dropout_rate, name='dropout_1'))
        
        # Layer 2
        model.add(layers.Dense(
            layer_2_size,
            activation='relu',
            name='dense_2'
        ))
        model.add(layers.Dropout(dropout_rate, name='dropout_2'))
        
        # Layer 3
        model.add(layers.Dense(
            layer_3_size,
            activation='relu',
            name='dense_3'
        ))
        model.add(layers.Dropout(dropout_rate, name='dropout_3'))
        
        # Output
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    @use_named_args(search_space)
    def objective(self, layer_1_size, layer_2_size, layer_3_size,
                 dropout_rate, learning_rate, batch_size):
        """
        Objective function to minimize (negative F1-score)
        
        Returns:
            Negative F1-score (to minimize)
        """
        
        self.iteration += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ITERATION {self.iteration}")
        logger.info(f"{'='*80}")
        logger.info(f"Hyperparameters:")
        logger.info(f"  Layer 1: {layer_1_size}")
        logger.info(f"  Layer 2: {layer_2_size}")
        logger.info(f"  Layer 3: {layer_3_size}")
        logger.info(f"  Dropout: {dropout_rate:.3f}")
        logger.info(f"  Learning rate: {learning_rate:.6f}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Build model
        model = self.build_model(
            layer_1_size, layer_2_size, layer_3_size,
            dropout_rate, learning_rate
        )
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=0
        )
        
        history = model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=20,  # Reduced for tuning
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(self.X_val, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(self.y_val, y_pred, zero_division=0)
        recall = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        
        logger.info(f"\nResults:")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        # Track best
        if f1 > self.best_score:
            self.best_score = f1
            self.best_params = {
                'layer_1_size': int(layer_1_size),
                'layer_2_size': int(layer_2_size),
                'layer_3_size': int(layer_3_size),
                'dropout_rate': float(dropout_rate),
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'f1_score': float(f1)
            }
            logger.info(f"  🎯 NEW BEST F1-SCORE: {f1:.4f}")
        
        # Clean up
        del model
        keras.backend.clear_session()
        
        # Return negative F1 (we minimize)
        return -f1
    
    def optimize(self, n_calls=50):
        """
        Run Bayesian Optimization
        
        Args:
            n_calls: Number of optimization iterations
        
        Returns:
            Best hyperparameters
        """
        
        logger.info("=" * 80)
        logger.info("STARTING BAYESIAN OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Search space:")
        logger.info(f"  Layer 1 size: [256, 1024]")
        logger.info(f"  Layer 2 size: [64, 512]")
        logger.info(f"  Layer 3 size: [32, 256]")
        logger.info(f"  Dropout rate: [0.1, 0.5]")
        logger.info(f"  Learning rate: [0.0001, 0.01]")
        logger.info(f"  Batch size: [32, 64, 128]")
        logger.info(f"\nOptimization iterations: {n_calls}")
        logger.info("=" * 80)
        
        # Run optimization
        result = gp_minimize(
            func=self.objective,
            dimensions=search_space,
            n_calls=n_calls,
            random_state=config.RANDOM_STATE,
            verbose=False
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\n🏆 Best F1-Score: {self.best_score:.4f}")
        logger.info(f"\n📊 Best Hyperparameters:")
        for key, value in self.best_params.items():
            if key != 'f1_score':
                logger.info(f"  {key}: {value}")
        
        return self.best_params


def run_bayesian_optimization(X_train, y_train, n_calls=50):
    """
    Run Bayesian Optimization to find best hyperparameters
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_calls: Number of optimization iterations
    
    Returns:
        Best hyperparameters dictionary
    """
    
    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )
    
    logger.info(f"Training set: {X_train_split.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    
    # Initialize tuner
    tuner = BayesianTuner(X_train_split, y_train_split, X_val, y_val)
    
    # Run optimization
    best_params = tuner.optimize(n_calls=n_calls)
    
    # Save best parameters
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"\n✅ Best hyperparameters saved to: {config.RESULTS_DIR / 'best_hyperparameters.json'}")
    
    return best_params


if __name__ == '__main__':
    # Test Bayesian Optimization
    logger.info("Testing Bayesian Optimization...")
    
    # Load preprocessed data
    X_train = np.load(config.PROCESSED_DATA_DIR / 'X_train_pca.npy')
    y_train = np.load(config.PROCESSED_DATA_DIR / 'y_train.npy')
    
    logger.info(f"Loaded data: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # Run optimization (small number for testing)
    best_params = run_bayesian_optimization(X_train, y_train, n_calls=10)
    
    print("\n" + "=" * 80)
    print("✅ BAYESIAN OPTIMIZATION TEST COMPLETE")
    print("=" * 80)
