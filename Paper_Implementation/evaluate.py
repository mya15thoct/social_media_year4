"""
Evaluation Script (IJISA-V17-N2-2 Paper Implementation)

Evaluate the trained model and compare with paper results:
- Target Accuracy: 99.42%
- Target Recall: 94.71%

Metrics calculated:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC
- Classification Report
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import logging
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: Path):
    """Load trained model"""
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info("✅ Model loaded successfully")
    return model


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model performance
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    
    logger.info("=" * 80)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 80)
    
    # Predict
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def print_metrics(metrics, compare_with_paper=True):
    """Print evaluation metrics"""
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\n📊 Performance Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    logger.info(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    if compare_with_paper:
        logger.info(f"\n📄 Comparison with Paper (IJISA-V17-N2-2):")
        
        acc_diff = (metrics['accuracy'] - config.TARGET_ACCURACY) * 100
        recall_diff = (metrics['recall'] - config.TARGET_RECALL) * 100
        
        logger.info(f"  Target Accuracy: {config.TARGET_ACCURACY:.4f} ({config.TARGET_ACCURACY*100:.2f}%)")
        logger.info(f"  Our Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  Difference:      {acc_diff:+.2f}%")
        
        logger.info(f"\n  Target Recall:   {config.TARGET_RECALL:.4f} ({config.TARGET_RECALL*100:.2f}%)")
        logger.info(f"  Our Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"  Difference:      {recall_diff:+.2f}%")
    
    logger.info(f"\n🔢 Confusion Matrix:")
    logger.info(f"\n{metrics['confusion_matrix']}")
    
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    logger.info(f"\n  True Negatives:  {tn:,}")
    logger.info(f"  False Positives: {fp:,}")
    logger.info(f"  False Negatives: {fn:,}")
    logger.info(f"  True Positives:  {tp:,}")


def plot_confusion_matrix(cm, save_path: Path):
    """Plot and save confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Fraud'],
        yticklabels=['Normal', 'Fraud']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png', dpi=config.PLOT_DPI)
    logger.info(f"📊 Confusion matrix saved to: {save_path / 'confusion_matrix.png'}")
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, roc_auc, save_path: Path):
    """Plot and save ROC curve"""
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'roc_curve.png', dpi=config.PLOT_DPI)
    logger.info(f"📊 ROC curve saved to: {save_path / 'roc_curve.png'}")
    plt.close()


def save_metrics_report(metrics, save_path: Path):
    """Save metrics to JSON and CSV"""
    
    import json
    
    # Save as JSON
    metrics_dict = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'roc_auc': float(metrics['roc_auc']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'paper_target_accuracy': float(config.TARGET_ACCURACY),
        'paper_target_recall': float(config.TARGET_RECALL),
        'accuracy_difference': float(metrics['accuracy'] - config.TARGET_ACCURACY),
        'recall_difference': float(metrics['recall'] - config.TARGET_RECALL)
    }
    
    with open(save_path / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    logger.info(f"📝 Metrics saved to: {save_path / 'evaluation_metrics.json'}")
    
    # Save as CSV
    df_metrics = pd.DataFrame([{
        'Metric': 'Accuracy',
        'Our Model': f"{metrics['accuracy']:.4f}",
        'Paper Target': f"{config.TARGET_ACCURACY:.4f}",
        'Difference': f"{(metrics['accuracy'] - config.TARGET_ACCURACY)*100:+.2f}%"
    }, {
        'Metric': 'Precision',
        'Our Model': f"{metrics['precision']:.4f}",
        'Paper Target': '-',
        'Difference': '-'
    }, {
        'Metric': 'Recall',
        'Our Model': f"{metrics['recall']:.4f}",
        'Paper Target': f"{config.TARGET_RECALL:.4f}",
        'Difference': f"{(metrics['recall'] - config.TARGET_RECALL)*100:+.2f}%"
    }, {
        'Metric': 'F1-Score',
        'Our Model': f"{metrics['f1_score']:.4f}",
        'Paper Target': '-',
        'Difference': '-'
    }, {
        'Metric': 'ROC-AUC',
        'Our Model': f"{metrics['roc_auc']:.4f}",
        'Paper Target': '-',
        'Difference': '-'
    }])
    
    df_metrics.to_csv(save_path / 'metrics_comparison.csv', index=False)
    logger.info(f"📝 Comparison table saved to: {save_path / 'metrics_comparison.csv'}")


def main():
    """Main evaluation execution"""
    
    # Load test data (for now, use validation split from training data)
    logger.info("Loading test data...")
    X_train = np.load(config.PROCESSED_DATA_DIR / 'X_train_pca.npy')
    y_train = np.load(config.PROCESSED_DATA_DIR / 'y_train.npy')
    
    # Use 20% as test set
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )
    
    logger.info(f"✅ Test data loaded:")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_test: {y_test.shape}")
    logger.info(f"  Fraud rate: {y_test.mean():.2%}")
    
    # Load model
    model_path = config.MODEL_DIR / 'best_model.keras'
    model = load_model(model_path)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, threshold=config.CLASSIFICATION_THRESHOLD)
    
    # Print results
    print_metrics(metrics, compare_with_paper=True)
    
    # Save visualizations
    figures_path = config.RESULTS_DIR / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(metrics['confusion_matrix'], figures_path)
    plot_roc_curve(y_test, metrics['y_pred_proba'], metrics['roc_auc'], figures_path)
    
    # Save metrics report
    metrics_path = config.RESULTS_DIR / 'metrics'
    metrics_path.mkdir(parents=True, exist_ok=True)
    save_metrics_report(metrics, metrics_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to:")
    logger.info(f"  📊 Figures: {figures_path}")
    logger.info(f"  📝 Metrics: {metrics_path}")


if __name__ == '__main__':
    main()
