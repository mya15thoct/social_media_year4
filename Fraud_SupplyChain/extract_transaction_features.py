"""
Extract Transaction-based Features from DataCo Supply Chain Dataset.

This script processes the raw supply chain dataset and aggregates transaction-based
features per customer for fraud detection. Features include order characteristics,
customer behavior patterns, and product information.

Usage:
    python extract_transaction_features.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FRAUD_STATUS = 'SUSPECTED_FRAUD'
CUSTOMER_ID_COL = 'Customer Id'
FRAUD_LABEL_COL = 'is_fraud'

# Feature column definitions
FRAUD_INDICATOR_COLS = [
    'Late_delivery_risk',
    'Benefit per order',
    'Order Profit Per Order',
    'Order Item Profit Ratio',
]

TRANSACTION_VALUE_COLS = [
    'Sales',
    'Order Item Total',
    'Order Item Quantity',
    'Order Item Discount',
    'Order Item Discount Rate',
]

TIME_FEATURE_COLS = [
    'order month',
    'order day',
    'Days for shipping (real)',
]

PAYMENT_DELIVERY_COLS = [
    'Type',
    'Delivery Status',
    'Shipping Mode',
]

CUSTOMER_INFO_COLS = [
    'Customer Segment',
    'Market',
]

PRODUCT_INFO_COLS = [
    'Category Name',
    'Department Name',
]

TARGET_COL = 'Order Status'

# Aggregate all feature columns
FEATURE_COLUMNS = (
    FRAUD_INDICATOR_COLS +
    TRANSACTION_VALUE_COLS +
    TIME_FEATURE_COLS +
    PAYMENT_DELIVERY_COLS +
    CUSTOMER_INFO_COLS +
    PRODUCT_INFO_COLS +
    [TARGET_COL, CUSTOMER_ID_COL]
)

CATEGORICAL_COLS = PAYMENT_DELIVERY_COLS + CUSTOMER_INFO_COLS + PRODUCT_INFO_COLS

NUMERICAL_COLS = FRAUD_INDICATOR_COLS + TRANSACTION_VALUE_COLS + TIME_FEATURE_COLS

AGGREGATION_FUNCTIONS = ['mean', 'sum', 'std', 'min', 'max']


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load the main dataset from CSV file.
    
    Args:
        file_path: Path to the dataset CSV file
        
    Returns:
        DataFrame containing the loaded dataset
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    logger.info(f"Loading dataset from {file_path}...")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path, encoding='latin1')
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features for fraud detection.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with selected features only
    """
    # Check which columns exist
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        feature_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
    else:
        feature_columns = FEATURE_COLUMNS
    
    df_selected = df[feature_columns].copy()
    logger.info(f"Selected {len(feature_columns)} columns")
    
    return df_selected


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical variables using Label Encoding.
    
    Args:
        df: Input DataFrame with categorical columns
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of label encoders)
    """
    logger.info("Encoding categorical variables...")
    
    le_dict = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
            logger.info(f"  - {col}: {len(le.classes_)} classes")
    
    return df, le_dict


def create_fraud_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary fraud label from Order Status column.
    
    Args:
        df: Input DataFrame with Order Status column
        
    Returns:
        DataFrame with added is_fraud binary label
    """
    logger.info("Creating fraud label...")
    
    # SUSPECTED_FRAUD = 1, others = 0
    df[FRAUD_LABEL_COL] = (df[TARGET_COL] == FRAUD_STATUS).astype(int)
    
    fraud_count = df[FRAUD_LABEL_COL].sum()
    total_count = len(df)
    fraud_rate = fraud_count / total_count * 100
    
    logger.info(f"  Fraud transactions: {fraud_count:,} ({fraud_rate:.2f}%)")
    logger.info(f"  Normal transactions: {total_count - fraud_count:,} ({100-fraud_rate:.2f}%)")
    
    return df


def build_aggregation_dict(df: pd.DataFrame) -> Dict:
    """
    Build aggregation dictionary for customer-level features.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Dictionary specifying aggregation functions for each column
    """
    agg_dict = {}
    
    # Numerical: mean, sum, std, min, max
    for col in NUMERICAL_COLS:
        if col in df.columns:
            agg_dict[col] = AGGREGATION_FUNCTIONS
    
    # Categorical: mode (most frequent)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            agg_dict[col] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    
    # Fraud label: max (if any transaction is fraud, customer is fraud)
    agg_dict[FRAUD_LABEL_COL] = 'max'
    
    return agg_dict


def aggregate_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction features by Customer Id.
    
    Args:
        df: DataFrame with transaction-level data
        
    Returns:
        DataFrame with customer-level aggregated features
    """
    logger.info("Aggregating features by customer...")
    
    agg_dict = build_aggregation_dict(df)
    
    # Group by customer
    df_agg = df.groupby(CUSTOMER_ID_COL).agg(agg_dict)
    
    # Flatten column names
    df_agg.columns = [
        '_'.join(col).strip() if isinstance(col, tuple) else col
        for col in df_agg.columns.values
    ]
    
    # Rename is_fraud column
    df_agg.rename(columns={f'{FRAUD_LABEL_COL}_max': FRAUD_LABEL_COL}, inplace=True)
    
    # Reset index to make Customer Id a column
    df_agg.reset_index(inplace=True)
    
    logger.info(f"  Aggregated to {len(df_agg):,} unique customers")
    logger.info(f"  Total features: {df_agg.shape[1] - 2} (excluding Customer Id and is_fraud)")
    
    return df_agg


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save aggregated features to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where to save the file
    """
    logger.info(f"Saving transaction features to {output_path}...")
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} customers with {df.shape[1]} columns")


def print_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics of the extracted features.
    
    Args:
        df: DataFrame with extracted features
    """
    logger.info("\nSample of transaction features:")
    logger.info(f"\n{df.head()}")
    
    # Show fraud distribution
    fraud_customers = df[FRAUD_LABEL_COL].sum()
    total_customers = len(df)
    fraud_rate = fraud_customers / total_customers * 100
    
    logger.info("\nFraud distribution:")
    logger.info(f"  Fraud customers: {fraud_customers:,} ({fraud_rate:.2f}%)")
    logger.info(f"  Normal customers: {total_customers - fraud_customers:,} ({100-fraud_rate:.2f}%)")


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Starting transaction feature extraction")
    logger.info("=" * 80)
    
    # Setup paths
    current_dir = Path(__file__).parent
    dataset_path = current_dir.parent / 'data' / 'raw' / 'DataCoSupplyChainDataset.csv'
    output_path = current_dir.parent / 'data' / 'processed' / 'transaction_features.csv'
    
    try:
        # Step 1: Load dataset
        df = load_dataset(dataset_path)
        
        # Step 2: Select features
        df = select_features(df)
        
        # Step 3: Create fraud label
        df = create_fraud_label(df)
        
        # Step 4: Encode categorical variables
        df, le_dict = encode_categorical(df)
        
        # Step 5: Aggregate by customer
        df_customer = aggregate_by_customer(df)
        
        # Step 6: Save to CSV
        save_features(df_customer, output_path)
        
        # Step 7: Print summary
        print_summary(df_customer)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Transaction features extraction complete!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        raise


if __name__ == '__main__':
    main()
