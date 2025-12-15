"""
Merge Transaction Features and Network Features.

This script merges transaction-based features with network-based features and creates
separate datasets for model comparison:
1. Transaction-only features
2. Network-only features
3. Combined features (transaction + network)

Usage:
    python merge_features.py
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CUSTOMER_ID_COL = 'Customer Id'
CUSTOMER_ID_ALT = 'customer_id'  # Alternative name in network features
FRAUD_LABEL_COL = 'is_fraud'

# Network feature columns
NETWORK_FEATURE_COLS = [
    'degree_centrality',
    'betweenness_centrality',
    'closeness_centrality',
    'pagerank',
    'eigenvector_centrality',
    'clustering_coefficient',
    'community_id',
]

# Link prediction feature columns
LINK_PREDICTION_COLS = [
    'avg_jaccard_unmade',
    'max_adamic_adar_unmade',
    'num_high_sim_unmade',
    'num_low_sim_made',
    'anomaly_score',
]


def load_transaction_features(file_path: Path) -> pd.DataFrame:
    """
    Load transaction features from CSV file.
    
    Args:
        file_path: Path to transaction features CSV
        
    Returns:
        DataFrame with transaction features
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    logger.info(f"Loading transaction features from {file_path}...")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Transaction features not found at {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"  Loaded {len(df):,} customers with {df.shape[1]} columns")
    
    return df


def load_network_features(file_path: Path) -> pd.DataFrame:
    """
    Load network features from CSV file.
    
    Args:
        file_path: Path to network features CSV
        
    Returns:
        DataFrame with network features
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    logger.info(f"Loading network features from {file_path}...")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Network features not found at {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"  Loaded {len(df):,} customers with {df.shape[1]} columns")
    
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names (e.g., customer_id -> Customer Id).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    if CUSTOMER_ID_ALT in df.columns:
        df.rename(columns={CUSTOMER_ID_ALT: CUSTOMER_ID_COL}, inplace=True)
    
    return df


def select_network_columns(df: pd.DataFrame, column_list: List[str]) -> pd.DataFrame:
    """
    Select only specified network feature columns plus Customer Id.
    
    Args:
        df: Input DataFrame with network features
        column_list: List of network feature column names
        
    Returns:
        DataFrame with selected columns only
    """
    # Check which columns exist
    existing_cols = [col for col in column_list if col in df.columns]
    selected_cols = [CUSTOMER_ID_COL] + existing_cols
    
    return df[selected_cols].copy()


def merge_features(
    df_transaction: pd.DataFrame,
    df_network: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge transaction and network features on Customer Id.
    
    Args:
        df_transaction: DataFrame with transaction features
        df_network: DataFrame with network features
        
    Returns:
        Merged DataFrame with both transaction and network features
    """
    logger.info("Merging features on Customer Id...")
    
    # Standardize column names
    df_network = standardize_column_names(df_network)
    
    # Select only network features (exclude is_fraud from network_features)
    df_network_selected = select_network_columns(df_network, NETWORK_FEATURE_COLS)
    
    # Merge
    df_merged = pd.merge(
        df_transaction,
        df_network_selected,
        on=CUSTOMER_ID_COL,
        how='inner'
    )
    
    logger.info(f"  Merged dataset: {len(df_merged):,} customers")
    logger.info(f"  Total features: {df_merged.shape[1] - 2} (excluding Customer Id and is_fraud)")
    
    # Count features
    transaction_features = df_transaction.shape[1] - 2  # exclude Customer Id and is_fraud
    network_features = len(df_network_selected.columns) - 1  # exclude Customer Id
    
    logger.info("\nFeature breakdown:")
    logger.info(f"  Transaction features: {transaction_features}")
    logger.info(f"  Network features: {network_features}")
    logger.info(f"  Combined features: {transaction_features + network_features}")
    
    return df_merged


def add_link_prediction_features(
    df_merged: pd.DataFrame,
    lp_file_path: Path
) -> pd.DataFrame:
    """
    Add link prediction features if available.
    
    Args:
        df_merged: Current merged DataFrame
        lp_file_path: Path to link prediction features CSV
        
    Returns:
        DataFrame with link prediction features added (if available)
    """
    if not lp_file_path.exists():
        logger.info("⚠️  Link prediction features not found (optional)")
        return df_merged
    
    logger.info(f"✓ Found link prediction features at {lp_file_path}")
    logger.info("Merging link prediction features...")
    
    df_lp = pd.read_csv(lp_file_path)
    
    # Standardize column names
    df_lp = standardize_column_names(df_lp)
    
    # Select only LP features
    df_lp_selected = select_network_columns(df_lp, LINK_PREDICTION_COLS)
    
    # Merge
    df_merged = pd.merge(df_merged, df_lp_selected, on=CUSTOMER_ID_COL, how='left')
    
    existing_lp_cols = [col for col in LINK_PREDICTION_COLS if col in df_lp_selected.columns]
    logger.info(f"  ✓ Added {len(existing_lp_cols)} link prediction features")
    
    return df_merged


def identify_feature_types(df_merged: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify different types of features in the merged dataset.
    
    Args:
        df_merged: Merged DataFrame
        
    Returns:
        Dictionary mapping feature type to list of column names
    """
    exclude_cols = [CUSTOMER_ID_COL, FRAUD_LABEL_COL]
    all_features = [col for col in df_merged.columns if col not in exclude_cols]
    
    # Filter network features that exist
    network_features = [col for col in NETWORK_FEATURE_COLS if col in df_merged.columns]
    
    # Transaction features = all features - network features
    transaction_features = [col for col in all_features if col not in network_features]
    
    return {
        'transaction': transaction_features,
        'network': network_features,
        'combined': all_features
    }


def save_separate_datasets(df_merged: pd.DataFrame, data_dir: Path) -> Dict[str, List[str]]:
    """
    Save 3 versions: transaction-only, network-only, combined.
    
    Args:
        df_merged: Merged DataFrame with all features
        data_dir: Directory where to save the datasets
        
    Returns:
        Dictionary mapping dataset type to feature list
    """
    logger.info("\nSaving datasets...")
    
    feature_dict = identify_feature_types(df_merged)
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Transaction-only
    transaction_cols = [CUSTOMER_ID_COL] + feature_dict['transaction'] + [FRAUD_LABEL_COL]
    df_transaction_only = df_merged[transaction_cols].copy()
    transaction_path = data_dir / 'transaction_only.csv'
    df_transaction_only.to_csv(transaction_path, index=False)
    logger.info(f"  ✅ Transaction-only: {len(feature_dict['transaction'])} features → {transaction_path.name}")
    
    # 2. Network-only
    network_cols = [CUSTOMER_ID_COL] + feature_dict['network'] + [FRAUD_LABEL_COL]
    df_network_only = df_merged[network_cols].copy()
    network_path = data_dir / 'network_only.csv'
    df_network_only.to_csv(network_path, index=False)
    logger.info(f"  ✅ Network-only: {len(feature_dict['network'])} features → {network_path.name}")
    
    # 3. Combined
    combined_path = data_dir / 'combined_features.csv'
    df_merged.to_csv(combined_path, index=False)
    logger.info(f"  ✅ Combined: {len(feature_dict['combined'])} features → {combined_path.name}")
    
    return feature_dict


def print_summary(df_merged: pd.DataFrame) -> None:
    """
    Print summary of the merged dataset.
    
    Args:
        df_merged: Merged DataFrame
    """
    logger.info("\nSample of combined features:")
    logger.info(f"\n{df_merged.head()}")
    
    # Show fraud distribution
    fraud_customers = df_merged[FRAUD_LABEL_COL].sum()
    total_customers = len(df_merged)
    fraud_rate = fraud_customers / total_customers * 100
    
    logger.info("\nFraud distribution in merged dataset:")
    logger.info(f"  Fraud customers: {fraud_customers:,} ({fraud_rate:.2f}%)")
    logger.info(f"  Normal customers: {total_customers - fraud_customers:,}")


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Starting feature merging process")
    logger.info("=" * 80)
    
    # Setup paths
    current_dir = Path(__file__).parent
    root_data_dir = current_dir.parent / 'data'
    processed_dir = root_data_dir / 'processed'
    
    transaction_path = processed_dir / 'transaction_features.csv'
    network_path = processed_dir / 'network_features.csv'
    lp_path = root_data_dir / 'link_prediction_features.csv'
    
    try:
        # Step 1: Load features
        df_transaction = load_transaction_features(transaction_path)
        df_network = load_network_features(network_path)
        
        # Step 2: Merge transaction + network
        df_merged = merge_features(df_transaction, df_network)
        
        # Step 3: Merge link prediction features if available
        df_merged = add_link_prediction_features(df_merged, lp_path)
        
        # Step 4: Save 3 versions
        feature_dict = save_separate_datasets(df_merged, processed_dir)
        
        # Step 5: Print summary
        print_summary(df_merged)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Feature merging complete!")
        logger.info("=" * 80)
        logger.info("\nNext step: Run model training to compare different feature sets")
        
    except Exception as e:
        logger.error(f"Error during feature merging: {e}")
        raise


if __name__ == '__main__':
    main()
