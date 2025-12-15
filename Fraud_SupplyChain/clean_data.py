"""
Clean corrupted data in combined_features.csv

This script cleans corrupted data entries (containing '/') in the combined features CSV file,
converts all numeric columns to proper types, and fills missing values with 0.

Usage:
    python clean_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_FILE = '../data/processed/combined_features.csv'
ID_COLUMN = 'Customer Id'
LABEL_COLUMN = 'is_fraud'
CORRUPTION_MARKER = '/'


def validate_file_exists(file_path: Path) -> bool:
    """
    Validate that the data file exists.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if file exists, False otherwise
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        logger.error("Please ensure combined_features.csv exists in the data/ folder")
        return False
    return True


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load CSV data with string type inference.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with all columns as strings
    """
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, dtype=str)
    logger.info(f"Loaded: {df.shape}")
    return df


def identify_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify columns that should be numeric (excluding ID and label columns).
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names that should be numeric
    """
    numeric_cols = [c for c in df.columns if c not in [ID_COLUMN, LABEL_COLUMN]]
    logger.info(f"Cleaning {len(numeric_cols)} numeric columns...")
    return numeric_cols


def clean_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Clean numeric columns by removing corrupted values and converting to float.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of column names to clean
        
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in numeric_cols:
        # Replace corrupted values (containing '/')
        df[col] = df[col].apply(
            lambda x: np.nan if isinstance(x, str) and CORRUPTION_MARKER in x else x
        )
        # Convert to float
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN with 0
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def convert_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the fraud label column to integer type.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with label column as int
    """
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce').fillna(0).astype(int)
    return df


def save_cleaned_data(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save cleaned DataFrame to CSV file.
    
    Args:
        df: Cleaned DataFrame
        file_path: Path where to save the file
    """
    df.to_csv(file_path, index=False)
    logger.info(f"✓ Cleaned and saved!")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Data types: \n{df.dtypes.value_counts()}")
    logger.info(f"\nSample:\n{df.head()}")


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Starting data cleaning process")
    logger.info("=" * 60)
    
    # Setup file path
    data_file_path = Path(DATA_FILE)
    
    # Validate file exists
    if not validate_file_exists(data_file_path):
        return
    
    try:
        # Load data
        df = load_data(data_file_path)
        
        # Identify numeric columns
        numeric_cols = identify_numeric_columns(df)
        
        # Clean numeric columns
        df = clean_numeric_columns(df, numeric_cols)
        
        # Convert label column
        df = convert_label_column(df)
        
        # Save cleaned data
        save_cleaned_data(df, data_file_path)
        
        logger.info("=" * 60)
        logger.info("Data cleaning completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise


if __name__ == '__main__':
    main()
