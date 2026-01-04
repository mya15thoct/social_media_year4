"""
Data Preprocessing Pipeline (IJISA-V17-N2-2 Paper Implementation)

This module implements the exact preprocessing steps from the paper:
1. Feature selection (remove redundant columns)
2. Label encoding for categorical features (17 features)
3. StandardScaler for numerical features (16 features)
4. DateTime decomposition (2 features → Year, Month, Day)
5. PCA dimensionality reduction (53 → 22 components)
6. SMOTE for class imbalance handling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from pathlib import Path
import pickle
import logging

import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperPreprocessor:
    """
    Preprocessing pipeline following IJISA-V17-N2-2 paper methodology
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.N_COMPONENTS, random_state=config.PCA_RANDOM_STATE)
        self.smote = SMOTE(
            sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
            random_state=config.SMOTE_RANDOM_STATE,
            k_neighbors=config.SMOTE_K_NEIGHBORS
        )
        
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load raw dataset"""
        logger.info(f"Loading dataset from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        df = pd.read_csv(file_path, encoding='latin1')
        logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        return df
    
    def create_fraud_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary fraud label from Order Status"""
        logger.info("Creating fraud label...")
        
        df['is_fraud'] = (df[config.TARGET_COLUMN] == config.FRAUD_LABEL).astype(int)
        
        fraud_count = df['is_fraud'].sum()
        total_count = len(df)
        fraud_rate = fraud_count / total_count * 100
        
        logger.info(f"  Fraud: {fraud_count:,} ({fraud_rate:.2f}%)")
        logger.info(f"  Normal: {total_count - fraud_count:,} ({100-fraud_rate:.2f}%)")
        
        return df
    
    def remove_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove redundant columns as per paper Section 3.1
        (emails, images, unique IDs, etc.)
        """
        logger.info("Removing redundant columns...")
        
        initial_cols = len(df.columns)
        
        # Remove columns that exist in the dataframe
        cols_to_remove = [col for col in config.COLUMNS_TO_REMOVE if col in df.columns]
        df = df.drop(columns=cols_to_remove, errors='ignore')
        
        removed_count = initial_cols - len(df.columns)
        logger.info(f"  Removed {removed_count} redundant columns")
        logger.info(f"  Remaining: {len(df.columns)} columns")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Label encoding for categorical features (Paper Section 3.2)
        17 categorical features as per paper
        """
        logger.info("Encoding categorical features...")
        
        categorical_cols = [col for col in config.CATEGORICAL_COLUMNS if col in df.columns]
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))
        
        logger.info(f"  Encoded {len(categorical_cols)} categorical features")
        
        return df
    
    def decompose_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose datetime features into Year, Month, Day (Paper Section 3.2)
        2 datetime features → 6 new features
        """
        logger.info("Decomposing datetime features...")
        
        datetime_cols = [col for col in config.DATETIME_COLUMNS if col in df.columns]
        
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Extract components
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                
                # Drop original datetime column
                df = df.drop(columns=[col])
        
        logger.info(f"  Decomposed {len(datetime_cols)} datetime columns → {len(datetime_cols)*3} features")
        
        return df
    
    def scale_numerical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        StandardScaler for numerical features (Paper Section 3.2)
        16 numerical features as per paper
        """
        logger.info("Scaling numerical features...")
        
        numerical_cols = [col for col in config.NUMERICAL_COLUMNS if col in df.columns]
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        logger.info(f"  Scaled {len(numerical_cols)} numerical features")
        logger.info(f"  Mean: 0.0, Std: 1.0")
        
        return df
    
    def apply_pca(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        PCA dimensionality reduction (Paper Section 3.2)
        Reduce to 22 principal components
        """
        logger.info("Applying PCA dimensionality reduction...")
        
        if fit:
            X_pca = self.pca.fit_transform(X)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            logger.info(f"  Reduced from {X.shape[1]} → {config.N_COMPONENTS} components")
            logger.info(f"  Explained variance: {explained_variance:.2%}")
        else:
            X_pca = self.pca.transform(X)
        
        return X_pca
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        SMOTE for class imbalance (Paper Section 3.3)
        Oversample minority class (fraud)
        """
        logger.info("Applying SMOTE for class imbalance...")
        
        original_fraud = y.sum()
        original_normal = len(y) - original_fraud
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        new_fraud = y_resampled.sum()
        new_normal = len(y_resampled) - new_fraud
        
        logger.info(f"  Before SMOTE:")
        logger.info(f"    Fraud: {original_fraud:,}, Normal: {original_normal:,}")
        logger.info(f"  After SMOTE:")
        logger.info(f"    Fraud: {new_fraud:,}, Normal: {new_normal:,}")
        logger.info(f"  ✅ Balanced dataset created")
        
        return X_resampled, y_resampled
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True, apply_smote: bool = False) -> tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Raw dataframe
            fit: Whether to fit transformers (True for train, False for test)
            apply_smote: Whether to apply SMOTE (only for training data)
        
        Returns:
            X_pca: PCA-transformed features
            y: Labels
        """
        logger.info("=" * 80)
        logger.info("STARTING PREPROCESSING PIPELINE (IJISA-V17-N2-2)")
        logger.info("=" * 80)
        
        # Step 1: Create fraud label
        df = self.create_fraud_label(df)
        
        # Step 2: Remove redundant columns
        df = self.remove_redundant_columns(df)
        
        # Step 3: Decompose datetime
        df = self.decompose_datetime(df)
        
        # Step 4: Encode categorical
        df = self.encode_categorical(df, fit=fit)
        
        # Step 5: Scale numerical
        df = self.scale_numerical(df, fit=fit)
        
        # Separate features and target
        y = df['is_fraud'].values
        X = df.drop(columns=['is_fraud'], errors='ignore').values
        
        logger.info(f"\nFeature matrix shape before PCA: {X.shape}")
        
        # Step 6: Apply PCA
        X_pca = self.apply_pca(X, fit=fit)
        
        logger.info(f"Feature matrix shape after PCA: {X_pca.shape}")
        
        # Step 7: Apply SMOTE (only for training)
        if apply_smote:
            X_pca, y = self.apply_smote(X_pca, y)
        
        logger.info("=" * 80)
        logger.info("✅ PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        
        return X_pca, y
    
    def save_preprocessor(self, save_path: Path):
        """Save fitted preprocessor objects"""
        logger.info(f"Saving preprocessor to {save_path}")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'pca': self.pca,
            'smote': self.smote
        }
        
        with open(save_path / 'preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info("✅ Preprocessor saved")
    
    def load_preprocessor(self, load_path: Path):
        """Load fitted preprocessor objects"""
        logger.info(f"Loading preprocessor from {load_path}")
        
        with open(load_path / 'preprocessor.pkl', 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.pca = preprocessor_data['pca']
        self.smote = preprocessor_data['smote']
        
        logger.info("✅ Preprocessor loaded")


def main():
    """Main preprocessing execution"""
    
    # Set random seed
    config.set_seed()
    
    # Initialize preprocessor
    preprocessor = PaperPreprocessor()
    
    # Load data
    df = preprocessor.load_data(config.RAW_DATA_PATH)
    
    # Preprocess (with SMOTE for training)
    X_train, y_train = preprocessor.preprocess(df, fit=True, apply_smote=True)
    
    # Save processed data
    logger.info("\nSaving processed data...")
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    np.save(config.PROCESSED_DATA_DIR / 'X_train_pca.npy', X_train)
    np.save(config.PROCESSED_DATA_DIR / 'y_train.npy', y_train)
    
    # Save preprocessor
    preprocessor.save_preprocessor(config.PROCESSED_DATA_DIR)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Processed data saved to: {config.PROCESSED_DATA_DIR}")
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    logger.info(f"  Fraud rate: {y_train.mean():.2%}")


if __name__ == '__main__':
    main()
