"""
Data loading and preprocessing module.
Handles loading the dataset, exporting to CSV, and splitting into train/test sets.
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset(data_dir: str = "data", save_csv: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load Breast Cancer Wisconsin Dataset from scikit-learn or local CSV.
    
    Args:
        data_dir: Directory where data is stored.
        save_csv: Whether to save the loaded data as a CSV file.
        
    Returns:
        X (DataFrame): Features matrix.
        y (Series): Binary target labels (0 = Malignant, 1 = Benign).
        metadata (Dict): Metadata including target names, feature names, and description.
    """
    csv_path = os.path.join(data_dir, "breast_cancer.csv")
    
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame
    
    if save_csv:
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
            
    X = raw.data
    y = raw.target
    
    metadata = {
        "feature_names": list(raw.feature_names),
        "target_names": list(raw.target_names),  # ['malignant', 'benign']
        "n_samples": len(df),
        "n_features": X.shape[1],
        "class_counts": y.value_counts().to_dict(),
        "csv_path": csv_path
    }
    
    return X, y, metadata


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets with stratification.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random state for reproducible results.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
