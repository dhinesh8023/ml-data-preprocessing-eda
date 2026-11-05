"""
Data Preprocessing Module
-------------------------
Handles missing values, duplicate detection, outlier detection and remediation,
feature transformations, categorical encoding, and feature scaling.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """End-to-end preprocessing pipeline for tabular datasets."""

    def __init__(self, target_col: str = "Churn", id_col: str = "customerID"):
        self.target_col = target_col
        self.id_col = id_col
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.encoded_df: Optional[pd.DataFrame] = None

    def fix_data_types_and_missing(
        self, df: pd.DataFrame, fill_strategy: str = "zero"
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Converts 'TotalCharges' from whitespace/object to float and handles missing values.
        
        Args:
            df: Input raw DataFrame.
            fill_strategy: 'zero' (since tenure==0 implies zero charges billed),
                           'median', or 'drop'.
        """
        df_clean = df.copy()
        
        # TotalCharges has empty spaces ' ' for new customers with tenure = 0
        initial_missing = df_clean.isnull().sum().to_dict()
        
        # Convert TotalCharges to numeric, coercing spaces to NaN
        if "TotalCharges" in df_clean.columns:
            df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
        
        nan_count = int(df_clean["TotalCharges"].isnull().sum())
        print(f"[Preprocessing] Detected {nan_count} missing values in 'TotalCharges'.")
        
        if fill_strategy == "zero":
            # Logical domain justification: customers with tenure == 0 have not accumulated charges yet
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)
            print("[Preprocessing] Imputed missing 'TotalCharges' with 0.0 (based on tenure=0 business logic).")
        elif fill_strategy == "median":
            median_val = df_clean["TotalCharges"].median()
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(median_val)
            print(f"[Preprocessing] Imputed missing 'TotalCharges' with median value: {median_val:.2f}")
        elif fill_strategy == "drop":
            df_clean = df_clean.dropna(subset=["TotalCharges"]).reset_index(drop=True)
            print(f"[Preprocessing] Dropped {nan_count} rows with missing 'TotalCharges'.")
        
        # Check any other numeric or categorical missing values
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        cat_cols = df_clean.select_dtypes(include=["object", "string", "str"]).columns
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        missing_summary = {
            "initial_coerced_nan": nan_count,
            "remaining_nulls": int(df_clean.isnull().sum().sum()),
        }
        return df_clean, missing_summary

    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Identifies and removes duplicate records based on ID and full row match."""
        df_clean = df.copy()
        
        # Row duplicates
        row_dups = int(df_clean.duplicated().sum())
        # ID duplicates
        id_dups = int(df_clean.duplicated(subset=[self.id_col]).sum()) if self.id_col in df_clean.columns else 0
        
        total_dups = max(row_dups, id_dups)
        if total_dups > 0:
            df_clean = df_clean.drop_duplicates(subset=[self.id_col] if self.id_col in df_clean.columns else None)
            df_clean = df_clean.reset_index(drop=True)
            print(f"[Preprocessing] Removed {total_dups} duplicate records.")
        else:
            print("[Preprocessing] No duplicate records found.")
            
        return df_clean, total_dups

    def detect_outliers_iqr(
        self, df: pd.DataFrame, columns: List[str], factor: float = 1.5
    ) -> Dict[str, Dict[str, Union[float, int, List[int]]]]:
        """
        Detects outliers using the Interquartile Range (IQR) method.
        Boundaries: [Q1 - factor * IQR, Q3 + factor * IQR]
        """
        outlier_report = {}
        for col in columns:
            if col not in df.columns:
                continue
            
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = round((outlier_count / len(df)) * 100, 2)
            
            self.outlier_bounds_[col] = (lower_bound, upper_bound)
            
            outlier_report[col] = {
                "q1": round(float(q1), 2),
                "q3": round(float(q3), 2),
                "iqr": round(float(iqr), 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_pct,
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
            }
        return outlier_report

    def detect_outliers_zscore(
        self, df: pd.DataFrame, columns: List[str], threshold: float = 3.0
    ) -> Dict[str, Dict[str, Union[float, int]]]:
        """Detects outliers using Z-score method (|Z| > threshold)."""
        z_report = {}
        for col in columns:
            if col not in df.columns:
                continue
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > threshold]
            z_report[col] = {
                "mean": round(float(mean), 2),
                "std": round(float(std), 2),
                "outlier_count": len(outliers),
                "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
            }
        return z_report

    def handle_outliers(
        self, df: pd.DataFrame, columns: List[str], method: str = "clip"
    ) -> pd.DataFrame:
        """
        Remediates outliers by clipping to IQR boundaries (Winsorization) or filtering.
        In Telco dataset, values represent legitimate tenure and billing amounts,
        so clipping/preserving is preferred over raw deletion.
        """
        df_clean = df.copy()
        if method == "clip":
            for col in columns:
                if col in self.outlier_bounds_:
                    low, high = self.outlier_bounds_[col]
                    df_clean[col] = df_clean[col].clip(lower=max(0.0, low), upper=high)
            print("[Preprocessing] Outliers handled via Winsorization/Clipping to IQR bounds.")
        elif method == "remove":
            for col in columns:
                if col in self.outlier_bounds_:
                    low, high = self.outlier_bounds_[col]
                    df_clean = df_clean[(df_clean[col] >= low) & (df_clean[col] <= high)]
            df_clean = df_clean.reset_index(drop=True)
            print("[Preprocessing] Outlier records removed from dataset.")
        return df_clean

    def clean_categories_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans redundant categorical values and performs domain feature engineering.
        """
        df_clean = df.copy()
        
        # Standardize redundant categorical values
        # e.g., 'No internet service' -> 'No', 'No phone service' -> 'No'
        replace_cols = [
            "MultipleLines", "OnlineSecurity", "OnlineBackup", 
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        for col in replace_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace({"No internet service": "No", "No phone service": "No"})
                
        # Feature Engineering: Tenure cohorts
        if "tenure" in df_clean.columns:
            bins = [-1, 12, 24, 48, 60, 100]
            labels = ["0-12 M", "13-24 M", "25-48 M", "49-60 M", ">60 M"]
            df_clean["TenureCohort"] = pd.cut(df_clean["tenure"], bins=bins, labels=labels)
            
        # Feature Engineering: Average Monthly Charges ratio
        if "TotalCharges" in df_clean.columns and "tenure" in df_clean.columns and "MonthlyCharges" in df_clean.columns:
            # For tenure > 0, estimate ratio of calculated avg monthly vs current monthly charges
            df_clean["ChargeRatio"] = np.where(
                df_clean["tenure"] > 0,
                df_clean["TotalCharges"] / (df_clean["tenure"] * df_clean["MonthlyCharges"] + 1e-5),
                1.0
            )

        return df_clean

    def encode_and_scale(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"],
        scale_method: str = "standard",
    ) -> pd.DataFrame:
        """
        Encodes binary and nominal categorical variables and scales numerical features.
        """
        df_encoded = df.copy()
        
        # Drop ID column if present
        if self.id_col in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=[self.id_col])
            
        # Binary target mapping
        if self.target_col in df_encoded.columns:
            df_encoded[self.target_col] = df_encoded[self.target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})

        # Binary categorical mapping
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map({"Yes": 1, "No": 0})
                
        if "gender" in df_encoded.columns:
            df_encoded["gender"] = df_encoded["gender"].map({"Female": 1, "Male": 0})

        # One-Hot Encoding for multi-class categorical features
        cat_cols = df_encoded.select_dtypes(include=["object", "string", "str", "category"]).columns.tolist()
        if cat_cols:
            df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

        # Feature Scaling
        valid_num_cols = [c for c in numeric_cols if c in df_encoded.columns]
        if valid_num_cols:
            if scale_method == "standard":
                self.scaler_ = StandardScaler()
            elif scale_method == "minmax":
                self.scaler_ = MinMaxScaler()
            
            scaled_vals = self.scaler_.fit_transform(df_encoded[valid_num_cols])
            for idx, col in enumerate(valid_num_cols):
                df_encoded[f"{col}_scaled"] = scaled_vals[:, idx]
                
        self.encoded_df = df_encoded
        return df_encoded
