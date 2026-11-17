"""
Unit tests for data preprocessing module using standard unittest framework.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_baseline import BaselineModelBenchmark


class TestDataPreprocessing(unittest.TestCase):
    """Test suite for data loading, cleaning, and transformation."""

    def setUp(self):
        """Set up synthetic sample DataFrame mirroring the Telco schema."""
        self.sample_df = pd.DataFrame({
            "customerID": ["0001-A", "0002-B", "0003-C", "0004-D", "0005-E"],
            "gender": ["Female", "Male", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 0, 1, 0, 0],
            "Partner": ["Yes", "No", "No", "Yes", "No"],
            "Dependents": ["No", "No", "No", "Yes", "No"],
            "tenure": [0, 12, 24, 48, 72],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No internet service", "Yes", "No", "Yes", "No internet service"],
            "OnlineBackup": ["Yes", "No", "Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "Yes", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)", "Mailed check"],
            "MonthlyCharges": [29.85, 89.10, 95.50, 45.20, 20.00],
            "TotalCharges": [" ", "1069.20", "2292.00", "2169.60", "1440.00"],
            "Churn": ["No", "Yes", "No", "No", "No"],
        })
        self.preprocessor = DataPreprocessor(target_col="Churn", id_col="customerID")

    def test_missing_values_imputation(self):
        """Test whitespace coercion and 0.0 domain imputation in TotalCharges."""
        clean_df, summary = self.preprocessor.fix_data_types_and_missing(self.sample_df, fill_strategy="zero")
        
        self.assertEqual(summary["initial_coerced_nan"], 1)
        self.assertEqual(summary["remaining_nulls"], 0)
        self.assertEqual(clean_df.loc[0, "TotalCharges"], 0.0)
        self.assertEqual(clean_df["TotalCharges"].dtype, float)

    def test_duplicate_removal(self):
        """Test duplicate detection and removal."""
        df_with_dups = pd.concat([self.sample_df, self.sample_df.iloc[[0]]], ignore_index=True)
        self.assertEqual(len(df_with_dups), 6)
        
        clean_df, dups_removed = self.preprocessor.remove_duplicates(df_with_dups)
        self.assertEqual(dups_removed, 1)
        self.assertEqual(len(clean_df), 5)

    def test_outlier_detection_iqr(self):
        """Test IQR outlier calculation."""
        clean_df, _ = self.preprocessor.fix_data_types_and_missing(self.sample_df)
        iqr_report = self.preprocessor.detect_outliers_iqr(clean_df, columns=["tenure", "MonthlyCharges"])
        
        self.assertIn("tenure", iqr_report)
        self.assertIn("MonthlyCharges", iqr_report)
        self.assertGreater(iqr_report["tenure"]["iqr"], 0)
        self.assertLess(iqr_report["tenure"]["lower_bound"], iqr_report["tenure"]["upper_bound"])

    def test_feature_engineering(self):
        """Test TenureCohort and ChargeRatio creation."""
        clean_df, _ = self.preprocessor.fix_data_types_and_missing(self.sample_df)
        clean_df = self.preprocessor.clean_categories_and_engineer_features(clean_df)
        
        self.assertIn("TenureCohort", clean_df.columns)
        self.assertIn("ChargeRatio", clean_df.columns)
        self.assertEqual(str(clean_df.loc[0, "TenureCohort"]), "0-12 M")
        self.assertEqual(clean_df.loc[0, "ChargeRatio"], 1.0)

    def test_encoding_and_scaling(self):
        """Test dummy encoding and standard scaling."""
        clean_df, _ = self.preprocessor.fix_data_types_and_missing(self.sample_df)
        clean_df = self.preprocessor.clean_categories_and_engineer_features(clean_df)
        encoded_df = self.preprocessor.encode_and_scale(clean_df, numeric_cols=["tenure", "MonthlyCharges", "TotalCharges"])
        
        self.assertNotIn("customerID", encoded_df.columns)
        self.assertIn("Churn", encoded_df.columns)
        self.assertTrue(set(encoded_df["Churn"].unique()).issubset({0, 1}))
        self.assertIn("tenure_scaled", encoded_df.columns)
        self.assertIn("MonthlyCharges_scaled", encoded_df.columns)
        self.assertIn("TotalCharges_scaled", encoded_df.columns)

    def test_baseline_benchmark(self):
        """Test baseline benchmark model training and evaluation."""
        bigger_df = pd.concat([self.sample_df] * 5, ignore_index=True)
        clean_df, _ = self.preprocessor.fix_data_types_and_missing(bigger_df)
        clean_df = self.preprocessor.clean_categories_and_engineer_features(clean_df)
        encoded_df = self.preprocessor.encode_and_scale(clean_df, numeric_cols=["tenure", "MonthlyCharges", "TotalCharges"])
        
        # Save temp test csv
        tmp_csv = Path(__file__).resolve().parent / "tmp_test_encoded.csv"
        encoded_df.to_csv(tmp_csv, index=False)
        try:
            benchmark = BaselineModelBenchmark(data_path=tmp_csv, output_dir=tmp_csv.parent)
            X_train, X_test, y_train, y_test = benchmark.load_and_split_data(target_col="Churn", test_size=0.4)
            summary = benchmark.train_and_evaluate(X_train, X_test, y_train, y_test)
            self.assertIn("Model", summary.columns)
            self.assertIn("ROC-AUC", summary.columns)
            self.assertEqual(len(summary), 3)
        finally:
            if tmp_csv.exists():
                tmp_csv.unlink()


if __name__ == "__main__":
    unittest.main()
