"""
Unit tests for data loader, models, and evaluation metrics using standard library unittest.
"""

import unittest
import pandas as pd
import numpy as np
from src.data_loader import load_dataset, split_data
from src.models import build_logistic_regression, build_decision_tree, train_all_models
from src.evaluate import evaluate_single_model, compare_models


class TestMLPipeline(unittest.TestCase):

    def test_load_dataset(self):
        X, y, metadata = load_dataset(data_dir="data", save_csv=False)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape[0], 569)
        self.assertEqual(X.shape[1], 30)
        self.assertEqual(len(metadata["target_names"]), 2)
        self.assertEqual(set(y.unique()), {0, 1})

    def test_split_data(self):
        X, y, _ = load_dataset(save_csv=False)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(X_test), 114)
        self.assertEqual(len(X_train), 455)
        # Stratification check (ratio should be preserved)
        train_ratio = (y_train == 1).mean()
        test_ratio = (y_test == 1).mean()
        self.assertLess(abs(train_ratio - test_ratio), 0.02)

    def test_model_training_and_evaluation(self):
        X, y, _ = load_dataset(save_csv=False)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
        models = train_all_models(X_train, y_train, random_state=42)
        
        self.assertIn("Logistic Regression", models)
        self.assertIn("Decision Tree", models)
        
        summary_df, results = compare_models(models, X_test, y_test)
        self.assertEqual(summary_df.shape[0], 2)
        
        for name, res in results.items():
            self.assertIn("accuracy", res)
            self.assertIn("precision", res)
            self.assertIn("recall", res)
            self.assertIn("f1_score", res)
            self.assertTrue(0.0 <= res["accuracy"] <= 1.0)
            self.assertTrue(0.0 <= res["precision"] <= 1.0)
            self.assertTrue(0.0 <= res["recall"] <= 1.0)
            self.assertTrue(0.0 <= res["f1_score"] <= 1.0)
            self.assertEqual(res["confusion_matrix"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
