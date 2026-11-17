"""
Baseline Machine Learning Modeling Module
-----------------------------------------
Evaluates predictive performance on the preprocessed dataset across multiple classifiers,
computes metrics (ROC-AUC, F1, Precision, Recall), plots ROC/PR curves,
and extracts top predictive feature importances.
"""

from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


class BaselineModelBenchmark:
    """Benchmark suite for validating preprocessed data with machine learning models."""

    def __init__(self, data_path: Path, output_dir: Path):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        }
        self.results: Dict[str, Dict[str, float]] = {}
        self.trained_models = {}

    def load_and_split_data(self, target_col: str = "Churn", test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads preprocessed data and returns stratified train-test splits."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Encoded dataset not found at: {self.data_path.resolve()}")

        df = pd.read_csv(self.data_path)
        
        # Exclude unscaled original numerical columns if scaled versions exist to avoid collinearity
        drop_cols = [target_col]
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if f"{col}_scaled" in df.columns:
                drop_cols.append(col)

        # Keep features
        X = df.drop(columns=drop_cols)
        y = df[target_col].astype(int)

        can_stratify = (y.value_counts() >= 2).all()
        stratify_param = y if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        print(f"[ModelBenchmark] Split dataset into {len(X_train)} train and {len(X_test)} test samples (Stratified {test_size*100:.0f}% test).")
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
        """Trains models, evaluates metrics, and returns summary performance table."""
        records = []
        for name, clf in self.models.items():
            clf.fit(X_train, y_train)
            self.trained_models[name] = clf

            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall_pts, precision_pts)

            self.results[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
            }

            records.append({
                "Model": name,
                "Accuracy (%)": round(acc * 100, 2),
                "Precision (%)": round(prec * 100, 2),
                "Recall (%)": round(rec * 100, 2),
                "F1-Score (%)": round(f1 * 100, 2),
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
            })
            print(f"  * {name:22s} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | Accuracy: {acc*100:.2f}%")

        summary_df = pd.DataFrame(records)
        return summary_df

    def plot_roc_and_pr_curves(self, X_test: pd.DataFrame, y_test: pd.Series) -> Path:
        """Plots side-by-side ROC Curves and Precision-Recall Curves for all benchmarked models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "Gradient Boosting": "#ff7f0e"}

        # Subplot 1: ROC Curve
        for name, clf in self.trained_models.items():
            y_proba = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            score = self.results[name]["ROC-AUC"]
            axes[0].plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})", color=colors.get(name, "gray"), linewidth=2)

        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Baseline (AUC = 0.500)")
        axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
        axes[0].legend(frameon=True, facecolor="white")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: Precision-Recall Curve
        baseline_pr = y_test.mean()
        for name, clf in self.trained_models.items():
            y_proba = clf.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            score = self.results[name]["PR-AUC"]
            axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {score:.3f})", color=colors.get(name, "gray"), linewidth=2)

        axes[1].axhline(baseline_pr, color="k", linestyle="--", alpha=0.6, label=f"No Skill (Prevalence = {baseline_pr:.2f})")
        axes[1].set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Recall", fontsize=11)
        axes[1].set_ylabel("Precision", fontsize=11)
        axes[1].legend(frameon=True, facecolor="white")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / "07_model_benchmark_roc.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[ModelBenchmark] Saved ROC/PR Curves to: {save_path}")
        return save_path

    def plot_feature_importances(self, X_train: pd.DataFrame, top_n: int = 15) -> Path:
        """Extracts and plots top feature importances from Gradient Boosting Classifier."""
        gbm = self.trained_models.get("Gradient Boosting")
        if gbm is None:
            raise ValueError("Gradient Boosting model has not been trained yet.")

        importances = pd.Series(gbm.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6.5))
        sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="viridis", legend=False, ax=ax, edgecolor="black")
        
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.003, p.get_y() + p.get_height() / 2, f"{width:.3f}", va="center", fontsize=9, fontweight="bold")

        ax.set_title(f"Top {top_n} Predictive Features for Customer Churn (Gradient Boosting)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Gini / Mean Decrease in Impurity Importance", fontsize=11)
        ax.set_ylabel("Feature Name", fontsize=11)
        ax.set_xlim(0, max(importances.values) * 1.15)

        plt.tight_layout()
        save_path = self.output_dir / "08_feature_importances.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[ModelBenchmark] Saved Feature Importances plot to: {save_path}")
        return save_path
