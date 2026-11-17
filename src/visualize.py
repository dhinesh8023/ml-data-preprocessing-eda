"""
Visualization module.
Generates publication-quality charts for confusion matrices, ROC curves,
metric comparisons, feature importances, and decision tree structures.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.tree import plot_tree


# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]], 
    class_names: List[str],
    output_path: str = "reports/confusion_matrices.png"
) -> str:
    """
    Plot side-by-side confusion matrix heatmaps for all evaluated models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_models = len(results)
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            cbar=False,
            xticklabels=class_names, 
            yticklabels=class_names,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f} | Prec: {res['precision']:.3f} | Rec: {res['recall']:.3f}", fontsize=12, pad=10)
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]], 
    y_test: pd.Series, 
    output_path: str = "reports/roc_curves.png"
) -> str:
    """
    Plot Receiver Operating Characteristic (ROC) curves comparing models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    
    for name, res in results.items():
        if res.get("y_proba") is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            auc = res.get("roc_auc", 0.0)
            plt.plot(fpr, tpr, lw=2.5, label=f"{name} (AUC = {auc:.3f})")
            
    # Diagonal random baseline
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    plt.title("ROC Curves Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_metrics_comparison(
    results: Dict[str, Dict[str, Any]], 
    output_path: str = "reports/metrics_comparison.png"
) -> str:
    """
    Plot grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    data = []
    for model_name, res in results.items():
        for m_key, m_disp in zip(metrics_names, display_names):
            data.append({
                "Model": model_name,
                "Metric": m_disp,
                "Score": res[m_key]
            })
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="Metric", y="Score", hue="Model", palette="crest")
    
    plt.ylim(0.80, 1.02)
    plt.title("Model Performance Metrics Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Score", fontweight="bold")
    plt.xlabel("Evaluation Metric", fontweight="bold")
    
    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", 
                va="bottom",
                fontsize=9, 
                fontweight="bold",
                xytext=(0, 2), 
                textcoords="offset points"
            )
            
    plt.legend(title="Algorithm", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_tree_structure(
    tree_model: Any, 
    feature_names: List[str], 
    class_names: List[str],
    output_path: str = "reports/decision_tree_structure.png"
) -> str:
    """
    Plot the graphical decision tree structure showing split nodes and leaves.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(18, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3
    )
    plt.title("Decision Tree Visualization (Truncated to Depth 3)", fontsize=16, fontweight="bold", pad=12)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_feature_importances(
    tree_model: Any, 
    feature_names: List[str],
    top_n: int = 10,
    output_path: str = "reports/feature_importance.png"
) -> str:
    """
    Plot top N most important features identified by the Decision Tree.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importances = tree_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]
    
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top_scores, y=top_features, hue=top_features, palette="viridis", legend=False)
    plt.title(f"Top {top_n} Feature Importances (Decision Tree)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Relative Gini Importance", fontweight="bold")
    plt.ylabel("Feature Name", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path
