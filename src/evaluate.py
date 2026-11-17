"""
Model evaluation module.
Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_single_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str = "Model",
    pos_label: int = 1
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for a single classification model.
    
    Args:
        model: Trained classifier or pipeline.
        X_test: Testing features.
        y_test: True target values.
        model_name: Display name for the model.
        pos_label: Target label considered positive (1 = Benign, 0 = Malignant).
        
    Returns:
        Dictionary of computed metrics, predictions, probabilities, and reports.
    """
    y_pred = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        auc = None
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, zero_division=0)
    
    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict
    }


def compare_models(
    models: Dict[str, Any], 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluate multiple models and compile a comparison table.
    
    Returns:
        summary_df: DataFrame with model comparison metrics.
        results_dict: Detailed metrics dictionary per model.
    """
    results = {}
    rows = []
    
    for name, model in models.items():
        metrics = evaluate_single_model(model, X_test, y_test, model_name=name)
        results[name] = metrics
        rows.append({
            "Model": name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1-Score": f"{metrics['f1_score']:.4f}",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
        })
        
    summary_df = pd.DataFrame(rows)
    return summary_df, results
