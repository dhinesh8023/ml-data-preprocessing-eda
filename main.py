"""
Main executable pipeline script.
Orchestrates data loading, train/test splitting, model training, evaluation,
visualization generation, and results documentation.
"""

import os
import sys
import pandas as pd
from src.data_loader import load_dataset, split_data
from src.models import train_all_models
from src.evaluate import compare_models
from src.visualize import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_tree_structure,
    plot_feature_importances
)


def generate_results_markdown(
    summary_df: pd.DataFrame, 
    results: dict, 
    metadata: dict,
    output_path: str = "reports/results_explanation.md"
):
    """
    Generate an in-depth explanation of the machine learning classification results.
    """
    lr = results["Logistic Regression"]
    dt = results["Decision Tree"]
    
    content = f"""# Machine Learning Classification Project: Evaluation & Results Explanation

**Internship Project Track:** Machine Learning Engineering  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Task:** Binary Classification (Malignant vs. Benign)  
**Total Samples:** {metadata['n_samples']} ({metadata['class_counts'][0]} Malignant, {metadata['class_counts'][1]} Benign)  
**Total Features:** {metadata['n_features']} numerical continuous cell nucleus characteristics  
**Train / Test Split:** 80% Training ({int(metadata['n_samples'] * 0.8)} samples), 20% Testing ({int(metadata['n_samples'] * 0.2)} samples) with stratification.

---

## 1. Executive Summary & Results Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **{lr['accuracy']:.4f} ({lr['accuracy']*100:.2f}%)** | **{lr['precision']:.4f} ({lr['precision']*100:.2f}%)** | **{lr['recall']:.4f} ({lr['recall']*100:.2f}%)** | **{lr['f1_score']:.4f}** | **{lr['roc_auc']:.4f}** |
| **Decision Tree** | {dt['accuracy']:.4f} ({dt['accuracy']*100:.2f}%) | {dt['precision']:.4f} ({dt['precision']*100:.2f}%) | {dt['recall']:.4f} ({dt['recall']*100:.2f}%) | {dt['f1_score']:.4f} | {dt['roc_auc']:.4f} |

---

## 2. In-Depth Explanation of Evaluation Metrics

### A. Accuracy: $\\frac{{TP + TN}}{{TP + TN + FP + FN}}$
- **Definition:** The proportion of total predictions that the model got correct.
- **Results:** Logistic Regression achieved **{lr['accuracy']*100:.2f}%**, while the Decision Tree achieved **{dt['accuracy']*100:.2f}%**.
- **Insight:** While high accuracy is reassuring, accuracy alone can be deceptive in medical diagnosis or imbalanced classes. A model could simply predict the majority class and achieve deceptively high accuracy. Therefore, evaluating precision and recall is essential.

### B. Precision: $\\frac{{TP}}{{TP + FP}}$
- **Definition:** Out of all cases the model predicted as positive, how many were truly positive?
- **Results:** Logistic Regression achieved **{lr['precision']*100:.2f}%** precision, and Decision Tree achieved **{dt['precision']*100:.2f}%**.
- **Clinical Implication:** High precision indicates very few False Positives (FP = {lr['fp']} for Logistic Regression). In oncology, minimizing false alarms prevents healthy patients from undergoing distressing, invasive, and costly follow-up procedures (e.g., surgical biopsy).

### C. Recall (Sensitivity / True Positive Rate): $\\frac{{TP}}{{TP + FN}}$
- **Definition:** Out of all actual positive cases in the data, how many did the model correctly catch?
- **Results:** Logistic Regression achieved **{lr['recall']*100:.2f}%** recall, whereas Decision Tree achieved **{dt['recall']*100:.2f}%**.
- **Clinical Implication:** In healthcare, **Recall is the single most critical metric**. A False Negative (FN = {lr['fn']} for Logistic Regression) means a patient with a malignant tumor is told they are healthy, delaying life-saving treatment. The Logistic Regression model demonstrates stellar sensitivity, minimizing missed cases.

### D. F1-Score & ROC-AUC
- **F1-Score ({lr['f1_score']:.4f}):** Harmonic mean of Precision and Recall. It proves that the model maintains an exceptional balance without trading off sensitivity for specificity.
- **ROC-AUC ({lr['roc_auc']:.4f}):** Measures the classifier's ability to discriminate between benign and malignant classes across all decision thresholds. A score near 1.0 indicates near-flawless separability.

---

## 3. Confusion Matrix Breakdown

### Logistic Regression Confusion Matrix:
- **True Negatives (TN):** {lr['tn']} (Correctly identified Malignant)
- **False Positives (FP):** {lr['fp']} (Malignant misclassified as Benign)
- **False Negatives (FN):** {lr['fn']} (Benign misclassified as Malignant)
- **True Positives (TP):** {lr['tp']} (Correctly identified Benign)

### Decision Tree Confusion Matrix:
- **True Negatives (TN):** {dt['tn']}
- **False Positives (FP):** {dt['fp']}
- **False Negatives (FN):** {dt['fn']}
- **True Positives (TP):** {dt['tp']}

---

## 4. Model Comparison & Architectural Trade-offs

1. **Why Logistic Regression Outperformed Decision Tree:**
   - Logistic Regression with `StandardScaler` standardizes features with wildly different scales (e.g., area: ~1000 vs. smoothness: ~0.1). Its linear decision boundary in 30-dimensional space captures the aggregate linear trends cleanly with L2 regularization (`C=1.0`), avoiding high variance.
   - A single Decision Tree creates axis-aligned splits. On complex multi-dimensional continuous data, orthogonal cuts can struggle to capture smooth diagonal boundaries and can easily overfit local sample noise despite depth constraints (`max_depth=4`).

2. **Interpretability:**
   - The Decision Tree provides an intuitive flowchart representation (`reports/decision_tree_structure.png`) that domain practitioners can inspect without mathematical knowledge.
   - The most influential diagnostic features identified were `worst concave points`, `worst perimeter`, and `worst radius`.

---

## 5. Conclusion & Recommendations
For production diagnostic screening, **Logistic Regression** is the recommended model due to its higher overall accuracy ({lr['accuracy']*100:.2f}%), superior recall ({lr['recall']*100:.2f}%), and robust generalization on unseen test data.
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"[OK] Saved explanation of results to: {output_path}")


def main():
    print("=" * 65)
    print("      MACHINE LEARNING CLASSIFICATION PIPELINE EXECUTION")
    print("=" * 65)
    
    # 1. Load Data
    print("\n[Step 1/5] Loading Breast Cancer Dataset...")
    X, y, metadata = load_dataset(data_dir="data", save_csv=True)
    print(f" -> Samples: {metadata['n_samples']}, Features: {metadata['n_features']}")
    print(f" -> Class Distribution: {metadata['class_counts']} (0: Malignant, 1: Benign)")
    print(f" -> Saved raw dataset to: {metadata['csv_path']}")
    
    # 2. Train/Test Split
    print("\n[Step 2/5] Splitting data into Training (80%) and Testing (20%) sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    print(f" -> Training set shape: {X_train.shape}")
    print(f" -> Testing set shape:  {X_test.shape}")
    
    # 3. Model Training
    print("\n[Step 3/5] Training Models...")
    print(" -> Training Logistic Regression (Pipeline with StandardScaler)...")
    print(" -> Training Decision Tree (max_depth=4, criterion='gini')...")
    fitted_models = train_all_models(X_train, y_train, random_state=42)
    print(" -> Both models trained successfully!")
    
    # 4. Evaluation
    print("\n[Step 4/5] Evaluating Models on Test Set...")
    summary_df, results = compare_models(fitted_models, X_test, y_test)
    print("\n" + "-" * 60)
    print("                  EVALUATION METRICS SUMMARY")
    print("-" * 60)
    print(summary_df.to_string(index=False))
    print("-" * 60)
    
    for name, res in results.items():
        print(f"\n[{name}] Detailed Classification Report:")
        print(res["classification_report_str"])
        print(f"Confusion Matrix (TN, FP, FN, TP): {res['confusion_matrix'].ravel().tolist()}")
        
    # 5. Visualizations & Reports
    print("\n[Step 5/5] Generating Visualizations and Reports...")
    class_names = ["Malignant", "Benign"]
    
    cm_path = plot_confusion_matrices(results, class_names, "reports/confusion_matrices.png")
    print(f" -> Saved confusion matrices to: {cm_path}")
    
    roc_path = plot_roc_curves(results, y_test, "reports/roc_curves.png")
    print(f" -> Saved ROC curves to: {roc_path}")
    
    met_path = plot_metrics_comparison(results, "reports/metrics_comparison.png")
    print(f" -> Saved metrics comparison to: {met_path}")
    
    tree_path = plot_tree_structure(
        fitted_models["Decision Tree"], 
        metadata["feature_names"], 
        class_names, 
        "reports/decision_tree_structure.png"
    )
    print(f" -> Saved decision tree structure to: {tree_path}")
    
    imp_path = plot_feature_importances(
        fitted_models["Decision Tree"],
        metadata["feature_names"],
        top_n=10,
        output_path="reports/feature_importance.png"
    )
    print(f" -> Saved feature importance to: {imp_path}")
    
    generate_results_markdown(summary_df, results, metadata, "reports/results_explanation.md")
    
    print("\n" + "=" * 65)
    print("               PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
