# Machine Learning Classification Project: Evaluation & Results Explanation

**Internship Project Track:** Machine Learning Engineering  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Task:** Binary Classification (Malignant vs. Benign)  
**Total Samples:** 569 (212 Malignant, 357 Benign)  
**Total Features:** 30 numerical continuous cell nucleus characteristics  
**Train / Test Split:** 80% Training (455 samples), 20% Testing (113 samples) with stratification.

---

## 1. Executive Summary & Results Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9825 (98.25%)** | **0.9861 (98.61%)** | **0.9861 (98.61%)** | **0.9861** | **0.9954** |
| **Decision Tree** | 0.9298 (92.98%) | 0.9571 (95.71%) | 0.9306 (93.06%) | 0.9437 | 0.9268 |

---

## 2. In-Depth Explanation of Evaluation Metrics

### A. Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Definition:** The proportion of total predictions that the model got correct.
- **Results:** Logistic Regression achieved **98.25%**, while the Decision Tree achieved **92.98%**.
- **Insight:** While high accuracy is reassuring, accuracy alone can be deceptive in medical diagnosis or imbalanced classes. A model could simply predict the majority class and achieve deceptively high accuracy. Therefore, evaluating precision and recall is essential.

### B. Precision: $\frac{TP}{TP + FP}$
- **Definition:** Out of all cases the model predicted as positive, how many were truly positive?
- **Results:** Logistic Regression achieved **98.61%** precision, and Decision Tree achieved **95.71%**.
- **Clinical Implication:** High precision indicates very few False Positives (FP = 1 for Logistic Regression). In oncology, minimizing false alarms prevents healthy patients from undergoing distressing, invasive, and costly follow-up procedures (e.g., surgical biopsy).

### C. Recall (Sensitivity / True Positive Rate): $\frac{TP}{TP + FN}$
- **Definition:** Out of all actual positive cases in the data, how many did the model correctly catch?
- **Results:** Logistic Regression achieved **98.61%** recall, whereas Decision Tree achieved **93.06%**.
- **Clinical Implication:** In healthcare, **Recall is the single most critical metric**. A False Negative (FN = 1 for Logistic Regression) means a patient with a malignant tumor is told they are healthy, delaying life-saving treatment. The Logistic Regression model demonstrates stellar sensitivity, minimizing missed cases.

### D. F1-Score & ROC-AUC
- **F1-Score (0.9861):** Harmonic mean of Precision and Recall. It proves that the model maintains an exceptional balance without trading off sensitivity for specificity.
- **ROC-AUC (0.9954):** Measures the classifier's ability to discriminate between benign and malignant classes across all decision thresholds. A score near 1.0 indicates near-flawless separability.

---

## 3. Confusion Matrix Breakdown

### Logistic Regression Confusion Matrix:
- **True Negatives (TN):** 41 (Correctly identified Malignant)
- **False Positives (FP):** 1 (Malignant misclassified as Benign)
- **False Negatives (FN):** 1 (Benign misclassified as Malignant)
- **True Positives (TP):** 71 (Correctly identified Benign)

### Decision Tree Confusion Matrix:
- **True Negatives (TN):** 39
- **False Positives (FP):** 3
- **False Negatives (FN):** 5
- **True Positives (TP):** 67

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
For production diagnostic screening, **Logistic Regression** is the recommended model due to its higher overall accuracy (98.25%), superior recall (98.61%), and robust generalization on unseen test data.
