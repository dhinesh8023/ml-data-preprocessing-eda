# 📊 Machine Learning Internship Project Report: Data Preprocessing & Exploratory Data Analysis (EDA)

**Author:** Dhinesh Jayavel  
**Email:** dhineshjayavel35@gmail.com  
**GitHub:** [@dhinesh8023](https://github.com/dhinesh8023)  
**Project Domain:** Customer Retention Analytics & Machine Learning Pipeline Engineering  
**Dataset:** IBM Telco Customer Churn ([Public Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))  
**Submission Date:** November 2026  

---

## 📑 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Dataset Overview & Ingestion](#2-dataset-overview--ingestion)
3. [Data Cleaning Methodology](#3-data-cleaning-methodology)
   - [3.1 Missing Value Diagnosis & Domain Imputation](#31-missing-value-diagnosis--domain-imputation)
   - [3.2 Duplicate Record Auditing](#32-duplicate-record-auditing)
   - [3.3 Outlier Detection & Treatment](#33-outlier-detection--treatment)
4. [Feature Engineering & Transformation](#4-feature-engineering--transformation)
5. [Exploratory Data Analysis (EDA) & Key Findings](#5-exploratory-data-analysis-eda--key-findings)
   - [5.1 Summary Statistics](#51-summary-statistics)
   - [5.2 Target Variable & Class Imbalance](#52-target-variable--class-imbalance)
   - [5.3 Univariate Distributions](#53-univariate-distributions)
   - [5.4 Bivariate Churn Drivers](#54-bivariate-churn-drivers)
   - [5.5 Multivariate Interactions & Correlation](#55-multivariate-interactions--correlation)
6. [Machine Learning Readiness: Encoding & Scaling](#6-machine-learning-readiness-encoding--scaling)
7. [Business Recommendations](#7-business-recommendations)
8. [Project Architecture & Execution Guide](#8-project-architecture--execution-guide)

---

## 1. Executive Summary
Data preprocessing and exploratory data analysis (EDA) represent the most crucial phase of the machine learning lifecycle. High-capacity machine learning models are fundamentally constrained by the quality and representation of the input training data ("garbage in, garbage out").

This project implements an end-to-end, reproducible Data Preprocessing and EDA system on the **IBM Telco Customer Churn dataset** (7,043 customer observations, 21 attributes). 

### Key Highlights:
- **Discovered and resolved implicit missing values**: Uncovered 11 whitespace strings in `TotalCharges` disguised as valid string objects; resolved them using mathematically sound, domain-grounded zero imputation ($0.0) based on new subscriber tenure (`tenure == 0`).
- **Statistical Outlier Auditing**: Conducted dual-method outlier detection via **Tukey's IQR method** ($1.5 \times \text{IQR}$) and **Z-score testing** ($|Z| > 3.0$), verifying that numerical extremes represented authentic customer tenure and pricing structures.
- **Engineered Domain Features**: Created tenure lifetime cohorts (`TenureCohort`) and billing velocity ratios (`ChargeRatio`) that surfaced churn patterns.
- **Discovered Critical Churn Drivers**:
  1. **Contract Type**: Month-to-month contracts exhibit a **42.7% churn rate**, compared to just **2.8%** for two-year contracts (a $15\times$ risk differential).
  2. **Tenure Vulnerability**: Customers in their first 12 months show an alarming **47.4% churn rate**.
  3. **High-Value Fiber Optic Churn**: Fiber optic customers churn at **41.9%**, despite paying the highest monthly fees.
  4. **Payment Friction**: Customers paying via **Electronic Check churn at 45.3%**, compared to 15.2% for automated bank transfer.
- **Production Pipeline**: Developed modular Python source code (`src/`), a Jupyter Notebook (`notebooks/`), and automated unit tests (`tests/`).

---

## 2. Dataset Overview & Ingestion
The dataset captures customer demographics, service subscriptions, account configurations, and subscription status:

| Metric | Value |
| :--- | :--- |
| **Total Customer Records (Rows)** | 7,043 |
| **Total Features (Columns)** | 21 (1 ID, 19 Features, 1 Target) |
| **Data Types** | 18 Object / Categorical, 2 Integer, 1 Float |
| **Raw Memory Footprint** | ~6.98 MB |
| **Target Variable** | `Churn` (Binary: `"Yes"` vs `"No"`) |

### Feature Categorization:
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Subscribed Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account & Contract Details**: `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target Label**: `Churn` (Customer attrition status within the last month)

---

## 3. Data Cleaning Methodology

### 3.1 Missing Value Diagnosis & Domain Imputation
A standard pandas `df.isnull().sum()` check initially reports `0` null values across all columns. However, schema inspection indicated that `TotalCharges` was loaded with an `object` (string) dtype rather than numerical float.

```python
# Uncovering whitespace strings in TotalCharges
spaces_mask = df['TotalCharges'].astype(str).str.strip() == ''
print(f"Whitespace strings in TotalCharges: {spaces_mask.sum()}")
# Result: 11 records
```

#### Root Cause Analysis:
All 11 records containing whitespace in `TotalCharges` correspond exactly to customers with `tenure == 0`. These are brand-new customers who enrolled in the current billing cycle and have not yet completed their first 30 days of service.

#### Imputation Strategy & Justification:
Rather than dropping these observations or applying mean/median imputation (which would distort cumulative billing statistics), `TotalCharges` was logically imputed as **`0.00`**, exactly reflecting zero accrued charges.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
```

### 3.2 Duplicate Record Auditing
Data duplicates can cause data leakage and artificially inflate validation metrics. We audited for both identical rows and duplicate primary keys (`customerID`):
- **Full row duplicates:** `0`
- **Duplicate customer IDs:** `0`

The dataset contains 7,043 unique customer entities with zero duplicate records.

### 3.3 Outlier Detection & Treatment
Continuous numerical variables (`tenure`, `MonthlyCharges`, `TotalCharges`) were audited using two statistical bounds:

#### 1. Tukey's Interquartile Range (IQR) Method
- $\text{IQR} = Q_3 - Q_1$
- $\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$
- $\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$

#### 2. Standardized Z-Score Method
- $Z = \frac{X - \mu}{\sigma}, \quad \text{Flag if } |Z| > 3.0$

#### Outlier Audit Results:
| Feature | Min | Q1 (25%) | Median | Q3 (75%) | Max | IQR | Lower Bound | Upper Bound | Outlier Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tenure`** | 0.00 | 9.00 | 29.00 | 55.00 | 72.00 | 46.00 | -60.00 | 124.00 | 0 (0.0%) |
| **`MonthlyCharges`** | 18.25 | 35.50 | 70.35 | 89.85 | 118.75 | 54.35 | -46.02 | 171.38 | 0 (0.0%) |
| **`TotalCharges`** | 0.00 | 398.55 | 1394.55 | 3786.60 | 8684.80 | 3388.05 | -4683.52 | 8868.67 | 0 (0.0%) |

#### Decision on Outlier Remediation:
No data points exceed Tukey's fences or $|Z| > 3.0$. The values span authentic domain boundaries (tenure from 0 to 72 months, monthly charges between $18.25 and $118.75, and cumulative charges up to $8,684.80). No artificial clipping or deletion is required, preventing loss of natural variance.

---

## 4. Feature Engineering & Transformation

### 4.1 Categorical Standardization
Several categorical columns contained redundant negative classifications:
- `"No internet service"` across `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- `"No phone service"` in `MultipleLines`.

These were standardized into uniform `"No"` responses to eliminate redundant degrees of freedom during one-hot encoding.

### 4.2 Domain-Specific Engineered Features
1. **`TenureCohort` (Customer Lifecycle Stage)**: Segmented continuous tenure into 5 business lifecycle buckets:
   - `0-12 M` (Early Stage - High Churn Window)
   - `13-24 M` (Established Stage)
   - `25-48 M` (Mature Stage)
   - `49-60 M` (Loyal Stage)
   - `>60 M` (Brand Champions)
2. **`ChargeRatio`**: Ratio of actual lifetime billed amount to expected theoretical charges ($\frac{\text{TotalCharges}}{\text{tenure} \times \text{MonthlyCharges}}$). Helps flag customer plan adjustments, promotional discounts, or historical fee changes.

---

## 5. Exploratory Data Analysis (EDA) & Key Findings

### 5.1 Summary Statistics

#### Continuous Features Summary:
| Feature | Count | Mean | Std Dev | Median | IQR | Min | Max | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tenure`** | 7,043 | 32.37 | 24.56 | 29.00 | 46.00 | 0.00 | 72.00 | +0.24 | -1.39 |
| **`MonthlyCharges`** | 7,043 | 64.76 | 30.09 | 70.35 | 54.35 | 18.25 | 118.75 | -0.22 | -1.26 |
| **`TotalCharges`** | 7,043 | 2279.73 | 2266.79 | 1394.55 | 3388.05 | 0.00 | 8684.80 | +0.96 | -0.23 |

- **`tenure`** exhibits a bimodal distribution: heavy concentration at the very beginning (0-6 months) and at the contractual ceiling (66-72 months).
- **`MonthlyCharges`** has a significant cluster around $20 (basic landline phone service) and another broad distribution between $70-$105 (fiber optic broadband + TV streaming bundles).
- **`TotalCharges`** is moderately right-skewed (+0.96) as expected for cumulative lifetime value.

#### Categorical Features Overview:
| Feature | Distinct Classes | Dominant Class (Mode) | Mode Frequency | Mode Percentage |
| :--- | :---: | :--- | :---: | :---: |
| **`gender`** | 2 | Male | 3,555 | 50.48% |
| **`SeniorCitizen`** | 2 | 0 (Non-senior) | 5,901 | 83.79% |
| **`Partner`** | 2 | No | 3,641 | 51.70% |
| **`Dependents`** | 2 | No | 4,933 | 70.04% |
| **`Contract`** | 3 | Month-to-month | 3,875 | 55.02% |
| **`InternetService`** | 3 | Fiber optic | 3,096 | 43.96% |
| **`PaymentMethod`** | 4 | Electronic check | 2,365 | 33.58% |
| **`Churn`** | 2 | No | 5,174 | 73.46% |

---

### 5.2 Target Variable & Class Imbalance
- **Active Customers (`Churn = No`):** 5,174 customers (73.5%)
- **Churned Customers (`Churn = Yes`):** 1,869 customers (26.5%)

```
Churn Class Balance:
[============================= No (73.5%) =============================][==== Yes (26.5%) ====]
```

**Takeaway:** A ~3:1 class imbalance exists. Downstream classification algorithms will require stratified splitting (`StratifiedKFold`), class weighting (`class_weight='balanced'`), or evaluation using ROC-AUC and PR-AUC instead of raw accuracy.

---

### 5.3 Univariate Distributions
*Refer to visualization:* `reports/figures/01_numerical_distributions.png`

```
  Tenure Distribution                Monthly Charges                  Total Charges
   |\             /|                  |   |\     /|                   |\
   | \___________/ |                  |___| \___/ |                   | \_________
   0M    36M     72M                 $20     $70 $105                 $0   $4k  $8.6k
 (New vs Long-term)                 (Basic vs Fiber)              (Right-skewed LTV)
```

---

### 5.4 Bivariate Churn Drivers
*Refer to visualization:* `reports/figures/04_churn_by_key_drivers.png`

#### 1. Contract Structure
- **Month-to-month:** **42.7% Churn Rate**
- **One-year:** **11.3% Churn Rate**
- **Two-year:** **2.8% Churn Rate**
- *Insight:* Month-to-month customers represent 55% of all subscribers but drive over 88% of all churn events. Long-term contractual commitments effectively neutralize customer attrition.

#### 2. Internet Service Type
- **Fiber Optic:** **41.9% Churn Rate**
- **DSL:** **19.0% Churn Rate**
- **No Internet:** **7.4% Churn Rate**
- *Insight:* Fiber optic is the highest revenue product ($70-$100/mo) but suffers from more than double the churn rate of DSL.

#### 3. Technical Support & Online Security
- **No Tech Support:** **41.6% Churn Rate**
- **With Tech Support:** **15.2% Churn Rate**
- **No Online Security:** **41.8% Churn Rate**
- **With Online Security:** **14.6% Churn Rate**
- *Insight:* Value-added security and support services act as major retention anchors.

#### 4. Payment Method
- **Electronic Check:** **45.3% Churn Rate**
- **Mailed Check:** **19.1% Churn Rate**
- **Bank Transfer (Automatic):** **16.7% Churn Rate**
- **Credit Card (Automatic):** **15.2% Churn Rate**
- *Insight:* Electronic check users churn at nearly triple the rate of customers utilizing automated payment options.

---

### 5.5 Multivariate Interactions & Correlation
*Refer to visualization:* `reports/figures/05_correlation_heatmap.png` & `06_tenure_vs_charges_scatter.png`

- **Tenure vs. Churn Correlation ($r = -0.35$):** The longer a customer stays, the probability of churn drops precipitously.
- **Tenure vs. TotalCharges ($r = +0.83$):** Strong expected linear alignment between tenure duration and cumulative revenue.
- **Monthly Charges vs. Churn ($r = +0.19$):** Higher monthly bills moderately correlate with higher churn probability, particularly when paired with Month-to-month contracts.
- **Scatter Plot Insight:** Churned customers (red markers) cluster heavily in the upper-left quadrant: **low tenure (< 12 months) + high monthly charges ($70 - $110)**.

---

## 6. Machine Learning Readiness: Encoding & Scaling
To prepare the dataset for predictive machine learning algorithms (Logistic Regression, Random Forest, Gradient Boosting):
1. **Target Encoding**: Mapped `Churn` to `{No: 0, Yes: 1}`.
2. **Binary Categorical Mapping**: Converted binary features (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`) to `{0, 1}`.
3. **One-Hot Encoding**: Expanded nominal multi-class features (`Contract`, `InternetService`, `PaymentMethod`, `TenureCohort`) via `pd.get_dummies(..., drop_first=True)` to prevent multicollinearity.
4. **Feature Scaling**: Fitted `StandardScaler` to continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`), normalizing them to $\mu = 0, \sigma = 1$.
5. **Final Output**: Encoded dataset exported to `data/processed/telco_customer_churn_encoded.csv` (7,043 rows $\times$ 31 columns).

### 6.1 Supervised Machine Learning Benchmark Results
We validated model readiness on an 80/20 stratified split across three baseline models:

| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **80.34%** | **67.38%** | 50.27% | 57.58% | **0.8466** | **0.6568** |
| **Logistic Regression** | 79.84% | 65.20% | 51.60% | 57.61% | 0.8429 | 0.6355 |
| **Gradient Boosting** | 79.70% | 64.57% | **52.14%** | **57.69%** | 0.8422 | 0.6489 |

*Refer to visualizations:* `reports/figures/07_model_benchmark_roc.png` & `08_feature_importances.png`

- **Key Takeaways**:
  - All three models achieve an ROC-AUC of $\sim 0.84-0.85$, demonstrating strong predictive capability.
  - The top predictive features align with our EDA findings: **Contract commitment**, **Tenure duration**, **Monthly Charges**, **Internet Service Type**, and **Tech Support availability**.

---

## 7. Business Recommendations

Based on empirical data findings, the following high-ROI operational interventions are recommended:

```
+---------------------------------------------------------------------------------------+
|                              STRATEGIC RETENTION ROADMAP                              |
+------------------------------------+--------------------------------------------------+
| Immediate Operational Fixes        | Long-Term Structural Interventions               |
+------------------------------------+--------------------------------------------------+
| 1. Auto-Pay Migration Incentives   | 1. Month-to-Month Contract Conversion Campaign   |
|    Offer a $3-$5 bill credit for   |    Offer 1-year contract discounts or free speed  |
|    switching from Electronic Check |    boosts to move customers off month-to-month.  |
|    to automated bank transfer.     |                                                  |
|                                    | 2. Fiber Optic Service Quality & Pricing Audit   |
| 2. First-90-Day Onboarding Program |    Investigate uptime and customer complaints    |
|    Deploy proactive customer care  |    in fiber optic clusters to address the        |
|    during the high-risk 0-12 month |    41.9% churn rate.                             |
|    window to ensure satisfaction.  |                                                  |
|                                    | 3. Bundle Tech Support & Security Packages       |
|                                    |    Incorporate security features into baseline   |
|                                    |    tiers to lower churn from 41.6% to 15.2%.     |
+------------------------------------+--------------------------------------------------+
```

---

## 8. Project Architecture & Execution Guide

### Directory Tree
```
ml-data-preprocessing-eda/
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv            # Original raw dataset
│   └── processed/
│       ├── telco_customer_churn_cleaned.csv    # Cleaned tabular dataset
│       └── telco_customer_churn_encoded.csv    # Preprocessed, scaled ML dataset
├── notebooks/
│   └── eda_and_preprocessing.ipynb             # Interactive Jupyter Notebook
├── reports/
│   ├── EDA_and_Preprocessing_Report.md         # Comprehensive internship report
│   ├── numerical_summary.csv                   # Numeric descriptive statistics
│   ├── categorical_summary.csv                 # Categorical frequency tables
│   └── figures/                                # Generated high-res visual plots
│       ├── 01_numerical_distributions.png
│       ├── 02_outlier_boxplots.png
│       ├── 03_target_distribution.png
│       ├── 04_churn_by_key_drivers.png
│       ├── 05_correlation_heatmap.png
│       └── 06_tenure_vs_charges_scatter.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py                          # Data ingestion & schema validation
│   ├── preprocessing.py                        # Cleaning, imputations, encoding, scaling
│   └── eda.py                                  # Statistical summaries & visualizations
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py                   # Automated unit tests
├── main.py                                     # End-to-end pipeline execution script
├── requirements.txt                            # Pinned Python package dependencies
├── .gitignore                                  # Git exclusion rules
└── README.md                                   # GitHub repository showcase documentation
```

### Reproducibility Commands
To run the automated data preprocessing and exploratory data analysis pipeline:
```bash
# Clone the repository
git clone https://github.com/dhinesh8023/ml-data-preprocessing-eda.git
cd ml-data-preprocessing-eda

# Install dependencies
pip install -r requirements.txt

# Execute the complete automated pipeline
python main.py

# Run unit test suite
pytest tests/
```
