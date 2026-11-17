# 📊 Data Preprocessing & Exploratory Data Analysis (EDA)
### *Machine Learning Internship Project*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-darkblue?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling%20%26%20Preprocessing-orange?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-black?logo=flask&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhinesh8023/ml-data-preprocessing-eda/blob/main/notebooks/eda_and_preprocessing.ipynb)

---

## 📌 Executive Summary
In real-world machine learning engineering, data is messy, incomplete, and rarely model-ready. This project delivers an end-to-end, production-grade **Data Preprocessing & Exploratory Data Analysis (EDA)** pipeline built on the **IBM Telco Customer Churn dataset** (from Kaggle / UCI Machine Learning Repository).

The repository showcases systematic workflows for:
1. **Schema Validation & Ingestion**: Type verification and deep memory profiling.
2. **Missing Value Auditing**: Uncovering implicit whitespace NaNs in `TotalCharges` and applying domain-grounded zero imputation ($0.0) based on newly acquired customer tenure (`tenure == 0`).
3. **Duplicate Detection**: Full-row and primary key uniqueness verification.
4. **Statistical Outlier Detection**: Dual-approach diagnostics utilizing **Tukey's Interquartile Range (IQR)** ($1.5 \times \text{IQR}$) and **Z-score testing** ($|Z| > 3.0$).
5. **Feature Engineering**: Creating customer lifecycle cohorts (`TenureCohort`) and billing velocity ratios (`ChargeRatio`).
6. **Exploratory Data Analysis (EDA)**: Descriptive statistics, univariate distributions, bivariate churn driver analysis, correlation matrices, and multivariate interactions.
7. **ML Preprocessing**: Binary encoding, One-Hot Encoding (drop-first), and `StandardScaler` feature normalization.
8. **Predictive ML Validation**: Benchmarking Logistic Regression, Random Forest, and Gradient Boosting (achieving **0.847 ROC-AUC**).
9. **Interactive Web Dashboard**: Full Flask application featuring live data browsing and a real-time customer churn risk scoring calculator.
10. **Automated Unit Testing**: 9 comprehensive unit tests verifying data integrity, preprocessing transformations, modeling, and API endpoints.

---

## 📂 Project Architecture

```
ml-data-preprocessing-eda/
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv            # Original raw dataset
│   └── processed/
│       ├── telco_customer_churn_cleaned.csv    # Cleaned tabular dataset
│       └── telco_customer_churn_encoded.csv    # Preprocessed & scaled ML-ready matrix
├── notebooks/
│   └── eda_and_preprocessing.ipynb             # Interactive Jupyter Notebook with embedded outputs
├── reports/
│   ├── EDA_and_Preprocessing_Report.md         # Comprehensive internship technical report
│   ├── internship_report.html                  # Self-contained, printable interactive HTML report
│   ├── numerical_summary.csv                   # Numeric descriptive statistics table
│   ├── categorical_summary.csv                 # Categorical frequency & cardinality table
│   ├── model_benchmark_summary.csv             # Classifier performance benchmarks
│   └── figures/                                # High-resolution visualization plots
│       ├── 01_numerical_distributions.png
│       ├── 02_outlier_boxplots.png
│       ├── 03_target_distribution.png
│       ├── 04_churn_by_key_drivers.png
│       ├── 05_correlation_heatmap.png
│       ├── 06_tenure_vs_charges_scatter.png
│       ├── 07_model_benchmark_roc.png
│       └── 08_feature_importances.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py                          # Data ingestion & schema inspection
│   ├── preprocessing.py                        # Cleaning, imputation, outliers, encoding, scaling
│   ├── eda.py                                  # Summary statistics & visualization generator
│   ├── model_baseline.py                       # Supervised ML benchmark & ROC/PR curves
│   └── report_generator.py                     # Self-contained HTML report compiler
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py                   # Automated tests for data pipeline & models
│   └── test_app.py                             # Automated tests for Flask dashboard & API
├── app.py                                      # Interactive Flask Web Dashboard & Live Predictor
├── main.py                                     # End-to-end pipeline execution runner
├── requirements.txt                            # Project package dependencies
├── .gitignore                                  # Git exclusion rules
└── README.md                                   # Project documentation
```

---

## 🔬 Key Data Preprocessing Findings & Methodology

### 1. Diagnosing Hidden Missing Values
- **Issue**: Standard `df.isnull().sum()` reported `0` null values because `TotalCharges` was loaded as an `object` string.
- **Root Cause**: 11 records contained blank whitespace characters (`' '`). All 11 records corresponded to new subscribers where `tenure == 0`.
- **Solution**: Imputed `TotalCharges = 0.00` because customers who joined during the current cycle have not yet accrued their first monthly invoice.

### 2. Statistical Outlier Detection
We evaluated numerical features using Tukey's fences ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$) and Standard Z-Scores ($|Z| > 3.0$):

| Feature | Min | Q1 (25%) | Median | Q3 (75%) | Max | IQR | Lower Bound | Upper Bound | Outliers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tenure`** | 0.00 | 9.00 | 29.00 | 55.00 | 72.00 | 46.00 | -60.00 | 124.00 | 0 (0.0%) |
| **`MonthlyCharges`** | 18.25 | 35.50 | 70.35 | 89.85 | 118.75 | 54.35 | -46.02 | 171.38 | 0 (0.0%) |
| **`TotalCharges`** | 0.00 | 398.55 | 1394.55 | 3786.60 | 8684.80 | 3388.05 | -4683.52 | 8868.67 | 0 (0.0%) |

**Conclusion:** All values fall within natural, valid business bounds. Retaining true natural variance preserves predictive signal for machine learning algorithms.

---

## 📈 Exploratory Data Analysis & Visual Insights

### 1. Target Imbalance Analysis
- **Non-Churn (`No`):** 5,174 customers (73.5%)
- **Churn (`Yes`):** 1,869 customers (26.5%)
- *Modeling Implication:* Class imbalance requires stratified cross-validation and evaluation via PR-AUC and ROC-AUC rather than accuracy.

### 2. Primary Churn Drivers
1. **Contract Structure**:
   - Month-to-Month contracts have an alarming **42.7% churn rate**.
   - One-Year contracts: **11.3% churn rate**.
   - Two-Year contracts: **2.8% churn rate** ($15\times$ lower risk).
2. **Early Tenure Vulnerability**:
   - Customers in their first 12 months experience a **47.4% churn rate**.
   - Churn drops dramatically below 15% once customers cross 24 months.
3. **Fiber Optic Service Issues**:
   - Fiber optic users churn at **41.9%**, compared to 19.0% for DSL.
4. **Payment Method Friction**:
   - Customers paying by **Electronic Check churn at 45.3%**, compared to 15.2% for automated bank transfer or credit card.
5. **Protective Role of Technical Support**:
   - Customers without `TechSupport` or `OnlineSecurity` churn at **>41%**, whereas bundled support cuts churn down to **<15%**.

---

## 🤖 Machine Learning Validation & Benchmark Results

To confirm that the preprocessed feature representations generalize well to predictive modeling, we trained three distinct supervised classifiers on an 80/20 stratified split:

| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **80.34%** | **67.38%** | 50.27% | 57.58% | **0.8466** | **0.6568** |
| **Logistic Regression** | 79.84% | 65.20% | 51.60% | 57.61% | 0.8429 | 0.6355 |
| **Gradient Boosting** | 79.70% | 64.57% | **52.14%** | **57.69%** | 0.8422 | 0.6489 |

- **Top Predictive Features:** Contract Type (`Contract_Two year`, `Contract_One year`), `tenure_scaled`, `MonthlyCharges_scaled`, `PaymentMethod_Electronic check`, and `TechSupport_Yes`.

---

## 🚀 Getting Started & Execution

### ☁️ Run in Google Colab (One-Click Cloud Execution)
Run the entire Jupyter Notebook in the cloud with zero local setup or installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhinesh8023/ml-data-preprocessing-eda/blob/main/notebooks/eda_and_preprocessing.ipynb)

👉 **Direct Link:** [https://colab.research.google.com/github/dhinesh8023/ml-data-preprocessing-eda/blob/main/notebooks/eda_and_preprocessing.ipynb](https://colab.research.google.com/github/dhinesh8023/ml-data-preprocessing-eda/blob/main/notebooks/eda_and_preprocessing.ipynb)

---

### Local Prerequisites
- Python 3.10 or higher
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/dhinesh8023/ml-data-preprocessing-eda.git
cd ml-data-preprocessing-eda

# 2. Install dependencies
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
Execute data ingestion, cleaning, outlier testing, feature engineering, summary tables, plots, and ML benchmarking:
```bash
python main.py
```

### Launching the Interactive Web Dashboard
Run the local Flask web application with interactive charts and a live customer churn calculator:
```bash
python app.py
# Open your browser at http://127.0.0.1:5000
```

### Running the Automated Unit Test Suite
Execute automated unit tests covering all cleaning steps, outlier boundaries, encoding, models, and dashboard routes:
```bash
python -m unittest discover -s tests
```

### Viewing the Standalone HTML Report
Open `reports/internship_report.html` directly in any web browser (Chrome, Edge, Firefox, Safari) to view a formatted, printable report with embedded charts and KPI cards.

---

## 💡 Strategic Business Recommendations

```
+-----------------------------------------------------------------------------------------+
|                                BUSINESS ACTION PLAN                                     |
+-------------------------------------+---------------------------------------------------+
| Retention Focus                     | Strategic Intervention                            |
+-------------------------------------+---------------------------------------------------+
| 1. Month-to-Month Attrition         | Incentive campaigns (discounts / free speed boost)|
|                                     | to convert month-to-month users into 1-year plans.|
| 2. First-90-Day Drop-Off            | Proactive onboarding check-ins and dedicated      |
|                                     | technical support in months 1-3.                  |
| 3. Fiber Optic Experience           | Audit fiber network stability and resolve quality |
|                                     | bottlenecks for high-value subscribers.           |
| 4. Payment Modernization            | Provide a $3-$5 monthly credit to incentivize     |
|                                     | transitioning from Electronic Check to Auto-Pay.  |
| 5. Security & Support Bundling      | Bundle TechSupport & OnlineSecurity into baseline |
|                                     | tiers to leverage their 2.7x retention multiplier.|
+-------------------------------------+---------------------------------------------------+
```

---

## 👤 Author
- **Dhinesh Jayavel**  
- **Email:** [dhineshjayavel35@gmail.com](mailto:dhineshjayavel35@gmail.com)  
- **GitHub:** [@dhinesh8023](https://github.com/dhinesh8023)  
