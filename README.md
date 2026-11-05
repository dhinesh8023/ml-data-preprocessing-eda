# 📊 Data Preprocessing & Exploratory Data Analysis (EDA)
### *Machine Learning Internship Project*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-darkblue?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-orange?logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

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
8. **Automated Unit Testing**: Comprehensive test suite covering cleaning, imputation, outlier boundaries, and scaling.

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
│   └── eda_and_preprocessing.ipynb             # Interactive Jupyter Notebook with rich visuals
├── reports/
│   ├── EDA_and_Preprocessing_Report.md         # Comprehensive internship technical report
│   ├── numerical_summary.csv                   # Numeric descriptive statistics table
│   ├── categorical_summary.csv                 # Categorical frequency & cardinality table
│   └── figures/                                # High-resolution visualization plots
│       ├── 01_numerical_distributions.png
│       ├── 02_outlier_boxplots.png
│       ├── 03_target_distribution.png
│       ├── 04_churn_by_key_drivers.png
│       ├── 05_correlation_heatmap.png
│       └── 06_tenure_vs_charges_scatter.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py                          # Data ingestion & schema inspection
│   ├── preprocessing.py                        # Cleaning, imputation, outliers, encoding, scaling
│   └── eda.py                                  # Summary statistics & visualization generator
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py                   # Automated unit tests
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

## 🚀 Getting Started & Execution

### Prerequisites
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
Execute the complete ingestion, cleaning, feature engineering, and visualization pipeline:
```bash
python main.py
```

### Running the Unit Test Suite
Execute automated tests verifying missing value imputation, outlier detection, and scaling logic:
```bash
python -m unittest discover -s tests
```

### Exploring the Interactive Jupyter Notebook
Launch the notebook to inspect interactive visualizations and markdown documentation:
```bash
jupyter notebook notebooks/eda_and_preprocessing.ipynb
```

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
