"""
Generates a standalone, self-contained HTML report with embedded CSS, summary tables,
and base64-encoded figures.
"""

import base64
from pathlib import Path
import pandas as pd


def generate_html_report(project_root: Path = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    
    # Load summary CSVs
    num_df = pd.read_csv(reports_dir / "numerical_summary.csv")
    cat_df = pd.read_csv(reports_dir / "categorical_summary.csv")
    model_df = pd.read_csv(reports_dir / "model_benchmark_summary.csv")

    # Encode figures as base64
    figures_b64 = {}
    figure_files = sorted(figures_dir.glob("*.png"))
    for fig_path in figure_files:
        with open(fig_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
            figures_b64[fig_path.name] = b64_str

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Learning Internship Project Report | Dhinesh Jayavel</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --secondary: #0f766e;
            --accent: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2.5rem 1.5rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #0369a1 100%);
            color: white;
            padding: 3rem 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            margin-bottom: 2.5rem;
        }}
        header h1 {{
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
            letter-spacing: -0.025em;
        }}
        header p {{
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 1.25rem;
        }}
        .badge-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1rem;
        }}
        .badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.18);
            backdrop-filter: blur(8px);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.25);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }}
        .kpi-card {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.03);
            transition: transform 0.2s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.06);
        }}
        .kpi-title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            color: var(--text-muted);
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .kpi-val {{
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--primary);
        }}
        .kpi-sub {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        .section-card {{
            background: var(--card-bg);
            border-radius: 14px;
            border: 1px solid var(--border);
            padding: 2.25rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        }}
        h2 {{
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--primary);
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.75rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        h3 {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #1e293b;
            margin: 1.5rem 0 0.75rem 0;
        }}
        p, li {{
            color: #334155;
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }}
        ul {{
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }}
        .table-responsive {{
            overflow-x: auto;
            margin: 1.25rem 0;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            color: #1e293b;
            font-weight: 700;
            padding: 0.85rem 1rem;
            border-bottom: 2px solid var(--border);
        }}
        td {{
            padding: 0.8rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        tr:hover {{
            background-color: #f1f5f9;
        }}
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }}
        .figure-box {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        }}
        .figure-box img {{
            width: 100%;
            height: auto;
            display: block;
            border-bottom: 1px solid var(--border);
        }}
        .figure-caption {{
            padding: 1rem 1.25rem;
            font-size: 0.95rem;
            font-weight: 600;
            color: #334155;
            background: #fafafa;
        }}
        .recommendation-box {{
            background: #ecfdf5;
            border-left: 4px solid #10b981;
            padding: 1.25rem 1.5rem;
            border-radius: 0 10px 10px 0;
            margin: 1rem 0;
        }}
        .recommendation-box h4 {{
            color: #065f46;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        footer {{
            text-align: center;
            padding: 2.5rem 0 1rem 0;
            color: var(--text-muted);
            font-size: 0.9rem;
            border-top: 1px solid var(--border);
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .section-card {{
                box-shadow: none;
                border: 1px solid #ccc;
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Machine Learning Internship Project Report</h1>
        <p>Comprehensive Data Preprocessing, Exploratory Data Analysis (EDA), and Machine Learning Benchmarking</p>
        <div class="badge-bar">
            <span class="badge">Author: Dhinesh Jayavel</span>
            <span class="badge">Dataset: IBM Telco Customer Churn</span>
            <span class="badge">Kaggle / UCI Repository</span>
            <span class="badge">November 2026</span>
            <span class="badge">Status: Production Ready</span>
        </div>
    </header>

    <!-- Key Performance Indicators -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-val">7,043</div>
            <div class="kpi-sub">Customers Analyzed</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Overall Churn Rate</div>
            <div class="kpi-val" style="color: #dc2626;">26.5%</div>
            <div class="kpi-sub">1,869 Churned Customers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Missing Values Fixed</div>
            <div class="kpi-val" style="color: #0f766e;">11 (100%)</div>
            <div class="kpi-sub">Imputed $0.0 for tenure=0</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Top ML ROC-AUC</div>
            <div class="kpi-val" style="color: #2563eb;">0.847</div>
            <div class="kpi-sub">Random Forest Benchmark</div>
        </div>
    </div>

    <!-- Section 1: Executive Summary -->
    <div class="section-card">
        <h2>📌 1. Executive Summary</h2>
        <p>
            Data Preprocessing and Exploratory Data Analysis (EDA) are foundational stages in production machine learning.
            Raw tabular data often presents implicit missing values, formatting anomalies, duplicates, distribution outliers, and class imbalance.
        </p>
        <p>
            In this internship project, we constructed an end-to-end, automated machine learning pipeline utilizing the <strong>IBM Telco Customer Churn dataset</strong>.
            Through domain-grounded data wrangling, Tukey's outlier fences, comprehensive bivariate segmentations, and predictive benchmarking, we demonstrated how rigorous data preprocessing translates raw tabular inputs into actionable business intelligence and high-performance ML models.
        </p>
    </div>

    <!-- Section 2: Data Cleaning & Preprocessing -->
    <div class="section-card">
        <h2>🧹 2. Data Cleaning & Feature Transformation</h2>
        <h3>2.1 Uncovering Implicit Missing Values</h3>
        <p>
            Standard null-checks (<code>df.isnull().sum()</code>) initially returned zero missing values. However, detailed schema auditing revealed that <code>TotalCharges</code> was stored with an <code>object</code> (string) dtype.
            11 records contained whitespace strings (<code>' '</code>).
        </p>
        <p>
            <strong>Root Cause Analysis:</strong> All 11 customers had <code>tenure == 0</code>. These customers joined the company during the current billing cycle and have not yet completed their first monthly billing cycle.
            <br>
            <strong>Imputation Decision:</strong> Imputing with <strong>$0.00</strong> preserves mathematical integrity, avoiding distortion from arbitrary mean/median values.
        </p>

        <h3>2.2 Outlier Auditing via Tukey's IQR & Z-Score Tests</h3>
        <p>
            Numerical features (<code>tenure</code>, <code>MonthlyCharges</code>, <code>TotalCharges</code>) were tested using <strong>Tukey's IQR method</strong> (1.5 &times; IQR) and <strong>Z-score bounds</strong> (|Z| &gt; 3.0).
            Zero artificial anomalies were detected; all continuous observations fall within authentic customer billing lifespans ($18.25 - $118.75/month, 0 - 72 months).
        </p>

        <h3>2.3 Summary Statistics Table</h3>
        <div class="table-responsive">
            {num_df.to_html(classes="table", index=False)}
        </div>

        <div class="table-responsive">
            {cat_df.to_html(classes="table", index=False)}
        </div>
    </div>

    <!-- Section 3: Visual Analytics -->
    <div class="section-card">
        <h2>📈 3. Visual Exploratory Data Analysis</h2>
        <div class="figure-grid">
            <div class="figure-box">
                <img src="data:image/png;base64,{figures_b64.get('01_numerical_distributions.png', '')}" alt="Numerical Distributions">
                <div class="figure-caption">Figure 1: Univariate Distributions & Kernel Density Estimation (KDE)</div>
            </div>
            <div class="figure-box">
                <img src="data:image/png;base64,{figures_b64.get('02_outlier_boxplots.png', '')}" alt="Outlier Boxplots">
                <div class="figure-caption">Figure 2: Distribution Boxplots segmented by Customer Churn Status</div>
            </div>
            <div class="figure-box">
                <img src="data:image/png;base64,{figures_b64.get('03_target_distribution.png', '')}" alt="Target Distribution">
                <div class="figure-caption">Figure 3: Target Class Balance (73.5% Retained vs 26.5% Churned)</div>
            </div>
            <div class="figure-box">
                <img src="data:image/png;base64,{figures_b64.get('04_churn_by_key_drivers.png', '')}" alt="Key Churn Drivers">
                <div class="figure-caption">Figure 4: Bivariate Churn Rates by Contract, Internet, Support, & Payment Method</div>
            </div>
            <div class="figure-box">
                <img src="data:image/png;base64,{figures_b64.get('05_correlation_heatmap.png', '')}" alt="Correlation Heatmap">
                <div class="figure-caption">Figure 5: Feature Correlation Matrix Heatmap</div>
            </div>
            <div class="figure-box">
                <img src="data:image/png;base64,{figures_b64.get('06_tenure_vs_charges_scatter.png', '')}" alt="Tenure vs Charges">
                <div class="figure-caption">Figure 6: Interaction Scatter Plot (Tenure vs Monthly Charges)</div>
            </div>
        </div>
    </div>

    <!-- Section 4: Machine Learning Validation -->
    <div class="section-card">
        <h2>🤖 4. Machine Learning Benchmarking & Feature Importances</h2>
        <p>
            To evaluate the modeling readiness of our preprocessed dataset, we trained three distinct supervised classifiers:
            <strong>Logistic Regression (L2)</strong>, <strong>Random Forest</strong>, and <strong>Gradient Boosting</strong> on an 80/20 stratified split.
        </p>

        <div class="table-responsive">
            {model_df.to_html(classes="table", index=False)}
        </div>

        <div class="figure-grid">
            <div class="figure-box">
                <img src="data:image/png;base64,{figures_b64.get('07_model_benchmark_roc.png', '')}" alt="ROC and PR Curves">
                <div class="figure-caption">Figure 7: Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves</div>
            </div>
            <div class="figure-box">
                <img src="data:image/png;base64,{figures_b64.get('08_feature_importances.png', '')}" alt="Feature Importances">
                <div class="figure-caption">Figure 8: Top 15 Predictive Churn Features from Gradient Boosting</div>
            </div>
        </div>
    </div>

    <!-- Section 5: Strategic Business Roadmap -->
    <div class="section-card">
        <h2>💡 5. Strategic Retention Roadmap</h2>
        <div class="recommendation-box">
            <h4>1. Contract Migration Campaign (15x Churn Reduction)</h4>
            <p>Month-to-month subscribers experience an alarming 42.7% churn rate compared to 2.8% for two-year commitments. Offer targeted incentives (first month free or free speed boost) to convert short-term users.</p>
        </div>
        <div class="recommendation-box">
            <h4>2. First-90-Day VIP Onboarding</h4>
            <p>The 0-12 month window accounts for 47.4% customer attrition. Establish automated milestone touchpoints, check-ins, and onboarding support within the first 90 days.</p>
        </div>
        <div class="recommendation-box">
            <h4>3. Fiber Optic Experience & Quality Audit</h4>
            <p>Fiber optic subscribers churn at 41.9% despite being the highest-paying segment ($70-$100+/month). Audit technical tickets and localized network stability to address subscriber friction.</p>
        </div>
        <div class="recommendation-box">
            <h4>4. Automated Payment Enrollment (3x Churn Difference)</h4>
            <p>Electronic Check users churn at 45.3% vs. ~15% for automated Bank Transfer and Credit Card. Provide a recurring $3-$5 monthly credit for enrolling in Auto-Pay.</p>
        </div>
    </div>

    <footer>
        <p>© 2026 Dhinesh Jayavel · Machine Learning Internship Project · <a href="https://github.com/dhinesh8023/ml-data-preprocessing-eda" target="_blank" style="color: var(--primary); text-decoration: none; font-weight: 600;">GitHub Repository</a></p>
    </footer>
</div>
</body>
</html>
"""

    report_path = reports_dir / "internship_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[HTML Report] Self-contained report successfully generated at: {report_path.resolve()}")


if __name__ == "__main__":
    generate_html_report()
