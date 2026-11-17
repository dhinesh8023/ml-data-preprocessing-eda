"""
Interactive Flask Web Application for Telco Customer Churn EDA & Preprocessing
Author: Dhinesh Jayavel
"""

import json
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
project_root = Path(__file__).resolve().parent

# Load datasets
raw_path = project_root / "data" / "raw" / "telco_customer_churn.csv"
clean_path = project_root / "data" / "processed" / "telco_customer_churn_cleaned.csv"
encoded_path = project_root / "data" / "processed" / "telco_customer_churn_encoded.csv"

# Pre-train a lightweight risk scoring model for live calculator
ml_df = pd.read_csv(encoded_path)
drop_cols = ["Churn", "tenure", "MonthlyCharges", "TotalCharges"]
feature_cols = [c for c in ml_df.columns if c not in drop_cols]
X_train = ml_df[feature_cols]
y_train = ml_df["Churn"].astype(int)

risk_model = LogisticRegression(max_iter=1000, random_state=42)
risk_model.fit(X_train, y_train)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telco Churn Analytics Dashboard | Dhinesh Jayavel</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .hero { background: linear-gradient(135deg, #1e3a8a, #0284c7); color: white; padding: 2.5rem 1.5rem; border-radius: 12px; margin-bottom: 2rem; }
        .card { border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.03); margin-bottom: 1.5rem; }
        .kpi { font-size: 2rem; font-weight: 800; }
        .nav-tabs .nav-link.active { font-weight: 700; border-bottom: 3px solid #0284c7; }
        .risk-gauge { height: 18px; border-radius: 9px; }
    </style>
</head>
<body>
<div class="container py-4">
    <div class="hero">
        <h1 class="display-6 fw-bold">Telco Customer Churn & EDA Dashboard</h1>
        <p class="lead mb-2">Interactive Machine Learning Analytics & Churn Prediction Portal</p>
        <span class="badge bg-light text-dark me-1">Dhinesh Jayavel</span>
        <span class="badge bg-light text-dark me-1">Dataset: 7,043 Customers</span>
        <span class="badge bg-light text-dark">ROC-AUC: 0.847</span>
    </div>

    <!-- KPI Cards -->
    <div class="row g-3 mb-4">
        <div class="col-md-3">
            <div class="card p-3">
                <div class="text-muted small fw-bold">TOTAL CUSTOMERS</div>
                <div class="kpi text-primary">7,043</div>
                <div class="small text-muted">Audited & Cleaned</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card p-3">
                <div class="text-muted small fw-bold">CHURN RATE</div>
                <div class="kpi text-danger">26.5%</div>
                <div class="small text-muted">1,869 Customers Lost</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card p-3">
                <div class="text-muted small fw-bold">AVG MONTHLY CHARGE</div>
                <div class="kpi text-success">$64.76</div>
                <div class="small text-muted">Median: $70.35</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card p-3">
                <div class="text-muted small fw-bold">MISSING CHARGES FIXED</div>
                <div class="kpi text-info">11 / 11</div>
                <div class="small text-muted">Domain Imputed $0.0</div>
            </div>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <ul class="nav nav-tabs mb-4" id="dashTabs" role="tablist">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#overview">📊 Visual Analytics</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#calculator">🎯 Real-time Churn Predictor</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#data">📋 Data Explorer</button></li>
    </ul>

    <div class="tab-content" id="dashTabsContent">
        <!-- Tab 1: Visual Analytics -->
        <div class="tab-pane fade show active" id="overview">
            <div class="row g-4">
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6 class="fw-bold">Target Distribution & Balance</h6>
                        <img src="/static/figures/03_target_distribution.png" class="img-fluid rounded border" alt="Target Distribution">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6 class="fw-bold">Key Churn Drivers</h6>
                        <img src="/static/figures/04_churn_by_key_drivers.png" class="img-fluid rounded border" alt="Churn Drivers">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6 class="fw-bold">Feature Correlation Matrix</h6>
                        <img src="/static/figures/05_correlation_heatmap.png" class="img-fluid rounded border" alt="Correlation Heatmap">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card p-3">
                        <h6 class="fw-bold">Top Predictive Features (Gradient Boosting)</h6>
                        <img src="/static/figures/08_feature_importances.png" class="img-fluid rounded border" alt="Feature Importances">
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 2: Live Churn Predictor -->
        <div class="tab-pane fade" id="calculator">
            <div class="card p-4">
                <h5 class="fw-bold mb-3">Live Customer Churn Risk Calculator</h5>
                <form id="riskForm">
                    <div class="row g-3">
                        <div class="col-md-4">
                            <label class="form-label fw-bold">Tenure (Months)</label>
                            <input type="number" class="form-control" id="tenure" value="6" min="0" max="72" required>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label fw-bold">Monthly Charges ($)</label>
                            <input type="number" step="0.01" class="form-control" id="monthly" value="85.50" min="15" max="150" required>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label fw-bold">Contract Type</label>
                            <select class="form-select" id="contract">
                                <option value="Month-to-month" selected>Month-to-month</option>
                                <option value="One year">One year</option>
                                <option value="Two year">Two year</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label fw-bold">Internet Service</label>
                            <select class="form-select" id="internet">
                                <option value="Fiber optic" selected>Fiber optic</option>
                                <option value="DSL">DSL</option>
                                <option value="No">No Internet</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label fw-bold">Tech Support</label>
                            <select class="form-select" id="tech_support">
                                <option value="No" selected>No</option>
                                <option value="Yes">Yes</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label fw-bold">Payment Method</label>
                            <select class="form-select" id="payment">
                                <option value="Electronic check" selected>Electronic check</option>
                                <option value="Mailed check">Mailed check</option>
                                <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                                <option value="Credit card (automatic)">Credit card (automatic)</option>
                            </select>
                        </div>
                        <div class="col-12 mt-4">
                            <button type="button" class="btn btn-primary px-4 fw-bold" onclick="calculateRisk()">Predict Churn Risk</button>
                        </div>
                    </div>
                </form>

                <div id="resultBox" class="mt-4 p-3 border rounded bg-light" style="display: none;">
                    <h6 class="fw-bold">Estimated Churn Probability: <span id="riskScore" class="fs-4"></span></h6>
                    <div class="progress risk-gauge my-2">
                        <div id="riskBar" class="progress-bar" role="progressbar" style="width: 0%;"></div>
                    </div>
                    <p id="riskRecommendation" class="mb-0 small text-muted"></p>
                </div>
            </div>
        </div>

        <!-- Tab 3: Data Explorer -->
        <div class="tab-pane fade" id="data">
            <div class="card p-3">
                <h6 class="fw-bold mb-3">Cleaned Dataset Sample (First 15 Records)</h6>
                <div class="table-responsive">
                    {{ table_html | safe }}
                </div>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
function calculateRisk() {
    const tenure = parseFloat(document.getElementById('tenure').value);
    const monthly = parseFloat(document.getElementById('monthly').value);
    const contract = document.getElementById('contract').value;
    const internet = document.getElementById('internet').value;
    const tech = document.getElementById('tech_support').value;
    const payment = document.getElementById('payment').value;

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tenure, monthly, contract, internet, tech, payment })
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('resultBox').style.display = 'block';
        const pct = (data.probability * 100).toFixed(1);
        document.getElementById('riskScore').innerText = pct + '% (' + data.risk_tier + ')';
        
        const bar = document.getElementById('riskBar');
        bar.style.width = pct + '%';
        if (data.probability > 0.5) {
            bar.className = 'progress-bar bg-danger';
            document.getElementById('riskRecommendation').innerText = 'High Retention Risk: Recommend immediate contract incentive discount or proactive customer success call.';
        } else if (data.probability > 0.25) {
            bar.className = 'progress-bar bg-warning text-dark';
            document.getElementById('riskRecommendation').innerText = 'Moderate Risk: Offer value-add add-on (free OnlineSecurity or TechSupport package).';
        } else {
            bar.className = 'progress-bar bg-success';
            document.getElementById('riskRecommendation').innerText = 'Low Risk: Customer exhibits loyal tenure and service indicators.';
        }
    });
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    df_clean = pd.read_csv(clean_path).head(15)
    cols_show = ["customerID", "gender", "tenure", "Contract", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]
    table_html = df_clean[cols_show].to_html(classes="table table-sm table-striped table-hover", index=False)
    return render_template_string(HTML_TEMPLATE, table_html=table_html)


@app.route("/static/figures/<path:filename>")
def serve_figure(filename):
    from flask import send_from_directory
    figures_dir = project_root / "reports" / "figures"
    return send_from_directory(figures_dir, filename)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    tenure = float(data.get("tenure", 12))
    monthly = float(data.get("monthly", 65.0))
    contract = data.get("contract", "Month-to-month")
    internet = data.get("internet", "Fiber optic")
    tech = data.get("tech", "No")
    payment = data.get("payment", "Electronic check")

    # Build input vector matching X_train
    row = pd.DataFrame(0.0, index=[0], columns=feature_cols)

    # Scaled estimates based on training means & stds
    tenure_scaled = (tenure - ml_df["tenure"].mean()) / (ml_df["tenure"].std() + 1e-5)
    monthly_scaled = (monthly - ml_df["MonthlyCharges"].mean()) / (ml_df["MonthlyCharges"].std() + 1e-5)
    total_est = tenure * monthly
    total_scaled = (total_est - ml_df["TotalCharges"].mean()) / (ml_df["TotalCharges"].std() + 1e-5)

    if "tenure_scaled" in row.columns:
        row["tenure_scaled"] = tenure_scaled
    if "MonthlyCharges_scaled" in row.columns:
        row["MonthlyCharges_scaled"] = monthly_scaled
    if "TotalCharges_scaled" in row.columns:
        row["TotalCharges_scaled"] = total_scaled

    # Dummy mappings
    if f"Contract_{contract}" in row.columns:
        row[f"Contract_{contract}"] = 1.0
    if f"InternetService_{internet}" in row.columns:
        row[f"InternetService_{internet}"] = 1.0
    if "TechSupport_Yes" in row.columns and tech == "Yes":
        row["TechSupport_Yes"] = 1.0
    if f"PaymentMethod_{payment}" in row.columns:
        row[f"PaymentMethod_{payment}"] = 1.0

    prob = float(risk_model.predict_proba(row)[0, 1])
    tier = "High Risk" if prob > 0.5 else ("Moderate Risk" if prob > 0.25 else "Low Risk")

    return jsonify({"probability": prob, "risk_tier": tier})


if __name__ == "__main__":
    print("[Flask Dashboard] Starting local web application on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
