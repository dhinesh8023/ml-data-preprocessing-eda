"""
Flask Interactive Dashboard & Prediction Web App.
Allows users and evaluators to interactively inspect metrics, visualizations,
and test real-time predictions with Logistic Regression and Decision Tree models.
"""

import os
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
from src.data_loader import load_dataset, split_data
from src.models import train_all_models
from src.evaluate import compare_models

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Initialize and cache models
print("[App] Loading dataset and training models...")
X, y, metadata = load_dataset(data_dir=os.path.join(BASE_DIR, "data"), save_csv=True)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
models = train_all_models(X_train, y_train, random_state=42)
summary_df, results = compare_models(models, X_test, y_test)

# Typical benchmark samples for testing
sample_malignant = X_test[y_test == 0].iloc[0].to_dict()
sample_benign = X_test[y_test == 1].iloc[0].to_dict()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Classification Model Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #f8fafc; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
        .hero { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; padding: 2.5rem 0; margin-bottom: 2rem; border-bottom: 3px solid #38bdf8; }
        .card { border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 1.5rem; transition: transform 0.2s; }
        .metric-badge { font-size: 1.1rem; padding: 0.4rem 0.8rem; border-radius: 8px; font-weight: 600; }
        .table-custom th { background-color: #f1f5f9; color: #334155; }
        .img-preview { border-radius: 8px; border: 1px solid #e2e8f0; width: 100%; height: auto; transition: 0.3s; }
        .img-preview:hover { transform: scale(1.02); }
        .btn-predict { background: #0284c7; color: white; border: none; font-weight: 600; }
        .btn-predict:hover { background: #0369a1; color: white; }
    </style>
</head>
<body>

<div class="hero">
    <div class="container">
        <div class="row align-items-center">
            <div class="col-md-8">
                <span class="badge bg-primary mb-2"><i class="fa-solid fa-graduation-cap"></i> ML Internship Project</span>
                <h1 class="fw-bold mb-1">Simple Classification Model Pipeline</h1>
                <p class="lead text-light opacity-75 mb-0">Breast Cancer Wisconsin (Diagnostic) Dataset • Logistic Regression vs. Decision Tree</p>
            </div>
            <div class="col-md-4 text-md-end mt-3 mt-md-0">
                <a href="https://github.com/dhinesh8023/ml-classification-model" target="_blank" class="btn btn-outline-light me-2">
                    <i class="fab fa-github"></i> GitHub Repo
                </a>
                <a href="https://colab.research.google.com/github/dhinesh8023/ml-classification-model/blob/main/notebooks/classification_model.ipynb" target="_blank" class="btn btn-warning">
                    <i class="fa-solid fa-bolt"></i> Google Colab
                </a>
            </div>
        </div>
    </div>
</div>

<div class="container pb-5">

    <!-- Row 1: Key Performance Metrics -->
    <div class="row">
        <div class="col-12">
            <div class="card p-4">
                <h4 class="fw-bold text-dark mb-3"><i class="fa-solid fa-chart-line text-primary"></i> Evaluation Metrics Comparison</h4>
                <div class="table-responsive">
                    <table class="table table-bordered table-hover align-middle table-custom">
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th class="text-center">Accuracy</th>
                                <th class="text-center">Precision</th>
                                <th class="text-center">Recall (Sensitivity)</th>
                                <th class="text-center">F1-Score</th>
                                <th class="text-center">ROC-AUC</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in summary %}
                            <tr>
                                <td class="fw-bold">{{ row['Model'] }}</td>
                                <td class="text-center"><span class="badge bg-success-subtle text-success border border-success-subtle metric-badge">{{ (row['Accuracy']|float * 100)|round(2) }}%</span></td>
                                <td class="text-center"><span class="badge bg-primary-subtle text-primary border border-primary-subtle metric-badge">{{ (row['Precision']|float * 100)|round(2) }}%</span></td>
                                <td class="text-center"><span class="badge bg-warning-subtle text-dark border border-warning-subtle metric-badge">{{ (row['Recall']|float * 100)|round(2) }}%</span></td>
                                <td class="text-center"><span class="badge bg-info-subtle text-dark border border-info-subtle metric-badge">{{ row['F1-Score'] }}</span></td>
                                <td class="text-center"><span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle metric-badge">{{ row['ROC-AUC'] }}</span></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                <div class="alert alert-info mb-0 mt-2">
                    <i class="fa-solid fa-circle-info"></i> <strong>Key Takeaway:</strong> In healthcare diagnosis, <strong>Recall</strong> is the single most vital metric to minimize false negatives (missed cancer cases). Logistic Regression achieved <strong>98.61% Recall</strong> and <strong>0.995 ROC-AUC</strong>, proving to be the optimal model.
                </div>
            </div>
        </div>
    </div>

    <!-- Row 2: Visual Diagnostics -->
    <div class="row">
        <div class="col-md-6">
            <div class="card p-3">
                <h5 class="fw-bold"><i class="fa-solid fa-table-cells text-primary"></i> Confusion Matrices</h5>
                <img src="/reports/confusion_matrices.png" class="img-preview" alt="Confusion Matrices">
            </div>
        </div>
        <div class="col-md-6">
            <div class="card p-3">
                <h5 class="fw-bold"><i class="fa-solid fa-chart-area text-primary"></i> ROC Curves</h5>
                <img src="/reports/roc_curves.png" class="img-preview" alt="ROC Curves">
            </div>
        </div>
    </div>

    <div class="row">
        <div class="col-md-6">
            <div class="card p-3">
                <h5 class="fw-bold"><i class="fa-solid fa-chart-bar text-primary"></i> Metric Comparison Chart</h5>
                <img src="/reports/metrics_comparison.png" class="img-preview" alt="Metrics Comparison">
            </div>
        </div>
        <div class="col-md-6">
            <div class="card p-3">
                <h5 class="fw-bold"><i class="fa-solid fa-ranking-star text-primary"></i> Top Diagnostic Features (Decision Tree)</h5>
                <img src="/reports/feature_importance.png" class="img-preview" alt="Feature Importance">
            </div>
        </div>
    </div>

    <!-- Row 3: Interactive Real-time Prediction -->
    <div class="row">
        <div class="col-12">
            <div class="card p-4">
                <h4 class="fw-bold mb-3"><i class="fa-solid fa-stethoscope text-primary"></i> Test Real-Time Diagnosis Prediction</h4>
                <p class="text-muted">Select a preset sample or click predict to run inference across both models:</p>
                <div class="mb-3">
                    <button class="btn btn-outline-danger me-2" onclick="loadSample('malignant')"><i class="fa-solid fa-virus"></i> Load Malignant Sample</button>
                    <button class="btn btn-outline-success" onclick="loadSample('benign')"><i class="fa-solid fa-heart-pulse"></i> Load Benign Sample</button>
                </div>
                
                <div id="predictionResult" class="p-3 rounded border bg-light d-none mb-3">
                    <h5 class="fw-bold mb-2">Inference Results:</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="p-3 border rounded bg-white">
                                <h6>Logistic Regression:</h6>
                                <p class="h4" id="lrPred"></p>
                                <small class="text-muted" id="lrProb"></small>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="p-3 border rounded bg-white">
                                <h6>Decision Tree:</h6>
                                <p class="h4" id="dtPred"></p>
                                <small class="text-muted" id="dtProb"></small>
                            </div>
                        </div>
                    </div>
                </div>

                <button class="btn btn-predict btn-lg" onclick="runPrediction()"><i class="fa-solid fa-play"></i> Run Model Inference</button>
            </div>
        </div>
    </div>

</div>

<script>
    let currentSample = {{ sample_malignant|tojson }};

    function loadSample(type) {
        if(type === 'malignant') {
            currentSample = {{ sample_malignant|tojson }};
            alert("Loaded Typical Malignant Benchmark Sample.");
        } else {
            currentSample = {{ sample_benign|tojson }};
            alert("Loaded Typical Benign Benchmark Sample.");
        }
    }

    async function runPrediction() {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentSample)
        });
        const data = await response.json();
        
        document.getElementById('predictionResult').classList.remove('d-none');
        
        const lrEl = document.getElementById('lrPred');
        lrEl.textContent = data['Logistic Regression'].prediction;
        lrEl.className = data['Logistic Regression'].prediction === 'Benign' ? 'h4 text-success fw-bold' : 'h4 text-danger fw-bold';
        document.getElementById('lrProb').textContent = "Benign Probability: " + (data['Logistic Regression'].probability * 100).toFixed(2) + "%";

        const dtEl = document.getElementById('dtPred');
        dtEl.textContent = data['Decision Tree'].prediction;
        dtEl.className = data['Decision Tree'].prediction === 'Benign' ? 'h4 text-success fw-bold' : 'h4 text-danger fw-bold';
        document.getElementById('dtProb').textContent = "Benign Probability: " + (data['Decision Tree'].probability * 100).toFixed(2) + "%";
    }
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, summary=summary_df.to_dict(orient="records"), sample_malignant=sample_malignant, sample_benign=sample_benign)

@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    df_sample = pd.DataFrame([data])
    
    response = {}
    class_map = {0: "Malignant", 1: "Benign"}
    
    for name, model in models.items():
        pred = model.predict(df_sample)[0]
        prob = model.predict_proba(df_sample)[0][1]
        response[name] = {
            "prediction": class_map[pred],
            "probability": float(prob)
        }
        
    return jsonify(response)


if __name__ == "__main__":
    print("[App] Starting web dashboard on http://localhost:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
