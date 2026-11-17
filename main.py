"""
Main Pipeline Runner
--------------------
Orchestrates end-to-end Data Preprocessing & Exploratory Data Analysis.
"""

import json
from pathlib import Path
import pandas as pd

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.eda import ExploratoryAnalyzer
from src.model_baseline import BaselineModelBenchmark
from src.report_generator import generate_html_report


def run_pipeline():
    project_root = Path(__file__).resolve().parent
    raw_path = project_root / "data" / "raw" / "telco_customer_churn.csv"
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" MACHINE LEARNING INTERNSHIP PROJECT: DATA PREPROCESSING & EDA")
    print("=" * 70)

    # 1. LOAD DATA
    print("\n--- PHASE 1: DATA INGESTION ---")
    loader = DataLoader(raw_path)
    raw_df = loader.load_data()
    schema = loader.inspect_schema()
    print(f"Dataset Shape: {schema['rows']} rows, {schema['columns']} columns")
    print(f"Memory Usage: {schema['memory_kb']} KB")

    # 2. PREPROCESSING
    print("\n--- PHASE 2: DATA CLEANING & PREPROCESSING ---")
    preprocessor = DataPreprocessor(target_col="Churn", id_col="customerID")

    # Step 2a: Missing values & type casting
    clean_df, missing_summary = preprocessor.fix_data_types_and_missing(raw_df, fill_strategy="zero")

    # Step 2b: Duplicate records
    clean_df, dups_removed = preprocessor.remove_duplicates(clean_df)

    # Step 2c: Outlier Detection
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    print("\n[Outlier Analysis - IQR Method (factor=1.5)]")
    iqr_report = preprocessor.detect_outliers_iqr(clean_df, columns=num_cols, factor=1.5)
    for col, metrics in iqr_report.items():
        print(f"  * {col:15s} | Q1={metrics['q1']:7.2f} | Q3={metrics['q3']:7.2f} | IQR={metrics['iqr']:7.2f} | "
              f"Lower={metrics['lower_bound']:7.2f} | Upper={metrics['upper_bound']:7.2f} | "
              f"Outliers={metrics['outlier_count']} ({metrics['outlier_percentage']}%)")

    print("\n[Outlier Analysis - Z-Score Method (|Z| > 3.0)]")
    z_report = preprocessor.detect_outliers_zscore(clean_df, columns=num_cols, threshold=3.0)
    for col, metrics in z_report.items():
        print(f"  * {col:15s} | Mean={metrics['mean']:7.2f} | Std={metrics['std']:7.2f} | "
              f"Outliers={metrics['outlier_count']} ({metrics['outlier_percentage']}%)")

    # Remediate / Winsorize outliers
    clean_df = preprocessor.handle_outliers(clean_df, columns=num_cols, method="clip")

    # Step 2d: Categorical cleaning & Feature Engineering
    clean_df = preprocessor.clean_categories_and_engineer_features(clean_df)

    # Export cleaned dataset
    clean_csv_path = processed_dir / "telco_customer_churn_cleaned.csv"
    clean_df.to_csv(clean_csv_path, index=False)
    print(f"\n[Export] Cleaned dataset saved to: {clean_csv_path}")

    # Step 2e: Encoding & Scaling
    encoded_df = preprocessor.encode_and_scale(clean_df, numeric_cols=num_cols, scale_method="standard")
    encoded_csv_path = processed_dir / "telco_customer_churn_encoded.csv"
    encoded_df.to_csv(encoded_csv_path, index=False)
    print(f"[Export] Encoded & scaled dataset saved to: {encoded_csv_path}")

    # 3. EXPLORATORY DATA ANALYSIS
    print("\n--- PHASE 3: EXPLORATORY DATA ANALYSIS & VISUALIZATION ---")
    analyzer = ExploratoryAnalyzer(output_dir=figures_dir)

    # 3a: Numerical Summary
    num_summary = analyzer.generate_numerical_summary(clean_df, num_cols)
    print("\n[Numerical Features Summary Table]")
    print(num_summary.to_string(index=False))
    num_summary.to_csv(reports_dir / "numerical_summary.csv", index=False)

    # 3b: Categorical Summary
    cat_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "InternetService", "PaymentMethod", "Churn"]
    cat_summary = analyzer.generate_categorical_summary(clean_df, cat_cols)
    print("\n[Categorical Features Summary Table]")
    print(cat_summary.to_string(index=False))
    cat_summary.to_csv(reports_dir / "categorical_summary.csv", index=False)

    # 3c: Visualizations
    analyzer.plot_numerical_distributions(clean_df, num_cols)
    analyzer.plot_outlier_boxplots(clean_df, num_cols, target_col="Churn")
    analyzer.plot_target_distribution(clean_df, target_col="Churn")
    analyzer.plot_churn_by_key_drivers(clean_df)
    analyzer.plot_correlation_heatmap(clean_df)
    analyzer.plot_tenure_vs_charges_scatter(clean_df)

    # 4. BASELINE ML BENCHMARK
    print("\n--- PHASE 4: MACHINE LEARNING VALIDATION & BENCHMARKING ---")
    benchmark = BaselineModelBenchmark(data_path=encoded_csv_path, output_dir=figures_dir)
    X_train, X_test, y_train, y_test = benchmark.load_and_split_data(target_col="Churn", test_size=0.2)
    model_summary = benchmark.train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\n[Model Benchmark Evaluation Summary]")
    print(model_summary.to_string(index=False))
    model_summary.to_csv(reports_dir / "model_benchmark_summary.csv", index=False)

    benchmark.plot_roc_and_pr_curves(X_test, y_test)
    benchmark.plot_feature_importances(X_train, top_n=15)

    # 5. GENERATE INTERACTIVE HTML REPORT
    print("\n--- PHASE 5: EXPORTING STANDALONE INTERACTIVE HTML REPORT ---")
    generate_html_report(project_root)

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" Clean Data  : {clean_csv_path}")
    print(f" Encoded Data: {encoded_csv_path}")
    print(f" Figures     : {figures_dir}")
    print(f" Reports     : {reports_dir}")
    print(f" HTML Report : {reports_dir / 'internship_report.html'}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
