"""
Exploratory Data Analysis (EDA) Module
-------------------------------------
Generates statistical summaries, bivariate comparisons, correlation analyses,
and publication-ready visualization figures.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ExploratoryAnalyzer:
    """Performs statistical analysis and generates visualizations for tabular datasets."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set stylish aesthetic
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    def generate_numerical_summary(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        """Computes comprehensive summary statistics for numeric features."""
        stats = []
        for col in num_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            stats.append({
                "Feature": col,
                "Count": int(series.count()),
                "Mean": round(float(series.mean()), 2),
                "Std Dev": round(float(series.std()), 2),
                "Median": round(float(series.median()), 2),
                "IQR": round(float(q3 - q1), 2),
                "Min": round(float(series.min()), 2),
                "25%": round(float(q1), 2),
                "75%": round(float(q3), 2),
                "Max": round(float(series.max()), 2),
                "Skewness": round(float(series.skew()), 2),
                "Kurtosis": round(float(series.kurt()), 2),
            })
        summary_df = pd.DataFrame(stats)
        return summary_df

    def generate_categorical_summary(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Computes summary statistics for categorical features."""
        stats = []
        for col in cat_cols:
            if col not in df.columns:
                continue
            series = df[col].astype(str).dropna()
            val_counts = series.value_counts()
            stats.append({
                "Feature": col,
                "Distinct Values": int(series.nunique()),
                "Top Mode": str(val_counts.index[0]),
                "Top Mode Freq": int(val_counts.iloc[0]),
                "Top Mode %": round((val_counts.iloc[0] / len(series)) * 100, 2),
            })
        return pd.DataFrame(stats)

    def plot_numerical_distributions(self, df: pd.DataFrame, num_cols: List[str]) -> Path:
        """Plots distribution histograms with KDE overlays."""
        fig, axes = plt.subplots(1, len(num_cols), figsize=(6 * len(num_cols), 5))
        if len(num_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, num_cols):
            data_series = pd.to_numeric(df[col], errors="coerce").dropna()
            sns.histplot(data_series, kde=True, ax=ax, color="#1f77b4", bins=30, edgecolor="black", alpha=0.6)
            mean_val = float(data_series.mean())
            median_val = float(data_series.median())
            ax.axvline(mean_val, color="#d62728", linestyle="--", label=f"Mean: {mean_val:.1f}")
            ax.axvline(median_val, color="#2ca02c", linestyle="-", label=f"Median: {median_val:.1f}")
            ax.set_title(f"Distribution of {col}", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.legend(frameon=True, facecolor="white")

        plt.tight_layout()
        save_path = self.output_dir / "01_numerical_distributions.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA] Saved distribution plot to: {save_path}")
        return save_path

    def plot_outlier_boxplots(self, df: pd.DataFrame, num_cols: List[str], target_col: str = "Churn") -> Path:
        """Plots boxplots to visualize outliers and differences across target classes."""
        fig, axes = plt.subplots(1, len(num_cols), figsize=(6 * len(num_cols), 5))
        if len(num_cols) == 1:
            axes = [axes]

        palette = {"No": "#2ca02c", "Yes": "#d62728", 0: "#2ca02c", 1: "#d62728", "0": "#2ca02c", "1": "#d62728"}
        for ax, col in zip(axes, num_cols):
            sns.boxplot(
                data=df,
                x=target_col if target_col in df.columns else None,
                y=col,
                hue=target_col if target_col in df.columns else None,
                palette=palette if target_col in df.columns else None,
                legend=False,
                ax=ax,
                width=0.45,
                flierprops={"marker": "o", "color": "red", "alpha": 0.5},
            )
            ax.set_title(f"Boxplot of {col} by {target_col}", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel(target_col, fontsize=11)
            ax.set_ylabel(col, fontsize=11)

        plt.tight_layout()
        save_path = self.output_dir / "02_outlier_boxplots.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA] Saved outlier boxplots to: {save_path}")
        return save_path

    def plot_target_distribution(self, df: pd.DataFrame, target_col: str = "Churn") -> Path:
        """Visualizes the target class balance with count and donut chart."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        counts = df[target_col].value_counts()
        labels = [str(x) for x in counts.index]
        colors = ["#2ca02c", "#d62728"]

        # Bar chart
        bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
        for bar in bars:
            yval = bar.get_height()
            pct = (yval / len(df)) * 100
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                yval + 50,
                f"{yval:,} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        axes[0].set_title(f"{target_col} Frequency Count", fontsize=13, fontweight="bold")
        axes[0].set_xlabel(target_col, fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_ylim(0, max(counts.values) * 1.15)

        # Donut chart
        wedges, texts, autotexts = axes[1].pie(
            counts.values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.4},
        )
        for t in texts:
            t.set_fontsize(11)
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
            at.set_color("white")
        axes[1].set_title(f"{target_col} Proportion", fontsize=13, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / "03_target_distribution.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA] Saved target distribution to: {save_path}")
        return save_path

    def plot_churn_by_key_drivers(self, df: pd.DataFrame) -> Path:
        """Analyzes churn rate percentages across Contract, InternetService, TechSupport, PaymentMethod."""
        features = ["Contract", "InternetService", "TechSupport", "PaymentMethod"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        axes = axes.flatten()

        for idx, col in enumerate(features):
            if col not in df.columns:
                continue
            ax = axes[idx]
            
            # Calculate churn percentage safely
            sub_df = df[[col, "Churn"]].copy()
            sub_df["IsChurn"] = sub_df["Churn"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
            
            churn_pct = (
                sub_df.groupby(col, observed=False)["IsChurn"]
                .mean()
                .mul(100)
                .reset_index(name="ChurnRate")
            )
            churn_pct = churn_pct.sort_values(by="ChurnRate", ascending=False)
            
            bars = ax.bar(
                churn_pct[col].astype(str),
                churn_pct["ChurnRate"],
                color="#e377c2",
                edgecolor="black",
                width=0.5,
            )
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
            ax.set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Churn Rate (%)", fontsize=10)
            max_rate = float(churn_pct["ChurnRate"].max()) if not churn_pct.empty else 50.0
            ax.set_ylim(0, max(max_rate * 1.25, 20.0))
            ax.tick_params(axis="x", rotation=25)

        plt.tight_layout()
        save_path = self.output_dir / "04_churn_by_key_drivers.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA] Saved key drivers plot to: {save_path}")
        return save_path

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> Path:
        """Plots correlation matrix heatmap for numerical variables and churn."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        plot_df = df.copy()
        if "Churn" in plot_df.columns:
            plot_df["Churn"] = plot_df["Churn"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
            
        corr_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"] if c in plot_df.columns]
        for c in corr_cols:
            plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
            
        corr_matrix = plot_df[corr_cols].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Correlation Heatmap (Key Features & Churn)", fontsize=13, fontweight="bold", pad=12)

        plt.tight_layout()
        save_path = self.output_dir / "05_correlation_heatmap.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA] Saved correlation heatmap to: {save_path}")
        return save_path

    def plot_tenure_vs_charges_scatter(self, df: pd.DataFrame) -> Path:
        """Visualizes interaction between Tenure, MonthlyCharges, and Churn."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_df = df.copy()
        palette = {"No": "#2ca02c", "Yes": "#d62728", 0: "#2ca02c", 1: "#d62728", "0": "#2ca02c", "1": "#d62728"}
        
        sns.scatterplot(
            data=plot_df,
            x="tenure",
            y="MonthlyCharges",
            hue="Churn",
            palette=palette,
            alpha=0.6,
            s=40,
            ax=ax,
        )
        ax.set_title("Customer Tenure vs Monthly Charges segmented by Churn", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Tenure (Months)", fontsize=11)
        ax.set_ylabel("Monthly Charges ($)", fontsize=11)
        ax.legend(title="Churn", frameon=True, facecolor="white")

        plt.tight_layout()
        save_path = self.output_dir / "06_tenure_vs_charges_scatter.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA] Saved scatter plot to: {save_path}")
        return save_path
