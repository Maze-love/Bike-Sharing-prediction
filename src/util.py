"""
=============================================================================
Project: Bike Sharing Demand Regression & Prediction Pipeline
Dataset: day.csv (Group 5 - Target Variable: 'cnt')
=============================================================================
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

TARGET_COLUMN = "cnt"
SCALED_COLUMN= ['holiday','workingday','mnth_sin',
                'mnth_cos','weekday_sin','weekday_cos','is_month_end','is_month_start']

# =============================================================================
# 1. DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def get_data_path(filename: str = "day.csv") -> Path:
    """Return absolute path to a dataset file."""
    return DATA_DIR / filename

def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    for directory in (FIGURES_DIR, METRICS_DIR, REPORTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)



def load_and_explore_data(path:Path | None= None) -> pd.DataFrame:
    """Loads dataset and prints essential EDA metrics."""
    print("=" * 60)
    print("STEP 1: DATASET OVERVIEW & EDA")
    print("=" * 60)
    
    data_path= path or get_data_path()
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    # print("\nMissing Values Count per Column:")
    # print(df.isnull().sum())
    
    # print("\nStatistical Summary:")
    # print(df.describe().T[["mean", "std", "min", "50%", "max"]])
    
    return df


def plot_eda_visualizations(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Generates and saves exploratory data visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Target Variable Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(df["cnt"], kde=True, color="skyblue")
    plt.title("Distribution of Daily Bike Rentals (Target: 'cnt')")
    plt.xlabel("Total Rentals (cnt)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/target_distribution.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 5))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # Temperature vs Demand Boxplot across Seasons
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="season", y="cnt", data=df,hue='season',palette='Set2')
    plt.title("Bike Demand across Seasons")
    plt.xlabel("Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)")
    plt.ylabel("Rental Count (cnt)")
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/season_vs_cnt_boxplot.png")
    plt.close()

def plot_distribution_plots(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """
    Plots and saves distribution plots (Histograms + KDE) for key continuous numerical variables.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    continuous_cols = ["temp", "weathersit", "hum", "windspeed", "cnt"]
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(continuous_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], kde=True, color="skyblue", bins=30)
        plt.title(f"Distribution of {col.capitalize()}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "distribution_plots.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"[INFO] Distribution plots saved to '{plot_path}'")


def plot_scatter_plots(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """
    Plots and saves scatter plots showing relationships between numerical features and target variable ('cnt').
    """
    os.makedirs(output_dir, exist_ok=True)
    
    features = ["temp", "holiday", "hum", "windspeed"]
    
    plt.figure(figsize=(14, 8))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(
            x=df[feature], 
            y=df["cnt"], 
            hue=df["season"], 
            palette="viridis", 
            alpha=0.7
        )
        plt.title(f"{feature.capitalize()} vs Target ('cnt')")
        plt.xlabel(f"{feature.capitalize()}")
        plt.ylabel("Rental Count (cnt)")
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "scatter_plots.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"[INFO] Scatter plots saved to '{plot_path}'")

def plot_coefficients(
    model: Any,
    feature_names: list[str],
    model_name: str,
) -> Path | None:
    """Plot coefficients for linear models."""
    estimator = model.named_steps.get("regressor", model)
    if not hasattr(estimator, "coef_"):
        return None

    coefs = np.abs(estimator.coef_.ravel())
    indices = np.argsort(coefs)[::-1][:20]
    top_names = [feature_names[i] for i in indices]
    top_values = estimator.coef_.ravel()[indices]

    plt.figure(figsize=(10, 8))
    colors = ["green" if v >= 0 else "red" for v in top_values]
    plt.barh(top_names, top_values, color=colors)
    plt.gca().invert_yaxis()
    plt.title(f"Top Coefficients — {model_name}")
    plt.xlabel("Coefficient Value")
    safe_name = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / "evaluation" / f"coefficients{safe_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
) -> Path | None:
    """Plot feature importance for tree-based models."""
    estimator = model.named_steps.get("regressor", model)
    if not hasattr(estimator, "feature_importances_"):
        return None

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_values, y=top_names, orient="h")
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance")
    safe_name = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / "evaluation" / f"feature_importance{safe_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# additional plots
def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Path:
    """Plot residual distribution and residuals vs predicted."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title(f"Residual Distribution — {model_name}")
    axes[0].set_xlabel("Residual ($)")

    sns.scatterplot(x=y_pred, y=residuals, alpha=0.4, ax=axes[1])
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title(f"Residuals vs Predicted — {model_name}")
    axes[1].set_xlabel("Predicted Price ($)")
    axes[1].set_ylabel("Residual ($)")

    ensure_output_dirs()
    safe_name = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / "evaluation" / f"residuals_{safe_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Path:
    """Plot predicted vs actual values."""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title(f"Predicted vs Actual — {model_name}")
    plt.legend()
    safe_name = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / "evaluation" / f"pred_vs_actual{safe_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_learning_curve(scores: dict[str, list[float]], model_name: str) -> Path:
    """Plot learning curve from cross-validation scores."""
    plt.figure(figsize=(10, 6))
    train_sizes = scores.get("train_sizes", list(range(1, len(scores["train_scores"]) + 1)))
    plt.plot(train_sizes, scores["train_scores"], label="Train R²", marker="o")
    plt.plot(train_sizes, scores["val_scores"], label="Validation R²", marker="o")
    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.title(f"Learning Curve — {model_name}")
    plt.legend()
    safe_name = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / "evaluation" / f"learning_curve{safe_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path



def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Inspect dataset and return structured summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned dataframe.

    Returns
    -------
    dict
        Summary including shape, dtypes, missing values, duplicates, describe.
    """
    summary: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().astype(int).to_dict(),
        "missing_percentage": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "descriptive_statistics": df.describe(include="all").to_dict(),
    }
    return summary

def generate_data_understanding_report(df: pd.DataFrame, output_path: Path | None = None) -> str:
    """
    Generate and save a markdown data understanding report.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to inspect.
    output_path : Path, optional
        Destination markdown file. Defaults to outputs/reports/data_understanding.md.

    Returns
    -------
    str
        Markdown report content.
    """
    # ensure_output_dirs()
    summary = inspect_dataset(df)
    report_path = output_path or (REPORTS_DIR / "data_understanding.md")

    lines = [
        "# Data Understanding Report",
        "",
        "## Dataset Shape",
        f"- **Rows:** {summary['shape']['rows']:,}",
        f"- **Columns:** {summary['shape']['columns']}",
        "",
        "## Columns",
        "",
    ]
    for col in summary["columns"]:
        lines.append(f"- `{col}` ({summary['dtypes'][col]})")

    lines.extend(["", "## Missing Values", ""])
    for col, count in summary["missing_values"].items():
        pct = summary["missing_percentage"][col]
        lines.append(f"- **{col}:** {count} ({pct}%)")

    lines.extend([
        "",
        "## Duplicated Rows",
        f"- **Count:** {summary['duplicated_rows']}",
        "",
        "## Descriptive Statistics",
        "",
        "```",
        pd.DataFrame(summary["descriptive_statistics"]).to_string(),
        "```",
        "",
    ])

    content = "\n".join(lines)
    save_markdown(content, report_path)

    # Also print to stdout for CLI usage
    print("=" * 60)
    print("DATA UNDERSTANDING")
    print("=" * 60)
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print(f"Dtypes:\n{pd.Series(summary['dtypes'])}")
    print(f"Missing values:\n{pd.Series(summary['missing_values'])}")
    print(f"Duplicated rows: {summary['duplicated_rows']}")
    print(f"Descriptive statistics:\n{pd.DataFrame(summary['descriptive_statistics'])}")
    print(f"\nReport saved to: {report_path}")

    return content



# untested functions...
def save_markdown(content: str, path: Path) -> None:
    """Write markdown content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")





# if __name__ == "__main__":
    # filepath = "../data/day.csv"
    
    # 1. EDA
    # df = load_and_explore_data(filepath)
    # plot_eda_visualizations(df)
    
    # 2. Preprocessing
    # X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 3. Model Comparison
    # comparison_table, scaler = train_and_evaluate_models(X_train, X_test, y_train, y_test)

