"""Model evaluation metrics and comparison utilities for bike rental prediction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import KFold, cross_val_score, learning_curve

from src.util import METRICS_DIR, REPORTS_DIR, save_markdown


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float = 0.0,
    predict_time: float = 0.0,
) -> dict[str, float]:
    """Compute MAE, MSE, RMSE, and R² for regression predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
        "train_time_sec": float(train_time),
        "predict_time_sec": float(predict_time),
    }


def evaluate_model(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> tuple[dict[str, float], Any, np.ndarray]:
    """Train a pipeline and evaluate on the test set."""
    start_train = time.perf_counter()
    fitted = pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_pred = fitted.predict(X_test)
    predict_time = time.perf_counter() - start_pred

    metrics = compute_regression_metrics(y_test.values, y_pred, train_time, predict_time)
    metrics["model"] = model_name
    return metrics, fitted, y_pred


def run_cross_validation(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "r2",
) -> dict[str, Any]:
    """Run k-fold cross-validation and return aggregated scores."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "scoring": scoring,
        "n_splits": n_splits,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "fold_scores": scores.tolist(),
    }


def compute_learning_curve_data(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute learning curve train/validation R² scores."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        train_sizes=train_sizes,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
    )
    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores": train_scores.mean(axis=1).tolist(),
        "val_scores": val_scores.mean(axis=1).tolist(),
    }


def select_best_model(metrics_df: pd.DataFrame) -> str:
    """Select best model using composite ranking penalizing high RMSE/MAE and runtime."""
    df = metrics_df.copy()
    df["rmse_rank"] = df["rmse"].rank()
    df["mae_rank"] = df["mae"].rank()
    df["r2_rank"] = df["r2"].rank(ascending=False)
    df["time_rank"] = df["train_time_sec"].rank()
    df["composite"] = (
        0.40 * df["rmse_rank"]
        + 0.35 * df["mae_rank"]
        + 0.15 * df["r2_rank"]
        + 0.10 * df["time_rank"]
    )
    best_row = df.loc[df["composite"].idxmin()]
    return str(best_row["model"])


def save_comparison_results(metrics_df: pd.DataFrame) -> Path:
    """Save model comparison metrics to outputs/metrics/comparison.csv."""
    # ensure_output_dirs()
    path = METRICS_DIR / "comparison.csv"
    metrics_df.to_csv(path, index=False)
    return path


def generate_comparison_report(metrics_df: pd.DataFrame, best_model: str) -> str:
    """Generate Markdown model comparison summary."""
    # ensure_output_dirs()
    sorted_df = metrics_df.sort_values("rmse")

    lines = [
        "# Bike Sharing Model Comparison Report",
        "",
        f"**Best Model:** {best_model}",
        "",
        "## Selection Criteria",
        "",
        "Selected via composite score: RMSE (40%), MAE (35%), R² (15%), Train Time (10%).",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | RMSE | MAE | R² | Train Time (s) | Predict Time (s) |",
        "|------|-------|------|-----|----|----------------|------------------|",
    ]

    for rank, (_, row) in enumerate(sorted_df.iterrows(), start=1):
        marker = " ⭐" if row["model"] == best_model else ""
        lines.append(
            f"| {rank} | {row['model']}{marker} | "
            f"{row['rmse']:,.2f} | {row['mae']:,.2f} | "
            f"{row['r2']:.4f} | {row['train_time_sec']:.3f} | "
            f"{row['predict_time_sec']:.4f} |"
        )

    best_row = metrics_df.loc[metrics_df["model"] == best_model].iloc[0]
    lines.extend([
        "",
        "## Summary",
        "",
        f"The **{best_model}** model achieved the highest overall score "
        f"(RMSE={best_row['rmse']:,.2f}, MAE={best_row['mae']:,.2f}, R²={best_row['r2']:.4f}).",
        "",
    ])

    content = "\n".join(lines)
    save_markdown(content, REPORTS_DIR / "model_comparison.md")
    return content