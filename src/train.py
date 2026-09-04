"""Model training pipeline for bike sharing demand prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
# from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.evaluate import (
    compute_learning_curve_data,
    evaluate_model,
    generate_comparison_report,
    run_cross_validation,
    save_comparison_results,
    select_best_model,
)

from src.feature_engineering import create_engineered_features
from src.preprocessing import clean_dataframe,build_preprocessor,get_feature_columns

from src.util import (
    METRICS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
    generate_data_understanding_report,
    load_and_explore_data,
    save_json,
    plot_residuals,
    plot_learning_curve,
    plot_prediction_vs_actual,
    plot_coefficients,
    plot_feature_importance
)

# Models that benefit from feature scaling
SCALED_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression", "Support Vector Regressor (SVR)"}

TREE_MODELS = {"Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor","EXtra Trees", "AdaBoost"}
LINEAR_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}




def get_model_registry() -> dict[str, Any]:
    """Return all 9 regression algorithms for bike rental estimation."""
    return {
        
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.5, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
        "Support Vector Regressor (SVR)": SVR(kernel="rbf", C=1000, epsilon=50),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        # "XGBoost Regressor (Bonus)": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0),
    }


def build_pipeline( 
    model: Any,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str
) -> Pipeline:
    """Build full sklearn pipeline with dynamic feature scaling if required."""

    scale_numeric = model_name in SCALED_MODELS
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )



def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extract transformed feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps["encoder"]
            cat_names = encoder.get_feature_names_out(columns).tolist()
            feature_names.extend(cat_names)

    return feature_names

def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load, clean, engineer features, and perform train/test split."""
    
    raw_df = load_and_explore_data()
    generate_data_understanding_report(raw_df)

    # 1. Clean Data
    cleaned_df = clean_dataframe(raw_df)

    # 2. Engineer Features
    engineered_df = create_engineered_features(cleaned_df)

    # 3. Features & Target Extraction
    feature_cols = [c for c in engineered_df.columns if c != TARGET_COLUMN]
    
    X = engineered_df[feature_cols]
    y = engineered_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, engineered_df


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Train all regression models and evaluate each.

    Returns metrics dataframe, fitted models dict, and predictions dict.
    """
    numeric_features, categorical_features = get_feature_columns(X_train)

    registry = get_model_registry()

    all_metrics: list[dict[str, float]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    cv_results: dict[str, dict] = {}
    learning_curves: dict[str, dict] = {}

    ensure_output_dirs()

    for model_name, estimator in registry.items():
        print(f"Training {model_name}...")
        pipeline = build_pipeline(estimator, numeric_features, categorical_features,model_name)
        
        metrics, fitted, y_pred = evaluate_model(
            pipeline, X_train, X_test, y_train, y_test, model_name
        )

        all_metrics.append(metrics)
        fitted_models[model_name] = fitted
        predictions[model_name] = y_pred

         # Cross-validation
        cv_results[model_name] = run_cross_validation(
            build_pipeline(
                clone_estimator(estimator),
                numeric_features,
                categorical_features,
                model_name,
            ),
            X_train,
            y_train,
        )

        # Learning curves for best candidates (all models for completeness)
        lc_pipeline = build_pipeline(
            clone_estimator(estimator),
            numeric_features,
            categorical_features,
            model_name,
        )
        learning_curves[model_name] = compute_learning_curve_data(lc_pipeline, X_train, y_train)

        # # Bonus plots
        plot_residuals(y_test.values, y_pred, model_name)
        plot_prediction_vs_actual(y_test.values, y_pred, model_name)
        plot_learning_curve(learning_curves[model_name], model_name)

        # Feature importance / coefficients
        feature_names = get_feature_names_from_pipeline(fitted)
        if model_name in TREE_MODELS:
            plot_feature_importance(fitted, feature_names, model_name)
        elif model_name in LINEAR_MODELS:
            plot_coefficients(fitted, feature_names, model_name)

    metrics_df = pd.DataFrame(all_metrics)
    save_comparison_results(metrics_df)
    save_json(cv_results, METRICS_DIR / "cross_validation.json")

    return metrics_df, fitted_models,predictions



def clone_estimator(estimator: Any) -> Any:
    """Clone sklearn estimator with same parameters."""
    from sklearn.base import clone
    return clone(estimator)


def save_best_model(
    fitted_models: dict[str, Pipeline],
    best_model_name: str,
    feature_cols: list[str],
) -> Path:
    """Save winning model bundle and metadata JSON."""
    
    ensure_output_dirs()
    model = fitted_models[best_model_name]
    model_path = MODELS_DIR / "best_model.pkl"

    metadata = {
        "model_name": best_model_name,
        "target_column": TARGET_COLUMN,
    }

    joblib.dump({"model": model, "metadata": metadata}, model_path)
    save_json(metadata, MODELS_DIR / "model_metadata.json")

    feature_info = {
        "all_features": feature_cols,
        "numeric_features": feature_cols,
        "best_model": best_model_name,
    }
    save_json(feature_info, MODELS_DIR / "feature_info.json")

    print(f"\n[INFO] Best model ({best_model_name}) saved to {model_path}")
    return model_path


def run_training_pipeline() -> dict[str, Any]:
    """Execute complete end-to-end training process."""
    X_train, X_test, y_train, y_test, engineered_df = prepare_data()

    metrics_df, fitted_models,predictions = train_all_models(X_train, X_test, y_train, y_test)

    best_model_name = select_best_model(metrics_df)
    generate_comparison_report(metrics_df, best_model_name)

    model_path = save_best_model(
        fitted_models, best_model_name, X_train.columns.tolist()
    )

    summary = {
        "best_model": best_model_name,
        "model_path": str(model_path),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    save_json(summary, REPORTS_DIR / "training_summary.json")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Selected Model: {best_model_name}")
    print(metrics_df.sort_values("rmse").to_string(index=False))

    return summary


if __name__ == "__main__":
    run_training_pipeline()