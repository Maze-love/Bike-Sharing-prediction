"""Inference utilities for bike sharing demand prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import clean_dataframe
from src.feature_engineering import create_engineered_features
from src.util import MODELS_DIR


def load_model(model_path: Path | None = None) -> dict[str, Any]:
    """
    Load saved model bundle from disk.

    Returns dict with 'model' (Pipeline) and 'metadata'.
    """
    path = model_path or (MODELS_DIR / "best_model.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)


def load_feature_info() -> dict[str, Any]:
    """Load feature column metadata saved during training."""
    import json
    
    info_path = MODELS_DIR / "feature_info.json"
    if not info_path.exists():
        raise FileNotFoundError("Feature info not found. Run training first.")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_input(raw_input: dict[str, Any]) -> pd.DataFrame:
    """
    Prepare a single prediction input from raw feature dict.

    Applies necessary data type parsing and feature engineering.
    """
    df = pd.DataFrame([raw_input])

    # Ensure date string is parsed to datetime for feature engineering
    # if "dteday" in df.columns:
    #     df["dteday"] = pd.to_datetime(df["dteday"])


    # Ensuring cleaning
    df= clean_dataframe(df)
    # Apply the same feature engineering steps used during training
    df = create_engineered_features(df)

    # Pad missing columns based on the exact features the model was trained on
    feature_info = load_feature_info()

    all_features = feature_info["all_features"]
    
    for col in all_features:
        if col not in df.columns:
            df[col] = 0 if col in feature_info.get("numeric_features", []) else "unknown"
    
    print ("resulted data \n: ",df[all_features])
    return df[all_features]


def predict_demand(raw_input: dict[str, Any], model_path: Path | None = None) -> float:
    """
    Predict bike rental demand from raw feature dictionary.

    Parameters
    ----------
    raw_input : dict
        Feature name to value mapping (same columns as training data minus cnt).
    model_path : Path, optional
        Path to saved model pickle.

    Returns
    -------
    float
        Predicted total rentals (cnt).
    """
    bundle = load_model(model_path)
    model = bundle["model"]

    
    X = prepare_input(raw_input)
    prediction = model.predict(X)[0]
    
    gbr = model.steps[-1][1]
    print((model.steps))
    print(gbr.feature_importances_)

    # Rental demand cannot be negative, cap the floor at 0
    return float(max(prediction, 0))


def predict_batch(df: pd.DataFrame, model_path: Path | None = None) -> pd.Series:
    """Predict rental demand for a batch of raw inputs."""
    bundle = load_model(model_path)
    model = bundle["model"]
    
    feature_info = load_feature_info()
    all_features = feature_info["all_features"]

    processed_rows = []
    for _, row in df.iterrows():
        processed = prepare_input(row.to_dict())
        processed_rows.append(processed.iloc[0])

    X = pd.DataFrame(processed_rows)[all_features]
    predictions = model.predict(X)
    
    # Ensure no negative predictions in batch
    predictions = [float(max(p, 0)) for p in predictions]
    
    return pd.Series(predictions, index=df.index, name="predicted_demand")