"""Data cleaning and preprocessing functions."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Machine Learning utilities
from sklearn.model_selection import train_test_split

from src.util import TARGET_COLUMN,SCALED_COLUMN


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing whitespace from object columns."""
    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns

    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return cleaned

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataframe."""
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with domain-aware imputation.

    Numeric: median imputation for model_year, mileage; drop rows with missing target.
    Categorical: fill with 'unknown'.
    """
    cleaned = df.copy()

    if TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()

    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)

    for col in numeric_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    categorical_cols = cleaned.select_dtypes(include=["object", "string", "category"]).columns
    for col in categorical_cols:
        cleaned[col] = cleaned[col].fillna("unknown")


    return cleaned




def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Split dataframe columns into numeric and categorical feature lists.

    Excludes target column.
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        c for c in feature_cols if c not in numeric_features
    ]

    # print("******from get_features_columns********")
    # print(len(feature_cols))
    # print(feature_cols)
    # print(numeric_features)
    # print(categorical_features)

    return numeric_features, categorical_features


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline returning a cleaned dataframe.

    Steps: trim, normalize, parse fields, detect invalids, handle missing, dedupe.
    """
    print('cleaning the dataframe')
    cleaned = df.copy()

    cleaned = trim_whitespace(cleaned)

    cleaned = handle_missing_values(cleaned)

    cleaned = remove_duplicates(cleaned)

    return cleaned

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer with imputation, scaling, and one-hot encoding.

    Parameters
    ----------
    numeric_features : list[str]
        Numeric column names.
    categorical_features : list[str]
        Categorical column names.
    scale_numeric : bool
        If True, apply StandardScaler to numeric features (for linear/SVR models).
        Tree models should set this to False.

    Returns
    -------
    ColumnTransformer
        Fitted-ready preprocessing transformer.
    """
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        # selected_numerical_features= [item for item in numeric_features if (not(item in SCALED_COLUMN))]
        # print("from build preprocessor")
        # print(selected_numerical_features)

        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")

