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

from src.util import TARGET_COLUMN


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


def get_feature_columns(df: pd.DataFrame)->pd.DataFrame:
    
    # Drop leakage and identifier features
    # 'casual' + 'registered' == 'cnt', which causes strict data leakage
    # features_to_drop = ["instant", "dteday", "casual", "registered"]

    features_to_drop = ["instant", "dteday", "casual", "registered"]
    cleaned_df = df.drop(columns=[col for col in features_to_drop if col in df.columns])

    return cleaned_df

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

    # cleaned=  get_feature_columns (cleaned)

    return cleaned


# =============================================================================
# 2. DATA PREPROCESSING (Split train/test)
# =============================================================================
def preprocess_data(df: pd.DataFrame):
    """
    Cleans data, handles potential leakage columns ('casual', 'registered'),
    removes identifiers, and splits data into train/test sets.
    """
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING & FEATURE SELECTION")
    print("=" * 60)
    
    
    cleaned_df = df

    # Training file
    # Feature & Target separation
    X = cleaned_df.drop(columns=["cnt"])
    y = cleaned_df["cnt"]
    
    # Train / Test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Features matrix shape: {X.shape}")
    print(f"Training split: {X_train.shape[0]} samples")
    print(f"Testing split:  {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test
