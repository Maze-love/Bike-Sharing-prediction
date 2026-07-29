"""Feature engineering for bike sharing prediction."""

from __future__ import annotations

import pandas as pd

# feature_engineering.py
import numpy as np

def remove_columns(df: pd.DataFrame)->pd.DataFrame:
    
    # Drop leakage and identifier features
    # 'casual' + 'registered' == 'cnt', which causes strict data leakage
    # features_to_drop = ["instant", "dteday", "casual", "registered"]

    features_to_drop = ["instant", "dteday", "casual", "registered","atemp"]
    cleaned_df = df.drop(columns=[col for col in features_to_drop if col in df.columns])

    return cleaned_df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar-based features from the date column."""
    df_out = df.copy()
    
    if "dteday" in df_out.columns:
        df_out["dteday"] = pd.to_datetime(df_out["dteday"], format="%d-%m-%Y")
        # df_out["day_of_month"] = df_out["dteday"].dt.day
        # df_out["quarter"] = df_out["dteday"].dt.quarter
        df_out["is_month_start"] = df_out["dteday"].dt.is_month_start.astype(int)
        df_out["is_month_end"] = df_out["dteday"].dt.is_month_end.astype(int)
        
    return df_out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates temperature difference and weather interaction metrics."""
    df_out = df.copy()

    # 1. Difference between 'feels like' temp and real temp
    if "temp" in df_out.columns and "atemp" in df_out.columns:
        df_out["temp_diff"] = df_out["atemp"] - df_out["temp"]

    # 2. Combined bad weather metric (Humidity * Windspeed)
    if "hum" in df_out.columns and "windspeed" in df_out.columns:
        df_out["bad_weather_index"] = df_out["hum"] * df_out["windspeed"]

    # 3. Wind chill effect / Thermal interaction (Windspeed * Temperature)
    if "windspeed" in df_out.columns and "temp" in df_out.columns:
        df_out["wind_temp_interaction"] = df_out["windspeed"] * df_out["temp"]

    return df_out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Sine/Cosine transformations to cyclical features (Month, Weekday).
    This helps linear models recognize that December (12) connects to January (1).
    """
    df_out = df.copy()

    if "mnth" in df_out.columns:
        df_out["mnth_sin"] = np.sin(2 * np.pi * df_out["mnth"] / 12.0)
        df_out["mnth_cos"] = np.cos(2 * np.pi * df_out["mnth"] / 12.0)

    if "weekday" in df_out.columns:
        df_out["weekday_sin"] = np.sin(2 * np.pi * df_out["weekday"] / 7.0)
        df_out["weekday_cos"] = np.cos(2 * np.pi * df_out["weekday"] / 7.0)

    return df_out


def add_temperature_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Categorizes continuous temperature into 4 discrete buckets."""
    df_out = df.copy()

    if "temp" in df_out.columns:
        df_out["temp_bin"] = pd.cut(
            df_out["temp"],
            bins=[-np.inf, 10, 20, 30, np.inf],
            labels=[0, 1, 2, 3]  # 0: Cold, 1: Mild, 2: Warm, 3: Hot
        ).astype(int)

    return df_out


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function that runs all feature engineering steps sequentially.
    """
    df_transformed = df.copy()
    
    # Run individual transformation functions
    df_transformed = add_temperature_bins(df_transformed)
    df_transformed = add_weather_features(df_transformed)

    df_transformed = add_date_features(df_transformed)
    df_transformed= remove_columns(df_transformed)
    # df_transformed = add_cyclical_features(df_transformed)
    
    return df_transformed