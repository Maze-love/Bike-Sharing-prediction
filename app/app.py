"""Streamlit application for Bike Sharing Demand Prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_feature_info, load_model,predict_demand

from src.util import (
    FIGURES_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    load_and_explore_data,
    get_data_path
)

st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Home", "Dataset Overview", "EDA", "Model Comparison", "Predict Demand", "About"]


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned dataset produced during training."""
    path = REPORTS_DIR / "cleaned_data.csv"
    if path.exists():
        return pd.read_csv(path)
    from src.feature_engineering import create_engineered_features
    from src.preprocessing import clean_dataframe
    raw = load_and_explore_data()
    return create_engineered_features(clean_dataframe(raw))


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load model comparison metrics."""
    path = METRICS_DIR / "comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def get_model_bundle():
    """Load trained model bundle."""
    try:
        return load_model()
    except FileNotFoundError:
        return None


def render_home():
    st.title("🚲 Bike Sharing Demand Prediction")
    st.markdown(
        """
        Welcome to the **Bike Sharing Demand Prediction** application.

        This end-to-end machine learning project predicts total daily bike rental counts
        (`cnt`) based on environmental, seasonal, and weather conditions such as temperature,
        humidity, windspeed, season, and day of the week.

        ### What you can do
        - Explore the dataset and key statistics
        - View exploratory data analysis visualizations
        - Compare regression model performance
        - **Predict bike demand** with custom weather and date inputs

        Use the sidebar to navigate between pages.
        """
    )

    metrics = load_metrics()
    if not metrics.empty:
        best = metrics.sort_values("rmse").iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", best["model"])
        col2.metric("RMSE", f"{best['rmse']:,.0f} rentals")
        col3.metric("MAE", f"{best['mae']:,.0f} rentals")
        col4.metric("R² Score", f"{best['r2']:.4f}")


def render_dataset_overview():
    st.title("📊 Dataset Overview")
    df = load_cleaned_data()

    st.subheader("Shape & Columns")
    st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
    st.dataframe(pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str).values}))

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in cleaned dataset.")
    else:
        st.bar_chart(missing)


def render_eda():
    st.title("📈 Exploratory Data Analysis")
    figures_dir = FIGURES_DIR

    figure_files = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []

    if not figure_files:
        st.warning("EDA figures not found. Run `python -m src.train` first.")
        return

    for fig_path in figure_files:
        st.subheader(fig_path.stem.replace("_", " ").title())
        st.image(str(fig_path), use_container_width=True)


def render_model_comparison():
    st.title("🏆 Model Comparison")
    metrics = load_metrics()

    if metrics.empty:
        st.warning("Metrics not found. Run `python -m src.train` first.")
        return

    st.subheader("Comparison Table")
    display_df = metrics.sort_values("rmse").copy()
    for col in ["mae", "mse", "rmse"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}")
    display_df["r2"] = display_df["r2"].apply(lambda x: f"{x:.4f}")
    display_df["train_time_sec"] = display_df["train_time_sec"].apply(lambda x: f"{x:.3f}s")
    display_df["predict_time_sec"] = display_df["predict_time_sec"].apply(lambda x: f"{x:.4f}s")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    rmse_path = FIGURES_DIR / "evaluation" / "model_comparison_rmse.png"
    leaderboard_path = FIGURES_DIR / "evaluation" / "model_leaderboard.png"
    if rmse_path.exists():
        col1.image(str(rmse_path), caption="RMSE Comparison", use_container_width=True)
    if leaderboard_path.exists():
        col2.image(str(leaderboard_path), caption="Model Leaderboard", use_container_width=True)

    report_path = REPORTS_DIR / "model_comparison.md"
    if report_path.exists():
        st.subheader("Summary Report")
        st.markdown(report_path.read_text(encoding="utf-8"))


def render_predict():
    st.title("🚲 Predict Rental Demand")

    bundle = get_model_bundle()
    if bundle is None:
        st.error("Model not loaded. Run `python -m src.train` to train and save the model.")
        return

    st.markdown(f"**Active model:** `{bundle['metadata']['model_name']}`")

    try:
        _ = load_feature_info()
    except FileNotFoundError:
        st.error("Feature info not found.")
        return

    with st.form("prediction_form"):
        st.subheader("Date & Weather Conditions")
        col1, col2 = st.columns(2)

        month_map= {
            1:"Januaray", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June",
            7:"July", 8:"August", 9:"September", 10:"October", 11:"November",12:"December"
        }

        season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
        weather_map = {
            1: "Clear / Few Clouds",
            2: "Mist + Cloudy",
            3: "Light Snow / Rain",
            4: "Heavy Rain / Ice Fog",
        }


        with col1:
            selected_year= st.selectbox("Year",[2018,2019,2020,2023])
            # date_input = st.date_input("Date")
            selected_month_label= st.selectbox("Month",list(month_map.values()))

            season_label = st.selectbox("Season", list(season_map.values()))
            weathersit_label = st.selectbox("Weather Condition", list(weather_map.values()))
            holiday_label = st.selectbox("Is Holiday?", ["No", "Yes"])
            workingday_label = st.selectbox("Is Working Day?", ["Yes", "No"])

        with col2:
            # Normalized bike dataset ranges (scaled 0.0 to 1.0)
            temp = st.slider("Normalized Temperature (temp)",min_value=-5.0, max_value=40.0, value=20.0, step=0.5)
            # temp = st.slider("Normalized 'Feels Like' Temp (atemp)", 0.0, 1.0, 0.5, step=0.01)
            hum = st.slider("Normalized Humidity (hum)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
            windspeed = st.slider("Normalized Windspeed", min_value=0.0, max_value=50.0, value=10.0, step=0.5)

        submitted = st.form_submit_button("Predict Rental Count", type="primary")

    if submitted:
        # Map human labels back to raw dataset encoding values
        mnth = [k for k, v in month_map.items() if v == selected_month_label][0]
        season = [k for k, v in season_map.items() if v == season_label][0]
        weathersit = [k for k, v in weather_map.items() if v == weathersit_label][0]

        
        yr = selected_year % 2018

        holiday = 1 if holiday_label == "Yes" else 0
        workingday = 1 if workingday_label == "Yes" else 0

        raw_input = {
            "season": season,
            "yr": yr,
            "mnth": mnth,
            "holiday": holiday,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp": temp,
            "hum": hum,
            "windspeed": windspeed,
            # "bad_weather_index":hum* windspeed,
            # "wind_temp_interaction": windspeed * temp,
        }



        try:
            predicted_cnt = predict_demand(raw_input)
            st.success(f"### Predicted Total Bike Rentals: **{predicted_cnt:.2f} bikes**")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_about():
    st.title("ℹ️ About")
    st.markdown(
        """
        ## Bike Sharing Demand Prediction

        **Objective:** Predict the daily total count of bike rentals (`cnt`) using regression algorithms.

        **Target Variable:** `cnt` (Total Rentals = Casual + Registered)

        ### Tech Stack
        - Python 3.12
        - Pandas, NumPy, Matplotlib, Seaborn
        - Scikit-Learn, Joblib
        - Streamlit

        ### Models Evaluated
        1. Linear Regression
        2. Ridge Regression
        3. Lasso Regression
        4. Decision Tree Regressor
        5. Random Forest Regressor
        6. Gradient Boosting Regressor
        7. Extra Trees Regressor
        8. AdaBoost Regressor
        9. Support Vector Regressor (SVR)

        ### Key Features & Engineering
        - **Date Extractions:** Day of month, quarter, start/end of month
        - **Weather Interaction:** Temperature difference (`atemp - temp`), bad weather index (`hum * windspeed`)

        ### Project Structure
        ```
        bike-sharing-demand/
        ├── data/raw/day.csv
        ├── notebooks/eda.ipynb
        ├── src/
        ├── models/best_model.pkl
        ├── outputs/
        ├── app/app.py
        └── tests/
        ```

        Built as a production-quality end-to-end ML regression project.
        """
    )


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES)

    page_map = {
        "Home": render_home,
        "Dataset Overview": render_dataset_overview,
        "EDA": render_eda,
        "Model Comparison": render_model_comparison,
        "Predict Demand": render_predict,
        "About": render_about,
    }
    page_map[page]()


if __name__ == "__main__":
    main()