# 🚲 Bike Sharing Demand — Regression Project (Group 5)

**Course:** Intelligent AI and Data Engineering
**Assignment:** Model Comparison, Regression Prediction, and Web Application
**Group:** 5
**Dataset:** Bike Sharing Demand (`day.csv`) — UCI Machine Learning Repository
**Target Variable:** `cnt` (total daily bike rentals)

## 📌 Business Problem

A bike-sharing company needs to forecast daily rental demand to plan inventory,
staffing, and bike redistribution across stations. This is a regression problem
because the target (`cnt`) is a continuous numeric count.

## 📂 Repository Structure

```
Group-5 Bike_Sharing/
├── notebook/
│   └── bsp.ipynb                               # Full ML pipeline (EDA -> Model -> Evaluation)
├── app/
│   ├── app.py                                  # Streamlit web application
├── models/                                     # Saved model artifacts (source copies)
|   |__ best_model.joblib                             # Trained best model
|   |__ feature_info                                  # feature_info
|   |__ model_metadata                                # Model data
├── outputs/                                     # EDA charts, comparison charts, results
├── slides/                                      # Presentation slides (.pptx)
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run the Notebook

```bash
jupyter notebook notebook/Group5_Bike_Sharing_Regression.ipynb
```

## ▶️ Run the Web Application

```bash
cd webapp
streamlit run app/app.py
```

Then open the local URL Streamlit prints (default: http://localhost:8501).

## 🧠 Machine Learning Workflow

1. **Data Understanding** — explored all 16 columns, identified `cnt` as target
2. **Preprocessing** — checked for missing values/duplicates (none found), removed
   `casual`/`registered` to prevent data leakage (they sum exactly to `cnt`), dropped `instant`
3. **Feature Engineering** — cyclical (sin/cos) encoding for month and weekday, day-of-month extraction
4. **EDA** — histograms, boxplots, scatter plots, correlation heatmap, time trend, pair plot
5. **Model Training** — 8 regression algorithms:
   Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR, XGBoost (bonus)
6. **Evaluation** — MAE, MSE, RMSE, R² for every model
7. **Hyperparameter Tuning** — GridSearchCV (5-fold) on the top ensemble model
8. **Bonus additions** — cross-validation, learning curve, residual analysis, feature importance
9. **Model Selection** — best model saved via `joblib`
10. **Web Application** — Streamlit app with Home / Predict / Model Comparison / About Team pages

## 🏆 Results Summary

See `outputs/metrics/comparison.csv` and `model/model_metadata.json` for exact figures.
The best model was selected by highest R² Score on the held-out test set (20% split),
cross-validated with 5-fold CV.

## 👥 Team — Group 5

_Add group member names and roles here before submission._

## 📄 License

Academic project — for coursework submission only.
