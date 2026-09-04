# Bike Sharing Model Comparison Report

**Best Model:** Random Forest Regressor

## Selection Criteria

Selected via composite score: RMSE (40%), MAE (35%), R² (15%), Train Time (10%).

## Leaderboard

| Rank | Model | RMSE | MAE | R² | Train Time (s) | Predict Time (s) |
|------|-------|------|-----|----|----------------|------------------|
| 1 | Random Forest Regressor ⭐ | 542.31 | 413.93 | 0.9141 | 0.707 | 0.1084 |
| 2 | Support Vector Regressor (SVR) | 559.11 | 414.79 | 0.9087 | 0.088 | 0.0265 |
| 3 | Gradient Boosting Regressor | 559.27 | 431.41 | 0.9086 | 0.573 | 0.0054 |
| 4 | Extra Trees | 568.73 | 418.97 | 0.9055 | 0.257 | 0.0470 |
| 5 | Decision Tree Regressor | 719.81 | 539.56 | 0.8486 | 0.031 | 0.0117 |
| 6 | Lasso Regression | 739.38 | 575.03 | 0.8403 | 0.030 | 0.0117 |
| 7 | Ridge Regression | 739.38 | 574.93 | 0.8403 | 0.054 | 0.0082 |
| 8 | Linear Regression | 739.70 | 575.35 | 0.8401 | 0.116 | 0.0047 |
| 9 | AdaBoost | 784.63 | 628.39 | 0.8201 | 0.238 | 0.0323 |

## Summary

The **Random Forest Regressor** model achieved the highest overall score (RMSE=542.31, MAE=413.93, R²=0.9141).
