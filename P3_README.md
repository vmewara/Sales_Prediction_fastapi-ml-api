# Superstore Sales Prediction

## Overview

An end-to-end regression project to predict Sales from the Sample Superstore dataset. The project covers data loading, exploratory data analysis, log transformation, feature engineering, model training, evaluation, and model export.

---

## Project Details

| Item | Detail |
|------|--------|
| **Problem Type** | Regression |
| **Target Variable** | Sales (log-transformed via np.log1p) |
| **Dataset** | Sample - Superstore.csv (encoding: latin) |
| **Best Model** | XGBoost Regressor |
| **Saved Models** | trained_pipeline-0.1.0.pkl (pickle) and xgb_model.pkl (joblib) |
| **Train / Test Split** | 80% / 20% — random_state=42 |

---

## Model Comparison (Recorded Results)

| Model | R² Score | MAE |
|-------|----------|-----|
| Linear Regression | 0.93 | 0.28 |
| XGBoost Regressor ✅ Best | — | — |

> Linear Regression R² 0.93 and MAE 0.28 recorded from execution.
> XGBoost Regressor trained and saved as the final deployment model.

---

## XGBoost Configuration

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)
```

---

## Pipeline

```
Sample - Superstore.csv (latin encoding)
  └── EDA (describe, isnull, duplicates, skew, value_counts)
        └── Log-transform Sales using np.log1p (reduces right skew)
              └── Visualisation (boxplot, histplot, heatmap, barplot, scatterplot)
                    └── Feature split: numerical (StandardScaler) + categorical (OneHotEncoder)
                          └── ColumnTransformer → Pipeline
                                └── Train/Test Split (80/20)
                                      └── XGBRegressor → Evaluate (R², MAE, RMSE) → Export .pkl
```

---

## Key EDA Findings

- Sales was strongly right-skewed — log-transformed before modelling
- Profit contains high positive and negative outliers
- Higher discounts negatively correlate with profit
- Higher discounts associate with increased order quantity

---

## Setup

```bash
pip install -r requirements.txt
```

Open the notebook in Google Colab or Jupyter and run cells top to bottom.

---

## File Structure

```
├── Copy_of_Welcome_To_Colab__3_.ipynb     # Main notebook
├── Sample - Superstore.csv                # Dataset (latin encoding)
├── trained_pipeline-0.1.0.pkl             # Saved model (pickle)
├── xgb_model.pkl                          # Saved model (joblib)
├── requirements.txt
└── README.md
```

---

## Notes

- Dataset must be loaded with encoding='latin' due to special characters in the CSV
- Sales predictions should be inverse-transformed using np.expm1() for real-world values
- Both pickle and joblib exports available — joblib preferred for large models
- Linear Regression R² 0.93 shows strong linear relationship in the data
