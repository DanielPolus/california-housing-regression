# California Housing Regression

Predicting median house prices in California districts using linear models and tree-based ensembles, with evaluation via RMSE/R² and feature importance analysis.

## Dataset

- **Source:** Kaggle – [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Rows:** 20,433 (после удаления строк с NaN в `total_bedrooms`)
- **Features (used):**
  - `longitude`, `latitude`
  - `housing_median_age`
  - `total_rooms`, `total_bedrooms`
  - `population`
  - `households`
  - `median_income`
- **Target:** `median_house_value`
- **Dropped:**  
  - `ocean_proximity` (категориальный признак, не кодировался в этой версии проекта)

## Preprocessing

- Removed rows with missing values in `total_bedrooms`.
- Train/test split: 80% / 20%, `random_state=42`.
- Numerical features scaled with `StandardScaler` (for linear/regularized models).

## Models

Classical regression models on tabular data:

- **Linear Regression** (baseline)
- **Ridge Regression**
- **Lasso Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

Hyperparameters for tree-based models were tuned with `GridSearchCV` (3-fold CV) using a regression scoring metric.

## Results (test set)

Main metrics: coefficient of determination (R²) and error in absolute price units (MSE / RMSE).

| Model              | R²    | MSE          | RMSE   |
|--------------------|-------|--------------|--------|
| Linear Regression  | 0.640 | 4.93e9       | 70,211 |
| Ridge Regression   | 0.640 | 4.92e9       | 70,163 |
| Lasso Regression   | 0.640 | 4.93e9       | 70,181 |
| Decision Tree      | 0.742 | 3.53e9       | 59,415 |
| Random Forest      | 0.806 | 2.66e9       | 51,530 |

**Summary:**

- Linear and regularized models explain ~64% of variance with RMSE ≈ 70k USD.
- A single decision tree captures non-linear relationships and improves RMSE to ≈ 59k USD.
- Random Forest performs best, achieving **R² ≈ 0.81** and **RMSE ≈ 51k USD**, significantly reducing prediction error compared to linear models.

## Feature Importance (Random Forest)

Random Forest feature importances highlight the most influential predictors:

- **median_income** — strongest predictor of house prices.
- **longitude, latitude** — location has a major effect (coastal vs inland, regional clusters).
- **housing_median_age, population, total_rooms, households** — contribute to refinement of predictions.

This matches intuition: wealthier areas and specific geographic regions tend to have higher house prices.

## How to Run

```bash
git clone https://github.com/DanielPolus/california-housing-regression.git
cd california-housing-regression

# (optional) create and activate a virtual environment

pip install -r requirements.txt
python main.py
