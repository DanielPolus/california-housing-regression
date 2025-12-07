import os
import pandas as pd

import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
)

path = kagglehub.dataset_download("camnugent/california-housing-prices")

print("Path to dataset files:", path)
print("Files in this folder:", os.listdir(path))

data_path = Path(path) / "housing.csv"
df = pd.read_csv(data_path)
df = df.dropna(subset=['total_bedrooms'])

print(f"Dataframe's shape: {df.shape}")
print(f"Dataframe's dtypes: {df.dtypes}")
print(f"Dataframe's head: {df.head()}")
print(f"Dataframe's NaN: {df.isna().mean().sort_values(ascending=False)}")

X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
y = df['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_small, _, y_train_small, _ = train_test_split(
    x_train_scaled,
    y_train,
    random_state=42,
)

log_res = LinearRegression()
param_grid_lin_reg = {
    'fit_intercept': [True, False],
}
grid_lin = GridSearchCV(
    estimator=log_res,
    param_grid=param_grid_lin_reg,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_lin.fit(x_train_small, y_train_small)
y_pred_lin = grid_lin.predict(x_test_scaled)
print(f"LinearRegression R^2: {r2_score(y_test, y_pred_lin)}")
print(f"LinearRegression MSE: {mean_squared_error(y_test, y_pred_lin)}")
print(f"LinearRegression RMSE: {root_mean_squared_error(y_test, y_pred_lin)}\n")


ridge = RidgeCV()
ridge.fit(x_train_scaled, y_train)
y_pred_ridge = ridge.predict(x_test_scaled)
print(f"Ridge R^2: {r2_score(y_test, y_pred_ridge)}")
print(f"Ridge MSE: {mean_squared_error(y_test, y_pred_ridge)}")
print(f"Ridge RMSE: {root_mean_squared_error(y_test, y_pred_ridge)}\n")


lasso = LassoCV()
lasso.fit(x_train_scaled, y_train)
y_pred_lasso = lasso.predict(x_test_scaled)
print(f"Lasso R^2: {r2_score(y_test, y_pred_lasso)}")
print(f"Lasso MSE: {mean_squared_error(y_test, y_pred_lasso)}")
print(f"Lasso RMSE: {root_mean_squared_error(y_test, y_pred_lasso)}\n")


dt = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 10, 50],
    'min_samples_leaf': [1, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
}
grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_dt.fit(x_train_small, y_train_small)
y_pred_dt = grid_dt.predict(x_test_scaled)
print(f"DecisionTree R^2: {r2_score(y_test, y_pred_dt)}")
print(f"DecisionTree MSE: {mean_squared_error(y_test, y_pred_dt)}")
print(f"DecisionTree RMSE: {root_mean_squared_error(y_test, y_pred_dt)}\n")


rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)
param_grid_rf = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt', 'log2'],
}
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(x_train_small, y_train_small)
y_pred_rf = grid_rf.predict(x_test_scaled)
print(f"RandomForest R^2 Report: {r2_score(y_test, y_pred_rf)}")
print(f"RandomForest MSE: {mean_squared_error(y_test, y_pred_rf)}")
print(f"RandomForest RMSE: {root_mean_squared_error(y_test, y_pred_rf)}\n")
best_rf = grid_rf.best_estimator_
print(f"RandomForest feature_importance: {best_rf.feature_importances_}\n")

