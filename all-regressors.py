import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Load Dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

# Decision Tree
dt_param_grid = {'max_depth': [10, None], 'min_samples_split': [2, 5]}
dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_param_grid, cv=2, scoring='neg_mean_squared_error')
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_
y_pred_dt = dt_best.predict(X_test)
results['Decision Tree'] = mean_squared_error(y_test, y_pred_dt)

# Random Forest
rf_param_grid = {'n_estimators': [50], 'max_depth': [10, None]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=2, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
results['Random Forest'] = mean_squared_error(y_test, y_pred_rf)

# AdaBoost
ab_param_grid = {'n_estimators': [50], 'learning_rate': [0.1]}
ab_grid = GridSearchCV(AdaBoostRegressor(random_state=42), ab_param_grid, cv=2, scoring='neg_mean_squared_error')
ab_grid.fit(X_train, y_train)
ab_best = ab_grid.best_estimator_
y_pred_ab = ab_best.predict(X_test)
results['AdaBoost'] = mean_squared_error(y_test, y_pred_ab)

# XGBoost
xgb_param_grid = {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [3, 5]}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42, objective='reg:squarederror'), xgb_param_grid, cv=2, scoring='neg_mean_squared_error')
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
results['XGBoost'] = mean_squared_error(y_test, y_pred_xgb)

# CatBoost
cat_param_grid = {'iterations': [100], 'learning_rate': [0.1], 'depth': [6]}
cat_grid = GridSearchCV(CatBoostRegressor(verbose=0, random_state=42), cat_param_grid, cv=2, scoring='neg_mean_squared_error')
cat_grid.fit(X_train, y_train)
cat_best = cat_grid.best_estimator_
y_pred_cat = cat_best.predict(X_test)
results['CatBoost'] = mean_squared_error(y_test, y_pred_cat)

# Compare and Save Best Model
best_model_name = min(results, key=results.get)
print(f"Best Model: {best_model_name} with MSE = {results[best_model_name]:.4f}")

if best_model_name == 'Decision Tree':
    best_model = dt_best
elif best_model_name == 'Random Forest':
    best_model = rf_best
elif best_model_name == 'AdaBoost':
    best_model = ab_best
elif best_model_name == 'XGBoost':
    best_model = xgb_best
elif best_model_name == 'CatBoost':
    best_model = cat_best

with open('best_regression_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Best model saved as 'best_regression_model.pkl'")
