"""
Student Performance Prediction System
Regression Model Training and Evaluation Pipeline
Implements:
  1. Simple Linear Regression
  2. Multiple Linear Regression
  3. Polynomial Regression (degree 2)
  4. Ridge Regression (L2 Regularization)
  5. Lasso Regression (L1 Regularization)
  6. Support Vector Regressor (SVR) - Linear Kernel
  7. Support Vector Regressor (SVR) - Polynomial Kernel
  8. Support Vector Regressor (SVR) - RBF Kernel
  9. Decision Tree Regressor
  10. Random Forest Regressor
Includes K-Fold Cross Validation and GridSearchCV hyperparameter tuning.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    set_plot_style,
    save_figure
)

def load_data():
    """Loads processed train and test sets for regression."""
    X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train_reg.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test_reg.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def plot_actual_vs_predicted(y_true, y_pred, model_name, r2, rmse):
    """Generates and saves an Actual vs Predicted scatter plot with reference line."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, color="#2980b9", alpha=0.7, edgecolors="k", s=50, label="Predictions")
    
    # 45-degree reference line
    min_val = min(min(y_true), min(y_pred)) - 1
    max_val = max(max(y_true), max(y_pred)) + 1
    ax.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2, label="Ideal (y = x)")
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title(f"Actual vs Predicted - {model_name}\n$R^2$: {r2:.3f} | RMSE: {rmse:.3f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Exam Score (G3)", fontsize=11)
    ax.set_ylabel("Predicted Exam Score (G3)", fontsize=11)
    ax.legend(loc="upper left")
    
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    filename = f"reg_actual_vs_pred_{safe_name}.png"
    save_figure(fig, filename)
    return filename

def plot_residuals(y_true, y_pred, model_name):
    """Generates and saves a Residual Distribution and Residuals vs Predicted plot."""
    set_plot_style()
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, color="#8e44ad", alpha=0.7, edgecolors="k", s=50)
    axes[0].axhline(0, color="#e74c3c", linestyle="--", linewidth=1.8)
    axes[0].set_title(f"Residuals vs Predicted ({model_name})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Score", fontsize=11)
    axes[0].set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    
    # Residual Distribution
    sns.histplot(residuals, kde=True, color="#27ae60", ax=axes[1], edgecolor="black")
    axes[1].axvline(0, color="#e74c3c", linestyle="--", linewidth=1.8)
    axes[1].set_title(f"Residual Error Distribution ({model_name})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    filename = f"reg_residuals_{safe_name}.png"
    save_figure(fig, filename)
    return filename

def train_all_regressors():
    """
    Trains and tunes all 10 regression algorithms using K-Fold CV and GridSearchCV.
    Returns dictionary of results and best trained model.
    """
    print("=" * 70)
    print("          REGRESSION MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = load_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models_config = {
        "Simple Linear Regression": {
            "type": "simple_linear",
            "features": ["G2"],
            "model": LinearRegression()
        },
        "Multiple Linear Regression": {
            "type": "direct",
            "model": LinearRegression()
        },
        "Polynomial Regression (Degree 2)": {
            "type": "polynomial",
            "features": ["G1", "G2"],
            "model": Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linreg", LinearRegression())
            ])
        },
        "Ridge Regression": {
            "type": "grid",
            "estimator": Ridge(),
            "param_grid": {"alpha": [0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0]}
        },
        "Lasso Regression": {
            "type": "grid",
            "estimator": Lasso(max_iter=10000, random_state=42),
            "param_grid": {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}
        },
        "SVR (Linear Kernel)": {
            "type": "grid",
            "estimator": SVR(kernel="linear"),
            "param_grid": {
                "C": [0.1, 1.0, 10.0],
                "epsilon": [0.05, 0.1, 0.2]
            }
        },
        "SVR (Polynomial Kernel)": {
            "type": "grid",
            "estimator": SVR(kernel="poly"),
            "param_grid": {
                "degree": [2, 3],
                "C": [0.1, 1.0, 10.0],
                "epsilon": [0.05, 0.1, 0.2]
            }
        },
        "SVR (RBF Kernel)": {
            "type": "grid",
            "estimator": SVR(kernel="rbf"),
            "param_grid": {
                "C": [0.1, 1.0, 10.0, 50.0],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "epsilon": [0.05, 0.1, 0.2]
            }
        },
        "Decision Tree Regressor": {
            "type": "grid",
            "estimator": DecisionTreeRegressor(random_state=42),
            "param_grid": {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "Random Forest Regressor": {
            "type": "grid",
            "estimator": RandomForestRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        }
    }
    
    results = []
    trained_models = {}
    
    for name, cfg in models_config.items():
        print(f"\n--> Training: {name}...")
        
        if cfg["type"] == "simple_linear":
            feature_cols = cfg["features"]
            model = cfg["model"]
            cv_scores = cross_val_score(model, X_train[feature_cols], y_train, cv=kf, scoring="r2")
            model.fit(X_train[feature_cols], y_train)
            y_pred = model.predict(X_test[feature_cols])
            best_params = {"feature": feature_cols[0], "intercept": float(model.intercept_), "slope": float(model.coef_[0])}
            final_model = model
            
        elif cfg["type"] == "polynomial":
            feature_cols = cfg["features"]
            model = cfg["model"]
            cv_scores = cross_val_score(model, X_train[feature_cols], y_train, cv=kf, scoring="r2")
            model.fit(X_train[feature_cols], y_train)
            y_pred = model.predict(X_test[feature_cols])
            best_params = {"degree": 2, "features": feature_cols}
            final_model = model
            
        elif cfg["type"] == "direct":
            model = cfg["model"]
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            best_params = {"features": "all_41"}
            final_model = model
            
        elif cfg["type"] == "grid":
            grid = GridSearchCV(cfg["estimator"], cfg["param_grid"], cv=kf, scoring="r2", n_jobs=1)
            grid.fit(X_train, y_train)
            final_model = grid.best_estimator_
            best_params = grid.best_params_
            cv_scores = cross_val_score(final_model, X_train, y_train, cv=kf, scoring="r2")
            y_pred = final_model.predict(X_test)
            
        # Evaluation Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"    Best Params: {best_params}")
        print(f"    5-Fold CV R2: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"    Test R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        
        # Save visualizations
        plot_actual_vs_predicted(y_test, y_pred, name, r2, rmse)
        plot_residuals(y_test, y_pred, name)
        
        # Save model pickle
        safe_key = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        model_filename = f"reg_model_{safe_key}.pkl"
        joblib.dump(final_model, os.path.join(MODELS_DIR, model_filename))
        trained_models[name] = final_model
        
        results.append({
            "Model": name,
            "CV_R2_Mean": round(cv_mean, 4),
            "CV_R2_Std": round(cv_std, 4),
            "Test_R2": round(r2, 4),
            "Test_MAE": round(mae, 4),
            "Test_MSE": round(mse, 4),
            "Test_RMSE": round(rmse, 4),
            "Best_Params": best_params,
            "Model_File": model_filename
        })
        
    df_results = pd.DataFrame(results).sort_values(by="Test_R2", ascending=False).reset_index(drop=True)
    
    # Save metrics
    df_results.to_csv(os.path.join(MODELS_DIR, "regression_metrics.csv"), index=False)
    with open(os.path.join(MODELS_DIR, "regression_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 70)
    print("                    REGRESSION RESULTS SUMMARY")
    print("=" * 70)
    print(df_results[["Model", "Test_R2", "Test_RMSE", "Test_MAE", "CV_R2_Mean"]].to_string(index=False))
    
    # Identify and save best regression model
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_regression_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_regression_meta.json"), "w") as f:
        json.dump(best_row.to_dict(), f, indent=4)
        
    print(f"\n[WINNER] Best Regression Model: '{best_name}' with Test R2 = {best_row['Test_R2']:.4f}")
    print(f"[SAVED] Saved to 'models/best_regression_model.pkl'.")
    return df_results

if __name__ == "__main__":
    train_all_regressors()

