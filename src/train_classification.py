"""
Student Performance Prediction System
Classification Model Training and Evaluation Pipeline
Implements:
  1. Support Vector Classifier (SVC) - Linear Kernel
  2. Support Vector Classifier (SVC) - Polynomial Kernel
  3. Support Vector Classifier (SVC) - RBF Kernel
  4. Decision Tree Classifier
  5. Random Forest Classifier
Includes Stratified K-Fold Cross Validation and GridSearchCV hyperparameter tuning.
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

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.utils import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    CLASS_NAMES,
    set_plot_style,
    save_figure
)

def load_data():
    """Loads processed train and test sets for classification."""
    X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train_clf.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test_clf.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm, model_name, acc, f1):
    """Generates and saves a publication-quality Confusion Matrix heatmap."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    # Calculate percentage annotations
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{c}\n({p:.1f}%)"
                
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                linewidths=1.2, linecolor="white", annot_kws={"fontsize": 11, "fontweight": "bold"})
    
    ax.set_title(f"Confusion Matrix - {model_name}\nAccuracy: {acc*100:.2f}% | F1-Macro: {f1:.3f}", 
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    filename = f"clf_confusion_matrix_{safe_name}.png"
    save_figure(fig, filename)
    return filename

def plot_feature_importance(importances, feature_names, model_name, top_n=15):
    """Generates and saves a bar chart of top N feature importances."""
    set_plot_style()
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = sns.barplot(data=df_imp, x="Importance", y="Feature", palette="viridis", ax=ax, edgecolor="black")
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gini / Mean Decrease in Impurity", fontsize=11)
    ax.set_ylabel("Feature Name", fontsize=11)
    
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    filename = f"clf_feature_importance_{safe_name}.png"
    save_figure(fig, filename)
    return filename

def train_all_classifiers():
    """
    Trains and tunes all 5 classification algorithms using Stratified K-Fold CV and GridSearchCV.
    Returns summary DataFrame and saves the best model.
    """
    print("=" * 70)
    print("        CLASSIFICATION MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = load_data()
    feature_names = list(X_train.columns)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_config = {
        "SVC (Linear Kernel)": {
            "estimator": SVC(kernel="linear", probability=True, random_state=42),
            "param_grid": {
                "C": [0.1, 1.0, 10.0]
            }
        },
        "SVC (Polynomial Kernel)": {
            "estimator": SVC(kernel="poly", probability=True, random_state=42),
            "param_grid": {
                "degree": [2, 3],
                "C": [0.1, 1.0, 10.0]
            }
        },
        "SVC (RBF Kernel)": {
            "estimator": SVC(kernel="rbf", probability=True, random_state=42),
            "param_grid": {
                "C": [0.1, 1.0, 10.0, 20.0],
                "gamma": ["scale", "auto", 0.01, 0.1]
            }
        },
        "Decision Tree Classifier": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 8, 12, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "Random Forest Classifier": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        }
    }
    
    results = []
    trained_models = {}
    
    for name, cfg in models_config.items():
        print(f"\n--> Training: {name}...")
        grid = GridSearchCV(cfg["estimator"], cfg["param_grid"], cv=skf, scoring="accuracy", n_jobs=1)
        grid.fit(X_train, y_train)
        
        final_model = grid.best_estimator_
        best_params = grid.best_params_
        cv_scores = cross_val_score(final_model, X_train, y_train, cv=skf, scoring="accuracy")
        y_pred = final_model.predict(X_test)
        
        # Performance Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
        report = classification_report(y_test, y_pred, labels=CLASS_NAMES, output_dict=True, zero_division=0)
        
        print(f"    Best Params: {best_params}")
        print(f"    5-Fold CV Accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
        print(f"    Test Accuracy: {acc*100:.2f}% | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")
        
        # Save Confusion Matrix figure
        plot_confusion_matrix(cm, name, acc, f1_macro)
        
        # Save Feature Importance if tree-based
        if hasattr(final_model, "feature_importances_"):
            plot_feature_importance(final_model.feature_importances_, feature_names, name)
            
        # Save model pickle
        safe_key = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        model_filename = f"clf_model_{safe_key}.pkl"
        joblib.dump(final_model, os.path.join(MODELS_DIR, model_filename))
        trained_models[name] = final_model
        
        results.append({
            "Model": name,
            "CV_Accuracy_Mean": round(cv_mean, 4),
            "CV_Accuracy_Std": round(cv_std, 4),
            "Test_Accuracy": round(acc, 4),
            "Precision_Macro": round(prec_macro, 4),
            "Recall_Macro": round(rec_macro, 4),
            "F1_Macro": round(f1_macro, 4),
            "Precision_Weighted": round(prec_weighted, 4),
            "Recall_Weighted": round(rec_weighted, 4),
            "F1_Weighted": round(f1_weighted, 4),
            "Confusion_Matrix": cm.tolist(),
            "Best_Params": best_params,
            "Classification_Report": report,
            "Model_File": model_filename
        })
        
    df_results = pd.DataFrame(results).sort_values(by="Test_Accuracy", ascending=False).reset_index(drop=True)
    
    # Save metrics
    df_results.to_csv(os.path.join(MODELS_DIR, "classification_metrics.csv"), index=False)
    with open(os.path.join(MODELS_DIR, "classification_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 70)
    print("                    CLASSIFICATION RESULTS SUMMARY")
    print("=" * 70)
    print(df_results[["Model", "Test_Accuracy", "F1_Macro", "Precision_Macro", "Recall_Macro", "CV_Accuracy_Mean"]].to_string(index=False))
    
    # Identify and save best classification model
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_classification_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_classification_meta.json"), "w") as f:
        json.dump(best_row.to_dict(), f, indent=4)
        
    print(f"\n[WINNER] Best Classification Model: '{best_name}' with Test Accuracy = {best_row['Test_Accuracy']*100:.2f}%")
    print(f"[SAVED] Saved to 'models/best_classification_model.pkl'.")
    return df_results

if __name__ == "__main__":
    train_all_classifiers()
