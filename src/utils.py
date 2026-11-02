"""
Student Performance Prediction System
Utility functions and configuration constants.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "student_data.csv")

# Ensure output directories exist
for path in [DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(path, exist_ok=True)

# Performance Category Definitions
CLASS_NAMES = ["Low", "Medium", "High"]
CLASS_DESCRIPTIONS = {
    "Low": "Score 0 - 9 (At-Risk / Needs Academic Intervention)",
    "Medium": "Score 10 - 13 (Average / Satisfactory Performance)",
    "High": "Score 14 - 20 (Good / High Academic Achievement)"
}

def categorize_score(score):
    """
    Categorizes final exam score (G3: 0-20) into performance classes.
    """
    if score < 10:
        return "Low"
    elif score <= 13:
        return "Medium"
    else:
        return "High"

# Binary feature mappings
BINARY_COLUMNS = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
    "schoolsup": {"no": 0, "yes": 1},
    "famsup": {"no": 0, "yes": 1},
    "paid": {"no": 0, "yes": 1},
    "activities": {"no": 0, "yes": 1},
    "nursery": {"no": 0, "yes": 1},
    "higher": {"no": 0, "yes": 1},
    "internet": {"no": 0, "yes": 1},
    "romantic": {"no": 0, "yes": 1}
}

NOMINAL_COLUMNS = ["Mjob", "Fjob", "reason", "guardian"]

NUMERICAL_COLUMNS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2"
]

def set_plot_style():
    """Sets a clean, modern aesthetic for matplotlib and seaborn visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#eaeaea"
    plt.rcParams["grid.linestyle"] = "--"

def save_figure(fig, filename):
    """Saves a matplotlib figure to the reports/figures directory with high DPI."""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filepath

def preprocess_single_input(input_dict, feature_columns_path=None, scaler_path=None):
    """
    Transforms a single student input dictionary into preprocessed, scaled feature vector
    ready for model inference.
    """
    if feature_columns_path is None:
        feature_columns_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    if scaler_path is None:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    
    feature_columns = joblib.load(feature_columns_path)
    scaler = joblib.load(scaler_path)
    
    df_single = pd.DataFrame([input_dict])
    
    # Map binary features
    for col, mapping in BINARY_COLUMNS.items():
        if col in df_single.columns:
            df_single[col] = df_single[col].map(mapping)
    
    # One-hot encode nominal features
    df_encoded = pd.get_dummies(df_single, columns=NOMINAL_COLUMNS, drop_first=True, dtype=int)
    
    # Reindex columns to match training feature columns exactly
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[feature_columns]
    
    # Scale features
    scaled_array = scaler.transform(df_encoded)
    return pd.DataFrame(scaled_array, columns=feature_columns)

