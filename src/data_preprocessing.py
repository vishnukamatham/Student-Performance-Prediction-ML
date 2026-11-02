"""
Student Performance Prediction System
Data Preprocessing Pipeline
"""

import os
import sys

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import (
    RAW_DATA_FILE,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    BINARY_COLUMNS,
    NOMINAL_COLUMNS,
    categorize_score
)

def load_raw_data(filepath=RAW_DATA_FILE):
    """Loads raw student performance dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure data/raw contains student_data.csv.")
    df = pd.read_csv(filepath)
    print(f"[INFO] Raw dataset loaded successfully. Shape: {df.shape}")
    return df

def clean_and_encode_features(df):
    """
    Cleans, maps binary features, and one-hot encodes nominal categorical features.
    Returns:
        X (DataFrame): Processed feature matrix
        y_reg (Series): Regression target (G3)
        y_clf (Series): Classification target (Low, Medium, High)
    """
    df = df.copy()
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[WARN] Found {missing_count} missing values. Handling with forward-fill.")
        df = df.ffill()
    else:
        print("[INFO] Zero missing values detected in the dataset.")
        
    # Extract targets
    y_reg = df["G3"].astype(float)
    y_clf = df["G3"].apply(categorize_score)
    
    # Drop target column for features
    X_raw = df.drop(columns=["G3"])
    
    # Map binary features
    X_encoded = X_raw.copy()
    for col, mapping in BINARY_COLUMNS.items():
        if col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].map(mapping)
            
    # One-hot encode nominal columns
    X_encoded = pd.get_dummies(X_encoded, columns=NOMINAL_COLUMNS, drop_first=True, dtype=int)
    
    print(f"[INFO] Feature encoding completed. Processed feature matrix shape: {X_encoded.shape}")
    return X_encoded, y_reg, y_clf

def split_and_scale_data(X, y_reg, y_clf, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets (stratified on classification target),
    scales numerical features with StandardScaler, and saves artifacts.
    """
    # Stratified split to ensure balanced classes in both train and test sets
    X_train_raw, X_test_raw, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf
    )
    
    feature_names = list(X.columns)
    
    # Fit StandardScaler on training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Convert scaled arrays back to DataFrames with feature names
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test_raw.index)
    
    # Save artifacts for inference
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    
    # Save processed CSV files
    X_train_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"), index=False)
    X_train_raw.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_train_unscaled.csv"), index=False)
    X_test_raw.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_test_unscaled.csv"), index=False)
    
    y_train_reg.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_train_reg.csv"), index=False)
    y_test_reg.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_test_reg.csv"), index=False)
    y_train_clf.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_train_clf.csv"), index=False)
    y_test_clf.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_test_clf.csv"), index=False)
    
    print(f"[INFO] Train split: {X_train_df.shape[0]} samples, Test split: {X_test_df.shape[0]} samples.")
    print(f"[INFO] Artifacts saved to '{MODELS_DIR}' and processed data saved to '{DATA_PROCESSED_DIR}'.")
    
    return {
        "X_train": X_train_df,
        "X_test": X_test_df,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf,
        "feature_names": feature_names,
        "scaler": scaler
    }

def run_preprocessing_pipeline():
    """Runs the complete preprocessing workflow."""
    print("=" * 60)
    print("      STUDENT PERFORMANCE PREPROCESSING PIPELINE")
    print("=" * 60)
    df = load_raw_data()
    X, y_reg, y_clf = clean_and_encode_features(df)
    processed_data = split_and_scale_data(X, y_reg, y_clf)
    print("[SUCCESS] Preprocessing completed successfully!\n")
    return processed_data

if __name__ == "__main__":
    run_preprocessing_pipeline()
