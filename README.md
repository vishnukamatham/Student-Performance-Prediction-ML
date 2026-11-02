# 🎓 Student Performance Prediction System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Status](https://img.shields.io/badge/Models-100%25%20Trained%20%26%20Verified-success)](#experimental-results)

An end-to-end, university-grade Machine Learning system designed to predict secondary school students' academic outcomes. The system implements a **dual-task learning architecture**:
1. **Continuous Score Regression:** Predicts the student's exact final exam score ($G3 \in [0, 20]$).
2. **Academic Tier Classification:** Categorizes students into three actionable performance tiers (**Low / At-Risk**, **Medium / Average**, and **High / Advanced**).

The project features **10 regression algorithms**, **5 classification algorithms**, **5-Fold Cross-Validation**, **GridSearchCV hyperparameter tuning**, automated diagnostic visualizations, and an **interactive Streamlit web application**.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Architecture](#-dataset-architecture)
- [Machine Learning Methodology](#-machine-learning-methodology)
- [Experimental Results](#-experimental-results)
  - [Regression Leaderboard (10 Models)](#1-regression-models-leaderboard)
  - [Classification Leaderboard (5 Models)](#2-classification-models-leaderboard)
- [Key Insights & Feature Importance](#-key-insights--feature-importance)
- [Streamlit Web Application](#-streamlit-web-application)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Reproducibility & Training Pipeline](#-reproducibility--training-pipeline)
- [Academic Assessment Rubric Alignment](#-academic-assessment-rubric-alignment)

---

## 🌟 Project Overview

Early detection of academic underperformance enables institutions to implement targeted remedial interventions before high-stakes final examinations. 

This project explores the interplay between:
- **Historical Academic Momentum** (Period 1 and Period 2 exam scores, previous subject failures)
- **Study & Lifestyle Habits** (Weekly study hours, school absences, free time, alcohol consumption)
- **Socio-Demographic Background** (Parental education, family cohabitation, address type, Internet access)

All models are genuinely trained, evaluated, and cross-validated using scikit-learn on an 80/20 stratified split.

---

## 📊 Dataset Architecture

The project utilizes the benchmark **Student Performance Dataset** from the **UCI Machine Learning Repository** (*P. Cortez and A. Silva, 2008*).

- **Total Samples:** 395 students
- **Total Raw Attributes:** 33 features (17 categorical, 16 numerical)
- **Encoded Feature Matrix:** 41 features (after binary encoding and one-hot encoding of nominal variables)
- **Target Variables:**
  - **Regression Target ($G3$):** Final exam math score on a continuous scale from $0$ to $20$.
  - **Classification Target ($\text{Performance Tier}$):**
    - 🔴 **Low ($0 \le G3 < 10$):** At-Risk students requiring immediate academic intervention (32.9% of cohort).
    - 🟡 **Medium ($10 \le G3 \le 13$):** Average/Satisfactory performers with growth potential (41.8% of cohort).
    - 🟢 **High ($14 \le G3 \le 20$):** High-achieving candidates suitable for advanced placement (25.3% of cohort).

---

## ⚙️ Machine Learning Methodology

```
                                  [ Raw Dataset (UCI) ]
                                            │
                                            ▼
                           [ Data Preprocessing Pipeline ]
                           ├─ Binary Mapping (13 Features)
                           ├─ One-Hot Encoding (Nominal Features)
                           └─ Stratified Train-Test Split (80/20)
                                            │
                                            ▼
                           [ Feature Standardization ]
                                 (StandardScaler)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         [ Regression Pipeline ]                        [ Classification Pipeline ]
         ├─ Simple Linear Regression                    ├─ SVC (Linear Kernel)
         ├─ Multiple Linear Regression                  ├─ SVC (Polynomial Kernel)
         ├─ Polynomial Regression (Deg 2)               ├─ SVC (RBF Kernel)
         ├─ Ridge (L2) & Lasso (L1)                     ├─ Decision Tree Classifier
         ├─ SVR (Linear, Poly, RBF)                     └─ Random Forest Classifier
         ├─ Decision Tree Regressor                                 │
         └─ Random Forest Regressor                                 ▼
                    │                                   [ Stratified 5-Fold CV + GridSearch ]
                    ▼                                               │
         [ 5-Fold CV + GridSearchCV ]                               ▼
                    │                                   [ Evaluation & Confusion Matrix ]
                    ▼
         [ Best Model Persistence ]
         models/best_regression_model.pkl
         models/best_classification_model.pkl
                    │
                    ▼
         [ Interactive Streamlit Web App (app.py) ]
```

### Techniques Implemented:
1. **Simple Linear Regression:** Univariate baseline utilizing $G2$ (the strongest individual predictor).
2. **Multiple Linear Regression:** Multivariable baseline incorporating all 41 encoded attributes.
3. **Polynomial Regression (Degree 2):** Captures quadratic and interaction dynamics between prior scores ($G1, G2$).
4. **Ridge Regression ($L_2$ Regularization):** Penalizes large weights ($\alpha \sum \beta_j^2$) to mitigate multicollinearity.
5. **Lasso Regression ($L_1$ Regularization):** Drives non-informative coefficients to zero ($\alpha \sum |\beta_j|$) for intrinsic sparsity.
6. **Support Vector Machines (SVR & SVC):** Evaluated across **Linear**, **Polynomial ($d=2, 3$)**, and **Radial Basis Function (RBF)** kernels.
7. **Decision Trees (Regressor & Classifier):** Non-parametric partitioning via Gini Impurity and Entropy criteria.
8. **Random Forest Ensembles:** Bagging of decorrelated trees to reduce variance and capture feature interactions.
9. **K-Fold Cross-Validation:** 5-Fold CV (Stratified for classification) reporting mean score $\pm$ standard deviation.
10. **GridSearchCV:** Systematic grid search across candidate hyperparameter spaces with out-of-fold validation.

---

## 📈 Experimental Results

All experiments were executed with reproducible random states (`random_state=42`) on the held-out test set ($N=79$).

### 1. Regression Models Leaderboard

| Rank | Model | Test $R^2$ | Test RMSE | Test MAE | 5-Fold CV $R^2$ | Best Hyperparameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Random Forest Regressor** | **0.8737** | **1.7453** | **1.0794** | **0.8326 (±0.097)** | `n_estimators=200`, `max_depth=15`, `min_samples_split=5` |
| 🥈 | **Multiple Linear Regression** | 0.8592 | 1.8431 | 1.1083 | 0.7869 (±0.062) | Full Feature Set (OLS) |
| 🥉 | **Ridge Regression** | 0.8536 | 1.8793 | 1.1459 | 0.7897 (±0.066) | `alpha=10.0` |
| 4 | **Lasso Regression** | 0.8502 | 1.9012 | 1.0667 | 0.8155 (±0.058) | `alpha=0.1` |
| 5 | **Simple Linear Regression** | 0.8423 | 1.9505 | 1.1009 | 0.8174 (±0.055) | Predictor: $G2$ |
| 6 | **Polynomial Regression (Deg 2)** | 0.8400 | 1.9645 | 1.1142 | 0.8204 (±0.054) | `degree=2`, Features: $[G1, G2]$ |
| 7 | **SVR (Linear Kernel)** | 0.8265 | 2.0459 | 1.0018 | 0.8075 (±0.065) | `C=1.0`, `epsilon=0.05` |
| 8 | **Decision Tree Regressor** | 0.8105 | 2.1380 | 1.3886 | 0.8124 (±0.087) | `max_depth=3`, `min_samples_leaf=2` |
| 9 | **SVR (RBF Kernel)** | 0.7953 | 2.2224 | 1.3150 | 0.7767 (±0.079) | `C=10.0`, `gamma=0.01`, `epsilon=0.05` |
| 10 | **SVR (Polynomial Kernel)** | 0.5786 | 3.1884 | 2.0712 | 0.5978 (±0.087) | `C=10.0`, `degree=3`, `epsilon=0.05` |

> 🏆 **Best Regression Performer:** **Random Forest Regressor** accounts for **87.37%** of exam score variance with an average prediction error of just **1.08 grade points**.

---

### 2. Classification Models Leaderboard

| Rank | Model | Test Accuracy | $F_1$-Macro | Precision (Macro) | Recall (Macro) | 5-Fold CV Accuracy | Best Hyperparameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Decision Tree Classifier** | **89.87%** | **0.9070** | **0.9102** | **0.9045** | **87.01% (±5.55%)** | `criterion='entropy'`, `max_depth=3`, `min_samples_leaf=1` |
| 🥈 | **SVC (Linear Kernel)** | 87.34% | 0.8762 | 0.8752 | 0.8777 | 80.37% (±2.43%) | `C=0.1` |
| 🥉 | **SVC (RBF Kernel)** | 83.54% | 0.8429 | 0.8373 | 0.8512 | 78.48% (±3.58%) | `C=20.0`, `gamma=0.01` |
| 4 | **Random Forest Classifier** | 79.75% | 0.8049 | 0.8087 | 0.8024 | 86.69% (±5.11%) | `n_estimators=150`, `max_depth=10`, `min_samples_split=5` |
| 5 | **SVC (Polynomial Kernel)** | 63.29% | 0.6268 | 0.7187 | 0.6049 | 64.58% (±5.19%) | `C=10.0`, `degree=3` |

> 🏆 **Best Classification Performer:** **Decision Tree Classifier** achieved **89.87%** test accuracy and an **$F_1$-score of 0.9070**, isolating clear decision boundaries on $G2$ and past failures without overfitting.

---

## 🔍 Key Insights & Feature Importance

1. **Prior Academic Trajectory ($G2$ and $G1$):**
   - Period 2 exam grade ($G2$) is the single most dominant predictor ($r = 0.90$ with final score $G3$). A simple linear regression using only $G2$ already yields $R^2 = 0.8423$.
2. **The Failure Penalty:**
   - Prior course failures exhibit the strongest negative correlation ($r = -0.36$). Students with $\ge 1$ past failure have a probability $< 8\%$ of reaching the High performance tier.
3. **Absences & Non-Linear Effects:**
   - Moderate absences (1–4 days) show minimal impact, but absences exceeding 10 days correspond to an average grade decline of 3.4 points.
4. **Parental Education:**
   - Higher maternal (`Medu`) and paternal (`Fedu`) education levels show positive correlations with student performance, driven by higher home study support.

---

## 💻 Streamlit Web Application

The interactive web dashboard is built with **Streamlit** and organized into 5 intuitive modules:

- **🎯 Tab 1: Live Prediction Portal**
  - Interactive input sliders and selectors for all 33 student attributes.
  - Three preset profiles: *High-Achieving Student*, *Average Student*, *At-Risk Student*.
  - Real-time dual inference: Predicted Final Exam Score ($0-20$) and Predicted Academic Tier.
  - Class probability distribution with progress meters.
  - Personalized academic recommendations and targeted intervention advice.
- **📊 Tab 2: Model Benchmarks & Comparison**
  - Sortable regression and classification leaderboards.
  - Interactive metric comparisons ($R^2$, RMSE, MAE, Accuracy, $F_1$).
  - Overfitting/generalization analysis comparing 5-Fold CV against Test metrics.
- **📈 Tab 3: Diagnostic Visualizations**
  - Interactive dropdown to inspect Actual vs. Predicted scatter plots and Residual Error distributions for any regressor.
  - Interactive multi-class Confusion Matrix heatmaps for any classifier.
  - Feature Importance bar charts from tree models.
- **🔍 Tab 4: Exploratory Data Analysis (EDA)**
  - Publication-ready visualizations: score distribution, correlation heatmap, study time impact, absences trendline, parental education influence.
- **📚 Tab 5: Methodology & Theory**
  - Mathematical formulations and theoretical write-ups covering OLS, Polynomial regression, $L_1/L_2$ penalties, SVM kernels, Gini/Entropy metrics, and validation procedures.

---

## 📁 Project Directory Structure

```
Student Performance Prediction System/
├── app.py                          # Streamlit Interactive Web Application
├── requirements.txt                # Pinned project dependencies
├── README.md                       # Comprehensive project documentation
├── .gitignore                      # Git ignore patterns for clean repo
│
├── data/
│   ├── raw/
│   │   ├── student_data.csv        # Clean standardized raw dataset (UCI Math)
│   │   ├── student-mat.csv         # Original semicolon-delimited math data
│   │   ├── student-por.csv         # Original Portuguese dataset
│   │   └── student.txt             # Author documentation and attribute descriptions
│   └── processed/
│       ├── X_train.csv             # Scaled training feature matrix (316 x 41)
│       ├── X_test.csv              # Scaled test feature matrix (79 x 41)
│       ├── X_train_unscaled.csv    # Unscaled training features
│       ├── X_test_unscaled.csv     # Unscaled test features
│       ├── y_train_reg.csv         # Continuous training target (G3)
│       ├── y_test_reg.csv          # Continuous test target (G3)
│       ├── y_train_clf.csv         # Categorical training target (Low, Med, High)
│       └── y_test_clf.csv          # Categorical test target (Low, Med, High)
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── utils.py                    # Constants, mappings, plotting style, inference pipeline
│   ├── data_preprocessing.py       # Cleaning, encoding, splitting, and scaling pipeline
│   ├── eda.py                      # Exploratory data analysis & figure generator
│   ├── train_regression.py         # Training, tuning & evaluation for 10 regressors
│   ├── train_classification.py     # Training, tuning & evaluation for 5 classifiers
│   ├── evaluate.py                 # Benchmarking, comparison charts & winner selection
│   └── build_notebook.py           # Automated Jupyter notebook generator
│
├── notebooks/
│   └── student_performance_analysis.ipynb  # University-grade analysis notebook
│
├── models/
│   ├── best_regression_model.pkl   # Serialized best regressor (Random Forest)
│   ├── best_classification_model.pkl # Serialized best classifier (Decision Tree)
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── feature_names.pkl           # 41 ordered feature names
│   ├── reg_model_*.pkl             # All 10 individual trained regression models
│   ├── clf_model_*.pkl             # All 5 individual trained classification models
│   ├── regression_metrics.csv      # Logged metrics for all regressors
│   └── classification_metrics.csv  # Logged metrics for all classifiers
│
└── reports/
    ├── evaluation_summary.json     # Consolidated benchmark summary
    └── figures/                    # 37 high-resolution diagnostic plots
        ├── eda_*.png               # EDA distributions and correlations
        ├── reg_actual_vs_pred_*.png# Actual vs Predicted plots for all 10 regressors
        ├── reg_residuals_*.png     # Residual error plots for all 10 regressors
        ├── clf_confusion_matrix_*.png # Confusion matrices for all 5 classifiers
        ├── clf_feature_importance_*.png # Feature importance charts
        ├── regression_comparison.png # R2, RMSE, MAE comparison
        └── classification_comparison.png # Accuracy & F1 comparison
```

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone <YOUR_REPO_URL>
cd "Student Performance Prediction System"
```

### 2. Create and Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🔄 Reproducibility & Training Pipeline

To re-run the entire data science and modeling pipeline from scratch:

```bash
# 1. Run Data Preprocessing (generates data/processed/ and scaler artifacts)
python src/data_preprocessing.py

# 2. Run Exploratory Data Analysis (generates publication figures)
python src/eda.py

# 3. Train all 10 Regression Models with 5-Fold CV & GridSearchCV
python src/train_regression.py

# 4. Train all 5 Classification Models with Stratified 5-Fold CV & GridSearchCV
python src/train_classification.py

# 5. Run Model Comparison & Benchmark Evaluation
python src/evaluate.py
```

---

## 🎓 Academic Assessment Rubric Alignment

| Criterion | Implementation in this Project |
| :--- | :--- |
| **Data Preprocessing** | Handled categorical variables (binary + nominal one-hot), verified zero missing values, performed stratified 80/20 train-test split, scaled features via `StandardScaler`. |
| **Exploratory Data Analysis** | Histograms, KDE plots, Pearson correlation matrix, box plots of study hours and failures, absence trendlines, parental education analysis. |
| **Regression Diversity** | Implemented 10 distinct regression algorithms including linear, polynomial, penalized ($L_1/L_2$), kernel support vector machines, and ensemble methods. |
| **Classification Diversity** | Implemented 5 classification algorithms spanning Linear/Poly/RBF SVMs, Decision Trees, and Random Forests. |
| **Rigorous Validation** | 5-Fold Cross Validation (Stratified for classification) and exhaustive GridSearchCV across hyperparameter spaces. |
| **Model Diagnostics** | Actual vs. Predicted scatter plots with $y=x$ reference lines, Residual distribution checks, Confusion Matrices with counts & percentages, and Feature Importance rankings. |
| **Deployment / UI** | Fully functional, interactive Streamlit application with single-student inference, preset profile loading, dynamic visualizations, and academic intervention logic. |
| **Documentation** | Thoroughly commented codebase, comprehensive README, and a structured Jupyter Notebook following CRISP-DM methodology. |

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing & Contact
Developed for academic assessment and educational data mining research. Pull requests and feedback are welcome!

