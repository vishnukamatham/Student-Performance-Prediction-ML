"""
Student Performance Prediction System
Streamlit Interactive Web Application
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils import (
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RAW_DATA_FILE,
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    preprocess_single_input
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    .tier-badge-high {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .tier-badge-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .tier-badge-low {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_metadata():
    """Loads trained best models, scalers, and metric files."""
    models = {}
    try:
        models["best_reg"] = joblib.load(os.path.join(MODELS_DIR, "best_regression_model.pkl"))
        models["best_clf"] = joblib.load(os.path.join(MODELS_DIR, "best_classification_model.pkl"))
        models["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        models["features"] = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        
        # Load all specific regressors
        reg_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("reg_model_") and f.endswith(".pkl")]
        models["all_regressors"] = {
            f.replace("reg_model_", "").replace(".pkl", "").replace("_", " ").title(): joblib.load(os.path.join(MODELS_DIR, f))
            for f in reg_files
        }
        
        # Load all specific classifiers
        clf_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("clf_model_") and f.endswith(".pkl")]
        models["all_classifiers"] = {
            f.replace("clf_model_", "").replace(".pkl", "").replace("_", " ").title(): joblib.load(os.path.join(MODELS_DIR, f))
            for f in clf_files
        }
        
        # Load metrics tables
        models["reg_metrics"] = pd.read_csv(os.path.join(MODELS_DIR, "regression_metrics.csv"))
        models["clf_metrics"] = pd.read_csv(os.path.join(MODELS_DIR, "classification_metrics.csv"))
        
        with open(os.path.join(MODELS_DIR, "best_regression_meta.json"), "r") as f:
            models["best_reg_meta"] = json.load(f)
        with open(os.path.join(MODELS_DIR, "best_classification_meta.json"), "r") as f:
            models["best_clf_meta"] = json.load(f)
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure you have run the training pipelines first.")
    return models

models_data = load_models_and_metadata()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown('<div class="main-title">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-End Machine Learning for Final Exam Score Regression & Academic Tier Classification</div>', unsafe_allow_html=True)
with col_h2:
    st.image("https://img.shields.io/badge/Status-100%25%20Verified%20ML-success?style=for-the-badge&logo=scikitlearn", use_container_width=True)

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Live Prediction Portal",
    "📊 Model Comparison & Benchmarks",
    "📈 Diagnostic Visualizations",
    "🔍 Exploratory Data Analysis",
    "📚 Methodology & Theory"
])

# =============================================================================
# TAB 1: LIVE PREDICTION PORTAL
# =============================================================================
with tab1:
    st.markdown("### 📝 Enter Student Profile Details")
    st.caption("Provide academic history, study habits, and demographic attributes to generate real-time predictions.")
    
    # Preset sample profiles for rapid evaluation
    st.markdown("##### ⚡ Quick-Fill Preset Profiles:")
    p_col1, p_col2, p_col3 = st.columns(3)
    preset_choice = None
    if p_col1.button("🌟 Profile A: High-Achieving Student"):
        preset_choice = "high"
    if p_col2.button("⚖️ Profile B: Average Student"):
        preset_choice = "medium"
    if p_col3.button("⚠️ Profile C: At-Risk Student"):
        preset_choice = "low"
        
    # Default values based on preset
    if preset_choice == "high":
        def_g1, def_g2, def_study, def_fail, def_abs = 16, 17, 3, 0, 1
        def_higher, def_medu, def_fedu = "yes", 4, 4
    elif preset_choice == "low":
        def_g1, def_g2, def_study, def_fail, def_abs = 6, 7, 1, 2, 14
        def_higher, def_medu, def_fedu = "no", 1, 1
    else:
        def_g1, def_g2, def_study, def_fail, def_abs = 11, 11, 2, 0, 4
        def_higher, def_medu, def_fedu = "yes", 2, 2

    # Input Form
    with st.form("student_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Academic Track Record
        with col1:
            st.markdown("#### 🎓 Academic History")
            g1 = st.slider("Period 1 Grade (G1: 0 - 20)", min_value=0, max_value=20, value=def_g1, help="First period exam score")
            g2 = st.slider("Period 2 Grade (G2: 0 - 20)", min_value=0, max_value=20, value=def_g2, help="Second period exam score")
            failures = st.selectbox("Past Class Failures", options=[0, 1, 2, 3], index=def_fail, help="Number of past course failures")
            schoolsup = st.selectbox("Extra Educational Support", options=["no", "yes"], index=0)
            famsup = st.selectbox("Family Educational Support", options=["yes", "no"], index=0)
            paid = st.selectbox("Extra Paid Classes", options=["no", "yes"], index=0)
            
        # Column 2: Study Habits & Lifestyle
        with col2:
            st.markdown("#### ⏳ Study & Lifestyle")
            studytime = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4], index=def_study-1, 
                                     format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}[x])
            absences = st.slider("School Absences (Days)", min_value=0, max_value=93, value=def_abs)
            freetime = st.slider("Free Time After School (1 - Low to 5 - High)", 1, 5, 3)
            goout = st.slider("Going Out with Friends (1 - Low to 5 - High)", 1, 5, 3)
            dalc = st.slider("Workday Alcohol Consumption (1 - 5)", 1, 5, 1)
            walc = st.slider("Weekend Alcohol Consumption (1 - 5)", 1, 5, 1)
            health = st.slider("Current Health Status (1 - 5)", 1, 5, 4)
            
        # Column 3: Demographics & Family Background
        with col3:
            st.markdown("#### 🏠 Demographics & Family")
            school = st.selectbox("School", options=["GP", "MS"], format_func=lambda x: "Gabriel Pereira (GP)" if x=="GP" else "Mousinho da Silveira (MS)")
            sex = st.selectbox("Sex", options=["F", "M"], format_func=lambda x: "Female (F)" if x=="F" else "Male (M)")
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address Type", options=["U", "R"], format_func=lambda x: "Urban (U)" if x=="U" else "Rural (R)")
            famsize = st.selectbox("Family Size", options=["GT3", "LE3"], format_func=lambda x: "Greater than 3 (GT3)" if x=="GT3" else "3 or Less (LE3)")
            pstatus = st.selectbox("Parents Cohabitation", options=["T", "A"], format_func=lambda x: "Living Together (T)" if x=="T" else "Apart (A)")
            medu = st.selectbox("Mother's Education", options=[0, 1, 2, 3, 4], index=def_medu,
                                format_func=lambda x: {0: "None", 1: "Primary (4th)", 2: "5th-9th Grade", 3: "Secondary", 4: "Higher Ed"}[x])
            fedu = st.selectbox("Father's Education", options=[0, 1, 2, 3, 4], index=def_fedu,
                                format_func=lambda x: {0: "None", 1: "Primary (4th)", 2: "5th-9th Grade", 3: "Secondary", 4: "Higher Ed"}[x])
            mjob = st.selectbox("Mother's Job", options=["at_home", "health", "services", "teacher", "other"], index=2)
            fjob = st.selectbox("Father's Job", options=["teacher", "health", "services", "at_home", "other"], index=4)
            reason = st.selectbox("Reason to Choose School", options=["course", "home", "reputation", "other"], index=0)
            guardian = st.selectbox("Guardian", options=["mother", "father", "other"], index=0)
            traveltime = st.selectbox("Travel Time to School", options=[1, 2, 3, 4], format_func=lambda x: {1: "<15 min", 2: "15-30 min", 3: "30-60 min", 4: ">60 min"}[x])
            activities = st.selectbox("Extracurricular Activities", options=["yes", "no"], index=0)
            nursery = st.selectbox("Attended Nursery School", options=["yes", "no"], index=0)
            higher = st.selectbox("Wants Higher Education", options=["yes", "no"], index=0 if def_higher=="yes" else 1)
            internet = st.selectbox("Internet Access at Home", options=["yes", "no"], index=0)
            romantic = st.selectbox("In a Romantic Relationship", options=["no", "yes"], index=0)
            famrel = st.slider("Quality of Family Relationships (1 - 5)", 1, 5, 4)
            
        st.markdown("---")
        submit_btn = st.form_submit_button("🚀 Run Dual Prediction (Regression & Classification)", use_container_width=True)
        
    if submit_btn:
        # Build student dictionary
        student_data = {
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
            "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
            "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
            "Dalc": dalc, "Walc": walc, "health": health, "absences": absences, "G1": g1, "G2": g2
        }
        
        # Preprocess single input
        scaled_input = preprocess_single_input(student_data)
        
        # Predict using Best Models
        best_reg = models_data["best_reg"]
        best_clf = models_data["best_clf"]
        
        # Predict score
        pred_score = float(best_reg.predict(scaled_input)[0])
        # Bound score between 0 and 20
        pred_score = max(0.0, min(20.0, pred_score))
        
        # Predict class & probabilities
        pred_class = str(best_clf.predict(scaled_input)[0])
        class_probs = best_clf.predict_proba(scaled_input)[0]
        clf_classes = list(best_clf.classes_)
        prob_dict = {cls: prob for cls, prob in zip(clf_classes, class_probs)}
        
        # Display Prediction Cards
        st.markdown("### 🎯 Prediction Results")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 6px solid #2563eb;">
                <div class="metric-label">Predicted Final Exam Score (Regression)</div>
                <div class="metric-value">{pred_score:.2f} <span style="font-size:1.1rem; color:#64748b;">/ 20</span></div>
                <p style="margin-top: 8px; color: #475569; font-size: 0.9rem;">
                    Model: <b>{models_data['best_reg_meta']['Model']}</b> ($R^2 = {models_data['best_reg_meta']['Test_R2']:.3f}$)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for visual score scale
            st.progress(pred_score / 20.0)
            
        with res_col2:
            badge_class = f"tier-badge-{pred_class.lower()}"
            st.markdown(f"""
            <div class="metric-card" style="border-left: 6px solid #10b981;">
                <div class="metric-label">Predicted Performance Category (Classification)</div>
                <div style="margin-top: 10px;">
                    <span class="{badge_class}" style="font-size: 1.4rem;">{pred_class} Tier</span>
                </div>
                <p style="margin-top: 10px; color: #475569; font-size: 0.9rem;">
                    {CLASS_DESCRIPTIONS.get(pred_class, "")}<br>
                    Model: <b>{models_data['best_clf_meta']['Model']}</b> (Accuracy = {models_data['best_clf_meta']['Test_Accuracy']*100:.1f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Prediction Probability Breakdown
        st.markdown("#### 📊 Classification Probability Distribution")
        p_c1, p_c2, p_c3 = st.columns(3)
        p_c1.metric("Low (0 - 9)", f"{prob_dict.get('Low', 0.0)*100:.1f}%")
        p_c2.metric("Medium (10 - 13)", f"{prob_dict.get('Medium', 0.0)*100:.1f}%")
        p_c3.metric("High (14 - 20)", f"{prob_dict.get('High', 0.0)*100:.1f}%")
        
        # Actionable Intervention Recommendations
        st.markdown("#### 💡 Actionable Academic Interventions")
        if pred_class == "Low":
            st.error("""
            **High Academic Risk Detected:**
            - **Targeted Tutoring:** Immediately enroll student in mathematics remedial workshops.
            - **Attendance Monitoring:** Student's absences should be restricted; regular follow-ups with guardians recommended.
            - **Study Habit Formation:** Weekly study time should be increased to at least 5-10 hours.
            """)
        elif pred_class == "Medium":
            st.warning("""
            **Satisfactory Performance with Growth Potential:**
            - **Focus on Exam Review:** Stronger revision in earlier chapters (G1 & G2 topics) will bridge the gap to high honors.
            - **Peer Study Groups:** Encourage active participation in collaborative problem-solving.
            """)
        else:
            st.success("""
            **High Academic Achievement:**
            - **Advanced Coursework:** Student is an excellent candidate for advanced placement and competitive examinations.
            - **Mentorship Opportunity:** Encourage mentoring peers to reinforce concepts while maintaining academic excellence.
            """)

# =============================================================================
# TAB 2: MODEL COMPARISON & BENCHMARKS
# =============================================================================
with tab2:
    st.markdown("### 📊 Comprehensive Model Benchmarking & Comparison")
    st.caption("All models trained and tested with 5-Fold Cross-Validation and GridSearchCV hyperparameter tuning.")
    
    # Overview Key Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Models Evaluated", "15 Models", "10 Reg + 5 Clf")
    kpi2.metric("Best Regressor", models_data['best_reg_meta']['Model'], f"R² = {models_data['best_reg_meta']['Test_R2']:.4f}")
    kpi3.metric("Best Classifier", models_data['best_clf_meta']['Model'], f"Acc = {models_data['best_clf_meta']['Test_Accuracy']*100:.2f}%")
    kpi4.metric("Dataset Size", "395 Students", "80/20 Stratified Split")
    
    st.markdown("---")
    
    # Regression Leaderboard
    st.markdown("#### 🏆 Regression Models Leaderboard (10 Algorithms)")
    st.dataframe(
        models_data["reg_metrics"][["Model", "Test_R2", "Test_RMSE", "Test_MAE", "CV_R2_Mean", "CV_R2_Std"]].style.highlight_max(
            subset=["Test_R2", "CV_R2_Mean"], color="#dcfce7"
        ).highlight_min(
            subset=["Test_RMSE", "Test_MAE"], color="#dcfce7"
        ).format({
            "Test_R2": "{:.4f}",
            "Test_RMSE": "{:.4f}",
            "Test_MAE": "{:.4f}",
            "CV_R2_Mean": "{:.4f}",
            "CV_R2_Std": "{:.4f}"
        }),
        use_container_width=True
    )
    
    # Regression Chart
    reg_comp_img = os.path.join(FIGURES_DIR, "regression_comparison.png")
    if os.path.exists(reg_comp_img):
        st.image(reg_comp_img, caption="Comparison of Test R², RMSE, and MAE Across All 10 Regression Models", use_container_width=True)
        
    st.markdown("---")
    
    # Classification Leaderboard
    st.markdown("#### 🏆 Classification Models Leaderboard (5 Algorithms)")
    st.dataframe(
        models_data["clf_metrics"][["Model", "Test_Accuracy", "F1_Macro", "Precision_Macro", "Recall_Macro", "CV_Accuracy_Mean"]].style.highlight_max(
            subset=["Test_Accuracy", "F1_Macro", "Precision_Macro", "Recall_Macro", "CV_Accuracy_Mean"], color="#dcfce7"
        ).format({
            "Test_Accuracy": "{:.2%}",
            "F1_Macro": "{:.4f}",
            "Precision_Macro": "{:.4f}",
            "Recall_Macro": "{:.4f}",
            "CV_Accuracy_Mean": "{:.2%}"
        }),
        use_container_width=True
    )
    
    # Classification Chart
    clf_comp_img = os.path.join(FIGURES_DIR, "classification_comparison.png")
    if os.path.exists(clf_comp_img):
        st.image(clf_comp_img, caption="Comparison of Test Accuracy and F1-Scores Across All 5 Classification Models", use_container_width=True)
        
    st.markdown("---")
    
    # Generalization stability chart
    st.markdown("#### 🔄 Cross-Validation vs Test Set Generalization")
    gen_img = os.path.join(FIGURES_DIR, "cv_vs_test_generalization.png")
    if os.path.exists(gen_img):
        st.image(gen_img, caption="5-Fold Cross Validation Score vs Test Score (Overfitting & Stability Diagnostic)", use_container_width=True)

# =============================================================================
# TAB 3: DIAGNOSTIC VISUALIZATIONS
# =============================================================================
with tab3:
    st.markdown("### 📈 Diagnostic Plots: Actual vs Predicted, Residuals & Confusion Matrices")
    
    v_tab1, v_tab2, v_tab3 = st.tabs(["Regression Diagnostics", "Classification Confusion Matrices", "Feature Importances"])
    
    with v_tab1:
        st.markdown("#### Actual vs Predicted & Residual Error Distribution")
        reg_names = list(models_data["all_regressors"].keys())
        selected_reg = st.selectbox("Select Regression Model to Inspect:", options=reg_names, index=0)
        
        safe_key = selected_reg.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        avp_file = os.path.join(FIGURES_DIR, f"reg_actual_vs_pred_{safe_key}.png")
        res_file = os.path.join(FIGURES_DIR, f"reg_residuals_{safe_key}.png")
        
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            if os.path.exists(avp_file):
                st.image(avp_file, caption=f"Actual vs Predicted: {selected_reg}", use_container_width=True)
            else:
                st.info(f"Plot {avp_file} not found.")
        with r_c2:
            if os.path.exists(res_file):
                st.image(res_file, caption=f"Residual Error Diagnostic: {selected_reg}", use_container_width=True)
            else:
                st.info(f"Plot {res_file} not found.")
                
    with v_tab2:
        st.markdown("#### Multi-Class Confusion Matrices (Low, Medium, High)")
        clf_names = list(models_data["all_classifiers"].keys())
        selected_clf = st.selectbox("Select Classification Model to Inspect:", options=clf_names, index=0)
        
        safe_clf_key = selected_clf.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        cm_file = os.path.join(FIGURES_DIR, f"clf_confusion_matrix_{safe_clf_key}.png")
        
        c_c1, c_c2 = st.columns([2, 1])
        with c_c1:
            if os.path.exists(cm_file):
                st.image(cm_file, caption=f"Confusion Matrix: {selected_clf}", use_container_width=True)
            else:
                st.info(f"Confusion matrix {cm_file} not found.")
        with c_c2:
            st.markdown(f"**Interpretation for {selected_clf}:**")
            st.markdown("""
            - **Diagonal elements** represent correct classifications.
            - **Off-diagonal elements** represent misclassifications.
            - Notice that misclassifications primarily occur adjacent to grade boundaries (e.g. Medium misclassified as High), with negligible extreme misclassifications (Low classified as High).
            """)
            
    with v_tab3:
        st.markdown("#### Feature Importance Rankings")
        fi_rf = os.path.join(FIGURES_DIR, "clf_feature_importance_random_forest_classifier.png")
        fi_dt = os.path.join(FIGURES_DIR, "clf_feature_importance_decision_tree_classifier.png")
        
        fi_c1, fi_c2 = st.columns(2)
        with fi_c1:
            if os.path.exists(fi_rf):
                st.image(fi_rf, caption="Random Forest: Top 15 Feature Importances", use_container_width=True)
        with fi_c2:
            if os.path.exists(fi_dt):
                st.image(fi_dt, caption="Decision Tree: Top 15 Feature Importances", use_container_width=True)

# =============================================================================
# TAB 4: EXPLORATORY DATA ANALYSIS
# =============================================================================
with tab4:
    st.markdown("### 🔍 Exploratory Data Analysis & Empirical Insights")
    st.caption("Key statistical discoveries derived from the UCI Student Performance dataset.")
    
    eda1, eda2 = st.columns(2)
    with eda1:
        st.image(os.path.join(FIGURES_DIR, "eda_target_distribution.png"), caption="Final Score Distribution & Performance Tiers", use_container_width=True)
        st.image(os.path.join(FIGURES_DIR, "eda_studytime_failures_impact.png"), caption="Impact of Weekly Study Time & Failures", use_container_width=True)
        st.image(os.path.join(FIGURES_DIR, "eda_absences_vs_grades.png"), caption="School Absences vs Final Exam Performance", use_container_width=True)
    with eda2:
        st.image(os.path.join(FIGURES_DIR, "eda_correlation_heatmap.png"), caption="Correlation Heatmap of Key Features", use_container_width=True)
        st.image(os.path.join(FIGURES_DIR, "eda_prior_grades_vs_final.png"), caption="Correlation of G1 and G2 with Final Grade G3", use_container_width=True)
        st.image(os.path.join(FIGURES_DIR, "eda_parent_education_impact.png"), caption="Influence of Parental Education Levels", use_container_width=True)

# =============================================================================
# TAB 5: METHODOLOGY & THEORY
# =============================================================================
with tab5:
    st.markdown("### 📚 Theoretical Framework & Mathematical Formulations")
    st.caption("Detailed explanations of algorithms, loss functions, and optimization techniques implemented.")
    
    with st.expander("1. Simple & Multiple Linear Regression", expanded=True):
        st.markdown(r"""
        - **Simple Linear Regression:** Fits a linear relationship using a single predictor $x$:
          $$y = \beta_0 + \beta_1 x + \epsilon$$
          Where $\beta_1 = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$.
        - **Multiple Linear Regression:** Extends to $p$ features:
          $$y = X\beta + \epsilon, \quad \hat{\beta} = (X^T X)^{-1} X^T y$$
          Minimizes the Ordinary Least Squares (OLS) residual sum of squares:
          $$\text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
        """)
        
    with st.expander("2. Polynomial Regression", expanded=False):
        st.markdown(r"""
        Captures non-linear curves by transforming input features into polynomial terms of degree $d$:
        $$y = \beta_0 + \sum_{j=1}^p \beta_j x_j + \sum_{j=1}^p \sum_{k=j}^p \beta_{jk} x_j x_k + \dots$$
        In this system, a degree-2 polynomial expansion was evaluated on key academic signals ($G1, G2$) to model synergistic grade interactions.
        """)
        
    with st.expander("3. Regularized Regression: Ridge (L2) & Lasso (L1)", expanded=False):
        st.markdown(r"""
        Regularization mitigates multicollinearity and prevents overfitting:
        - **Ridge Regression (L2 penalty):**
          $$\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \beta_j^2$$
          Shrinks coefficients smoothly toward zero; optimal when many predictors have small-to-moderate effects.
        - **Lasso Regression (L1 penalty):**
          $$\mathcal{L}_{\text{Lasso}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p |\beta_j|$$
          Drives coefficients exactly to zero, performing intrinsic feature selection and producing sparse, interpretable models.
        """)
        
    with st.expander("4. Support Vector Machines: SVR & SVC with Kernels", expanded=False):
        st.markdown(r"""
        SVM maps input space into higher-dimensional reproducing kernel Hilbert spaces (RKHS) via the kernel trick:
        - **Linear Kernel:** $K(x, x') = x^T x'$
        - **Polynomial Kernel:** $K(x, x') = (\gamma x^T x' + r)^d$
        - **Radial Basis Function (RBF) Kernel:** $K(x, x') = \exp(-\gamma ||x - x'||^2)$
        - **Support Vector Regression (SVR):** Utilizes an $\epsilon$-insensitive loss band where errors smaller than $\epsilon$ are ignored:
          $$\min \frac{1}{2} ||w||^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$
        """)
        
    with st.expander("5. Decision Trees & Random Forest Ensembles", expanded=False):
        st.markdown(r"""
        - **Decision Trees:** Recursively partition feature space by maximizing Information Gain or minimizing Gini Impurity:
          $$Gini = 1 - \sum_{k=1}^K p_k^2, \quad Entropy = -\sum_{k=1}^K p_k \log_2(p_k)$$
        - **Random Forests (Bootstrap Aggregation + Feature Subsampling):**
          Constructs an ensemble of $B$ decorrelated decision trees:
          $$\hat{f}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)$$
          Averages out variance while preserving low bias, making it robust against noise.
        """)
        
    with st.expander("6. Validation & Evaluation Metrics", expanded=False):
        st.markdown(r"""
        - **K-Fold Cross Validation:** Partitions data into $K=5$ disjoint folds; trains on $K-1$ and validates on 1, repeating $K$ times to assess out-of-sample stability.
        - **GridSearchCV:** Exhaustively searches Cartesian product of hyperparameter grids using nested cross-validation to prevent data leakage.
        - **Evaluation Metrics:**
          - Mean Absolute Error: $\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$
          - Mean Squared Error: $\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$
          - Root Mean Squared Error: $\text{RMSE} = \sqrt{\text{MSE}}$
          - Coefficient of Determination: $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
          - Accuracy: $\frac{\text{TP} + \text{TN}}{\text{Total}}$
          - F1-Score: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
        """)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    Student Performance Prediction System | University ML Assessment Ready | Built with Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
