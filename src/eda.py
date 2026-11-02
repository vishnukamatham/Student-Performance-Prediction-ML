"""
Student Performance Prediction System
Exploratory Data Analysis (EDA) Module
Generates statistical summaries and publication-quality visualizations.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import (
    RAW_DATA_FILE,
    FIGURES_DIR,
    set_plot_style,
    save_figure,
    categorize_score
)

def run_eda(filepath=RAW_DATA_FILE):
    """
    Executes comprehensive exploratory data analysis, prints analytical insights,
    and saves visualizations to reports/figures/.
    """
    print("=" * 60)
    print("      STUDENT PERFORMANCE - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    set_plot_style()
    df = pd.read_csv(filepath)
    df["performance_tier"] = df["G3"].apply(categorize_score)
    
    print(f"[INFO] Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[INFO] Target Variable 'G3' Summary:")
    print(df["G3"].describe().round(2))
    
    print("\n[INFO] Performance Category Distribution:")
    print(df["performance_tier"].value_counts(normalize=True).round(3) * 100)
    
    # -------------------------------------------------------------
    # 1. Target Distribution & Category Breakdown
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram + KDE
    sns.histplot(df["G3"], kde=True, bins=20, color="#2b5c8f", ax=axes[0], edgecolor="black", alpha=0.6)
    axes[0].axvline(df["G3"].mean(), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {df['G3'].mean():.2f}")
    axes[0].axvline(df["G3"].median(), color="#27ae60", linestyle=":", linewidth=2, label=f"Median: {df['G3'].median():.2f}")
    axes[0].axvline(10, color="#f39c12", linestyle="-.", linewidth=1.5, label="Pass Threshold (10)")
    axes[0].set_title("Distribution of Final Exam Scores (G3)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Final Score (0 - 20)", fontsize=11)
    axes[0].set_ylabel("Student Count", fontsize=11)
    axes[0].legend(loc="upper right")
    
    # Categorical Countplot
    palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#27ae60"}
    order = ["Low", "Medium", "High"]
    counts = df["performance_tier"].value_counts()[order]
    bars = axes[1].bar(order, counts, color=[palette[k] for k in order], edgecolor="black", alpha=0.85)
    axes[1].set_title("Students per Performance Tier", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Academic Tier", fontsize=11)
    axes[1].set_ylabel("Number of Students", fontsize=11)
    
    # Add count labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{int(yval)} ({yval/len(df)*100:.1f}%)", 
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    save_figure(fig, "eda_target_distribution.png")
    print("[SAVED] reports/figures/eda_target_distribution.png")
    
    # -------------------------------------------------------------
    # 2. Correlation Heatmap (Key Features vs G3)
    # -------------------------------------------------------------
    numeric_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
        "G1", "G2", "G3"
    ]
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, annot=True, 
                fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap: Numerical Academic & Behavioral Features", fontsize=14, fontweight="bold", pad=12)
    save_figure(fig, "eda_correlation_heatmap.png")
    print("[SAVED] reports/figures/eda_correlation_heatmap.png")
    
    # -------------------------------------------------------------
    # 3. Prior Exam Grades (G1, G2) vs Final Exam (G3)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.regplot(data=df, x="G1", y="G3", ax=axes[0], color="#2980b9", 
                scatter_kws={"alpha": 0.5}, line_kws={"color": "#c0392b", "linewidth": 2})
    axes[0].set_title(f"Period 1 Grade (G1) vs Final Grade (G3)\n(Pearson r = {df['G1'].corr(df['G3']):.2f})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("G1 Score (0 - 20)", fontsize=11)
    axes[0].set_ylabel("G3 Score (0 - 20)", fontsize=11)
    
    sns.regplot(data=df, x="G2", y="G3", ax=axes[1], color="#8e44ad", 
                scatter_kws={"alpha": 0.5}, line_kws={"color": "#c0392b", "linewidth": 2})
    axes[1].set_title(f"Period 2 Grade (G2) vs Final Grade (G3)\n(Pearson r = {df['G2'].corr(df['G3']):.2f})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("G2 Score (0 - 20)", fontsize=11)
    axes[1].set_ylabel("G3 Score (0 - 20)", fontsize=11)
    
    save_figure(fig, "eda_prior_grades_vs_final.png")
    print("[SAVED] reports/figures/eda_prior_grades_vs_final.png")
    
    # -------------------------------------------------------------
    # 4. Study Time & Past Failures Impact on Score
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    studytime_labels = ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"]
    sns.boxplot(data=df, x="studytime", y="G3", hue="studytime", palette="Blues", legend=False, ax=axes[0])
    axes[0].set_xticks(range(len(studytime_labels)))
    axes[0].set_xticklabels(studytime_labels)
    axes[0].set_title("Weekly Study Time vs Final Exam Score", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Study Time Interval", fontsize=11)
    axes[0].set_ylabel("Final Score (G3)", fontsize=11)
    
    sns.boxplot(data=df, x="failures", y="G3", hue="failures", palette="Reds", legend=False, ax=axes[1])
    axes[1].set_title("Past Course Failures vs Final Exam Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Past Class Failures", fontsize=11)
    axes[1].set_ylabel("Final Score (G3)", fontsize=11)
    
    save_figure(fig, "eda_studytime_failures_impact.png")
    print("[SAVED] reports/figures/eda_studytime_failures_impact.png")
    
    # -------------------------------------------------------------
    # 5. Absences vs Final Exam Score
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="absences", y="G3", hue="performance_tier", 
                    palette=palette, alpha=0.7, s=60, ax=ax)
    sns.regplot(data=df, x="absences", y="G3", scatter=False, ax=ax, color="#34495e", line_kws={"linestyle": "--"})
    ax.set_title("School Absences vs Final Exam Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Absences", fontsize=11)
    ax.set_ylabel("Final Score (G3)", fontsize=11)
    ax.legend(title="Performance Tier")
    save_figure(fig, "eda_absences_vs_grades.png")
    print("[SAVED] reports/figures/eda_absences_vs_grades.png")
    
    # -------------------------------------------------------------
    # 6. Parental Education (Mother & Father) vs Score
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    edu_labels = ["None", "4th Grade", "5th-9th", "Secondary", "Higher Ed"]
    
    sns.barplot(data=df, x="Medu", y="G3", hue="Medu", palette="Purples", legend=False, ax=axes[0], capsize=0.1)
    axes[0].set_xticks(range(len(edu_labels)))
    axes[0].set_xticklabels(edu_labels, rotation=15)
    axes[0].set_title("Mother's Education Level vs Average G3 Score", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mother's Education", fontsize=11)
    axes[0].set_ylabel("Mean Final Score", fontsize=11)
    
    sns.barplot(data=df, x="Fedu", y="G3", hue="Fedu", palette="Greens", legend=False, ax=axes[1], capsize=0.1)
    axes[1].set_xticks(range(len(edu_labels)))
    axes[1].set_xticklabels(edu_labels, rotation=15)
    axes[1].set_title("Father's Education Level vs Average G3 Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Father's Education", fontsize=11)
    axes[1].set_ylabel("Mean Final Score", fontsize=11)
    
    save_figure(fig, "eda_parent_education_impact.png")
    print("[SAVED] reports/figures/eda_parent_education_impact.png")
    
    # -------------------------------------------------------------
    # 7. Alcohol Consumption (Workday vs Weekend)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    alc_labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    
    sns.barplot(data=df, x="Dalc", y="G3", hue="Dalc", palette="Oranges", legend=False, ax=axes[0], capsize=0.1)
    axes[0].set_xticks(range(len(alc_labels)))
    axes[0].set_xticklabels(alc_labels, rotation=15)
    axes[0].set_title("Workday Alcohol Consumption (Dalc) vs Score", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Workday Alcohol Level", fontsize=11)
    axes[0].set_ylabel("Mean Final Score", fontsize=11)
    
    sns.barplot(data=df, x="Walc", y="G3", hue="Walc", palette="Reds", legend=False, ax=axes[1], capsize=0.1)
    axes[1].set_xticks(range(len(alc_labels)))
    axes[1].set_xticklabels(alc_labels, rotation=15)
    axes[1].set_title("Weekend Alcohol Consumption (Walc) vs Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Weekend Alcohol Level", fontsize=11)
    axes[1].set_ylabel("Mean Final Score", fontsize=11)
    
    save_figure(fig, "eda_alcohol_impact.png")
    print("[SAVED] reports/figures/eda_alcohol_impact.png")
    
    print("\n[SUCCESS] All EDA visualizations generated and saved to reports/figures/.")

if __name__ == "__main__":
    run_eda()
