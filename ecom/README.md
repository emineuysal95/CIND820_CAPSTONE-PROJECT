# E-Commerce Customer Churn Prediction

This project aims to analyze and predict customer churn behavior in the e-commerce sector using a real-world customer dataset. The objective is to build explainable and reliable machine learning models that highlight key churn drivers and provide actionable insights.

---

# Project Overview

* **Domain:** E-Commerce
* **Objective:** Predict whether a customer will churn based on historical shopping behavior and demographics
* **Dataset:** E-Commerce Churn Dataset (cleaned and preprocessed)

---

# Repository Structure

```
├── E-Com_Churn_Analysis.ipynb        # Jupyter Notebook with full pipeline (EDA to SHAP)
├── E-Com_Churn_Analysis.py           # Python script version of the workflow
├── ecommerce_customer_data.csv       # Original customer churn dataset
├── README.md                         # Project overview and documentation
├── Eda Reports/                      # Automated profiling and EDA visuals
└── E-Com_Churn_Analysis.html         # Executed notebook in HTML format
```

---

# Methodology

## 1. Exploratory Data Analysis (EDA)

* Missing values, outliers, class imbalance overview
* Distribution analysis (churn vs non-churn)
* Correlation analysis (heatmap, Cramér’s V)

## 2. Feature Engineering

* Created features: `PriceToQuantity`, `AvgItemValue`, `AgeGroup`
* One-hot encoding for categorical variables
* Statistical tests:

  * Chi-Square for categorical features
  * ANOVA for numerical features

## 3. Modeling Pipeline

* Models Used:

  * Logistic Regression (baseline)
  * XGBoost (with GridSearchCV tuning)
* Evaluation Metrics:

  * Accuracy
  * ROC AUC
  * F1 Score
  * MCC
  * Confusion Matrix
* **Class Imbalance Handling:** ADASYN used to oversample the minority churn class

## 4. Model Interpretation

* SHAP values computed on test data using tuned XGBoost model
* SHAP summary plots (bar & beeswarm) to identify important features
* Business insights drawn from most impactful features

---

# Key Insights

* Features such as `Support Tickets`, `Product Category`, `Return Rate`, and `High Purchase Amount` are strong churn indicators
* Customers with frequent returns and low support satisfaction are more likely to churn
* A low AvgItemValue may also relate to lower loyalty

---

# Tools & Libraries Used

* Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
* XGBoost
* SHAP
* Imbalanced-learn (ADASYN)
* SciPy & Statsmodels (ANOVA, Chi-square)
* ydata-profiling for automated EDA

---

# Performance Summary

| Model               | Accuracy | ROC AUC | F1-Score | MCC  |
| ------------------- | -------- | ------- | -------- | ---- |
| Logistic Regression | 0.80     | 0.52    | 0.25     | 0.10 |
| XGBoost (tuned)     | 0.83     | 0.78    | 0.61     | 0.52 |

> The tuned XGBoost model significantly outperforms the baseline, especially in handling class imbalance and identifying churn risk with interpretable results.

---

# Dataset

* **Source:**
  Jagtap, S. (2023). *E-Commerce Customer for Behavior Analysis*
  [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)

---

# Author

**Emine Uysal**
Certificate Student, Data Analytics, Big Data & Predictive Analytics
Toronto Metropolitan University
