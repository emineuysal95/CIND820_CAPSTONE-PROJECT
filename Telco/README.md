# Telco Customer Churn Prediction

This project analyzes and predicts customer churn behavior in the telecommunications sector using the Telco Customer Churn dataset. The goal is to build interpretable machine learning models, address class imbalance, and provide actionable insights through explainable AI techniques.

---

# Project Overview

* **Domain:** Telecommunications
* **Objective:** Predict whether a customer will churn (leave the service)
* **Dataset:** Telco Customer Churn Dataset (publicly available)

---

# Repository Structure

```
├── Telco_Churn_Analysis.ipynb       # Jupyter Notebook: EDA, preprocessing, modeling, SHAP
├── Telco_Churn_Analysis.py          # Python script version of the full analysis
├── telco_customer_data.csv          # Original Telco customer churn dataset
├── README.md                        # Project description and methodology
├── Eda Reports/                     # Auto and manual EDA reports with visuals
├── Telco_Churn_Analysis.html        # Executed notebook exported as HTML
```

---

# Methodology

## 1. Exploratory Data Analysis (EDA)

* Descriptive statistics and churn class distribution
* Missing value imputation
* PCA-based churn visualization
* Correlation heatmaps, Chi-square and ANOVA tests

## 2. Feature Engineering

* Feature creation: `TenureByContract`, `EngagementScore`, `TotalChargesPerMonth`
* One-Hot Encoding of categorical variables
* Multicollinearity checked via Variance Inflation Factor (VIF)
* Chi-square (categorical) and ANOVA (numerical) tests for churn relationship

## 3. Modeling Pipeline

* Models: Logistic Regression, Random Forest, XGBoost
* Evaluation: Accuracy, F1, ROC AUC, MCC, Confusion Matrix, Training & Prediction time
* Class Imbalance handled with **SMOTE**

## 4. Model Interpretation

* Global and local interpretation using **SHAP** (on SMOTE-resampled data)
* Feature importance visualized via SHAP summary plots
* Correlation of SHAP importance with tree-based models

---

# Key Insights

* **Top Churn Predictors:** `Tenure`, `Contract`, `MonthlyCharges`, `TechSupport`, `EngagementScore`
* **Retention Patterns:** Customers with longer tenure, contract plans, and tech support are less likely to churn
* **Risk Profiles:** Customers with short tenure, month-to-month plans, no support, and high charges are at higher risk

---

# Tools & Libraries Used

* Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
* XGBoost
* SHAP
* Imbalanced-learn (SMOTE)
* Statsmodels (VIF)
* SciPy (ANOVA & Chi-square tests)
* Time module for training/prediction duration

---

# Performance Summary

| Model               | Accuracy | ROC AUC | F1-Score | MCC  |
| ------------------- | -------- | ------- | -------- | ---- |
| Logistic Regression | 0.80     | 0.84    | 0.65     | 0.52 |
| Random Forest       | 0.84     | 0.87    | 0.70     | 0.60 |
| XGBoost             | 0.86     | 0.89    | 0.72     | 0.64 |

> XGBoost achieved the best performance with balanced metrics and strong interpretability using SHAP.

---

# Dataset

* **Source:** Kaggle (2018), [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

# Author

**Emine Uysal**
Certificate Student, Data Analytics, Big Data & Predictive Analytics
Toronto Metropolitan University
