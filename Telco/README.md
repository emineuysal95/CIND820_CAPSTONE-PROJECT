# Telco Customer Churn Prediction

This project aims to analyze and predict customer churn behavior in the telecommunications sector using a real-world dataset. The goal is to build interpretable machine learning models, identify key churn drivers, and provide actionable insights through explainable AI techniques.

---

# Project Overview
Domain: Telecommunications
Objective: Predict whether a customer will churn (leave the service)
Dataset: Telco Customer Churn Dataset (publicly available)


---

# Repository Structure

├── Telco_Churn_Analysis.ipynb       # Main Jupyter Notebook (EDA + modeling)
├── telco_customer_data.csv          # Dataset used for analysis
├── README.md                        # Project overview and documentation
├── churn_model_outputs/             # Folder for saving model files, SHAP visuals, etc.
└── visuals/                         # Figures and plots used in reporting

---

# Methodology
#1. Exploratory Data Analysis (EDA)
Descriptive statistics and visualizations

Missing value treatment

Class imbalance overview

Churn distribution and correlation matrix

#2. Feature Engineering
- Encoding categorical variables with One-Hot Encoding
- Multicollinearity detection via Variance Inflation Factor (VIF)
- Statistical tests:
- Chi-square for categorical features
- ANOVA for numerical features

#3. Modeling Pipeline
Models trained and evaluated:
- Logistic Regression
- Random Forest
- XGBoost

 Evaluation Metrics:
- Accuracy
- ROC AUC Score
- F1-Score (added for robustness)
- Confusion Matrix

# Class Imbalance Handling:
Applied SMOTE to oversample minority class in training data

#4. Model Interpretation
SHAP (SHapley Additive exPlanations) to interpret feature importance and visualize local & global model explanations

Tree-based feature importance comparison

---

# Key Insights
Tenure, Contract Type, Monthly Charges, and Tech Support are among the top churn predictors.

Long-term customers with contract-based services and tech support are less likely to churn.

Short-tenure, month-to-month, and high-charge customers are at higher churn risk.

---

# Tools & Libraries Used
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- XGBoost
- SHAP
- Imbalanced-learn (SMOTE)
- Statsmodels (VIF)
- SciPy (ANOVA & Chi-square tests)

---

 Performance Summary
Model	Accuracy	ROC AUC	F1-Score
Logistic Regression	0.80	0.84	0.65
Random Forest	0.84	0.87	0.70
XGBoost	0.86	0.89	0.72

Note: XGBoost achieved the best performance with balanced metrics and strong interpretability using SHAP.

---

# Dataset


---

# Author
Emine Uysal
Certificate Student, Data Analytics, Big Data & Predictive Analytics
Toronto Metropolitan University
GitHub Profile
