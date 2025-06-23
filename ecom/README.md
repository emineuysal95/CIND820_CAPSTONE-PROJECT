# 🛍️ E-Commerce Customer Churn Analysis

This project analyzes customer churn behavior in an e-commerce setting, with the goal of identifying patterns and improving customer retention strategies through data-driven insights.

---

## 📊 Objective

To explore and prepare customer data for churn prediction by conducting in-depth exploratory data analysis (EDA), feature engineering, and applying supervised machine learning models.

---

## 🧠 Approach

The workflow follows these key steps:

1. **Exploratory Data Analysis (EDA)**  
   - Initial inspection, distribution plots, boxplots, and correlation matrix  
   - Categorical and numerical feature behavior across churned vs. retained customers  
   - Statistical tests: ANOVA, Chi-Square, Cramér’s V  

2. **Data Cleaning & Preprocessing**  
   - Handling missing values and duplicates  
   - Encoding categorical variables  
   - Outlier detection and variance inflation factor (VIF) to assess multicollinearity  

3. **Feature Engineering**  
   - Ratio-based features (e.g., `PriceToQuantity`, `AvgItemValue`)  
   - Age segmentation (`AgeGroup`)  
   - Month extraction from purchase date  

4. **Class Imbalance Handling**  
   - Applied ADASYN oversampling technique to balance churn vs. non-churn classes  

5. **Modeling**  
   - Baseline Logistic Regression  
   - ADASYN + Logistic Regression  
   - XGBoost with `scale_pos_weight` adjustment  
   - Model evaluation via accuracy, ROC AUC, Precision, Recall, and F1-score  

6. **Explainability**  
   - SHAP (SHapley Additive exPlanations) analysis to interpret XGBoost predictions and rank feature importance  

---

## 🔧 Techniques Used

- `Pandas`, `NumPy`, `Seaborn`, `Matplotlib`
- `scikit-learn`, `imblearn`, `xgboost`, `shap`, `ydata-profiling`
- Statistical tests from `scipy` and `statsmodels`

---

## 📁 Files

- `E-Com_Churn_Analysis.py`: Full script containing all EDA, feature engineering, modeling, and SHAP interpretation steps.
- `ecommerce_eda_report.html`: Auto-generated profiling report using `ydata-profiling`.

---

## 🔍 Dataset

- **Source**:  
  Jagtap, S. (2023). *E-Commerce Customer for Behavior Analysis* [Dataset]. Kaggle  
  [🔗 Kaggle Link](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)

---


