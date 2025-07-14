
#  E-Commerce Customer Churn Analysis

This project analyzes customer churn behavior in an e-commerce environment, aiming to uncover actionable insights that improve customer retention using data science techniques.

---

## Objective

To predict customer churn by performing in-depth data exploration, feature engineering, class balancing, and machine learning modeling.

---

## Project Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Visual exploration with distribution plots, boxplots, and heatmaps  
   - Comparison of churned vs. retained customers using statistical tests (ANOVA, Chi-Square, Cramér’s V)

2. **Data Cleaning & Preprocessing**  
   - Removal of missing values and duplicates  
   - Label encoding for categorical features  
   - Multicollinearity check using VIF

3. **Feature Engineering**  
   - Derived features: `PriceToQuantity`, `AvgItemValue`, `AgeGroup`, `PurchaseMonth`  
   - Normalization and transformations for modeling readiness

4. **Class Imbalance Handling**  
   - Applied ADASYN to oversample minority (churned) class

5. **Modeling**  
   - Logistic Regression (baseline & with ADASYN)  
   - XGBoost with `scale_pos_weight`  
   - Evaluation metrics: ROC AUC, Accuracy, Precision, Recall, F1-score

6. **Model Explainability**  
   - SHAP analysis to interpret XGBoost model predictions and identify influential features

---

## Tools & Libraries

- **Languages**: Python  
- **Libraries**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `xgboost`, `shap`, `imblearn`, `ydata-profiling`, `scipy`, `statsmodels`

---

## Project Structure
📁 E-Com_Churn_Analysis/
├── E-Com_Churn_Analysis.py
├──E-Com_Churn_Analysis.ipny
├── E-Com_Churn_Analysis.html
├── e-commerce_Customer_Dataset.csv
├── eda_ecommerce_cleaned.html
├── eda_ecom_raw.html
└── README.md

## Project Files

- `E-Com_Churn_Analysis.py`: Complete Python script with EDA, preprocessing, modeling, and SHAP analysis  
- `ecommerce_eda_report.html`: Automated profiling report generated with `ydata-profiling`  
- `E-Com_Churn_Analysis.py`: Standalone Python script implementing the full churn prediction workflow.
- `E-Com_Churn_Analysis.ipny`:Executable notebook with code, outputs, and visualizations for interactive and reproducible analysis.
- `E-Com_Churn_Analysis.html`: HTML export with embedded plots and results

---

## Dataset

- **Source**:  
  Jagtap, S. (2023). *E-Commerce Customer for Behavior Analysis*  
  [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)

---

## Author

Emine Uysal  
Certificate Student, Data Analytics, Big Data & Predictive Analytics  
Toronto Metropolitan University  
[GitHub Profile](https://github.com/emineuysal95)
