# CIND820_CAPSTONE PROJECT : CUSTOMER CHURN PREDICTION IN E-COMMERCE AND TELECOMMUNICATION 

---

🧠 Project Overview
Customer churn presents an ongoing challenge in data-driven sectors such as e-commerce and telecommunications, where customer retention directly impacts profitability and long-term growth.

This project explores churn behavior by applying predictive analytics and machine learning to identify patterns that lead to customer attrition. Using open-source datasets from Kaggle, it compares churn factors between two industries and builds classification models to predict churn risk.

In addition, we include a complete modeling pipeline using the IBM Telco dataset with EDA, feature engineering, and model training in Python.

---

📌 Key Components
Exploratory Data Analysis (EDA)
Feature engineering
Data cleaning and preprocessing (missing values, encoding, transformation)
Model building (Logistic Regression, Random Forest, XGBoost)
Sector-based churn pattern comparison
Power BI dashboards for visualization and stakeholder reporting
End-to-end implementation with the IBM Telco dataset
Visualizing results using Seaborn
🛠️ Tools & Technologies
Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
Visual Studio Code (VS Code)
Power BI
🎯 Goal
To provide actionable, data-driven insights that help businesses better understand and reduce customer churn across multiple industries.

---

📂 Repository Structure
├── 1_Abstract
├── 2_Literature_Review # The articles you have used, literature notes, referenced sources 
├── 3_Datasets # Raw datasets(.csv), Cleaned datasets 
│ ├── E_Commerce 
│ └── Telecom 
├── 4_EDA Reports & Preprocessing # Exploratory data analysis, visaulizations, statistical analyzes(Cleaning the datasets, encoding, missing value handling, VIF) 
├── 5_Modeling # Logistic, Random Forest, XGBoost, SMOTE uygulamaları 
├── 6_Initial Results 
│ ├── Interpretability # SHAP, LIME, feature importance, açıklayıcı grafikler 
│ └── Evaluation # Model karşılaştırmaları, metrik görselleri (ROC, Confusion Matrix) 
├── 7_Presentation # PowerPoint presentation, Power BI visuals and dashboard screenshoots └
── README.md # Project Explanation

---

📊 Datasets

- **Telco Churn Dataset:**Kaggle - Blastchar. (2017). Telco Customer Churn [Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **E-Commerce Dataset:** Kaggle - Jagtap, S. (2023). E-Commerce Customer for Behavior Analysis[Dataset](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)
---
🧹 Data Cleaning & Preparation
The following steps were taken to prepare the Telco dataset for modeling:

Converted TotalCharges to numeric and removed rows with missing values.
Encoded target variable Churn as binary: 'Yes' → 1, 'No' → 0
Applied label encoding to all categorical features.
Removed non-predictive identifier column customerID.
Split the data into training and test sets using an 80/20 ratio.
The following steps were taken to prepare the E-commerce dataset for modeling:

Removed non-predictive identifier columns such as Customer ID and Customer Name.
Checked for missing values and handled them appropriately.
Converted categorical variables (e.g., Gender, Location, Preferred Payment Method) using label encoding or one-hot encoding as needed.
Standardized numerical features (e.g., Tenure, Avg Transaction Value, Purchase Frequency) using StandardScaler.
Balanced the dataset using SMOTE due to class imbalance in the target variable Churn.
Encoded the target variable Churn as binary: 'Yes' → 1, 'No' → 0.
Split the dataset into training and test sets with an 80/20 ratio.

---
Author: Emine Uysal 
CAPSTONE PROJECT: Customer Churn Prediction in E-commerce and Telecommunications Date: June 2025
---


