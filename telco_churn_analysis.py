# CIND 820 FINAL PROJECT : Customer Churn Prediction in E-commerce and Telecommunications

## ANALYSIS OF THE TELECOMMUNICATION DATASET

# 1. LOAD LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. LOAD THE DATASET
file_path = "C:\\Users\\emine\\OneDrive\\Masaüstü\\CIND820\\Telco-Customer-Churn.csv"
df_telco = pd.read_csv(file_path)

# 3. DISPLAY BASIC INFORMATION
print(df_telco.info())
print(df_telco.head())

# 4. CHECK NUMBER OF UNIQUE VALUES PER COLUMN
unique_values = df_telco.nunique().sort_values()
print("Unique values per column:\n", unique_values)

# 5. DROP 'customerID' COLUMN
df_telco.drop('customerID', axis=1, inplace=True)

# 6. CONVERT 'TotalCharges' TO NUMERIC
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')

# 7. CHECK FOR MISSING VALUES IN 'TotalCharges'
missing_total_charges = df_telco['TotalCharges'].isnull().sum()
print(f"Missing TotalCharges values: {missing_total_charges}")
print(df_telco[df_telco['TotalCharges'].isnull()])

# 8. DROP ROWS WHERE 'TotalCharges' IS NULL
df_telco = df_telco[df_telco['TotalCharges'].notnull()]

# 9. CONVERT 'Churn' TO BINARY
df_telco['Churn'] = df_telco['Churn'].map({'No': 0, 'Yes': 1})

# 10. CONFIRM DATA CLEANING
print(df_telco.info())
print(df_telco['Churn'].value_counts())

# 11. ENCODING CATEGORICAL VARIABLES
cat_cols = df_telco.select_dtypes(include=['object']).columns.tolist()
df_telco_encoded = pd.get_dummies(df_telco, columns=cat_cols, drop_first=True)

# 12. CONFIRM TRANSFORMATION
print(df_telco_encoded.info())

# 13. CORRELATION ANALYSIS
corr_matrix = df_telco_encoded.corr()

# HEATMAP
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of All Features")
plt.tight_layout()
plt.show()

# CORRELATION WITH CHURN - BARPLOT
plt.figure(figsize=(10, 8))
churn_corr_sorted = corr_matrix["Churn"].sort_values(ascending=False)
sns.barplot(x=churn_corr_sorted.values, y=churn_corr_sorted.index)
plt.title("Correlation of All Features with Churn")
plt.xlabel("Correlation with Churn")
plt.tight_layout()
plt.show()

# 14. TOP POSITIVE & NEGATIVE CORRELATIONS
top_positive = churn_corr_sorted[1:11]
top_negative = churn_corr_sorted[-10:]
top_corr = pd.concat([top_positive, top_negative])

plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr.values, y=top_corr.index)
plt.title("Top Features Correlated with Churn")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()

# INTERPRETATION COMMENT (Can be added in the report):
# Customers with longer tenure, security services, and long-term contracts are less likely to churn.
# Users with fiber internet, paperless billing, or electronic checks are more likely to churn.

# BOX PLOTS OF NUMERICAL FEATURES BY CHURN
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Tenure vs Churn
sns.boxplot(x='Churn', y='tenure', data=df_telco, ax=axes[0])
axes[0].set_title("Tenure by Churn")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Tenure (months)")

# MonthlyCharges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df_telco, ax=axes[1])
axes[1].set_title("Monthly Charges by Churn")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Monthly Charges")

# TotalCharges vs Churn
sns.boxplot(x='Churn', y='TotalCharges', data=df_telco, ax=axes[2])
axes[2].set_title("Total Charges by Churn")
axes[2].set_xlabel("Churn")
axes[2].set_ylabel("Total Charges")

plt.tight_layout()
plt.show()

# 15. CHURN CLASS DISTRIBUTION ANALYSIS
churn_counts = df_telco['Churn'].value_counts()
print("Churn class distribution:\n", churn_counts)
print("\nChurn class percentages (%):\n", churn_counts / len(df_telco) * 100)

# BAR CHART
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df_telco, hue='Churn', palette='Set2', legend=False)
plt.title("Churn Class Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout()
plt.show()

# PIE CHART
plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
plt.title("Churn Class Proportion (Pie Chart)")
plt.axis('equal')  # Ensure circle shape
plt.tight_layout()
plt.show()

