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

####HEATMAP SUMMARIZE:The correlation heatmap reveals key features associated with customer churn. Specifically, long-term customers, those with two-year contracts, and those using support services are less likely to churn. In contrast, customers using fiber optic internet, opting for paperless billing, and paying via electronic check show higher churn tendencies. The presence of strong correlations among service-related dummy variables suggests potential multicollinearity issues that should be addressed during model building.
# Positive Correlations with churn include: Fiber optic internet service, Paperless billing, Electronic check payment method
# Negative Correlations with churn include: Long tenure, Two-year contract, Security services

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

## INTERPRETATION COMMENT: The plots reveal that customers using fiber optic internet, paying via electronic check, or having high monthly charges are more likely to churn—suggesting a cost-sensitive, digital-savvy profile. In contrast, long-tenured users, those with two-year contracts, and those subscribed to support or security services are less likely to leave. These patterns highlight the importance of retention strategies focused on customer engagement, service bundling, and contract-based commitment.

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

## INTERPRETATION COMMENT: The box plots indicate clear differences in numerical features between churned and retained customers. Customers who churned tend to have shorter tenure, higher monthly charges, and lower total charges—suggesting they are relatively new customers dissatisfied with service costs. Conversely, customers with longer tenure and higher total charges are more loyal and less likely to leave. This highlights the importance of early engagement and pricing strategy in churn prevention.

# 15. CHURN CLASS DISTRIBUTION ANALYSIS
churn_counts = df_telco['Churn'].value_counts()
print("Churn class distribution:\n", churn_counts)
print("\nChurn class percentages (%):\n", churn_counts / len(df_telco) * 100)

## COMMENT: The churn distribution is imbalanced, with approximately 73% of customers not churning and 27% churning. This imbalance should be addressed during the modeling phase using techniques such as resampling, or by choosing algorithms that handle imbalance well.

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

# 16. SHAPIRO-WILK TEST FOR NORMALITY
# The Shapiro-Wilk test checks if the data is normally distributed.

from scipy.stats import shapiro

for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    stat, p = shapiro(df_telco[col])
    print(f"{col} - p-value: {p}")
    if p < 0.05:
        print(" Not normally distributed")
    else:
        print(" Likely normal")

# 17. OUTLIER ANALYSIS (Z-Score Method)

from scipy.stats import zscore

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
z_scores = df_telco[numeric_cols].apply(zscore)
outliers = (z_scores > 3) | (z_scores < -3)

print("\n Outlier Analysis using Z-Score Method:")
print(outliers.sum())  # Count of outliers in each column

# OUTLIER ANALYSIS:Using the Z-score method (threshold ±3), no outliers were detected in the numerical features tenure, MonthlyCharges, or TotalCharges. This suggests that the distributions of these variables fall within an acceptable range of variability. However, alternative methods may be applied if visualizations suggest skewness or hidden outliers.

# 18. SCALING NUMERICAL FEATURES
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_telco_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df_telco_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']])

print("\n Numerical features after scaling:")
print(df_telco_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']].head())

## INTERPRETATION COMMENT: Standardization was applied to the numerical features (tenure, MonthlyCharges, and TotalCharges) using the StandardScaler. After scaling, all features have a mean of approximately 0 and a standard deviation of 1. This ensures that each feature contributes equally to distance-based or weight-sensitive algorithms. For example, low tenure and total charges in the early rows indicate customers with lower engagement, which may correlate with a higher risk of churn.

# 19. MULTICOLLINEARITY ANALYSIS (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# SELECT NUMERICAL FEATURES FOR VIF
# Select only the numerical features for VIF calculation
X_vif = df_telco_encoded.select_dtypes(include=['float64', 'int64']).dropna()

# CALCULATE VIF
vif_df = pd.DataFrame()
vif_df["Feature"] = X_vif.columns
vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# FILTER VIF > 5
vif_df = vif_df[vif_df["VIF"] > 5]
print("\n Top 15 features with highest VIF values:")
print(vif_df.sort_values(by="VIF", ascending=False).head(15))

# INTERPRETATION COMMENT: The VIF analysis indicates potential multicollinearity issues among several features, particularly those related to internet service types and payment methods. Features like "Fiber optic internet service" and "Electronic check payment method" show high VIF values, suggesting redundancy in information. This could lead to instability in model coefficients and inflated standard errors. Addressing multicollinearity may involve removing or combining features, or using regularization techniques in modeling.

# 20. RANDOMFORESTCLASSIFIER ANALYSIS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# SPLIT DATA INTO X AND Y
X = df_telco_encoded.drop("Churn", axis=1)
y = df_telco_encoded["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# BUILD RANDOM FOREST MODEL
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

print("\n Top 20 Important Features from Random Forest:")
print(top_features)

## COMMENT: Although TotalCharges showed a relatively high VIF (9.65), feature importance analysis via Random Forest revealed it as the most significant predictor of churn. Given that the chosen model is tree-based and robust to multicollinearity, TotalCharges was retained in the final model.

# PLOT FEATURE IMPORTANCE
plt.figure(figsize=(12, 8))
top_features.plot(kind='barh', color='skyblue')
plt.title("Top 20 Important Features from Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
# INTERPRETATION COMMENT: The Random Forest model identified TotalCharges, MonthlyCharges, and tenure as the most important features for predicting churn. This aligns with previous analyses indicating that financial metrics and customer engagement duration are critical indicators of churn risk. Other significant features include internet service type, contract length, and payment method, which further highlight the importance of service characteristics in customer retention strategies.

