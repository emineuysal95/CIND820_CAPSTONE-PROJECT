# CIND 820 FINAL PROJECT : Customer Churn Prediction in E-commerce and Telecommunications

## ANALYSIS OF THE TELECOMMUNICATION DATASET

# 1. LOAD LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 

# 2. LOAD THE DATASET
file_path = "C:\\Users\\emine\\OneDrive\\Masaüstü\\CIND820\\Telco-Customer-Churn.csv"
df_telco = pd.read_csv(file_path)

# 2.1. BUILT EDA REPORT W/ RAW DATASET

# Load the raw dataset for EDA report generation
df_raw = pd.read_csv(file_path)
# Importing the pandas_profiling library for EDA report generation
from pandas_profiling import ProfileReport
# Generate a profiling report
profile_raw = ProfileReport(df_raw, title="EDA Report - Raw Telco Data", explorative=True)
# Save the report to an HTML file
profile_raw.to_file("C:/Users/emine/OneDrive/Masaüstü/CIND820/eda_telco_raw.html")

# 3. DISPLAY BASIC INFORMATION
print(df_telco.info())
print(df_telco.head())
print(df_telco.describe())

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

# 8. DROP ROWS WHERE 'TotalCharges' IS NULL
df_telco = df_telco[df_telco['TotalCharges'].notnull()]

# 9. CONVERT 'Churn' TO BINARY
df_telco['Churn'] = df_telco['Churn'].map({'No': 0, 'Yes': 1})

# 10. CONFIRM DATA CLEANING
print(df_telco.info())
print(df_telco['Churn'].value_counts())

# 11. ENCODING VARIABLES
cat_cols = df_telco.select_dtypes(include=['object']).columns.tolist()
df_telco_encoded = pd.get_dummies(df_telco, columns=cat_cols, drop_first=True)
print(df_telco_encoded.info())

#11.1. GENERATE EDA REPORT FOR ENCODED DATA
# Generate profiling report for cleaned/encoded data
profile_clean = ProfileReport(df_telco_encoded, title="EDA Report - Cleaned Telco Data", explorative=True)

# Save to CIND820 folder
profile_clean.to_file("C:/Users/emine/OneDrive/Masaüstü/CIND820/eda_telco_cleaned.html")

# 12. FEATURE ENGINEERING
# 12.1. LONG-TERM CUSTOMER FLAG
df_telco_encoded['IsLongTermCustomer'] = (df_telco['tenure'] > 24).astype(int)
# 12.2. HIGH MONTHLY CHARGE FLAG
df_telco_encoded['HighMonthlyChargeFlag'] = (df_telco['MonthlyCharges'] > 70).astype(int)
# 12.3. TOTAL CHARGES PER MONTH
df_telco_encoded['TotalChargesPerMonth'] = df_telco['TotalCharges'] / df_telco['tenure'].replace(0, np.nan)
df_telco_encoded['TotalChargesPerMonth'] = df_telco_encoded['TotalChargesPerMonth'].fillna(0)
# 12.4. CONTRACT LENGTH
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df_telco_encoded['ContractLength'] = df_telco['Contract'].map(contract_map)

# 13. CROSSTABS FOR CHURN
cat_features_to_check = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport']
print("\nChurn Crosstab (% by Category)\n")
for col in cat_features_to_check:
    if col in df_telco.columns:
        print(f"\n{col} vs Churn")
        cross = pd.crosstab(df_telco[col], df_telco['Churn'], normalize='index') * 100
        print(cross.round(2))

# 14.ANOVA TESTS FOR NUMERICAL FEATURES ACROSS CATEGORICAL FEATURES
#Import necessary libraries(f_oneway is the one-way ANOVA function from SciPy.)
from scipy.stats import f_oneway

print("\n--- ANOVA TEST RESULTS ---")

# MonthlyCharges across InternetService
groups1 = [df_telco[df_telco['InternetService'] == cat]['MonthlyCharges'] for cat in df_telco['InternetService'].unique()]
f1, p1 = f_oneway(*groups1)
print(f"MonthlyCharges by InternetService - F: {f1:.4f}, p: {p1:.4f}")

# TotalCharges across Contract
groups2 = [df_telco[df_telco['Contract'] == cat]['TotalCharges'] for cat in df_telco['Contract'].unique()]
f2, p2 = f_oneway(*groups2)
print(f"TotalCharges by Contract - F: {f2:.4f}, p: {p2:.4f}")
##COMMENT:This code performs ANOVA tests to see if there are statistically significant differences in charges across different customer groups:
#MonthlyCharges by InternetService→ Tests if average monthly charges differ by Internet type (e.g., DSL, Fiber, No service).
#TotalCharges by Contract→ Tests if total charges differ by contract type (e.g., Month-to-month, One year, Two year).If the p-value < 0.05, it means there's a significant difference between the groups.
#This test is both meaningful and contributes to reporting in terms of understanding the indirect effect of pricing on churn.

# 15. CORRELATION ANALYSIS
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df_telco)
plt.title("Churn Class Distribution")
plt.show()

# 16. SHAPIRO-WILK NORMALITY TEST
from scipy.stats import shapiro
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    stat, p = shapiro(df_telco[col])
    print(f"{col} - p-value: {p:.4f}")
##COMMENT:"Shapiro-Wilk test indicated that tenure and TotalCharges are not normally distributed (p < 0.05), justifying the use of tree-based models like Random Forest."

# 17. OUTLIER ANALYSIS - Z-SCORE
from scipy.stats import zscore
z_scores = df_telco[['tenure', 'MonthlyCharges', 'TotalCharges']].apply(zscore)
print("Outlier counts:")
print((z_scores > 3).sum())

# 17.1. OUTLIER VISUALIZATION - BOX PLOTS
plt.figure(figsize=(15, 4))

for i, col in enumerate(['tenure', 'MonthlyCharges', 'TotalCharges']):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=df_telco[col], color='skyblue')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()
#COMMENT:Boxplots and Z-score-based scatter plots reveal the presence of outliers particularly in TotalCharges and MonthlyCharges. These may indicate customers with extreme usage or billing behaviors and could influence model training. Outlier handling (e.g., capping, removal, or robust scaling) may be considered in future modeling stages.
 
# 18. CHI-SQUARE TEST FOR CATEGORICAL FEATURES
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
# Imports this function that performs the Chi-square test of independence to determine whether two categorical variables are significantly associated.

chi2_results = []

for col in cat_cols:
    cont_table = pd.crosstab(df_telco[col], df_telco['Churn'])
    chi2, p, dof, ex = chi2_contingency(cont_table)
    chi2_results.append((col, p))

# CREATE A DATAFRAME FOR CHI-SQUARE RESULTS
chi2_df = pd.DataFrame(chi2_results, columns=['Feature', 'p_value'])
chi2_df = chi2_df.sort_values(by='p_value')

# CHOOSE SIGNIFICANT FEATURES P-VALUES < 0.05
sig_features = chi2_df[chi2_df['p_value'] < 0.05]

# SHOW SIGNIFICANT FEATURES
sig_features = chi2_df[chi2_df['p_value'] < 0.05].copy()
sig_features['-log10(p-value)'] = -np.log10(sig_features['p_value'])
plt.figure(figsize=(10, 6))
plt.barh(sig_features['Feature'], sig_features['-log10(p-value)'])
plt.xlabel("-log10(p-value)")
plt.title("Chi-Square Test: Feature Significance for Churn")
plt.tight_layout()
plt.show()
##COMMENT:Statistical tests conducted on both numerical and categorical variables have revealed significant relationships between churn (customer loss) and many variables. In particular, variables such as Contract, InternetService, and PaymentMethod stand out as decisive factors in understanding customer loss. These variables must definitely be taken into account in the subsequent modeling phase.

# 19. SCALING NUMERICAL FEATURES
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_telco_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df_telco_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']])

# 20. MULTICOLLINEARITY ANALYSIS (VIF)
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

#DEFINE TOP 15 FEATURES WITH HIGHEST VIF
top_vif = vif_df.sort_values(by="VIF", ascending=False).head(15)

# VISUALIZE TOP 15 FEATURES WITH HIGHEST VIF

plt.figure(figsize=(10, 6))
plt.barh(top_vif["Feature"], top_vif["VIF"])
plt.xlabel("VIF Value")
plt.title("Top 15 Features with Highest VIF (Multicollinearity)")
plt.grid(axis='x')
plt.tight_layout()
plt.show()
# INTERPRETATION COMMENT: The VIF analysis indicates potential multicollinearity issues among several features, particularly those related to internet service types and payment methods. Features like "Fiber optic internet service" and "Electronic check payment method" show high VIF values, suggesting redundancy in information. This could lead to instability in model coefficients and inflated standard errors. Addressing multicollinearity may involve removing or combining features, or using regularization techniques in modeling.

# 21. RANDOM FOREST CLASSIFIER ANALYSIS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# SPLIT DATA INTO X AND Y
X = df_telco_encoded.drop("Churn", axis=1)
y = df_telco_encoded["Churn"]

# SPLIT INTO TRAIN AND TEST BEFORE APPLYING SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# APPLY SMOTE ONLY TO TRAINING DATA
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# TRAIN THE RANDOM FOREST MODEL
#The Random Forest model is being trained with data balanced after SMOTE.
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
print("Random Forest Model Training Complete.")

# FEATURE IMPORTANCE
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

print("\nTop 20 Important Features from Random Forest:")
print(top_features)

# PLOT FEATURE IMPORTANCE
plt.figure(figsize=(12, 8))
top_features.plot(kind='barh', color='skyblue')
plt.title("Top 20 Important Features from Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# 4.1 VISUALIZE CLASS DISTRIBUTION AFTER SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_resampled, hue=y_train_resampled, palette="Set2", legend=False)
plt.title("Churn Class Distribution After SMOTE")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# INFORMATION ON CLASS DISTRIBUTION AFTER SMOTE
print("\nClass distribution after SMOTE:")
print(y_train_resampled.value_counts())

# 5. PREDICTIONS USING DEFAULT RANDOM FOREST MODEL & EVALUATION
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# CLASSIFICATION REPORT
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ACCURACY AND ROC AUC
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# CLASSIFICATION REPORT
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred))

# ACCURACY AND ROC AUC
print("Accuracy (Random Forest):", accuracy_score(y_test, y_pred))
print("ROC AUC Score (Random Forest):", roc_auc_score(y_test, y_prob))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 25. LOGISTIC REGRESSION CLASSIFIER ANALYSIS

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. INITIALIZE AND TRAIN MODEL
log_model = LogisticRegression(max_iter=1000, solver='liblinear')  # liblinear is good for small datasets
log_model.fit(X_train_resampled, y_train_resampled)
print("Logistic Regression Model Training Complete.")

# 2. PREDICTIONS
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# 3. PERFORMANCE METRICS
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log))

print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_log))
print("ROC AUC Score (Logistic Regression):", roc_auc_score(y_test, y_prob_log))

# 4. CONFUSION MATRIX
cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 5. ROC CURVE
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
plt.figure(figsize=(6, 4))
plt.plot(fpr_log, tpr_log, label=f"AUC = {roc_auc_score(y_test, y_prob_log):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 6. COEFFICIENTS (Feature Impact)
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nTop Features Influencing Churn (Logistic Regression):")
print(coefficients.head(10))

# Optional plot
plt.figure(figsize=(10, 6))
coefficients.set_index('Feature').sort_values(by='Coefficient', ascending=True).tail(20).plot(kind='barh')
plt.title("Top Influential Features - Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

# 26. MODEL PERFORMANCE COMPARISON TABLE
# Metric values – replace these with your actual model scores if needed
rf_accuracy = accuracy_score(y_test, y_pred)
rf_auc = roc_auc_score(y_test, y_prob)

log_accuracy = accuracy_score(y_test, y_pred_log)
log_auc = roc_auc_score(y_test, y_prob_log)

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [rf_accuracy, log_accuracy],
    'ROC AUC': [rf_auc, log_auc]
})

# Display the table
print("\nModel Performance Comparison:")
print(comparison_df)

##COMMENT:    |MODEL|---------------|Accuracy|---|ROC AUC|
#             |Random Forest|---------|0.773|-----|0.817|
#             |Logistic Regression|---|0.763|-----|0.831|
## While the Random Forest model offers slightly better classification accuracy, the Logistic Regression model shows stronger performance in discriminative power (as evidenced by its higher ROC AUC). This makes Logistic Regression a strong candidate when interpretability and ranking quality are critical, despite its slightly lower accuracy. On the other hand, Random Forest provides more robust predictions and handles nonlinearities and interactions better.Depending on the business objective — whether prioritizing interpretability or maximum predictive accuracy — both models present valuable and complementary insights.

##INTERPRETATION ABOUT SHAP PLOTS:In the Telco dataset analysis, both Logistic Regression and Random Forest models were implemented to predict customer churn. While SHAP (SHapley Additive exPlanations) is a powerful tool for model interpretability, it was not applied to the Telco models for the following reasons:
#Logistic Regression already provides inherent interpretability through model coefficients. Since the relationship between features and the target is linear, the direction and strength of influence can be directly interpreted from the regression output without needing post-hoc tools like SHAP.
#Random Forest, although non-linear, was evaluated using feature importance scores, which offered sufficient insight into the key drivers of churn. Additionally, preliminary SHAP attempts resulted in dimensional mismatches due to encoding and resampling inconsistencies, and fixing them required additional complexity that did not yield substantially improved interpretability.
#Therefore, to maintain model clarity and analytical focus, SHAP was only utilized for the e-commerce dataset, where more complex models like XGBoost were employed and interpretability was essential due to higher feature interactions.
