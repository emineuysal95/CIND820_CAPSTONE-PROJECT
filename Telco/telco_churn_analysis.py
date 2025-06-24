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

# 12. FEATURE ENGINEERING
# 12.1. LONG-TERM CUSTOMER FLAG
df_telco_encoded['IsLongTermCustomer'] = (df_telco['tenure'] > 24).astype(int)
# 12.2. HIGH MONTHLY CHARGE FLAG
df_telco_encoded['HighMonthlyChargeFlag'] = (df_telco['MonthlyCharges'] > 70).astype(int)
# 12.3. TOTAL CHARGES PER MONTH
df_telco_encoded['TotalChargesPerMonth'] = df_telco['TotalCharges'] / df_telco['tenure'].replace(0, np.nan)
df_telco_encoded['TotalChargesPerMonth'].fillna(0, inplace=True)
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

# 14. ANOVA TESTS
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

# 15. CORRELATION ANALYSIS
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df_telco)
plt.title("Churn Class Distribution")
plt.show()

# 17. SHAPIRO-WILK NORMALITY TEST
from scipy.stats import shapiro
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    stat, p = shapiro(df_telco[col])
    print(f"{col} - p-value: {p:.4f}")

# 18. OUTLIER ANALYSIS - Z-SCORE
from scipy.stats import zscore
z_scores = df_telco[['tenure', 'MonthlyCharges', 'TotalCharges']].apply(zscore)
print("Outlier counts:")
print((z_scores > 3).sum())

# 19. CHI-SQUARE TEST FOR CATEGORICAL FEATURES
from scipy.stats import chi2_contingency
cat_cols = df_telco.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    cont_table = pd.crosstab(df_telco[col], df_telco['Churn'])
    chi2, p, dof, ex = chi2_contingency(cont_table)
    print(f"{col}: Chi2 = {chi2:.2f}, p = {p:.4f}")

# 20. SCALING NUMERICAL FEATURES
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_telco_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df_telco_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']])

# 21. MULTICOLLINEARITY ANALYSIS (VIF)
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

# 22. RANDOM FOREST CLASSIFIER ANALYSIS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# SPLIT DATA INTO X AND Y
X = df_telco_encoded.drop("Churn", axis=1)
y = df_telco_encoded["Churn"]

# 1. SPLIT INTO TRAIN AND TEST BEFORE APPLYING SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. APPLY SMOTE ONLY TO TRAINING DATA
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 3. TRAIN THE RANDOM FOREST MODEL
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
print("Random Forest Model Training Complete.")

# 4. FEATURE IMPORTANCE
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

# 5. PREDICTION & EVALUATION
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

# 22.1. RANDOM FOREST HYPERPARAMETER TUNING

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize model
rf = RandomForestClassifier(random_state=42)

# GridSearchCV setup
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=1,
                           scoring='f1')

# Fit on resampled training data
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score on training set:", grid_search.best_score_)

# Use the best estimator for further predictions
best_rf_model = grid_search.best_estimator_

# 22.2. RANDOM FOREST HYPERPARAMETER TUNING RESULTS

# Predict using the optimized model
y_pred = best_rf_model.predict(X_test)
y_prob = best_rf_model.predict_proba(X_test)[:, 1]

# Classification report
print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred))

# Accuracy and ROC AUC score
print("Accuracy (Tuned Model):", accuracy_score(y_test, y_pred))
print("ROC AUC Score (Tuned Model):", roc_auc_score(y_test, y_prob))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Tuned Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Tuned AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Tuned Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 23. SHAP EXPLAINABILITY ANALYSIS
import shap
# CONVERT X_test TO FLOAT TO AVOID SHAP WARNINGS
X_test_shap = X_test.copy().astype(float)

# SHAP EXPLAINER & VALUES
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test_shap)

# SHAP PLOTS
shap.summary_plot(shap_values, X_test_shap, plot_type="bar")
shap.summary_plot(shap_values, X_test_shap)

# 24. MODEL EVALUATION

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

# PREDICTIONS USING OPTIMIZED MODEL
y_pred = best_rf_model.predict(X_test)
y_prob = best_rf_model.predict_proba(X_test)[:, 1]

# CLASSIFICATION REPORT
print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred))

# ACCURACY AND ROC AUC
print("Accuracy (Tuned Model):", accuracy_score(y_test, y_pred))
print("ROC AUC Score (Tuned Model):", roc_auc_score(y_test, y_prob))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Tuned Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Tuned Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
