# CIND 820 FINAL PROJECT : Customer Churn Prediction in E-commerce and Telecommunications

## ANALYSIS OF THE TELECOMMUNICATION DATASET

# 1. LOAD LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from pandas_profiling import ProfileReport
from sklearn.impute import SimpleImputer

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
# COMMENT: Generates an exploratory data analysis (EDA) report for the raw dataset, providing insights into data structure, distributions, and potential issues.

# 3. DISPLAY BASIC INFORMATION
print(df_telco.info())
print(df_telco.head())
print(df_telco.describe())

# 4. CHECK NUMBER OF UNIQUE VALUES PER COLUMN
unique_values = df_telco.nunique().sort_values()
print("Unique values per column:\n", unique_values)
# COMMENT: Helps identify categorical vs. numerical columns, and detect constant or near-constant features.

# 5. DROP 'customerID' COLUMN
df_telco.drop('customerID', axis=1, inplace=True)
# COMMENT: 'customerID' is a unique identifier and doesn't contribute to predictive power.

# 6. CONVERT 'TotalCharges' TO NUMERIC
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')
# COMMENT: Converts TotalCharges column to numeric, coercing invalid entries to NaN.

# 7. CHECK FOR MISSING VALUES IN 'TotalCharges'
missing_total_charges = df_telco['TotalCharges'].isnull().sum()
print(f"Missing TotalCharges values: {missing_total_charges}")
# COMMENT: Identify how many entries failed conversion and now contain NaNs.

# 8. DROP ROWS WHERE 'TotalCharges' IS NULL
df_telco = df_telco[df_telco['TotalCharges'].notnull()]
# COMMENT: Dropping a small number of missing rows is preferable to imputation in this case.

# 9. CONVERT 'Churn' TO BINARY
df_telco['Churn'] = df_telco['Churn'].map({'No': 0, 'Yes': 1})
# COMMENT: Converts target variable into binary format for modeling.

# 10. CONFIRM DATA CLEANING
print(df_telco.info())
print(df_telco['Churn'].value_counts())

# 11. ENCODING VARIABLES
cat_cols = df_telco.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df_telco, columns=cat_cols, drop_first=True)
print(df_encoded.info())

# 11.1 PCA FOR DIMENSIONALITY REDUCTION (Visualization)
features = df_encoded.drop("Churn", axis=1)
target = df_encoded["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Churn'] = target.values

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Churn', palette='coolwarm', alpha=0.6)
plt.title("PCA - Customer Churn Visualization")
plt.tight_layout()
plt.show()

print(f"Explained Variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Explained Variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")

# 11.2. GENERATE EDA REPORT FOR ENCODED DATA
profile_clean = ProfileReport(df_encoded, title="EDA Report - Cleaned Telco Data", explorative=True)
profile_clean.to_file("C:/Users/emine/OneDrive/Masaüstü/CIND820/eda_telco_cleaned.html")
# COMMENT: Generates an EDA report for the cleaned and encoded dataset, providing insights into feature distributions, correlations, and potential issues.

# 12. SPLIT DATA FOR FEATURE ENGINEERING AND MODELING
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 12.1. FEATURE ENGINEERING
df_encoded['IsLongTermCustomer'] = (df_telco['tenure'] > 24).astype(int)
df_encoded['HighMonthlyChargeFlag'] = (df_telco['MonthlyCharges'] > 70).astype(int)
df_encoded['TotalChargesPerMonth'] = df_telco['TotalCharges'] / df_telco['tenure'].replace(0, np.nan)
df_encoded['TotalChargesPerMonth'] = df_encoded['TotalChargesPerMonth'].fillna(0)
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df_encoded['ContractLength'] = df_telco['Contract'].map(contract_map)

# 13. ENGAGEMENT SCORE (Bundled Services Index)
# This metrric shows how many value-added digital services (security, support, streaming) a customer uses.
# It quantifies customer engagement with the service ecosystem, which can be a strong predictor of churn.

bundled_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
bundled_cols_train = [col for col in X_train.columns if any(f"{f}_Yes" in col for f in bundled_features)]
bundled_cols_test = [col for col in X_test.columns if any(f"{f}_Yes" in col for f in bundled_features)]

X_train['EngagementScore'] = X_train[bundled_cols_train].sum(axis=1)
X_test['EngagementScore'] = X_test[bundled_cols_test].sum(axis=1)
# COMMENT: EngagementScore quantifies how many value-added digital services (security, support, streaming) a customer uses. Higher scores may signal greater customer retention due to stronger integration with the service ecosystem.

# 13.1. VISUALIZE ENGAGEMENT SCORE DISTRIBUTION FOR INTERPRETATION
plt.figure(figsize=(8, 4))
sns.histplot(X_train['EngagementScore'], bins=7, kde=False, color='skyblue', edgecolor='black')
plt.title("Distribution of Engagement Score (Train Set)")
plt.xlabel("Number of Active Digital Services")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.show()
# COMMENT: This histogram shows the distribution of Engagement Scores, indicating how many bundled services customers typically use. Most customers use 2-3 services, with fewer using all 6.
# COMMENT: EngagementScore quantifies how many value-added digital services (security, support, streaming) a customer uses. 
# Higher scores may signal greater customer retention due to stronger integration with the service ecosystem.

# 13.2. CHURN RATE DISTRUBITION BY ENGAGEMENT SCORE
plt.figure(figsize=(8, 5))
sns.boxplot(x=y_train, y=X_train['EngagementScore'], palette='pastel')
plt.title("Engagement Score by Churn Status")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Engagement Score")
plt.xticks([0, 1], ['Non-Churn', 'Churn'])
plt.tight_layout()
plt.show()

#13.3. CHECK AVERAGE ENGAGEMENT SCORE FOR CHURNED AND NON-CHURNED CUSTOMERS
# Calculate mean EngagementScore using training set only
mean_engaged_churn = X_train[y_train == 1]['EngagementScore'].mean()
mean_engaged_nonchurn = X_train[y_train == 0]['EngagementScore'].mean()

print(f"Average EngagementScore (Churned): {mean_engaged_churn:.2f}")
print(f"Average EngagementScore (Non-Churned): {mean_engaged_nonchurn:.2f}")
# COMMENT: The average EngagementScore shows that churned customers use fewer bundled digital services compared to non-churned ones.
# COMMENT: Calculating these averages from the training set ensures no data leakage while still revealing meaningful patterns.
#COMMENT: Churned customers had a noticeably lower average engagement score (1.80) compared to retained ones (2.13), suggesting that reduced customer-platform interaction is associated with higher churn risk.

# 14. EDA VISUALS FOR NUMERICAL DISTRIBUTIONS
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
plt.figure(figsize=(15, 4))
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df_telco[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# COMMENT: Visualizes the distribution of key numerical features, helping to spot skewness or multimodal patterns.

# 15. CATEGORICAL FEATURE DISTRIBUTION BY CHURN
cat_eda_cols = ['InternetService', 'Contract', 'PaymentMethod']
plt.figure(figsize=(15, 8))
for i, col in enumerate(cat_eda_cols):
    plt.subplot(2, 2, i + 1)
    sns.countplot(data=df_telco, x=col, hue='Churn')
    plt.title(f"{col} by Churn")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMENT: Reveals relationships between churn and key categorical features through side-by-side bar plots.

# 15.1. CHURN RATE BY SERVICE TYPE (Contract, TechSupport, OnlineSecurity, InternetService)

def churn_rate_by_category(df, column):
    churn_pct = pd.crosstab(df[column], df['Churn'], normalize='index') * 100
    churn_pct = churn_pct.rename(columns={0: 'Non-Churn %', 1: 'Churn %'})

    # Visualization
    churn_pct['Churn %'].plot(kind='bar', figsize=(8, 5), color='salmon', edgecolor='black')
    plt.title(f"Churn Rate by {column}")
    plt.ylabel("Churn Percentage")
    plt.xlabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"\nChurn Percentage by {column}:\n", churn_pct.round(2))

# Run for selected features
for col in ['Contract', 'TechSupport', 'OnlineSecurity', 'InternetService']:
    churn_rate_by_category(df_telco, col)

# COMMENT: This function calculates churn rates for a given categorical feature and visualizes the results, providing insights into how different categories relate to churn.
# COMMENT: This analysis shows how different service features impact churn rates, providing actionable insights for customer retention strategies.
# Contract : Customers with Month-to-month contracts exhibit a significantly higher churn rate (over 40%) compared to those on One year or Two year plans. This suggests that long-term commitment correlates with reduced churn, potentially due to early termination fees or perceived service satisfaction.
# TechSupport : Churn is notably higher among customers who do not have Tech Support services. The presence of technical support likely enhances customer retention by resolving issues quickly and improving user experience.
# OnlineSecurity : Similar to TechSupport, customers without OnlineSecurity are more prone to churn. This may reflect lower engagement levels or unmet expectations regarding bundled service value.
# InternetService : Among the InternetService categories, Fiber optic users have the highest churn rate—likely due to higher costs or competitive alternatives. DSL users show lower churn, and those with No internet service churn the least, possibly reflecting minimal telecom engagement.


# 16. CROSSTABS FOR CHURN
cat_features_to_check = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport']
print("\nChurn Crosstab (% by Category)\n")
for col in cat_features_to_check:
    if col in df_telco.columns:
        print(f"\n{col} vs Churn")
        cross = pd.crosstab(df_telco[col], df_telco['Churn'], normalize='index') * 100
        print(cross.round(2))

# 17.ANOVA TESTS FOR NUMERICAL FEATURES ACROSS CATEGORICAL FEATURES
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

# 18. CORRELATION ANALYSIS
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df_telco)
plt.title("Churn Class Distribution")
plt.show()

# 19. SHAPIRO-WILK NORMALITY TEST
from scipy.stats import shapiro
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    stat, p = shapiro(df_telco[col])
    print(f"{col} - p-value: {p:.4f}")
##COMMENT:"Shapiro-Wilk test indicated that tenure and TotalCharges are not normally distributed (p < 0.05), justifying the use of tree-based models like Random Forest."

# 20. OUTLIER ANALYSIS - Z-SCORE
from scipy.stats import zscore
z_scores = df_telco[['tenure', 'MonthlyCharges', 'TotalCharges']].apply(zscore)
print("Outlier counts:")
print((z_scores > 3).sum())

# 20.1. OUTLIER VISUALIZATION - BOX PLOTS
plt.figure(figsize=(15, 4))

for i, col in enumerate(['tenure', 'MonthlyCharges', 'TotalCharges']):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=df_telco[col], color='skyblue')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()
#COMMENT:Boxplots and Z-score-based scatter plots reveal the presence of outliers particularly in TotalCharges and MonthlyCharges. These may indicate customers with extreme usage or billing behaviors and could influence model training. Outlier handling (e.g., capping, removal, or robust scaling) may be considered in future modeling stages.
 
# 21. CHI-SQUARE TEST FOR CATEGORICAL FEATURES
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

# 22. SCALING NUMERICAL FEATURES
scaler = StandardScaler()
df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']])

# 23. MULTICOLLINEARITY ANALYSIS (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df_encoded.select_dtypes(include=['float64', 'int64']).dropna()

vif_df = pd.DataFrame()
vif_df["Feature"] = X_vif.columns
vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

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

# 24. CORRELATION HEATMAP
plt.figure(figsize=(14, 10))
corr = df_encoded.corr()
sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

##COMMENT: Correlation heatmap allows quick detection of redundant features or strong linear relationships.
##COMMENT: The heatmap illustrates the linear correlations between numerical features in the Telco churn dataset. Notably, there is a strong positive correlation between tenure and TotalCharges, as well as between MonthlyCharges and TotalCharges, indicating that customers who stay longer or pay more monthly tend to accumulate higher total charges. However, the Churn variable shows weak correlations with most features, suggesting that churn behavior may not be directly explained by linear relationships alone. This highlights the importance of using advanced modeling techniques, such as logistic regression or SHAP, to uncover more complex patterns behind customer churn.

# 24.1. TARGET-CORRELATION VISUALIZATION
target_corr = corr['Churn'].sort_values(ascending=False)[1:11]  # exclude self-correlation
plt.figure(figsize=(8, 6))
target_corr.plot(kind='barh')
plt.title("Top 10 Features Most Correlated with Churn")
plt.xlabel("Correlation with Churn")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
# COMMENT: The bar chart shows the top features most linearly correlated with churn, yet all exhibit very weak correlations. For example, tenure has a slightly positive correlation, implying customers who stay longer are marginally less likely to churn, while MonthlyCharges and SeniorCitizen have small negative correlations, suggesting higher monthly costs or being a senior might slightly increase churn risk. However, the extremely low absolute values indicate that no single numeric feature strongly drives churn on its own, reinforcing the need for non-linear modeling approaches to better capture underlying churn dynamics.
# COMMENT: Highlights which features are most associated with churn—useful for feature selection.

# 25. RANDOM FOREST CLASSIFIER ANALYSIS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# SPLIT DATA INTO X AND Y
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

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

# 25.1 VISUALIZE CLASS DISTRIBUTION AFTER SMOTE
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

# 25.2. PREDICTIONS USING DEFAULT RANDOM FOREST MODEL & EVALUATION
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

# 26. LOGISTIC REGRESSION CLASSIFIER ANALYSIS

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

# 26.1. INITIALIZE AND TRAIN MODEL
log_model = LogisticRegression(max_iter=1000, solver='liblinear')  # liblinear is good for small datasets
log_model.fit(X_train_resampled, y_train_resampled)
print("Logistic Regression Model Training Complete.")

# 26.2. PREDICTIONS
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# 26.3. PERFORMANCE METRICS
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log))

print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_log))
print("ROC AUC Score (Logistic Regression):", roc_auc_score(y_test, y_prob_log))

# 26.4. CONFUSION MATRIX
cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 26.5. ROC CURVE
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

# 26.6. COEFFICIENTS (Feature Impact)
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

# 27. MODEL PERFORMANCE COMPARISON TABLE
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
print(comparison_df.round(3))

##COMMENT:    |MODEL|---------------|Accuracy|---|ROC AUC|
#             |Random Forest|---------|0.773|-----|0.817|
#             |Logistic Regression|---|0.763|-----|0.831|
## While the Random Forest model offers slightly better classification accuracy, the Logistic Regression model shows stronger performance in discriminative power (as evidenced by its higher ROC AUC). This makes Logistic Regression a strong candidate when interpretability and ranking quality are critical, despite its slightly lower accuracy. On the other hand, Random Forest provides more robust predictions and handles nonlinearities and interactions better.Depending on the business objective — whether prioritizing interpretability or maximum predictive accuracy — both models present valuable and complementary insights.

##INTERPRETATION ABOUT SHAP PLOTS:In the Telco dataset analysis, both Logistic Regression and Random Forest models were implemented to predict customer churn. While SHAP (SHapley Additive exPlanations) is a powerful tool for model interpretability, it was not applied to the Telco models for the following reasons:
#Logistic Regression already provides inherent interpretability through model coefficients. Since the relationship between features and the target is linear, the direction and strength of influence can be directly interpreted from the regression output without needing post-hoc tools like SHAP.
#Random Forest, although non-linear, was evaluated using feature importance scores, which offered sufficient insight into the key drivers of churn. Additionally, preliminary SHAP attempts resulted in dimensional mismatches due to encoding and resampling inconsistencies, and fixing them required additional complexity that did not yield substantially improved interpretability.
#Therefore, to maintain model clarity and analytical focus, SHAP was only utilized for the e-commerce dataset, where more complex models like XGBoost were employed and interpretability was essential due to higher feature interactions.

# 27.1. SHAP EXPLAINABILITY
import shap

# Ensure X_test is a DataFrame with the same column structure used in training
X_test_shap = X_test.copy()
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_shap)

# SHAP plot handling based on structure
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test_shap, plot_type="bar")
    shap.summary_plot(shap_values[1], X_test_shap)
else:
    shap.summary_plot(shap_values, X_test_shap, plot_type="bar")
    shap.summary_plot(shap_values, X_test_shap)
# COMMENT: SHAP provides global understanding of feature importance and directionality. Using exact feature alignment with X_test ensures SHAP values match dimensions.
# COMMENT: Top SHAP features like 'Contract_Two year', 'MonthlyCharges', and 'tenure' suggest that customers with long contracts and lower monthly charges are less likely to churn, aligning with business expectations. Meanwhile, high monthly charges and short tenure increase churn probability, confirming prior EDA findings.

# 28. RANDOM FOREST DECISION TREE VISUALIZATION
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=X_train.columns, filled=True, max_depth=3, fontsize=10)
plt.title("Random Forest - First Tree Visualization (Depth=3)")
plt.tight_layout()
plt.show()
# COMMENT: This simplified tree helps interpret a single estimator's decision flow in the Random Forest. It visualizes key feature thresholds contributing to churn classification.

# COMMENT: This decision tree shows how the Random Forest model splits on features like ContractLength, TotalCharges, StreamingTV, and PaymentMethod to distinguish churn vs. non-churn.
# Customers with short contract lengths, low total charges, and specific service patterns are more likely to churn.
# On the other hand, customers with longer contracts and consistent payment histories are less likely to churn.
# Visualizing one tree helps interpret how the ensemble makes its decisions in a human-readable form.

# WHY THE MODEL PREDICTS CHURN:
# - ContractLength ≤ 0.5 → short-term (month-to-month) customers are more churn-prone.
# - No StreamingTV or StreamingMovies suggests low engagement.
# - PaymentMethod_Electronic check users have higher churn risk.
# - Customers with low tenure (new customers) tend to churn more.
 
# WHY THE MODEL PREDICTS NON-CHURN:
# - Longer contracts, high TotalCharges (longer relationship), stable payment method.
# - Dependents and bundled services (TV, internet) increase customer stickiness.

# 28.1. PRECISION-RECALL CURVE
from sklearn.metrics import precision_recall_curve

# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Take class predictions and probabilities scores
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]  # Class 1 için (churn) olasılık skorları

# Calculate Precision, Recall, and Thresholds
precision_vals, recall_vals, threshold_vals = precision_recall_curve(y_test, y_prob_rf)

# Visualize Precision-Recall Curve
plt.figure(figsize=(6, 4))
plt.plot(threshold_vals, precision_vals[:-1], label='Precision', linestyle='--')
plt.plot(threshold_vals, recall_vals[:-1], label='Recall', linestyle='-')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs. Recall Curve (Random Forest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMENT: The Precision-Recall curve illustrates the trade-off between precision and recall at various classification thresholds.
# It helps identify the optimal threshold based on business priorities, such as whether to prioritize precision (
# COMMENT: This curve visually demonstrates the trade-off between precision and recall across different thresholds, aiding in threshold selection based on business priorities.

# 28.2. MODEL COMPARISON TABLE
# CURRENT PERFORMANCE METRICS
comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression', 'XGBoost'],
    'Accuracy': [0.773, 0.763, 0.755],
    'ROC AUC': [0.817, 0.831, 0.805],
    'F1-score': [0.69, 0.64, 0.67]
})

# PERFORMANCE COMPARISON TABLE
plt.figure(figsize=(7, 3.5))
plt.table(cellText=comparison_df.round(3).values,
          colLabels=comparison_df.columns,
          loc='center', cellLoc='center')
plt.axis('off')
plt.title("Model Performance Comparison (Accuracy, AUC, F1-score)")
plt.tight_layout()
plt.show()
# COMMENT: This table summarizes the performance metrics of the Random Forest, Logistic Regression, and XGBoost models, allowing for quick comparison of their predictive capabilities.


# 29. XGBOOST MODEL (ADDITIONAL MODEL)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))

# 29.1. HYPERPARAMETER TUNING WITH GRIDSEARCHCV FOR RANDOM FOREST
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train_resampled, y_train_resampled)
print("\nBest Parameters for Random Forest:", grid_rf.best_params_)
print("Best ROC AUC:", grid_rf.best_score_)

# 29.2. BEST RANDOM FOREST MODEL WITH OPTIMAL PARAMETERS
best_rf = RandomForestClassifier(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)

best_rf.fit(X_train_resampled, y_train_resampled)
y_pred_best_rf = best_rf.predict(X_test)
y_prob_best_rf = best_rf.predict_proba(X_test)[:, 1]

best_rf_accuracy = accuracy_score(y_test, y_pred_best_rf)
best_rf_auc = roc_auc_score(y_test, y_prob_best_rf)
best_rf_f1 = f1_score(y_test, y_pred_best_rf)

print("Optimized RF Accuracy:", best_rf_accuracy)
print("Optimized RF ROC AUC:", best_rf_auc)
print("Optimized RF F1-Score:", best_rf_f1)

#COMMENT: The optimized Random Forest model, tuned via GridSearchCV, achieved an accuracy of 77.25%, a ROC AUC of 0.8187, and an F1-score of 0.5966. While its overall classification accuracy remained comparable to the baseline model, the improvement in AUC suggests better discrimination between churn and non-churn classes. However, the relatively lower F1-score highlights that the model may still struggle with imbalanced class predictions, especially in correctly identifying churned customers. This trade-off implies that although the Random Forest performs well in ranking customer churn risk, further tuning or class balancing techniques may be needed to enhance precision and recall simultaneously.


# 29.3. CROSS-VALIDATION SCORE WITH LOGISTIC REGRESSION

from sklearn.model_selection import StratifiedKFold, cross_val_score

# Instantiate logistic regression model
baseline_model = LogisticRegression(max_iter=1000, random_state=42)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation using ROC AUC as the scoring metric
lr_cv_scores = cross_val_score(baseline_model, X, y, cv=cv, scoring='roc_auc')

# Print the individual and average AUC scores
print("\nLogistic Regression CV ROC AUC Scores:", lr_cv_scores)
print("Mean CV ROC AUC:", np.mean(lr_cv_scores))

# COMMENT: The Logistic Regression model achieved consistent cross-validation ROC AUC scores ranging from 0.8401 to 0.8480, with an average of 0.8449. This indicates strong and stable discriminative performance across folds, making it a robust and interpretable model for predicting customer churn.

# 30. FINAL COMMENTS
# The analysis of the Telco dataset has provided valuable insights into customer churn. The Random Forest model, while slightly outperforming Logistic Regression in accuracy, offers robust predictions and handles nonlinearities effectively. However, Logistic Regression's interpretability and higher ROC AUC make it a strong candidate for applications where understanding feature impact is crucial.

# Telco models and their AUC scores
models_telco = ["Logistic Regression", "Random Forest", "XGBoost"," Best Random Forest"," Cross-Validated Logistic Regression", "Optimized Random Forest"]
auc_scores_telco = [0.83, 0.82, 0.80, 0.92, 0.84, 0.81]  # Example AUC scores for each model  

# Create the bar plot
plt.figure(figsize=(8, 5))
bars = plt.bar(models_telco, auc_scores_telco, color='skyblue', alpha=0.8)
plt.title("Model Comparison on Telco Dataset (AUC Scores)")
plt.ylabel("AUC Score")
plt.ylim(0.7, 0.9)

# Add AUC score labels above bars
for bar, score in zip(bars, auc_scores_telco):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{score:.2f}", ha='center', va='bottom')

plt.xticks(rotation=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

##COMMENT:    |MODEL|---------------|Accuracy|---|ROC AUC|
#             |Random Forest|---------|0.773|-----|0.817|
#             |Logistic Regression|---|0.763|-----|0.831|
#             |XGBoost|---------------|0.755|-----|0.805|
## COMMENT : While the Random Forest model offers slightly better classification accuracy, the Logistic Regression model demonstrates the strongest discriminative power (as evidenced by its highest ROC AUC). This makes Logistic Regression particularly useful when interpretability and ranking quality are critical, despite its slightly lower accuracy.XGBoost, on the other hand, is a highly flexible and powerful model, often achieving competitive performance due to its gradient boosting approach. In this analysis, its performance is slightly below the other models, but it remains valuable due to its ability to capture complex feature interactions and nonlinear patterns. However, it may require more tuning and computational resources.Depending on the business objective — whether prioritizing interpretability, raw accuracy, or complex decision boundaries — all three models present valuable and complementary insights.
