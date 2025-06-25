# CIND 820 FINAL PROJECT : Customer Churn Prediction in E-commerce and Telecommunications
## THE E-COMMERCE CHURN ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# LOAD DATA
file_path = r"C:\Users\emine\OneDrive\Masaüstü\CIND820\ecommerce_customer_data.csv"
df_ecom = pd.read_csv(file_path)

# EDA REPORT GENERATION USING ydata_profiling
from ydata_profiling import ProfileReport

# Build the EDA report
profile = ProfileReport(df_ecom, title="E-Commerce Customer Churn - EDA Report", explorative=True)
# Save the report to an HTML file
profile.to_file(r"C:\Users\emine\OneDrive\Masaüstü\CIND820\ecommerce_eda_report.html")

# INITIAL DATA INSPECTION
print(df_ecom.head())
print(df_ecom.info())
print(df_ecom.describe(include='all'))
print(df_ecom.dtypes)

# DROP UNNECESSARY COLUMNS
df_ecom.drop(columns=['Customer ID', 'Customer Name'], inplace=True, errors='ignore')

# PROCESS DATE
df_ecom['Purchase Date'] = pd.to_datetime(df_ecom['Purchase Date'], errors='coerce')
df_ecom['PurchaseMonth'] = df_ecom['Purchase Date'].dt.month
df_ecom.drop(columns=['Purchase Date'], inplace=True)

# HANDLE DUPLICATES
df_ecom.drop_duplicates(inplace=True)

# HANDLE MISSING VALUES
df = df_ecom.copy()  # work on a copy

num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# DROP RETURNS COLUMN IF EXISTS
if 'Returns' in df.columns:
    df.drop(columns=['Returns'], inplace=True)
    print("Dropped 'Returns' column.")

# REDEFINE COLUMNS AFTER TRANSFORMATIONS
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

# ENCODE CATEGORICAL FEATURES FOR VIF CALCULATION
df_encoded = df_ecom.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# RECALCULATE VIF (Variance Inflation Factor)
num_features_for_vif = [col for col in df_encoded.columns if df_encoded[col].dtype in [np.float64, np.int64] and col != 'Churn']

# CHECK FOR INF AND NaN VALUES BEFORE VIF
print("NaN count before VIF:\n", df_encoded[num_features_for_vif].isna().sum())
print("Inf count before VIF:\n", np.isinf(df_encoded[num_features_for_vif]).sum())

# FIX INF VALUES AND FILL NAs
df_encoded[num_features_for_vif] = df_encoded[num_features_for_vif].replace([np.inf, -np.inf], np.nan)
df_encoded[num_features_for_vif] = df_encoded[num_features_for_vif].fillna(df_encoded[num_features_for_vif].median())
df_encoded.drop(columns=['Age'], inplace=True)
# COMMENT: The 'Age' column was removed due to high multicollinearity with 'Customer Age'. Keeping both would distort model estimates and inflate standard errors.

# VIF DATAFRAME
vif_data = pd.DataFrame()
num_features_for_vif = [col for col in num_features_for_vif if col != 'Age']
vif_data['Feature'] = num_features_for_vif
vif_data['VIF'] = [variance_inflation_factor(df_encoded[num_features_for_vif].values, i) for i in range(len(num_features_for_vif))]
print(vif_data)

# PLOT VIF
plt.figure(figsize=(10, 6))
sns.barplot(x='VIF', y='Feature', data=vif_data.sort_values(by='VIF', ascending=False))
plt.title('Variance Inflation Factor (VIF) for Numerical Features')
plt.xlabel('VIF')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

##COMMENT:The Variance Inflation Factor (VIF) analysis reveals that the variable Age has an extremely high VIF score, likely due to it being a duplicate or highly collinear with Customer Age. Since both variables represent similar information, keeping both can distort model estimates and inflate standard errors. To mitigate multicollinearity, Age should be removed from the feature set, retaining only Customer Age for clarity and stability in modeling.

# CHURN DISTRIBUTION ANALYSIS
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

churn_rate = df['Churn'].value_counts(normalize=True) * 100
print("Churn distribution (%):\n", churn_rate)

# BOXPLOT OF NUMERICAL FEATURES
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[num_cols])
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# COMMENT ON BOXPLOT: The boxplot shows that "Total Purchase Amount" and "Product Price" have the highest variability, with potential outliers. Other features like "Quantity", "Age", and "Churn" are more uniformly distributed. Feature scaling may be needed before modeling.

# PLOT HISTOGRAMS OF NUMERICAL FEATURES
df[num_cols].hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

# CORRELATION HEATMAP (NUMERICAL FEATURES + CHURN)
if 'Churn' in df.columns:
    plt.figure(figsize=(12, 8))
    corr = df[num_cols + ['Churn']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix (Numerical Features and Churn)")
    plt.tight_layout()
    plt.show()
# COMMENT ON CORRELATION: The correlation matrix shows that all numerical features have very weak or negligible correlations with customer churn. Variables such as Product Price, Quantity, Total Purchase Amount, Age, and Purchase Month do not exhibit any meaningful linear relationship with churn. This suggests that churn behavior is likely influenced more by categorical factors, and further analysis should focus on those features or consider creating new engineered features for better predictive insights.

# INDIVIDUAL BOXPLOTS (SELECTED FEATURES)
for col in ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age']:
    if col in df.columns:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()
# COMMENT ON INDIVIDUAL BOXPLOTS:
# - Product Price: Shows a wide range with some outliers, indicating a diverse product range.
# - Quantity: Mostly clustered around lower values (1-5), with few high outliers.
# - Total Purchase Amount: Displays significant variability, with some high outliers indicating high-spending customers.
# - Customer Age: Fairly evenly distributed, with no extreme outliers, suggesting a balanced customer base.

# FEATURE ENGINEERING
df['PriceToQuantity'] = df['Product Price'] / (df['Quantity'] + 1)
df['AvgItemValue'] = df['Total Purchase Amount'] / (df['Quantity'] + 1)
df['AgeGroup'] = pd.cut(df['Customer Age'], bins=[0, 25, 40, 60, 100],
                        labels=['GenZ', 'Millennial', 'GenX', 'Boomer'])

# CROSS-TABULATIONS WITH CHURN
for col in cat_cols:
    if 'Churn' in df.columns:
        print(f"\n{col} vs Churn")
        print(pd.crosstab(df[col], df['Churn'], normalize='index') * 100)

# DISPLAY CATEGORICAL DISTRIBUTIONS
for col in cat_cols:
    print(f"\n{col} distribution:\n{df[col].value_counts()}")

## COMMENT ON CATEGORICAL FEATURES
# Product Price: Prices are evenly distributed with no significant clustering. This suggests a wide range of product offerings across different price segments.
# Quantity:Customers mostly purchase between 1 and 5 units. The limited range indicates this variable may be treated as categorical in analysis.
# Total Purchase Amount:Purchase amounts span a broad range, indicating the presence of both low- and high-spending customers. This feature can be valuable for segmentation and churn prediction.
# Customer Age/Age:Age distribution is fairly balanced, though there is a slight dip in the 25–35 age group. Both younger and older customer groups are well represented in the dataset.
# Churn:There is a noticeable class imbalance—most customers did not churn, while a smaller group did. This imbalance should be addressed during the modeling phase (e.g., with resampling techniques).
# Purchase Month:Sales are higher during the first half of the year (especially from March to August), with a decline in the fall and winter months. This reflects seasonal purchasing behavior.

# CATEGORICAL FEATURE VISUALIZATION (Churn)
original_data = pd.read_csv(file_path)
original_data['Purchase Date'] = pd.to_datetime(original_data['Purchase Date'], errors='coerce')
original_data['AgeGroup'] = pd.cut(original_data['Customer Age'], bins=[0, 25, 40, 60, 100],
                                   labels=['GenZ', 'Millennial', 'GenX', 'Boomer'])

original_cat_cols = ['Gender', 'Payment Method', 'Product Category', 'AgeGroup']
for col in original_cat_cols:
    if col in original_data.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, hue='Churn', data=original_data)
        plt.title(f'{col} vs Churn')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# EDA FOR CATEGORICAL FEATURES AND CHURN
from scipy.stats import f_oneway
# ANOVA TEST FOR DIFFERENCES IN MEANS
# Assuming 'Total Purchase Amount' is the numerical feature of interest
groups = [df[df['Product Category'] == cat]['Total Purchase Amount'] for cat in df['Product Category'].unique()]
f_stat, p_value = f_oneway(*groups)

print("F-statistic:", f_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Result: There is a statistically significant difference between groups.")
else:
    print("Result: There is no statistically significant difference between groups.")


# CHI-SQUARE TEST FOR INDEPENDENCE(CATEGORICAL FEATURES)
from scipy.stats import chi2_contingency

for col in cat_cols:
    if col != 'Churn':
        table = pd.crosstab(df[col], df['Churn'])
        chi2, p, dof, _ = chi2_contingency(table)
        print(f"{col} vs Churn - p-value: {p:.4f}")


# CRAMÉR'S V FUNCTION(STATISTICAL MEASURE OF ASSOCIATION)
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

# CALCULATE CRAMÉR'S V SCORES
cramers_scores = {}
for col in original_cat_cols:
    if col in original_data.columns:
        score = cramers_v(original_data[col], original_data['Churn'])
        cramers_scores[col] = round(score, 3)

# PLOT CRAMÉR'S V SCORES
plt.figure(figsize=(8, 5))
sns.barplot(x=list(cramers_scores.values()), y=list(cramers_scores.keys()))
plt.xlabel("Cramér's V")
plt.title("Cramér's V between Categorical Features and Churn")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()\

# BUILT EDA REPORT FOR CLEANED E-COMMERCE DATA
from ydata_profiling import ProfileReport
# Assuming 'df' already contains the cleaned e-commerce dataset
df_ecom = df.copy()
# Generate the profiling report
profile_ecom = ProfileReport(df_ecom, title="EDA Report - Cleaned E-Commerce Data", explorative=True)
# Save the report to your desktop folder
profile_ecom.to_file("C:/Users/emine/OneDrive/Masaüstü/CIND820/eda_ecommerce_cleaned.html")


## MODELING AND PREDICTION OF CUSTOMER CHURN
# UPLOAD NECESSARY LIBRARIES
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import ADASYN

# BASELINE LOGISTIC REGRESSION MODEL(SCALED FEATURES)
# This function runs a baseline logistic regression model on the dataset.
# It scales the numerical features, splits the data into training and testing sets,
# trains a logistic regression model, and evaluates its performance.
def run_baseline_logistic_regression(df):
    df['PurchaseMonth'] = pd.to_datetime(df['Purchase Date'], errors='coerce').dt.month
    to_scale = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age', 'PurchaseMonth']
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[to_scale] = scaler.fit_transform(df_scaled[to_scale])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled[to_scale], df_scaled['Churn'],
        test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("=== BASELINE MODEL ===")
    print(classification_report(
    y_test, y_pred,
    zero_division=0  # or 1 to avoid warnings for zero division
))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title("Baseline: Scaled Logistic Regression")
    plt.show()
  
## COMMENT: The baseline logistic regression model provides a starting point for understanding the relationship between features and customer churn. The classification report shows precision, recall, and F1-score for each class, while the confusion matrix visualizes the model's performance. This model serves as a benchmark for comparing more complex models or techniques like ADASYN oversampling.
def run_adasyn_logistic_regression(df):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
    from sklearn.impute import SimpleImputer
    from imblearn.over_sampling import ADASYN


# 1. LABEL ENCODING
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

# 2. FEATURE AND TARGET SELECTION
    X = df_enc.drop(columns=['Churn'])
    y = df_enc['Churn']

# 3. TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

# 4. CONTROL NaN AND INFINITE VALUES
    print("Before cleaning -> NaN:", X_train.isna().sum().sum(), "Inf:", np.isinf(X_train.values).sum())

# 5. FILL MISSING VALUES AND HANDLE INFINITE VALUES
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    print("After cleaning  -> NaN:", X_train.isna().sum().sum(), "Inf:", np.isinf(X_train.values).sum())

# 6. APPLY ADASYN FOR OVER-SAMPLING
    X_res, y_res = ADASYN(random_state=42).fit_resample(X_train, y_train)
    print("=== ADASYN APPLIED ===")
    print("Resampled class distribution (%):")
    print(pd.Series(y_res).value_counts(normalize=True) * 100)

#  VISUALIZE THE CHURN DISTRIBUTION AFTER ADASYN
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y_res)
    plt.title('Churn Distribution After ADASYN')
    plt.xlabel('Churn (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()

# 7. TRAIN LOGISTIC REGRESSION MODEL AND EVALUATE
    model = LogisticRegression()
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title("ADASYN: Logistic Regression")
    plt.show()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve - ADASYN Logistic")
    plt.show()


# 8. RUN THE FUNCTIONS
if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\emine\OneDrive\Masaüstü\CIND820\ecommerce_customer_data.csv")
    run_baseline_logistic_regression(df)
    run_adasyn_logistic_regression(df)

# XGBoost MODEL FOR CUSTOMER CHURN PREDICTION
# IMPORT NECESSARY LIBRARIES
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

# 1. LABEL ENCODING AND FEATURE SELECTION
def run_xgboost_churn(df):
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    
    X = df_enc.drop(columns=['Churn'])
    y = df_enc['Churn']
    
# 2. TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
# 3. FILL MISSING VALUES
    imp = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)
    
# 4. CALCULATE SCALE_POS_WEIGHT FOR IMBALANCE HANDLING
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    
# 5. BUILD XGBoost MODEL
    model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
# 6. EVALUATE MODEL WITH CROSS-VALIDATION WITH ROC AUC
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-val AUC (5 folds): {scores.mean():.3f} ± {scores.std():.3f}")

# 7. TRAIN AND PREDICT
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    
# 8. CONFUSION MATRIX AND ROC CURVE
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve - XGBoost")
    plt.show()

# 9. RUN THE FUNCTION
if __name__ == "__main__":
    run_xgboost_churn(df)

##| Method             | AUC    | Comment                                        |
#| --------------------| ------ | -----------------------------------------------|
#| ADASYN + Logistic   | \~0.51 | Almost random, the discrimination power is weak|
#| XGBoost             | \~0.73 | Acceptable "discrimination power" is quite good|
# COMMENT: The ADASYN + Logistic Regression model achieved an AUC of approximately 0.51, indicating nearly random performance with no real predictive power. In contrast, the XGBoost model demonstrated a significantly better AUC of around 0.73, suggesting it has a reasonable level of discrimination ability in predicting customer churn.
# This indicates that while the ADASYN + Logistic Regression model struggled to capture meaningful patterns in.


# SHAP ANALYSIS FOR XGBoost MODEL

import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score
)

def run_xgboost_with_shap(df):
    import shap

    # 1. DROP UNNECESSARY COLUMNS
    # Assuming 'Customer Name' and 'Customer ID' are not needed for modeling
    cols_to_drop = ['Customer Name', 'Customer ID']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 2. IF 'Customer Age' AND 'Age' ARE THE SAME, DROP 'Age'
    # This is to avoid multicollinearity issues
    if 'Customer Age' in df.columns and 'Age' in df.columns:
        if df['Customer Age'].equals(df['Age']):
            df = df.drop(columns=['Age'])

    # Label encoding
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    X = df_enc.drop(columns=['Churn'])
    y = df_enc['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Impute missing values
    imp = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

    # Handle class imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # Train XGBoost model
    model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Churn-specific metrics
    print("=== Class 1 (Churn) Focused Metrics ===")
    print("Precision:", precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve - XGBoost")
    plt.show()

    # === SHAP ANALYSIS ===
    print("=== SHAP Feature Importance ===")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar")
    shap.summary_plot(shap_values, X_test)


# RUN FUNCTION
if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\emine\OneDrive\Masaüstü\CIND820\ecommerce_customer_data.csv")
    run_xgboost_with_shap(df)

# COMMENT: The SHAP analysis provides insights into feature importance and how each feature contributes to the model's predictions. The summary plots show the global importance of features, while individual SHAP values help understand specific predictions. This can guide further feature engineering or model refinement.
