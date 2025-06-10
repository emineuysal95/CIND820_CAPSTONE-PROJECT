## EXPLORATORY DATA ANALYSIS (EDA) OF THE E-COMMERCE DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

# LOAD DATA
file_path = r"C:\Users\emine\OneDrive\Masaüstü\CIND820\ecommerce_customer_data.csv"
df_ecom = pd.read_csv(file_path)

# DROP UNNECESSARY COLUMNS
df_ecom.drop(columns=['Customer ID', 'Customer Name'], inplace=True)

# PROCESS DATE
df_ecom['Purchase Date'] = pd.to_datetime(df_ecom['Purchase Date'], errors='coerce')
df_ecom['PurchaseMonth'] = df_ecom['Purchase Date'].dt.month
df_ecom.drop(columns=['Purchase Date'], inplace=True)

# HANDLE DUPLICATES
duplicates = df_ecom.duplicated().sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicate rows. Dropping duplicates.")
    df_ecom.drop_duplicates(inplace=True)
else:
    print("No duplicate rows found.")

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

# CHECK DATA TYPES
print("Data Types:")
print(df.dtypes)

# DESCRIPTIVE STATISTICS
print("Descriptive Statistics:")
print(df.describe(include='all'))

# PLOT HISTOGRAMS OF NUMERICAL FEATURES
df[num_cols].hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

# DISPLAY CATEGORICAL DISTRIBUTIONS
for col in cat_cols:
    print(f"\n{col} distribution:\n{df[col].value_counts()}")

# CORRELATION HEATMAP (NUMERICAL FEATURES + CHURN)
if 'Churn' in df.columns:
    plt.figure(figsize=(12, 8))
    corr = df[num_cols + ['Churn']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix (Numerical Features and Churn)")
    plt.tight_layout()
    plt.show()

# BOXPLOT OF NUMERICAL FEATURES
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[num_cols])
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# INDIVIDUAL BOXPLOTS (SELECTED FEATURES)
for col in ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age']:
    if col in df.columns:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

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

# CRAMÉR'S V FUNCTION
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
plt.show()

