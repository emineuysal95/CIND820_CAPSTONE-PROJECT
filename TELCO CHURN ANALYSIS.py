## CIND 820 FINAL PROJECT : Customer Churn Prediction in E-commerce and Telecommunications

##ANALYSIS OF THE TELECOMUNICATION DATASET
# LOAD THE NECESSARY LIBRARIES
import pandas as pd
# DEFINE THE FILE PATH AND LOAD THE DATASET
file_path = "C:\\Users\\emine\\OneDrive\\Masaüstü\\CIND820\\IBM Telco Data.xlsx"
df_telco = pd.read_excel(file_path)

#DISPLAY THE STRUCTURE OF THE DATASET
df_telco.head()

# CHECK THE DATASET FOR MISSING VALUES AND DATA TYPES
df_telco.info()

# Also check for number of unique values per column to detect categorical/numerical split
unique_values = df_telco.nunique().sort_values()

unique_values

#DISPLAY THE FIRST FEW ROWS OF THE DATASET
print(df_telco.head())

# DROP CUSTOMERID COLUMN AS IT IS NOT NEEDED FOR ANALYSIS
df_telco.drop('customerID', axis=1, inplace=True)

# CONVERT 'TotalCharges' TO NUMERIC (coerce errors to NaN)
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')

#CHECK FOR MISSING VALUES IN 'TotalCharges'
missing_total_charges = df_telco['TotalCharges'].isnull().sum()

# Display number of missing values and rows with missing values
missing_total_charges, df_telco[df_telco['TotalCharges'].isnull()]

##Durum Analizi:TotalCharges sütununda 11 adet eksik değer var.Bu satırlarda tenure değeri 0, yani müşteriler hizmeti henüz hiç kullanmamış.tenure = 0 olanlar için TotalCharges doğal olarak NaN olmuş çünkü henüz fatura oluşmamış.Bu satırları veri setinden çıkarabiliriz çünkü:Çok az sayıda satırdır (%0.15).Model için anlamlı bir katkı sağlamazlar (henüz hizmet kullanımı olmamış).

# DROP ROWS WHERE 'TotalCharges' IS NULL
df_telco = df_telco[df_telco['TotalCharges'].notnull()]

# CONVERT 'Churn' TO BINARY VALUES (0 for 'No', 1 for 'Yes')
df_telco['Churn'] = df_telco['Churn'].map({'No': 0, 'Yes': 1})

# CONFIRM CHANGES
df_telco.info(), df_telco['Churn'].value_counts()

# IDENTIFIY CATEGORICAL VARIABLES
cat_cols = df_telco.select_dtypes(include=['object']).columns.tolist()

# Convert categorical variables to dummy/indicator variables
df_telco_encoded = pd.get_dummies(df_telco, columns=cat_cols, drop_first=True)

# CONFIRM TRANSFORMATION
df_telco_encoded.info()

#BUILDING A CORRELATION MATRIX AND PLOTTING THE HEATMAP
import matplotlib.pyplot as plt
import seaborn as sns

# COMPUTE THE CORRELATION MATRIX
corr_matrix = df_telco_encoded.corr()

# PLOTTING THE CORRELATION HEATMAP
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of All Features")
plt.tight_layout()
plt.show()

# PLOTTING THE CORRELATION OF ALL FEATURES WITH CHURN
plt.figure(figsize=(10, 8))
churn_corr_sorted = corr_matrix["Churn"].sort_values(ascending=False)
sns.barplot(x=churn_corr_sorted.values, y=churn_corr_sorted.index)
plt.title("Correlation of All Features with Churn")
plt.xlabel("Correlation with Churn")
plt.tight_layout()
plt.show()

# FOCUS ON FEATURES MOST CORRELATED WITH CHURN
churn_corr = corr_matrix['Churn'].sort_values(ascending=False)

# PICK TOP 10 POSITIVE AND NEGATIVE CORRELATIONS
top_positive = churn_corr[1:11]
top_negative = churn_corr[-10:]

# COMBINE TOP POSITIVE AND NEGATIVE CORRELATIONS
top_corr = pd.concat([top_positive, top_negative])

# PLOTTING THE TOP FEATURES CORRELATED WITH CHURN
plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr.values, y=top_corr.index)
plt.title('Top Features Correlated with Churn')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()

#Churn ile en yüksek pozitif ve negatif korelasyona sahip değişkenler:
##Pozitif Korelasyon (Churn artışıyla birlikte artan özellikler):InternetService_Fiber optic, OnlineSecurity_No internet service, TechSupport_No internet service, Contract_Month-to-month (dolaylı olarak), PaperlessBilling_Yes, PaymentMethod_Electronic check

##Negatif Korelasyon (Churn arttıkça azalan özellikler):Contract_Two year, OnlineSecurity_Yes, TechSupport_Yes, Partner_Yes, tenure (müşteri süresi), TotalCharges

#INTERPRETATION:Bu analiz, daha uzun süreli müşterilerin, güvenlik hizmeti alanların ve daha uzun kontratlı kullanıcıların ayrılma ihtimalinin düşük olduğunu gösteriyor. Bu bulgular modelleme ve iş stratejisi açısından oldukça anlamlı.

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# FEATUREA AND TARGET SEPARATION
X = df_telco_encoded.drop("Churn", axis=1)
y = df_telco_encoded["Churn"]

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# HANDLING CLASS IMBALANCE WITH SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# SCALING THE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Display the shapes of the resulting datasets
print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)

# Box plot for MonthlyCharges, TotalCharges, and tenure with respect to Churn
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. MonthlyCharges
sns.boxplot(x='Churn', y='MonthlyCharges', data=df_telco, ax=axes[0])
axes[0].set_title("Monthly Charges by Churn")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Monthly Charges")

# 2. TotalCharges
sns.boxplot(x='Churn', y='TotalCharges', data=df_telco, ax=axes[1])
axes[1].set_title("Total Charges by Churn")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Total Charges")

# 3. Tenure
sns.boxplot(x='Churn', y='tenure', data=df_telco, ax=axes[2])
axes[2].set_title("Tenure by Churn")
axes[2].set_xlabel("Churn")
axes[2].set_ylabel("Tenure (months)")

plt.tight_layout()
plt.show()

git init
git remote add origin https://github.com/KULLANICIADIN/telco-churn-analysis.git
git add telco_churn_analysis.py
git commit -m "Add Telco Churn Analysis Script"
git branch -M main
git push -u origin main


