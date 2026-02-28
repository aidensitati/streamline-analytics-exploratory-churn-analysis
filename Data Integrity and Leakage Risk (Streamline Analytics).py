import pandas as pd
import numpy as np

# ==============================
# 1. Load Dataset
# ==============================

file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline Analytics.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully\n")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

# ==============================
# 2. Basic Structural Validation
# ==============================

print("\n--- Checking for Duplicate user_id ---")
duplicate_users = df['user_id'].duplicated().sum()
print("Duplicate user_ids:", duplicate_users)

if duplicate_users > 0:
    print("WARNING: Unit of analysis violation detected.")

# ==============================
# 3. Data Type & Timestamp Integrity
# ==============================

print("\n--- Converting signup_date to datetime ---")
df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')

print("Null signup_date values:", df['signup_date'].isnull().sum())

# ==============================
# 4. Lifecycle Asymmetry Check
# ==============================

print("\n--- Lifecycle Distribution ---")
print(df['churn'].value_counts())

print("\n--- Average Tenure by Churn Status ---")
print(df.groupby('churn')['tenure_months'].describe())

print("\n--- Exposure Difference Check ---")
print(df.groupby('churn')['avg_weekly_usage_hours'].mean())

# ==============================
# 5. Missingness Analysis
# ==============================

print("\n--- Missingness Summary ---")
missing_summary = df.isnull().sum().sort_values(ascending=False)
print(missing_summary)

# Missingness by churn status
print("\n--- Missingness by Churn Status ---")
missing_by_churn = df.groupby('churn').apply(lambda x: x.isnull().sum())
print(missing_by_churn)

# ==============================
# 6. Exposure-Sensitive Variable Flagging
# ==============================

print("\n--- Exposure Sensitivity Checks ---")

exposure_sensitive_vars = [
    'avg_weekly_usage_hours',
    'support_tickets',
    'payment_failures'
]

for col in exposure_sensitive_vars:
    corr = df[[col, 'tenure_months']].corr().iloc[0,1]
    print(f"{col} correlation with tenure_months: {corr}")

# ==============================
# 7. Lifecycle-Terminal Risk Indicators
# ==============================

print("\n--- Last Login Distribution by Churn ---")
print(df.groupby('churn')['last_login_days_ago'].describe())

# Potential leakage if churned users always have high last_login_days_ago
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# ==============================
# 8. Plan Distribution & Structural Effects
# ==============================

print("\n--- Plan Type Distribution ---")
print(df['plan_type'].value_counts())

print("\n--- Plan Type vs Churn ---")
print(pd.crosstab(df['plan_type'], df['churn'], normalize='index'))

print("\n--- Monthly Fee Distribution by Plan ---")
print(df.groupby('plan_type')['monthly_fee'].describe())

# ==============================
# 9. Right-Censoring Proxy Check
# ==============================

print("\n--- Tenure Distribution by Plan Type ---")
print(df.groupby('plan_type')['tenure_months'].describe())

# ==============================
# 10. Save Cleaned / Updated Version (if needed)
# ==============================

# Example: remove duplicate users if found
if duplicate_users > 0:
    df = df.drop_duplicates(subset='user_id')
    print("\nDuplicates removed.")

# Save updated dataset in same directory
updated_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline Analytics.csv"
df.to_csv(updated_path, index=False)

print("\nUpdated dataset saved to original location.")
