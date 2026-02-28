# Determining Which Features need to be engineered

import pandas as pd
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================

file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline Analytics.csv"
df = pd.read_csv(file_path)

df['signup_date'] = pd.to_datetime(df['signup_date'])

# =========================================================
# FIX CHURN VARIABLE (CRITICAL)
# =========================================================

# Preserve original
df['churn_original'] = df['churn']

# Convert Yes/No to 1/0
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0}).astype(int)

print("\n================ BASIC STRUCTURE ================\n")
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# =========================================================
# TARGET DISTRIBUTION
# =========================================================

print("\n================ TARGET DISTRIBUTION ================\n")
print(df['churn'].value_counts())
print("\nChurn Rate:", df['churn'].mean())

# =========================================================
# NUMERICAL FEATURE SUMMARY
# =========================================================

numeric_cols = [
    'monthly_fee',
    'avg_weekly_usage_hours',
    'support_tickets',
    'payment_failures',
    'tenure_months',
    'last_login_days_ago'
]

print("\n================ NUMERICAL SUMMARY ================\n")
print(df[numeric_cols].describe())

# =========================================================
# BEHAVIORAL SEPARATION
# =========================================================

print("\n================ GROUPED MEANS BY CHURN ================\n")
print(df.groupby('churn')[numeric_cols].mean())

print("\n================ GROUPED MEDIANS BY CHURN ================\n")
print(df.groupby('churn')[numeric_cols].median())

print("\n================ GROUPED STD BY CHURN ================\n")
print(df.groupby('churn')[numeric_cols].std())

# =========================================================
# CORRELATION MATRIX
# =========================================================

print("\n================ CORRELATION MATRIX ================\n")
print(df[numeric_cols + ['churn']].corr())

# =========================================================
# TEMPORAL STRUCTURE CHECK
# =========================================================

print("\n================ TEMPORAL STRUCTURE ================\n")

df['tenure_days'] = df['tenure_months'] * 30
df['estimated_event_date'] = df['signup_date'] + pd.to_timedelta(df['tenure_days'], unit='D')

print("Signup Date Range:", df['signup_date'].min(), "to", df['signup_date'].max())
print("Estimated Event Date Range:", df['estimated_event_date'].min(), "to", df['estimated_event_date'].max())

snapshot_estimate = df['estimated_event_date'].max()
print("Estimated Snapshot Date:", snapshot_estimate)

df['estimated_last_login_date'] = snapshot_estimate - pd.to_timedelta(df['last_login_days_ago'], unit='D')

print("\nLast Login Date Range:",
      df['estimated_last_login_date'].min(),
      "to",
      df['estimated_last_login_date'].max())

df['login_gap_before_event'] = (
    df['estimated_event_date'] - df['estimated_last_login_date']
).dt.days

print("\nLogin Gap Before Event (Churned Only):")
print(df[df['churn'] == 1]['login_gap_before_event'].describe())

# =========================================================
# NORMALIZED BEHAVIOR (RATE FEATURES)
# =========================================================

df['ticket_rate'] = df['support_tickets'] / df['tenure_months'].replace(0, np.nan)
df['failure_rate'] = df['payment_failures'] / df['tenure_months'].replace(0, np.nan)

print("\n================ RATE FEATURES SUMMARY ================\n")
print(df[['ticket_rate', 'failure_rate']].describe())

print("\nGrouped Means (Rates) by Churn:\n")
print(df.groupby('churn')[['ticket_rate', 'failure_rate']].mean())

print("\nCorrelation with Churn (Rates):\n")
print(df[['ticket_rate', 'failure_rate', 'churn']].corr())

# =========================================================
# INTERACTION CANDIDATES
# =========================================================

df['tickets_x_failures'] = df['support_tickets'] * df['payment_failures']
df['usage_x_recency'] = df['avg_weekly_usage_hours'] * df['last_login_days_ago']
df['failures_x_recency'] = df['payment_failures'] * df['last_login_days_ago']
df['tickets_x_recency'] = df['support_tickets'] * df['last_login_days_ago']

interaction_cols = [
    'tickets_x_failures',
    'usage_x_recency',
    'failures_x_recency',
    'tickets_x_recency'
]

print("\n================ INTERACTION SUMMARY ================\n")
print(df[interaction_cols].describe())

print("\nGrouped Means (Interactions) by Churn:\n")
print(df.groupby('churn')[interaction_cols].mean())

print("\nCorrelation with Churn (Interactions):\n")
print(df[interaction_cols + ['churn']].corr())

# =========================================================
# STRUCTURAL CHECK
# =========================================================

print("\n================ PLAN TYPE DISTRIBUTION ================\n")
print(df['plan_type'].value_counts())
print("\nPlan Type vs Churn:\n")
print(pd.crosstab(df['plan_type'], df['churn'], normalize='index'))

# =========================================================
# VARIANCE / INFORMATION CHECK
# =========================================================

print("\n================ VARIANCE CHECK ================\n")
print(df[numeric_cols].var())

print("\nUnique Values Per Feature:\n")
for col in numeric_cols:
    print(col, ":", df[col].nunique())

# =========================================================
# SKEWNESS CHECK
# =========================================================

print("\n================ SKEWNESS ================\n")
print(df[numeric_cols].skew())

# =========================================================
# FINAL ENGINEERING READINESS SUMMARY
# =========================================================

print("\n================ ENGINEERING READINESS SUMMARY ================\n")

summary_table = pd.DataFrame({
    "Mean": df[numeric_cols].mean(),
    "Std": df[numeric_cols].std(),
    "Variance": df[numeric_cols].var(),
    "Skewness": df[numeric_cols].skew(),
    "Correlation_with_Churn": df[numeric_cols + ['churn']].corr()['churn'].drop('churn')
})

print(summary_table)

import pandas as pd
import numpy as np

# ===============================
# LOAD DATA
# ===============================
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline Analytics.csv"
df = pd.read_csv(file_path)

# ===============================
# CHURN TO BINARY (YES/NO → 1/0)
# ===============================
df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

# ===============================
# FEATURE ENGINEERING
# ===============================

# --- Log Transforms ---
df["log_support_tickets"] = np.log1p(df["support_tickets"])
df["log_payment_failures"] = np.log1p(df["payment_failures"])
df["log_last_login_days_ago"] = np.log1p(df["last_login_days_ago"])
df["log_usage"] = np.log1p(df["avg_weekly_usage_hours"])

# --- Quadratic Terms ---
df["recency_squared"] = df["last_login_days_ago"] ** 2
df["failures_squared"] = df["payment_failures"] ** 2
df["tickets_squared"] = df["support_tickets"] ** 2
df["usage_squared"] = df["avg_weekly_usage_hours"] ** 2

# --- Tenure-Normalized Rates ---
df["ticket_rate"] = df["support_tickets"] / df["tenure_months"]
df["failure_rate"] = df["payment_failures"] / df["tenure_months"]

df["ticket_rate_smooth"] = df["support_tickets"] / (df["tenure_months"] + 1)
df["failure_rate_smooth"] = df["payment_failures"] / (df["tenure_months"] + 1)

# --- Core Interactions ---
df["tickets_x_failures"] = df["support_tickets"] * df["payment_failures"]
df["tickets_x_recency"] = df["support_tickets"] * df["last_login_days_ago"]
df["failures_x_recency"] = df["payment_failures"] * df["last_login_days_ago"]
df["usage_x_recency"] = df["avg_weekly_usage_hours"] * df["last_login_days_ago"]

# --- Composite Friction ---
df["friction_total"] = df["support_tickets"] + df["payment_failures"]
df["friction_per_month"] = df["friction_total"] / df["tenure_months"]

df["engagement_gap"] = df["last_login_days_ago"] / (df["avg_weekly_usage_hours"] + 1)

df["stress_stack_raw"] = (
    df["payment_failures"] * 2 +
    df["support_tickets"] * 1.5 +
    df["last_login_days_ago"] * 0.5
)

# --- Tenure Interactions ---
df["tenure_x_failures"] = df["tenure_months"] * df["payment_failures"]
df["tenure_x_tickets"] = df["tenure_months"] * df["support_tickets"]
df["tenure_x_recency"] = df["tenure_months"] * df["last_login_days_ago"]

# --- Ratio Features ---
df["failures_to_tickets_ratio"] = df["payment_failures"] / (df["support_tickets"] + 1)
df["recency_to_tenure_ratio"] = df["last_login_days_ago"] / df["tenure_months"]
df["usage_to_fee_ratio"] = df["avg_weekly_usage_hours"] / df["monthly_fee"]

# ===============================
# SAVE NEW DATASET
# ===============================
output_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline_Analytics_Engineered.csv"
df.to_csv(output_path, index=False)

print("New engineered dataset saved successfully.")
print("Final shape:", df.shape)

#Feature Triage 

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# ===============================
# LOAD ENGINEERED DATA
# ===============================

file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline_Analytics_Engineered.csv"
df = pd.read_csv(file_path)

# ===============================
# DEFINE FEATURE SET
# ===============================

exclude_cols = ["user_id", "signup_date", "churn"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df["churn"]

# Ensure numeric only for MI
X_numeric = X.select_dtypes(include=[np.number])

# ===============================
# BASIC METRICS
# ===============================

variance = X_numeric.var()
unique_counts = X_numeric.nunique()
missing_rate = X_numeric.isnull().mean()

# Pearson correlation
correlation = X_numeric.corrwith(y)

# Mutual Information
mi = mutual_info_classif(X_numeric.fillna(0), y, random_state=42)
mi_series = pd.Series(mi, index=X_numeric.columns)

# ===============================
# EFFECT SIZE (Standardized Mean Difference)
# ===============================

mean_0 = X_numeric[df["churn"] == 0].mean()
mean_1 = X_numeric[df["churn"] == 1].mean()

std_pooled = np.sqrt(
    (X_numeric[df["churn"] == 0].var() +
     X_numeric[df["churn"] == 1].var()) / 2
)

effect_size = (mean_1 - mean_0) / std_pooled

# ===============================
# BUILD FEATURE SCAN TABLE
# ===============================

feature_scan = pd.DataFrame({
    "variance": variance,
    "unique_values": unique_counts,
    "missing_rate": missing_rate,
    "correlation_with_churn": correlation,
    "mutual_information": mi_series,
    "effect_size": effect_size
})

feature_scan["abs_correlation"] = feature_scan["correlation_with_churn"].abs()
feature_scan["abs_effect_size"] = feature_scan["effect_size"].abs()

# ===============================
# NORMALIZE SCORING METRICS
# ===============================

scaler = MinMaxScaler()

scoring_matrix = scaler.fit_transform(
    feature_scan[["abs_correlation", "mutual_information", "abs_effect_size"]]
)

feature_scan["triage_score"] = scoring_matrix.sum(axis=1)

# ===============================
# EMPIRICAL TIER ASSIGNMENT
# ===============================

feature_scan = feature_scan.sort_values("triage_score", ascending=False)

percentiles = feature_scan["triage_score"].rank(pct=True)

feature_scan["tier"] = pd.cut(
    percentiles,
    bins=[0, 0.25, 0.5, 0.75, 1.0],
    labels=["Tier 4 (Low)", "Tier 3 (Moderate)", "Tier 2 (Strong)", "Tier 1 (High)"]
)

# ===============================
# SAVE FEATURE SCAN TABLE
# ===============================

scan_output_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Feature_Triage_Scan.csv"
feature_scan.to_csv(scan_output_path)

print("Feature scan saved.")
print(feature_scan.head(10))

# ===============================
# CREATE CHURN-SPLIT DIAGNOSTIC DATASET
# ===============================

df_churned = df[df["churn"] == 1].copy()
df_not_churned = df[df["churn"] == 0].copy()

df_churned["churn_group"] = "Churned"
df_not_churned["churn_group"] = "Retained"

diagnostic_df = pd.concat([df_churned, df_not_churned], axis=0)

diagnostic_output_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline_Diagnostic_ChurnSplit.csv"
diagnostic_df.to_csv(diagnostic_output_path, index=False)

print("Diagnostic churn-split dataset saved.")
