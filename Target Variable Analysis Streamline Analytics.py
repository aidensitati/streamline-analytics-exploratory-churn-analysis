import pandas as pd

# File path
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline Analytics.csv"

# Load dataset
df = pd.read_csv(file_path)

# Basic info
print("Dataset Loaded Successfully\n")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

# Ensure churn is categorical
df['churn'] = df['churn'].astype('category')

# --- Class Balance ---
churn_counts = df['churn'].value_counts()
print("\n--- Churn Distribution ---")
print(churn_counts)

# Segment-level churn differences by plan_type
plan_churn = pd.crosstab(df['plan_type'], df['churn'], normalize='index')
print("\n--- Plan Type vs Churn ---")
print(plan_churn)

# Segment-level churn differences by monthly_fee (redundant with plan_type but included for completeness)
fee_churn = df.groupby('monthly_fee')['churn'].value_counts(normalize=True).unstack()
print("\n--- Monthly Fee vs Churn ---")
print(fee_churn)

# --- Temporal Churn Behavior ---
# Convert signup_date to datetime
df['signup_date'] = pd.to_datetime(df['signup_date'])

# Cohort churn over signup month
df['signup_month'] = df['signup_date'].dt.to_period('M')
cohort_churn = df.groupby('signup_month')['churn'].value_counts(normalize=True).unstack()
print("\n--- Cohort Churn Over Time ---")
print(cohort_churn)

# Early-life vs late-life churn patterns using tenure
tenure_summary = df.groupby('churn')['tenure_months'].describe()
print("\n--- Tenure Summary by Churn Status ---")
print(tenure_summary)

# --- Behavioral Variable Separation ---
behavioral_vars = ['avg_weekly_usage_hours', 'support_tickets', 'payment_failures', 'last_login_days_ago']
behavioral_summary = df.groupby('churn')[behavioral_vars].describe().T
print("\n--- Behavioral Variable Summary by Churn ---")
print(behavioral_summary)

# Correlation between behavioral variables and churn
# Encoding churn: Yes=1, No=0
df['churn_encoded'] = df['churn'].map({'No':0, 'Yes':1})
behavioral_corr = df[behavioral_vars + ['churn_encoded']].corr()['churn_encoded'].drop('churn_encoded')
print("\n--- Correlation of Behavioral Variables with Churn ---")
print(behavioral_corr)

# --- Baseline Performance Anchors ---
# Naive baseline: predict majority class
majority_class = df['churn'].mode()[0]
baseline_accuracy = (df['churn'] == majority_class).mean()
print(f"\n--- Naive Baseline Accuracy ---\nPredicting all '{majority_class}' -> Accuracy: {baseline_accuracy:.4f}")

# Simple heuristic baseline example: churn if last_login_days_ago > median
median_login = df['last_login_days_ago'].median()
df['heuristic_pred'] = df['last_login_days_ago'] > median_login
heuristic_accuracy = (df['heuristic_pred'] == df['churn_encoded']).mean()
print(f"\n--- Heuristic Baseline Accuracy (last_login_days_ago > median) ---: {heuristic_accuracy:.4f}")

# Cleanup temporary columns
df.drop(['churn_encoded', 'heuristic_pred', 'signup_month'], axis=1, inplace=True)

# Save processed dataset (optional)
df.to_csv(file_path, index=False)
print("\nUpdated dataset saved to original location.")
