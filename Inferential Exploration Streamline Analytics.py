import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==============================
# FILE PATHS
# ==============================

retained_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline analytics Retained.csv"
churned_path  = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline_Diagnostic_ChurnSplit.csv"

# ==============================
# LOAD DATA
# ==============================

retained = pd.read_csv(retained_path)
churned = pd.read_csv(churned_path)

# ==============================
# ADD CHURN LABEL
# ==============================

retained["churn"] = 0
churned["churn"] = 1

# ==============================
# INITIAL VALIDATION
# ==============================

print("\n===== INITIAL DATA VALIDATION =====")
print(f"Retained Shape: {retained.shape}")
print(f"Churned Shape:  {churned.shape}")

print("\nMissing Values (Retained):")
print(retained.isnull().sum().sum())

print("\nMissing Values (Churned):")
print(churned.isnull().sum().sum())

# ==============================
# 12–14 TENURE WINDOW FILTER
# ==============================

TENURE_COLUMN = "tenure_months"

retained_window = retained[
    (retained[TENURE_COLUMN] >= 12) &
    (retained[TENURE_COLUMN] <= 14)
]

churned_window = churned[
    (churned[TENURE_COLUMN] >= 12) &
    (churned[TENURE_COLUMN] <= 14)
]

print("\n===== AFTER 12–14 WINDOW FILTER =====")
print(f"Retained (12–14) Shape: {retained_window.shape}")
print(f"Churned (12–14) Shape:  {churned_window.shape}")

print("\nMissing Values After Window Filter")
print(f"Retained: {retained_window.isnull().sum().sum()}")
print(f"Churned:  {churned_window.isnull().sum().sum()}")

# ==============================
# CONJOIN DATA (12–14)
# ==============================

combined = pd.concat([retained_window, churned_window], axis=0)

print("\n===== COMBINED DATA =====")
print(f"Combined Shape: {combined.shape}")
print("Churn Distribution:")
print(combined["churn"].value_counts())

# ==============================
# FEATURES FOR INFERENTIAL TESTING
# ==============================

features = [
    "stress_stack_raw",
    "failures_x_recency",
    "tickets_x_recency",
    "payment_failures",
    "tickets_x_failures",
    "friction_total",
    "engagement_gap",
    "log_payment_failures",
    "log_last_login_days_ago",
    "failures_squared",
    "log_usage",
    "last_login_days_ago",
    "failure_rate_smooth",
    "tenure_x_failures",
    "recency_squared",
    "support_tickets"
]

# ==============================
# INFERENTIAL TESTING
# ==============================

results = []

for feature in features:
    if feature in combined.columns:
        churned_vals = combined[combined["churn"] == 1][feature].dropna()
        retained_vals = combined[combined["churn"] == 0][feature].dropna()
        
        if len(churned_vals) > 1 and len(retained_vals) > 1:
            t_stat, p_value = stats.ttest_ind(churned_vals, retained_vals, equal_var=False)
            
            results.append({
                "Feature": feature,
                "Churn_Mean": round(churned_vals.mean(), 4),
                "Retained_Mean": round(retained_vals.mean(), 4),
                "Mean_Diff": round(churned_vals.mean() - retained_vals.mean(), 4),
                "T_Statistic": round(t_stat, 4),
                "P_Value": round(p_value, 6)
            })

results_df = pd.DataFrame(results).sort_values(by="P_Value")

print("\n===== INFERENTIAL TEST RESULTS (Sorted by P-Value) =====")
print(results_df.to_string(index=False))

results_df.to_csv("inferential_results_12_14_window.csv", index=False)

# ==========================================================
# STRUCTURAL VALIDATION SECTION
# ==========================================================

print("\n\n===== MONOTONIC CHURN PROBABILITY (STRESS QUANTILES) =====")

combined["stress_quantile"] = pd.qcut(
    combined["stress_stack_raw"],
    q=5,
    duplicates="drop"
)

quantile_churn = (
    combined.groupby("stress_quantile")["churn"]
    .mean()
    .reset_index()
)

print(quantile_churn)

rho, pval = spearmanr(
    np.arange(len(quantile_churn)),
    quantile_churn["churn"]
)

print(f"\nSpearman rho = {round(rho,4)}, p = {round(pval,6)}")

# ==========================================================
# SLOPE PRESERVATION (FULL POPULATION vs 12–14)
# ==========================================================

print("\n===== STRESS SLOPE (FULL POPULATION) =====")

full_population = pd.concat([retained, churned], axis=0)

X_full = sm.add_constant(full_population[["stress_stack_raw"]])
y_full = full_population["churn"]

model_full = sm.Logit(y_full, X_full).fit(disp=False)
print(model_full.summary())

print("\n===== STRESS SLOPE (12–14 WINDOW) =====")

X_window = sm.add_constant(combined[["stress_stack_raw"]])
y_window = combined["churn"]

model_window = sm.Logit(y_window, X_window).fit(disp=False)
print(model_window.summary())

# ==========================================================
# TENURE × STRESS INTERACTION
# ==========================================================

print("\n===== TENURE × STRESS INTERACTION MODEL =====")

full_population["stress_x_tenure"] = (
    full_population["stress_stack_raw"] *
    full_population[TENURE_COLUMN]
)

X_interact = sm.add_constant(
    full_population[[
        "stress_stack_raw",
        TENURE_COLUMN,
        "stress_x_tenure"
    ]]
)

interaction_model = sm.Logit(
    full_population["churn"],
    X_interact
).fit(disp=False)

print(interaction_model.summary())

# ==========================================================
# STANDARDIZED EFFECT SIZES (COHEN'S D)
# ==========================================================

print("\n===== STANDARDIZED EFFECT SIZES (COHEN'S D) =====")

def cohens_d(g1, g2):
    mean1, mean2 = np.mean(g1), np.mean(g2)
    std1, std2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / pooled

for feature in features:
    if feature in combined.columns:
        g1 = combined[combined["churn"] == 1][feature].dropna()
        g2 = combined[combined["churn"] == 0][feature].dropna()
        if len(g1) > 1 and len(g2) > 1:
            d = cohens_d(g1, g2)
            print(f"{feature}: d = {round(d,4)}")

# ==========================================================
# MULTIVARIATE DOMINANCE STRUCTURE
# ==========================================================

print("\n===== MULTIVARIATE DOMINANCE (STANDARDIZED LOGISTIC) =====")

dominance_features = [
    f for f in features
    if f in full_population.columns
]

X_multi = full_population[dominance_features].dropna()
y_multi = full_population.loc[X_multi.index, "churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_scaled, y_multi)

coef_df = pd.DataFrame({
    "Feature": dominance_features,
    "Standardized_Coefficient": log_reg.coef_[0]
}).sort_values(by="Standardized_Coefficient", ascending=False)

print(coef_df.to_string(index=False))


