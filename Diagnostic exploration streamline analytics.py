# ============================================================
# STREAMLINE ANALYTICS — DIAGNOSTIC EXPLORATION
# Population: Churned customers only
# Mode: Descriptive Diagnostic (No predictive framing)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. LOAD DATA
# -----------------------------

file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\Streamline_Diagnostic_ChurnSplit.csv"
df = pd.read_csv(file_path)

# Filter churned customers only
df = df[df["churn"] == 1].copy()

print("Dataset shape (churned only):", df.shape)
print("="*60)

# -----------------------------
# Tier 1 & Tier 2 Features
# -----------------------------

tier1 = [
    "stress_stack_raw",
    "failures_x_recency",
    "tickets_x_recency",
    "payment_failures",
    "tickets_x_failures",
    "friction_total",
    "engagement_gap",
    "log_payment_failures"
]

tier2 = [
    "log_last_login_days_ago",
    "failures_squared",
    "log_usage",
    "last_login_days_ago",
    "failure_rate_smooth",
    "tenure_x_failures",
    "recency_squared",
    "support_tickets"
]

diagnostic_features = tier1 + tier2

# ============================================================
# MODULE 1 — INTERNAL STRESS ARCHITECTURE
# ============================================================

print("\nMODULE 1 — INTERNAL STRESS ARCHITECTURE")
print("="*60)

for feature in tier1:
    print(f"\nFeature: {feature}")
    
    # Plot
    plt.figure()
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} Distribution (Churned Users)")
    plt.show()
    
    # Numerical Summary
    summary = {
        "Mean": df[feature].mean(),
        "Median": df[feature].median(),
        "Std": df[feature].std(),
        "Min": df[feature].min(),
        "Max": df[feature].max(),
        "Skewness": skew(df[feature]),
        "Kurtosis": kurtosis(df[feature])
    }
    
    print(pd.Series(summary))


# ============================================================
# MODULE 2 — ESCALATION TRAJECTORIES (Aligned to Churn)
# ============================================================

print("\nMODULE 2 — ESCALATION TRAJECTORIES")
print("="*60)

# Assume tenure_months approximates lifecycle progression
df_sorted = df.sort_values(["user_id", "tenure_months"])

trajectory_features = [
    "stress_stack_raw",
    "failures_x_recency",
    "tickets_x_recency",
    "engagement_gap",
    "last_login_days_ago"
]

# Group by tenure stage
for feature in trajectory_features:
    print(f"\nTrajectory for: {feature}")
    
    grouped = df_sorted.groupby("tenure_months")[feature]
    
    mean_vals = grouped.mean()
    median_vals = grouped.median()
    p25 = grouped.quantile(0.25)
    p75 = grouped.quantile(0.75)
    
    # Plot
    plt.figure()
    plt.plot(mean_vals.index, mean_vals.values)
    plt.title(f"{feature} Mean by Tenure (Churned)")
    plt.xlabel("Tenure Months")
    plt.ylabel(feature)
    plt.show()
    
    # Numerical Reporting
    trajectory_df = pd.DataFrame({
        "Mean": mean_vals,
        "Median": median_vals,
        "P25": p25,
        "P75": p75,
        "Delta_Mean": mean_vals.diff()
    })
    
    print(trajectory_df.head(15))


# ============================================================
# MODULE 3 — INTERACTION REGIME MAPPING
# ============================================================

print("\nMODULE 3 — INTERACTION REGIME MAPPING")
print("="*60)

interaction_features = [
    "failures_x_recency",
    "tickets_x_recency",
    "tickets_x_failures",
    "stress_stack_raw"
]

# Pairwise heatmaps
for i in range(len(interaction_features)):
    for j in range(i+1, len(interaction_features)):
        f1 = interaction_features[i]
        f2 = interaction_features[j]
        
        plt.figure()
        sns.kdeplot(
            x=df[f1],
            y=df[f2],
            fill=True
        )
        plt.title(f"{f1} vs {f2} Density")
        plt.show()

# Correlation Matrix
corr_matrix = df[interaction_features].corr()
print("\nInteraction Correlation Matrix:")
print(corr_matrix)


# ============================================================
# MODULE 4 — BEHAVIORAL STATE CLUSTERING
# ============================================================

print("\nMODULE 4 — BEHAVIORAL STATE CLUSTERING")
print("="*60)

clustering_features = tier1.copy()

X = df[clustering_features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df.loc[X.index].copy()
df_clustered["cluster"] = clusters

# Plot PCA embedding
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Projection of Churned Behavioral States")
plt.show()

# Cluster Summary
cluster_summary = df_clustered.groupby("cluster")[clustering_features].mean()
cluster_counts = df_clustered["cluster"].value_counts()

print("\nCluster Counts:")
print(cluster_counts)

print("\nCluster Feature Means:")
print(cluster_summary)


# ============================================================
# MODULE 5 — ABSOLUTE vs SMOOTHED VALIDATION
# ============================================================

print("\nMODULE 5 — ABSOLUTE vs SMOOTHED VALIDATION")
print("="*60)

absolute_features = [
    "payment_failures",
    "support_tickets",
    "stress_stack_raw"
]

smoothed_features = [
    "log_payment_failures",
    "failure_rate_smooth",
    "tenure_x_failures"
]

comparison_features = absolute_features + smoothed_features

for feature in comparison_features:
    print(f"\nFeature: {feature}")
    
    plt.figure()
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} Distribution")
    plt.show()
    
    summary = {
        "Mean": df[feature].mean(),
        "Std": df[feature].std(),
        "Variance": df[feature].var(),
        "Coefficient_of_Variation": df[feature].std() / df[feature].mean() if df[feature].mean() != 0 else np.nan
    }
    
    print(pd.Series(summary))


print("\nDIAGNOSTIC EXPLORATION COMPLETE.")
print("="*60)
