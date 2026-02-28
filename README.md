# streamline-analytics-exploratory-churn-analysis

# Project Overview
# Stage 1 – Data Reality & Generative Context

The dataset was grounded in an explicit SaaS operating model:

Subscription-based analytics platform with recurring fees.
Behavioral signals (usage, support tickets, login recency).
Structural signals (plan type, tenure, billing, churn).
Churn defined as discrete, agency-driven, and probabilistic.

Key validation outcomes:

One row per user; lifecycle-consistent measurement.
Churn is finalized and time-bounded.
Retained users are right-censored.
Behavioral and structural features are clearly distinguished.
The dataset is internally coherent and operationally plausible.

# Stage 2–3 – Analytical Framing & Integrity

EDA was treated as diagnostic, not predictive or causal.

Constraints enforced:

User-level unit of analysis.
Strict temporal discipline (no leakage).
Behavioral signals evaluated only pre-churn.
No structural confounding from plan type or fees.

Integrity checks confirmed:

No duplication.
No missing values.
Minimal exposure-driven bias.
Plan-level churn rates nearly identical.
Behavioral features largely independent of tenure.

Conclusion: Structural leakage risk is minimal. Analytical constraints shift to disciplined interpretation.

# Stage 4–5 – Target Analysis & Feature Triage

Initial separation showed:

Churn rate ~57%.
Structural variables (plan, fee, tenure) are largely neutral.
Behavioral features show subtle but meaningful divergence.

Feature engineering expanded the feature space (11 → 43 variables), emphasizing:

Interaction terms (failures × recency, tickets × recency).
Composite stress metrics (stress_stack_raw, friction_total).
Non-linear transforms.

Empirical triage identified high-signal variables:

Tier 1 (Primary Drivers)

stress_stack_raw
friction_total
failures × recency
tickets × recency
tickets × failures
engagement_gap
payment_failures

Key insight: churn is driven by compounded negative behaviors rather than isolated events.

# Stage 6 – Structural Diagnostic Exploration

Objective: characterize churn geometry.

Core findings:

Continuous Escalation Manifold
PCA and clustering reveal elongated gradients rather than discrete archetypes. Churn is process-driven, not type-driven.
Interaction Dominance
Compounded metrics (failures × recency, tickets × recency) define stress structure more strongly than raw counts.
Mid-Band Stress Regime
Churned users cluster in moderate-to-high accumulated stress zones. Extreme outliers are not required.
12–14 Month Regime Transition
Multiple features exhibit synchronized instability in this tenure window, suggesting coordinated escalation.
Absolute Metrics Preserve Structure
Smoothed or normalized rates obscure gradient geometry; absolute intensities retain signal clarity.
Interpretation: churn behaves like a coupled stress system—stable under moderate load, escalating along a continuous gradient, and occasionally destabilized within a specific tenure window.

# Stage 7 – Inferential Validation

Statistical testing focused on the 12–14 month transition window and full-population slope behavior.

Key Results

Significant Differentiators (p < 0.001):

friction_total
stress_stack_raw
failures × recency
tickets × recency
tickets × failures
engagement_gap
payment_failures

Churned users consistently exhibit:

Higher stress and friction.
Greater interaction amplification.
Lower engagement.
Monotonic Stress Gradient
Churn probability increases from 43% (lowest stress quantile) to 77% (highest).
Spearman ρ = 0.9 confirms strong monotonicity.
Slope Preservation
Stress effect remains highly significant across full population.
Escalation gradient is not an artifact of tenure filtering.
Interaction Structure
Stress predicts churn largely independent of tenure.
Minor stress × tenure interaction suggests slight attenuation at extreme tenures.

Effect Sizes
Largest standardized effects:

friction_total (d = 0.614)
tickets × failures (0.578)
stress_stack_raw (0.531)

These represent practical, actionable risk signals.

# What Churn Means in This Dataset

Churn is not random exit behavior.
It is the observable outcome of a structured, gradient-driven escalation of:

Accumulated friction
Interaction failures
Engagement decline
Stress compounds over time. Once accumulated strain exceeds individual tolerance thresholds, churn occurs.
The 12–14 month window represents a regime where divergence becomes most visible—not because tenure causes churn, but because escalation has had time to compound.

Churn in this dataset is:

Continuous, not archetypal
Interaction-driven, not isolated
Behavioral, not structural
Structured, but probabilistic

# Practical Implications

Prevention should focus on:

Monitoring compounded interaction metrics rather than raw counts.
Identifying mid-to-high stress quantiles before threshold breach.
Treating months 12–14 as proactive surveillance checkpoints.
Increasing perceived support while resolving friction to preserve tolerance.
Churn emerges from escalating strain.
Reducing escalation velocity and increasing user tolerance flattens the gradient.

# Final Summary

This repository demonstrates that churn within Streamline Analytics behaves as a measurable escalation process rather than a discrete shock event.
Through disciplined framing, leakage control, structural diagnostics, and inferential validation, the project maps the geometry of churn:
A continuous stress gradient, amplified by interaction effects, culminating in threshold breach.
The result is a defensible, lifecycle-aware understanding of how churn forms—and where intervention can meaningfully alter its trajectory.
