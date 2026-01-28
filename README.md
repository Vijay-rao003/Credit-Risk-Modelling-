# Credit-Risk-Modelling-
credit-risk, risk-modeling, probability-of-default, pd-model, logistic-regression, xgboost, calibration, expected-loss, machine-learning, data-science, finance, banking, portfolio-analytics

Project Overview
. This project builds a full Credit Risk Modeling pipeline to estimate Probability of Default (PD) from a consumer loan dataset (cr_loan.csv).

.It benchmarks the industry-standard Logistic Regression model against a modern Gradient Boosting model (XGBoost), then translates risk scores into portfolio decisions using Expected Loss (EL = PD × LGD × EAD) and acceptance-rate strategy curves.

.Unlike “just training a classifier,” this project is built to mirror real financial risk management:

(a)leakage-safe preprocessing pipelines

(b)imbalanced-class handling (SMOTE)

(c)ROC/AUC + probability calibration (PD accuracy)

(d)strategy tables to quantify growth vs risk trade-offs

📊 Key Visualizations

1) Segment Risk Diagnostics (EDA)
.Default-rate cross-tabs & pivot tables reveal which borrower/loan segments carry the most risk (e.g., intent, grade, home ownership).
.Outlier plots (boxplots + scatter) identify extreme exposures and affordability risk patterns that can distort PD estimates.

What it proves: the model is built on clean, explainable risk signals—not noisy data artifacts.

2) Model Discrimination: ROC Curve Comparison (Logit vs XGBoost)
.ROC curves compare how well each model ranks risky borrowers above safe borrowers across thresholds.
This is a core “model power” diagnostic in credit scoring.
What it means: better rank-ordering enables stronger underwriting segmentation and pricing.

3) Calibration Curves (PD Accuracy — Crucial in Banking)
.AUC alone is not enough. Banks need calibrated PDs because PD feeds provisioning, pricing, and EL calculations.
.The calibration curve checks whether predicted PD bins match observed default rates.
What it proves: the model outputs are meaningful probabilities—not just scores.

4) Strategy & Economics: Acceptance Rate vs Bad Rate / Portfolio Value
A bank’s real question is:
“If we accept X% of applicants, what bad rate and expected loss do we take—and what portfolio value do we create?”

This project generates: a Strategy Table (PD cutoff by acceptance rate)
.Acceptance vs Bad Rate curve
.Acceptance vs Value Proxy (EAD − EL) curve
What it proves: the model is actionable for underwriting and risk appetite decisions.

❓ The Question (Problem Statement)
Credit portfolios face a constant trade-off between:

Growth: approving more loans increases portfolio size and revenue potential

Risk: higher acceptance increases defaults and expected loss

Most ML projects stop at metrics.This project goes further:

.Build PD models → validate (AUC + calibration) → design underwriting thresholds using Expected Loss.

The Goal: Create a bank-style decision framework that quantifies how changing PD thresholds impacts:

.default risk (bad rate)

.expected loss (EL)

.portfolio value (growth vs loss)

✅ The Answer (Results)
This project produces two competing PD models:

.Logistic Regression: interpretable, governance-friendly baseline

.XGBoost: stronger non-linear fit, often higher AUC

They are compared on:

.ROC/AUC (discrimination power)

.Calibration curves + Brier score (probability quality)

Then converted into strategy using:

.Expected Loss (PD × LGD × EAD)

.Acceptance-rate simulations to select thresholds consistent with risk appetit

segment-level default analysis 

phase 1- (cross-tabs/pivots):

.outlier detection (boxplots/scatter)

.missing data profiling and treatment

.data quality filters (e.g., unrealistic employment length)

Phase 2 — Logistic Regression (Industry Standard) leakage-safe preprocessing:

.numeric median imputation

.categorical mode imputation + one-hot encoding

.PD modeling using predict_proba

.confusion matrix + precision/recall trade-off

Phase 3 — XGBoost (Advanced Model):

.gradient boosting classifier

.class imbalance handling with SMOTE

.stratified k-fold cross-validation for stability

.risk driver analysis using Permutation Importance (AUC impact)

Phase 4 — Evaluation & Business Strategy:

.ROC/AUC comparison

.calibration curves (PD accuracy)

.strategy table:

(a)PD thresholds by acceptance rate

(b)observed bad rate

(c)EL and value proxy comparisons

.Predicting a New Customer PD (probability and Default) and Loan Approval
The trained model is used to score new loan applicants by estimating their Probability of Default (PD). 
For a new borrower, the same preprocessing pipeline (imputation and encoding) is applied, and the model outputs a PD value between 0 and 1, representing the likelihood of default.

In this project, a PD cutoff of 10% is selected based on the acceptance-rate vs bad-rate strategy curve, representing a balanced trade-off between portfolio growth and risk control.

.If PD ≤ 10% → Loan is Approved

.If PD > 10% → Loan is Rejected

.For example, a new applicant with a predicted PD ≈ 9.05% falls below the cutoff and would be approved.
The predicted PD is further translated into Expected Loss (EL = PD × LGD × EAD), allowing the bank to quantify potential loss and support pricing and underwriting decisions.

🛠️ Libraries & Tech Stack

.pandas,numpy- Data cleaning,feature prep,numerical computation

.matplotlib-all plots(EDA,ROC,calibration,strategy curves)

.scikit-learn- pipelines,preprocessing,metrices,calibration,CV

.xgboost-Gradient boosting model for PD prediction

.(imbalanced-learn) - SMOTE for class imbalance handling





