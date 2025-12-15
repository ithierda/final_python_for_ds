# 🎧 Churn Prediction on a Music Streaming Platform

This project focuses on **predicting user churn** for a music streaming service using event-level user logs.  
It was developed as part of a **Kaggle competition** and emphasizes correct problem formulation, temporal validation, and leakage-free feature engineering.

**Authors**  
- Ithier d’Aramon  
- Gaspard Baumelou  

---

## Project Overview

User retention is a key challenge for subscription-based platforms.  
The objective of this project is to **predict whether a user will churn in the 10 days following an observation window**, based on their historical usage behavior.

The dataset contains:
- ~17 million user events
- 19,140 users
- Logs between **October 1st and November 20th, 2018**

Each event includes user activity such as song plays, likes/dislikes, page visits, subscription changes, and cancellations.

---

## Key Challenges Addressed

- Correct **definition of churn** aligned with the Kaggle evaluation setup  
- Avoiding **temporal data leakage**
- Designing **behavioral and temporal features** that generalize
- Reconciling discrepancies between **local validation metrics** and **Kaggle leaderboard scores**

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Comparison of churners vs non-churners behavior
- Analysis of page visits, engagement, and activity over time
- Identification of strong but misleading signals (e.g. last activity date)

### 2. Baseline Model
- Simple churn definition based on `Cancellation Confirmation`
- Basic aggregated features
- XGBoost classifier  
=> Resulted in strong local ROC-AUC but poor Kaggle performance due to leakage

### 3. Leakage-Free Feature Engineering
- Aggregations over a **fixed observation window**
- Engagement, activity intensity, and negative interaction ratios
- Removal of features dependent on users’ last activity date

### 4. Advanced Features
- RFM (Recency, Frequency, Monetary) features
- Temporal trends (quartiles, early vs late activity)
- Rolling windows and activity acceleration
- Page-level ratios and engagement composition

### 5. Target Redefinition
- Churn defined relative to a **time cutoff**
- Prediction of churn occurring **after** the observation period
- Multiple cutoffs used to maximize data usage

---

## Models Used

- Logistic Regression (baseline)
- XGBoost
- LightGBM
- Ensemble (XGB + LGBM)
- AutoGluon (final model)

---

## Results

| Stage | Main Change | Validation ROC-AUC | Kaggle Score |
|------|-----------|------------------|-------------|
| Baseline | Naïve label + leaky features | Over-optimistic | ~0.51 |
| Step 1 | Leakage-free aggregates | ~0.77 | ~0.58 |
| Step 2 | Advanced temporal + RFM features | ~0.90 | ~0.63 |
| Step 3 | Tuned LightGBM / AutoGluon | ~0.72 | **~0.655** |

The final AutoGluon model achieved the **best Kaggle accuracy**.

---

## Key Takeaways

- Most performance gains came from **problem formulation**, not model complexity
- Correct label definition and temporal validation are critical
- Simple models with well-designed features can outperform complex pipelines
- Understanding the business objective matters more than optimizing metrics blindly

