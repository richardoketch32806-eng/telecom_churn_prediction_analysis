# Telecom Churn Prediction Analysis

## Overview
Customer churn is a major challenge in the telecom industry. This project uses machine learning to predict which customers are likely to leave, enabling proactive retention strategies.

**Objective:** Build a predictive model and generate actionable insights to reduce churn and improve business performance.

---

## Business and Data Understanding

### Business Problem
- 14.5% annual churn rate
- High cost of acquiring new customers
- No system to identify at-risk customers early

### Dataset
- 3,333 customers
- 21 features
- Target: Churn (Yes/No)

---

## Exploratory Data Analysis (EDA)

### Churn Distribution
![Churn Rate](visual_01_waffle_churn_rate.png)

- Shows class imbalance
- Majority of customers do not churn

---

### Usage Patterns
![Usage Distribution](visual_02_kde_usage_distributions.png)

- High usage customers more likely to churn

---

### Feature Relationships
![Violin Plots](visual_03_violin_plots.png)

![Service Calls](visual_04_strip_service_calls.png)

- Customers with more service calls churn more

---

### Multivariate Analysis
![Pair Plot](visual_05_pair_plot.png)

![Correlation Heatmap](visual_06_correlation_heatmap.png)

- Identifies relationships between variables
- Helps guide feature selection

---

## Data Preparation

- No missing values
- Encoded categorical variables
- Scaled numerical features
- Selected relevant predictors

---

## Modeling

### Models Tested
- Logistic Regression (Baseline & Tuned)
- Decision Tree (Baseline & Tuned)
- Random Forest (Final Model)

---

## Model Evaluation

### Confusion Matrices
![M1](visual_07a_m1_confusion_matrix.png)
![M2](visual_07b_m2_confusion_matrix.png)
![M3](visual_07c_m3_confusion_matrix.png)
![M4](visual_07d_m4_confusion_matrix.png)
![M5](visual_07e_m5_confusion_matrix.png)

---

### Decision Tree Visualization
![Decision Tree](visual_08_decision_tree.png)

---

### Model Comparison
![Recall Gap](visual_09_dumbbell_recall_gap.png)

![Radar Chart](visual_10_radar_chart.png)

---

### ROC Curve
![ROC Curve](visual_11_roc_curves.png)

---

### Precision-Recall Curve
![PR Curve](visual_12_pr_curve.png)

---

## Feature Importance

### Lollipop Chart
![Feature Importance](visual_13_feature_importance_lollipop.png)

### Treemap
![Feature Importance Treemap](visual_14_feature_importance_treemap.png)

---

## Key Insights

- Customers with **high service calls** are most likely to churn  
- **High daytime usage** increases churn risk  
- **International plan users** show higher churn  

---

## Conclusion & Recommendations

### Best Model
- Random Forest
- Recall: ~72%
- AUC: 0.88

---

### Business Recommendations

1. **Act after 3rd service call**
2. **Target high-usage customers**
3. **Review international plan pricing**

---

## Business Value

- Enables proactive retention  
- Reduces churn-related revenue loss  
- Improves customer satisfaction  

---

## Project Structure

