# Telecom Churn Prediction Analysis

## Overview
Customer churn is a critical challenge in the telecommunications industry, directly impacting revenue and growth. This project focuses on predicting customer churn using machine learning techniques to enable proactive retention strategies.

The objective of this analysis is to build a predictive model that identifies customers at risk of leaving, allowing the business to take timely action and improve customer retention.

---

## Business and Data Understanding

### Stakeholder Audience
This project is designed for:
- **Business Executives** – to understand churn impact on revenue  
- **Marketing Teams** – to design targeted retention campaigns  
- **Customer Retention Teams** – to prioritize high-risk customers  
- **Data Teams** – to support data-driven decision-making  

### Business Problem
Customer churn leads to revenue loss and increased acquisition costs. The company currently lacks a system to identify customers likely to churn before they leave.

### Dataset
The dataset contains information on **3,333 telecom customers** with **21 features**, including:
- Customer usage patterns (day, evening, night minutes)
- Customer service interactions
- Subscription plans (e.g., international plan)
- Account-related information

**Target Variable:**
- `Churn` (Yes/No)

### Key Considerations
- The dataset has **no missing values**
- Class imbalance exists (majority do not churn)
- Accuracy alone is not sufficient → focus on **Recall**

---

## Modeling

### Approach
The problem is treated as a **binary classification task**:
- Predict whether a customer will **churn or stay**

### Models Tested
Five models were developed and compared:
1. Logistic Regression (Baseline)
2. Logistic Regression (Tuned)
3. Decision Tree (Baseline)
4. Decision Tree (Tuned)
5. Random Forest (Final Model)

### Feature Engineering
- Data normalization/scaling applied
- Categorical variables encoded
- Train-test split used to prevent data leakage

### Model Selection
The **Random Forest model** was selected as the final model due to:
- Better generalization
- Higher recall
- Strong overall performance

---

## Evaluation

### Key Metrics Used
- **Recall** → Ability to detect churners (most important)
- **Precision** → Accuracy of churn predictions
- **AUC (ROC Score)** → Overall model performance

### Final Model Performance
- **Recall:** ~72%  
- **Precision:** ~55%  
- **AUC:** 0.88  

### Interpretation
- The model correctly identifies **72% of customers who churn**
- Some false positives exist, but acceptable for retention strategy
- Missing a churner is more costly than a false alarm

---

## Conclusion

### Key Insights
- Customers with **multiple service calls** are highly likely to churn  
- **High daytime usage** correlates with higher churn risk  
- **International plan users** show increased churn probability  

### Business Impact
- Enables **proactive customer retention**
- Reduces revenue loss from churn
- Improves efficiency of retention campaigns

### Recommendations
- Flag customers after multiple service calls  
- Target high-usage customers with pricing incentives  
- Review international plan offerings  

### Limitations
- Model misses some churners (~28%)  
- Limited features (no demographic or contract data)  
- Requires continuous updates and retraining  

### Next Steps
- Deploy model into production  
- Integrate with CRM systems  
- Continuously retrain with new data  

---


