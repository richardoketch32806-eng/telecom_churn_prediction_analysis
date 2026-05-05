#  OVERVIEW

Customer churn — when a subscriber cancels their service — is one of the most costly problems in the telecommunications industry. Acquiring a new customer costs **five to seven times more** than retaining an existing one. Even a modest improvement in identifying at-risk customers before they leave has a direct, measurable impact on revenue.

This project builds a supervised machine learning pipeline to **predict which customers are likely to churn** using historical account and usage data. The analysis follows the **CRISP-DM framework** (Cross-Industry Standard Process for Data Mining), moving from business problem definition through to a deployable model with concrete retention recommendations.

**Key outcomes:**
- A Random Forest classifier achieving **72.2% recall** and **ROC-AUC of 0.882** on held-out test data
- Identification of the top churn drivers with direct business action mapped to each
- A threshold-adjustable scoring system the retention team can tune to match their outreach capacity

---

## 1. Business & Data Understanding

### Stakeholder Audience

The primary audience for this project is the **Customer Retention & Marketing team** at a mid-sized US telecom operator. This team is responsible for proactively reaching out to at-risk customers with targeted offers, service upgrades, or loyalty incentives — *before* those customers cancel their subscription.

A secondary audience is senior commercial leadership, who need assurance that the model's outputs are trustworthy and that the recommendations are grounded in observable customer behaviour rather than a black-box prediction.

### Why Machine Learning?

A simple business rule such as "flag customers with more than 3 service calls" captures one signal but misses the combined, non-linear interactions between usage patterns, plan type, call volume, and account history. A classification model can capture all of these signals simultaneously, quantify each one's contribution, and score every account on a continuous probability scale — giving the retention team a ranked priority list, not just a binary flag.

### Dataset

The dataset contains **3,333 customer records** and **21 features** sourced from a US telecom operator. Each record represents one customer account, and the target variable `churn` indicates whether that customer cancelled their service.

**Feature categories:**

| Category | Features |
|----------|----------|
| Account info | Account length, area code, state |
| Plan subscriptions | International plan, voice mail plan |
| Usage — daytime | Total day minutes, calls, charges |
| Usage — evening | Total evening minutes, calls, charges |
| Usage — night | Total night minutes, calls, charges |
| Usage — international | Total intl minutes, calls, charges |
| Service interactions | Customer service calls, number of voicemail messages |
| Target | Churn (True/False) |

**Key findings from EDA:**

| Finding | Implication |
|---------|-------------|
| 14.49% churn rate | Class imbalance — accuracy is misleading; use Recall and F1 |
| International plan holders: 42.4% churn vs 11.5% without | Single strongest predictor in the dataset |
| 4+ customer service calls: 45%+ churn rate | Non-linear threshold effect — not captured by linear models |
| Churned customers average 207 vs 175 day minutes | Higher usage → higher bills → price sensitivity |
| Voice mail plan holders churn at 8.7% vs 16.7% | A protective factor and a potential free retention incentive |
| Charge columns have r = 1.00 with minute columns | Perfect collinearity — charge columns dropped to avoid redundancy |
| Zero missing values across all 21 features | No imputation required |

### Business Success Metric

The cost asymmetry between error types drives the metric choice:

- A **false negative** (missing a real churner) means the customer is lost — high cost
- A **false positive** (contacting a loyal customer) means a wasted retention offer — low cost

Given this, the models are optimised in the following priority order:

1. **Recall** (primary) — catch as many true churners as possible
2. **F1-score** (secondary) — balance recall against precision
3. **ROC-AUC** (tertiary) — evaluate ranking quality at every threshold

---

## 2. Data Preparation

The raw dataset required targeted cleaning before modelling. No values were missing, so preparation focused on removing redundant features and encoding categoricals.

**Features removed (7 columns):**

| Column | Reason for removal |
|--------|--------------------|
| `phone number` | Unique identifier with no predictive signal |
| `state` | 51 categories — too high cardinality for this dataset size |
| `area code` | Only 3 distinct values; confirmed no churn signal |
| `total day charge` | Pearson r = 1.00 with `total day minutes` — perfectly collinear |
| `total eve charge` | Pearson r = 1.00 with `total eve minutes` |
| `total night charge` | Pearson r = 1.00 with `total night minutes` |
| `total intl charge` | Pearson r = 1.00 with `total intl minutes` |

**Encoding:**
- `international plan` and `voice mail plan` encoded as binary integers (yes → 1, no → 0)
- `churn` encoded as binary integer (True → 1, False → 0)

**Train / test split:**
An 80/20 stratified split was applied, preserving the 14.49% churn rate in both subsets. This produced 2,666 training rows and 667 test rows.

**Feature scaling:**
A `StandardScaler` was fitted on the training set only and applied to both splits — preventing any data leakage from test statistics into the training process. Scaling was applied for the logistic regression models; tree-based models used unscaled features as they are scale-invariant.

The cleaned dataset (14 features, 3,333 rows, zero missing values) was exported to `churn_telecoms_cleaned.csv` for reproducibility.

---

## 3. Modeling

Five models were trained in a progressive sequence, with each iteration addressing the shortcomings identified in the previous one.

### Model Progression

| Model | Algorithm | Key Design Choice | Purpose |
|-------|-----------|-------------------|---------|
| M1 | Logistic Regression | Default parameters | Establish performance floor |
| M2 | Logistic Regression | `class_weight='balanced'` + GridSearchCV on `C` | Correct for class imbalance |
| M3 | Decision Tree | Default parameters | Test non-linear boundary hypothesis |
| M4 | Decision Tree | GridSearchCV on `max_depth`, `min_samples_leaf`, `class_weight` | Reduce overfitting |
| M5 | Random Forest ⭐ | 200 trees, `class_weight='balanced'`, GridSearchCV | Final model — ensemble generalisation |

### Rationale for Each Step

**M1 → M2 (Logistic Regression, Baseline → Tuned):** The baseline model scored only 24.7% recall despite an ROC-AUC of 0.817 — the model could rank customers by risk but was badly calibrated at the default 0.5 threshold. Adding `class_weight='balanced'` forces the algorithm to penalise missed churners more heavily, and GridSearchCV identified the optimal regularisation strength `C` across 5-fold cross-validation.

**M2 → M3 (Logistic → Decision Tree):** With recall at 74.2%, M2 had low precision (35%) — every flagged cohort contained many non-churners. The core limitation is structural: logistic regression draws linear decision boundaries, while churn drivers like the "4+ service calls" threshold are inherently non-linear. Decision Trees can model these directly.

**M3 → M4 (Decision Tree, Baseline → Tuned):** The baseline tree memorised training data perfectly (train recall = 1.00) but generalised poorly (test recall = 0.649), with a train-test gap of 0.351. GridSearchCV over tree depth and leaf constraints brought this gap down to 0.196 while raising test recall to 74.2% — matching M2 but with substantially better F1 (0.673 vs 0.474).

**M4 → M5 (Decision Tree → Random Forest):** Even the tuned tree showed a meaningful overfitting gap. A Random Forest addresses this by training 200 trees on random bootstrap samples of the data, with each split considering a random subset of features. Averaging across 200 diverse trees reduces variance significantly compared to any single tree.

---

## 4. Evaluation

### Model Comparison

| Model | Train Recall | Test Recall | Train-Test Gap | Test F1 | Test AUC |
|-------|:-----------:|:-----------:|:--------------:|:-------:|:--------:|
| M1: LR Baseline | 0.257 | 0.247 | 0.010 | 0.284 | 0.817 |
| M2: LR Tuned | 0.757 | 0.742 | 0.015 | 0.474 | 0.817 |
| M3: DT Baseline | 1.000 | 0.649 | 0.351 | 0.663 | 0.819 |
| M4: DT Tuned | 0.938 | 0.742 | 0.196 | 0.673 | 0.833 |
| **M5: RF Final ⭐** | **0.852** | **0.722** | **0.130** | **0.622** | **0.882** |

### Final Model Performance on Holdout Test Set

The Random Forest (200 estimators, max_depth=5, class_weight=balanced) was evaluated on the 667-record held-out test set it had never seen during training or tuning:

| Metric | Score |
|--------|-------|
| Recall | 0.722 |
| Precision | 0.527 |
| F1-Score | 0.622 |
| ROC-AUC | 0.882 |

**In plain business terms:** Of the 100 customers who would churn, the model correctly identifies approximately 72 of them. The retention team receives a prioritised list where roughly 1 in 2 flagged customers is a genuine churn risk — a meaningful signal compared to a 14.5% base rate with no model.

### Why Random Forest Over the Tuned Decision Tree?

On raw recall, M4 (Decision Tree Tuned) and M5 are statistically tied (0.742 vs 0.722). Two factors make M5 the stronger production choice:

1. **ROC-AUC 0.882 vs 0.833** — the retention team operates on a probability score, not a binary flag. They lower or raise the threshold based on how many customers they can contact that week. AUC measures ranking quality across every possible threshold, and M5 is meaningfully better at this.

2. **Smallest overfitting gap (0.130 vs 0.196)** — a model with a smaller train-test gap is more reliable when deployed on new production data drawn from a slightly different distribution (new months, pricing changes, competitive shifts).

### Evaluation Visualisations

The notebook contains 12 purpose-built visualisations. The key evaluation charts are:

- **Dumbbell chart** — train vs test recall gap per model, showing overfitting at a glance
- **Radar chart** — all five models across all four metrics simultaneously
- **ROC curves** — AUC comparison across all models
- **Precision-Recall curve** — threshold trade-off for the final model, colour-coded by threshold value so the team can identify their operating point
- **Feature importance lollipop chart and treemap** — ranked driver analysis from the final Random Forest

---

## 5. Conclusion

### Top Churn Drivers & Recommended Actions

| Rank | Feature | Why It Matters | Recommended Action |
|------|---------|----------------|-------------------|
| 1 | Total day minutes | Heavy users face higher bills — price sensitivity drives exit | Offer loyalty rate caps for customers exceeding the 75th percentile (>216 min/day) |
| 2 | Customer service calls | 4+ calls signals deep, unresolved dissatisfaction | Auto-flag after the 3rd call; route immediately to a senior retention agent |
| 3 | International plan | Holders churn at 42.4% vs 11.5% — plan may be poor value | Audit pricing and coverage; conduct exit interviews with former subscribers |
| 4 | Total eve minutes | Secondary usage signal correlated with day usage | Monitor alongside day minutes as a composite heavy-user flag |
| 5 | Total intl minutes | High international usage on an already-at-risk plan compounds churn risk | Proactively offer discounted international bundles to high intl-usage accounts |

**Protective factor:** Voice mail plan holders churn at only 8.7% compared to 16.7% for non-holders. Consider offering the voice mail plan as a free add-on during retention outreach — a low-cost intervention with a potentially high retention impact.

### Model Limitations

1. **No demographic or contract data** — age, income, and contract type are established churn drivers in the literature but are absent from this dataset. Including them would likely improve model performance and fairness assessment.
2. **No timestamps** — the dataset is a static snapshot. Without time-ordered data, churn cannot be modelled as a time-to-event problem, and seasonal patterns cannot be detected.
3. **Static model** — the competitive landscape, pricing, and customer behaviour evolve. The model must be retrained periodically to remain calibrated.
4. **27.8% missed churners** — the model still misses roughly 1 in 4 genuine churners at the default threshold. Lowering the probability threshold recovers more of these at the cost of more false positives.
5. **45% false positive rate among flagged customers** — outreach costs for non-churners must be factored into ROI calculations when designing the retention programme.

### Deployment Recommendations

- **Weekly batch scoring** — score all active accounts weekly and export a probability-ranked list to the CRM system for the retention team
- **Adjustable threshold** — the team should lower the threshold when they have capacity for a larger outreach campaign, and raise it when resource-constrained — the Precision-Recall curve in the notebook maps every possible operating point
- **Quarterly retraining** — refresh the model on the most recent 12 months of data every quarter to account for pricing changes and competitive shifts
- **A/B test** — hold out a randomly selected control group of high-risk customers to measure the true causal lift from model-driven outreach versus no intervention

---
