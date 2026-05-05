# Contributing to Telecom Customer Churn Prediction

Thank you for your interest in contributing! This document explains how to get involved, report issues, and submit changes.

---

## Table of Contents

- [Who Can Contribute](#who-can-contribute)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Code Style](#code-style)
- [Current Contributors](#current-contributors)

---

## Who Can Contribute

Anyone is welcome to contribute — whether it's fixing a typo, improving a visualisation, extending the model pipeline, adding new analysis objectives, or cleaning up documentation.

---

## Getting Started

### 1. Fork the repository
Click the **Fork** button at the top right of the repo page on GitHub.

### 2. Clone your fork
```bash
git clone https://github.com/your-username/telecom-churn-prediction
cd telecom-churn-prediction
```

### 3. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If a `requirements.txt` is not yet present, install the core dependencies manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn squarify
```

### 4. Create a new branch
Always work on a dedicated branch — never commit directly to `main`.
```bash
git checkout -b feature/your-feature-name
```

---

## How to Contribute

### Areas open for contribution

| Area | Examples |
|------|---------|
| New modelling | Additional classifiers (XGBoost, SVM, KNN), stacking ensembles, threshold optimisation |
| Feature engineering | Interaction terms, binning usage columns, polynomial features |
| Visualisations | Improved charts, interactive Plotly/Altair plots, SHAP value plots |
| Evaluation | Additional metrics (MCC, Brier score), calibration curves, learning curves |
| Data pipeline | More robust preprocessing, pipeline objects (`sklearn.pipeline.Pipeline`) |
| Documentation | Clearer cell explanations, better inline comments, README updates |
| Bug fixes | Broken code, incorrect calculations, chart rendering issues |

---

## Commit Message Guidelines

Use clear, descriptive commit messages in the imperative mood:

```
feat: add XGBoost classifier as Model 6
fix: correct precision calculation in evaluation helper
docs: update README with final model selection rationale
refactor: wrap preprocessing steps in sklearn Pipeline
data: add cleaned churn dataset export cell
style: align colour palette constants at top of notebook
```

**Prefixes to use:**

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature, model, or analysis |
| `fix:` | Bug fix or incorrect output |
| `docs:` | Documentation only |
| `refactor:` | Code restructuring without behaviour change |
| `data:` | Changes to datasets or data pipeline |
| `style:` | Formatting, whitespace, no logic change |

---

## Pull Request Process

1. Ensure your branch is up to date with the latest `main`:
```bash
git fetch origin
git rebase origin/main
```

2. **Run the notebook end-to-end** (`Kernel → Restart & Run All`) to confirm no errors before submitting.

3. Push your branch to your fork:
```bash
git push origin feature/your-feature-name
```

4. Open a **Pull Request** against the `main` branch of this repository.

5. In your PR description, include:
   - What change you made and why
   - Any datasets or dependencies added
   - Screenshots of any new charts or model outputs
   - Before/after metric comparison if modelling changes were made

6. At least one existing contributor must review and approve before merging.

7. Once approved, the PR will be **squash-merged** into `main`.

---

## Reporting Issues

Found a bug, a calculation error, or a broken chart? Please open a GitHub Issue:

1. Go to the **Issues** tab of this repository
2. Click **New Issue**
3. Use a clear title, e.g. `Bug: recall metric calculated on train set instead of test set`
4. Include:
   - Steps to reproduce
   - Expected vs. actual behaviour
   - Python version and relevant package versions (output of `pip freeze`)

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Use **4-space indentation** (no tabs)
- Add inline comments for non-obvious logic
- Keep notebook cells **focused** — one task or visual per cell
- Use descriptive variable names that match the domain (e.g. `churn_rate`, not `cr`)
- Use the **established colour palette** for all new charts to maintain visual consistency across the notebook:

```python
C_RETAIN = '#2ecc71'   # green  — retained customers
C_CHURN  = '#e74c3c'   # red    — churned customers
C_BLUE   = '#3498db'   # blue   — train metrics / Model 1
C_ORANGE = '#e67e22'   # orange — Model 2
C_PURPLE = '#9b59b6'   # purple — final model / Model 5
```

- Apply the same global plot settings already defined at the top of the notebook:

```python
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False
})
```

---

## Current Contributor

| Contributor | GitHub |
|-------------|--------|
| *Richard Duncan Oketch* | *@richardoketch32806-eng* |

---

*Thank you for helping improve this project!*

