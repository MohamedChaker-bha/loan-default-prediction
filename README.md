# Loan Default Prediction Using Supervised Learning

Predicting loan defaults using real-world Kaggle data, handling class imbalance, and comparing KNN, Logistic Regression, and a tuned pipeline.

---

## Overview

Banks need to identify customers likely to default on their loans. The challenge: in real data, defaults are rare (~10%), so standard models learn to predict "no default" for everyone — getting 90% accuracy while catching zero actual defaults. This project tackles that class imbalance problem.

**Dataset:** [Loan Default Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default) (255,347 records, sampled to 1,000)

---

## Dataset Features

| Feature | Description |
|---------|-------------|
| `Age` | Customer age |
| `Income` | Annual income |
| `LoanAmount` | Amount borrowed |
| `CreditScore` | Credit rating |
| `MonthsEmployed` | Months at current job |
| `NumCreditLines` | Number of open credit accounts |
| `InterestRate` | Loan interest rate (%) |
| `LoanTerm` | Loan duration (months) |
| `DTIRatio` | Debt-to-income ratio |
| `Default` | Target — 1 = defaulted, 0 = paid back |

---

## Approach

1. **Exploratory analysis** — identified ~10% default rate (class imbalance)
2. **Preprocessing** — split first to prevent data leakage, then impute and scale
3. **KNN** — model complexity curve for best k, threshold adjustment to catch defaults
4. **Logistic Regression** — `class_weight='balanced'` to handle imbalance, ROC/AUC analysis
5. **Pipeline + GridSearchCV** — SimpleImputer → StandardScaler → LogisticRegression (balanced), tuned with cross-validation
6. **Model comparison** — classification reports, ROC curves, and confusion matrices

---

## Key Challenge: Class Imbalance

| Approach | Result |
|----------|--------|
| Default LogReg | 90% accuracy, 0% default recall — useless |
| Balanced LogReg | Lower accuracy, but actually catches defaults |
| Threshold adjustment | Trade-off between catching defaults and false alarms |

A model that ignores all defaults is worse than useless for a bank — missing a default costs thousands. This project prioritizes **recall** (catching actual defaults) over raw accuracy.

---

## Tools & Libraries

- Python 3
- scikit-learn (KNN, LogisticRegression, GridSearchCV, Pipeline, SimpleImputer, StandardScaler)
- pandas, numpy, matplotlib

---

## Key Takeaways

- **Accuracy is misleading** with imbalanced data — always check precision, recall, and F1
- **class_weight='balanced'** is the simplest fix for imbalanced classification
- **Data leakage** inflates scores — always split before preprocessing
- **Real-world data is messy** — handling imbalance is a critical ML skill

---

## How to Run

```bash
git clone https://github.com/MohamedChaker-bha/loan-default-prediction.git
cd loan-default-prediction
pip install pandas scikit-learn matplotlib numpy
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default) and place `Loan_default.csv` in the project folder, then:

```bash
jupyter notebook loan_default_prediction.ipynb
```

---

## Author

**Mohamed Chaker Iben Hadj Amor**  
Data Engineering Student · FSB, Tunisia  
[LinkedIn](https://www.linkedin.com/in/mohamed-chaker-iben-hadj-amor/) · [GitHub](https://github.com/MohamedChaker-bha)
