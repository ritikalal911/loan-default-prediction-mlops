# Dataset Documentation

## 1. Dataset Overview

This project uses the **Credit Risk Dataset v0** from Kaggle for a **Loan Default Prediction** task.

- **Source:** Kaggle – Credit Risk Dataset v0
- **Machine learning task:** Binary classification
- **Rows:** 37,408
- **Original columns:** 10
- **Target variable:** `loan_quality`
- **Target classes:** `Good`, `Bad`

The objective is to classify a loan as **Good** or **Bad** using customer financial and demographic information.

---

## 2. Dataset Features

| Column | Type | Role in Project |
|---|---|---|
| `account_number` | Integer | Unique identifier; removed before model training |
| `total_investment` | Numeric | Predictive feature |
| `current_balance` | Numeric | Predictive feature |
| `marital_status` | Categorical | Predictive feature |
| `gender` | Categorical | Predictive feature |
| `due_payment` | Numeric | Predictive feature |
| `compensation_charged` | Categorical | Predictive feature |
| `client_type` | Categorical | Predictive feature |
| `loan_quality` | Categorical | **Target variable** |
| `repay_mode` | Categorical | Predictive feature |

`account_number` is excluded because it is an identifier rather than a meaningful predictive variable.

After preprocessing and one-hot encoding, the model receives **17 processed features**.

---

## 3. Target Distribution

| Class | Records | Percentage |
|---|---:|---:|
| Good | 33,254 | 88.90% |
| Bad | 4,154 | 11.10% |
| **Total** | **37,408** | **100%** |

The dataset is imbalanced, with substantially fewer `Bad` loans than `Good` loans. For this reason, the classification experiments use `class_weight: balanced`, and evaluation focuses not only on accuracy but also on precision, recall, F1-score, and ROC-AUC for the `Bad` class.

---

## 4. Data Quality Assessment

### 4.1 Missing Values

| Column | Missing Values |
|---|---:|
| `account_number` | 0 |
| `total_investment` | 0 |
| `current_balance` | 0 |
| `marital_status` | 2 |
| `gender` | 2 |
| `due_payment` | 0 |
| `compensation_charged` | 2 |
| `client_type` | 103 |
| `loan_quality` | 0 |
| `repay_mode` | 0 |

The target variable contains no missing values.

During preprocessing:

- Numeric missing values are handled with **median imputation**.
- Categorical missing values are handled with **most-frequent imputation**.
- The preprocessing transformer is fitted only on the training split to avoid data leakage.

### 4.2 Duplicate Rows

- **Duplicate rows found:** 0

No duplicate removal was required for the current dataset.

### 4.3 Outlier Assessment

The three numeric predictive variables were assessed using the **1.5 × IQR rule**.

| Numeric Feature | IQR-Flagged Records |
|---|---:|
| `total_investment` | 4,680 |
| `current_balance` | 6,636 |
| `due_payment` | 5,991 |

These records were **not automatically removed** because large monetary values may represent valid financial observations rather than data errors. In particular, `due_payment` has Q1 = Q3 = 0, so a conventional IQR rule flags any non-zero value; automatic deletion would therefore be inappropriate.

Instead, numeric variables are standardized using `StandardScaler`.

---

## 5. Preprocessing

Preprocessing is implemented in:

```text
src/prepare.py
```

The preparation stage performs the following steps:

1. Loads `data/raw/credit_risk_dataset.csv`
2. Validates the target column `loan_quality`
3. Checks for duplicate rows
4. Separates features and target
5. Drops `account_number`
6. Creates train, validation, and test splits
7. Identifies numeric and categorical features
8. Imputes missing values
9. One-hot encodes categorical variables
10. Standardizes numeric variables
11. Saves processed datasets
12. Saves the fitted preprocessing transformer

### Numeric Features

- `total_investment`
- `current_balance`
- `due_payment`

Numeric pipeline:

```text
Median Imputation → StandardScaler
```

### Categorical Features

- `marital_status`
- `gender`
- `compensation_charged`
- `client_type`
- `repay_mode`

Categorical pipeline:

```text
Most-Frequent Imputation → OneHotEncoder(handle_unknown="ignore")
```

---

## 6. Train / Validation / Test Split

The dataset is split using stratification so that the `Good`/`Bad` class distribution remains approximately consistent in each subset.

| Split | Percentage | Records | Purpose |
|---|---:|---:|---|
| Training | 70% | 26,184 | Fit machine learning models |
| Validation | 15% | 5,612 | Compare MLflow experiments and select the best model |
| Test | 15% | 5,612 | Final unbiased evaluation |
| **Total** | **100%** | **37,408** | |

A fixed seed is used:

```yaml
random_state: 42
```

### Training Distribution

| Class | Records |
|---|---:|
| Good | 23,276 |
| Bad | 2,908 |

### Validation Distribution

| Class | Records |
|---|---:|
| Good | 4,989 |
| Bad | 623 |

### Test Distribution

| Class | Records |
|---|---:|
| Good | 4,989 |
| Bad | 623 |

The validation split is used for experiment comparison. The test split is kept separate until the final `evaluate` stage.

---

## 7. Data Leakage Prevention

To avoid leakage:

- The preprocessing transformer is fitted on **training data only**.
- The validation set is used only for model/parameter comparison.
- The test set is not used during training or model selection.
- Final performance is calculated only after the best model is selected.

The workflow is:

```text
Raw Dataset
    ↓
prepare
    ↓
Train 70% ───────┐
Validation 15% ──┴──→ train / MLflow comparison
                          ↓
                     Best Model
                          ↓
Test 15% ───────────→ evaluate
```

---

## 8. DVC Data Versioning

The raw dataset is versioned using **DVC**.

Tracked dataset:

```text
data/raw/credit_risk_dataset.csv
```

DVC metadata file:

```text
data/raw/credit_risk_dataset.csv.dvc
```

The project also uses a DVC remote named:

```text
storage
```

Useful commands:

```bash
dvc remote list
dvc push
dvc pull
dvc status -c
```

The DVC pipeline can be reproduced using:

```bash
dvc repro
```

The main stages are:

```text
prepare → train → evaluate
```

---

## 9. Generated Data Artifacts

The preparation stage produces:

```text
data/processed/
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
├── y_test.csv
└── preprocessor.pkl
```

These outputs are generated reproducibly from the raw dataset and preprocessing configuration.

---

## 10. Dataset Summary

The final dataset used in Phase 1 contains:

- **37,408 records**
- **10 original columns**
- **8 predictive variables** after removing the identifier and target
- **3 numeric predictive variables**
- **5 categorical predictive variables**
- **2 target classes**
- **0 duplicate rows**
- Limited missing categorical data
- A meaningful class imbalance between `Good` and `Bad`
- **17 processed model features** after preprocessing

This dataset supports the project goal of developing and tracking a reproducible binary classification pipeline for loan default risk.
