# Model Card — Loan Default Prediction

## 1. Model Overview

**Model name:** Loan Default Prediction Model  
**Project:** Loan Default Prediction – MLOps  
**Task:** Binary classification  
**Target variable:** `loan_quality`  
**Target classes:** `Good`, `Bad`  
**Current production model:** Random Forest Classifier  
**Deployment:** FastAPI + Docker + Render  
**Model artifact:** `models/model.joblib`

The model predicts whether a loan record is more likely to belong to the `Good` or `Bad` loan-quality class using customer financial and demographic features.

---

## 2. Intended Use

The model was developed as an academic MLOps project to demonstrate:

- reproducible machine learning pipelines,
- experiment tracking,
- model deployment,
- CI/CD,
- drift monitoring,
- automated retraining,
- and model lifecycle documentation.

The deployed API provides a prediction and class probabilities for a supplied loan record.

### Intended users

- Course instructors
- Project team members
- Students evaluating the MLOps workflow

### Out-of-scope use

This model is **not intended for real-world lending approval, denial, pricing, or other financial decisions**.

The model should not be used as the sole basis for decisions affecting a real person's access to credit or financial services.

---

## 3. Dataset

**Source:** Kaggle – Credit Risk Dataset v0  
**Rows:** 37,408  
**Original columns:** 10  
**Predictive variables:** 8  
**Target:** `loan_quality`

### Model input features

- `total_investment`
- `current_balance`
- `marital_status`
- `gender`
- `due_payment`
- `compensation_charged`
- `client_type`
- `repay_mode`

The identifier column `account_number` is excluded from training.

### Target distribution

| Class | Records | Percentage |
|---|---:|---:|
| Good | 33,254 | 88.90% |
| Bad | 4,154 | 11.10% |

The dataset is imbalanced, so the selected Random Forest uses:

```text
class_weight = balanced
```

---

## 4. Data Preparation

The preprocessing pipeline is implemented in:

```text
src/prepare.py
```

### Numerical processing

Numerical features:

- `total_investment`
- `current_balance`
- `due_payment`

Processing:

```text
Median Imputation → StandardScaler
```

### Categorical processing

Categorical features:

- `marital_status`
- `gender`
- `compensation_charged`
- `client_type`
- `repay_mode`

Processing:

```text
Most-Frequent Imputation → OneHotEncoder(handle_unknown="ignore")
```

After preprocessing, the model receives **17 processed features**.

The fitted preprocessing artifact is stored as:

```text
data/processed/preprocessor.pkl
```

---

## 5. Data Split

The dataset is split using stratified sampling.

| Split | Percentage | Records | Purpose |
|---|---:|---:|---|
| Training | 70% | 26,184 | Model training |
| Validation | 15% | 5,612 | Experiment comparison |
| Test | 15% | 5,612 | Final evaluation |

A fixed random seed is used:

```text
random_state = 42
```

The test set is kept separate from training and experiment selection.

---

## 6. Experiment Tracking

Experiments are tracked using MLflow.

Three Phase 1 experiments were compared:

| Run | Model | Accuracy | Precision Bad | Recall Bad | F1 Bad | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| 01 Baseline | Logistic Regression | 0.6246 | 0.1791 | 0.6645 | 0.2821 | 0.6729 |
| 02 RF Experiment 1 | Random Forest | 0.6771 | 0.2104 | **0.6934** | 0.3229 | 0.7595 |
| **03 RF Experiment 2** | **Random Forest** | **0.7272** | **0.2400** | 0.6726 | **0.3537** | **0.7904** |

The third experiment was selected because it achieved the strongest overall validation performance, including the highest validation accuracy, precision, F1-score, and ROC-AUC.

Experiment 2 had slightly higher recall for the `Bad` class.

---

## 7. Selected Model Configuration

The selected model is a Random Forest Classifier with:

```yaml
n_estimators: 200
max_depth: 15
min_samples_split: 5
random_state: 42
class_weight: balanced
```

The selected model is stored as:

```text
models/model.joblib
```

---

## 8. Final Test Performance

The selected model was evaluated on the untouched test set.

| Metric | Result |
|---|---:|
| Accuracy | 0.7281 |
| Precision (`Bad`) | 0.2424 |
| Recall (`Bad`) | 0.6822 |
| F1-score (`Bad`) | 0.3577 |
| ROC-AUC | 0.7995 |
| Classification threshold | 0.50 |

### Evaluation artifacts

```text
reports/metrics.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
```

---

## 9. Deployment

The model is exposed through a FastAPI inference service.

### API endpoints

```text
GET  /
GET  /health
POST /predict
```

### Public deployment

Render is used for cloud deployment.

```text
https://loan-default-prediction-mlops-gpz6.onrender.com
```

Swagger documentation:

```text
https://loan-default-prediction-mlops-gpz6.onrender.com/docs
```

### Prediction response

The API returns:

- predicted class,
- probability of `Good`,
- probability of `Bad`.

Example structure:

```json
{
  "prediction": "Good",
  "probability_good": 0.748,
  "probability_bad": 0.252
}
```

---

## 10. CI/CD

GitHub Actions is used for automated validation.

The workflow performs:

1. Dependency installation
2. Ruff linting
3. Ruff formatting checks
4. FastAPI unit tests
5. Data validation tests
6. Docker image build validation
7. Automatic Render deployment after successful checks on `main`

The workflow file is:

```text
.github/workflows/ci-cd.yml
```

---

## 11. Monitoring

Data drift is monitored using EvidentlyAI.

Reference and simulated current data are compared across the eight model input features.

### Latest drift demonstration

| Metric | Result |
|---|---:|
| Monitored columns | 8 |
| Drifted columns | 5 |
| Drift share | 0.625 |
| Drift threshold | 0.50 |
| Dataset drift detected | Yes |

The drifted current dataset is **intentionally simulated for the Phase 2 demonstration**. It does not represent real production traffic.

Monitoring artifacts are stored in:

```text
reports/monitoring/drift_report.html
reports/monitoring/drift_report.json
reports/monitoring/drift_status.json
```

---

## 12. Retraining

When the monitored drift share exceeds the configured threshold, the monitoring workflow triggers model retraining.

The retraining workflow combines the original training and validation sets while leaving the test set unchanged.

### Retraining dataset

```text
Training + Validation = 31,796 records
Test = 5,612 records
```

### Current vs retrained candidate

| Metric | Current Model | Retrained Candidate |
|---|---:|---:|
| Accuracy | **0.7281** | 0.7274 |
| Precision (`Bad`) | 0.2424 | **0.2485** |
| Recall (`Bad`) | 0.6822 | **0.7191** |
| F1-score (`Bad`) | 0.3577 | **0.3693** |
| ROC-AUC | 0.7995 | **0.8078** |

ROC-AUC change:

```text
+0.0082
```

The retrained candidate is recommended for promotion because it improves ROC-AUC and all three minority-class metrics, although overall accuracy decreases slightly.

The candidate is stored separately as:

```text
models/retrained_model.joblib
```

The comparison report is stored as:

```text
reports/monitoring/retraining_comparison.json
```

The current production model should remain unchanged until the candidate is deliberately reviewed and promoted.

---

## 13. Limitations

The model has several important limitations.

### Class imbalance

Only about 11% of the dataset belongs to the `Bad` class. This makes minority-class performance particularly important and means accuracy alone can be misleading.

### Precision

The test precision for the `Bad` class is relatively low. A large number of records predicted as `Bad` may actually belong to the `Good` class.

### Dataset representativeness

The model is limited by the population, collection process, and quality of the original Kaggle dataset. Performance may not generalize to another lender, geography, time period, or customer population.

### Simulated drift

The Phase 2 drift demonstration uses intentionally shifted data rather than true production observations.

### No causal interpretation

Feature importance or model predictions should not be interpreted as causal relationships.

---

## 14. Fairness and Ethical Considerations

The dataset includes demographic variables such as:

- `gender`
- `marital_status`

These features can create fairness concerns in real lending systems.

This academic model has not undergone a complete fairness, bias, disparate-impact, regulatory, or legal assessment.

For a real financial application, sensitive attributes would require careful governance, legal review, fairness testing, explainability, human oversight, and potentially exclusion from the production decision model.

The model should therefore be treated strictly as an educational demonstration.

---

## 15. Monitoring Recommendations

If this system were used in a real production environment, monitoring should include:

- input feature drift,
- prediction distribution drift,
- missing-value rates,
- schema validation,
- API availability and latency,
- model performance after labels become available,
- class-specific precision and recall,
- fairness metrics,
- retraining frequency,
- and candidate-versus-production model comparison.

Retraining should not automatically replace the production model solely because drift is detected.

A candidate model should be promoted only after evaluation against defined acceptance criteria.

---

## 16. Reproducibility

The Phase 1 machine learning workflow is managed with DVC.

Main stages:

```text
prepare → train → evaluate
```

Run:

```bash
dvc repro
```

The Phase 2 monitoring workflow can be run with:

```bash
python monitoring/run_monitoring.py
```

This performs:

```text
Drift Detection
      ↓
Drift Threshold Decision
      ↓
Retraining if Required
      ↓
Current vs Candidate Comparison
```

---

## 17. Model Lifecycle

Current project lifecycle:

```text
Raw Data
   ↓
DVC Prepare
   ↓
Training
   ↓
MLflow Experiment Comparison
   ↓
Selected Model
   ↓
Final Test Evaluation
   ↓
FastAPI
   ↓
Docker
   ↓
Render Deployment
   ↓
GitHub Actions CI/CD
   ↓
EvidentlyAI Monitoring
   ↓
Drift Detection
   ↓
Candidate Retraining
   ↓
Performance Comparison
```

---

## 18. Model Card Status

**Phase 1 model:** Completed  
**Deployment:** Completed  
**CI/CD:** Completed  
**Monitoring:** Completed  
**Retraining demonstration:** Completed  
**Production approval:** Not applicable — academic project only
