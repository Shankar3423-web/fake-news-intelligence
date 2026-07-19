# Phase 12 — Automatic Model Retraining Report

*Generated: 2026-07-19 13:05:29 UTC*

---

## 1. Run Overview

| Field | Value |
| :--- | :--- |
| **Run ID** | `retrain_20260719_183430_f54e5499` |
| **Decision** | **REJECTED** |
| **Reason** | Candidate rejected: weighted_score 0.9986 ≤ production 0.9986 — production model retained. |
| **Pipeline Duration** | `59.65s` |

## 2. Approved Feedback Summary

| Metric | Value |
| :--- | :--- |
| **Approved Records Used** | `5` |
| **Pre-Merge Rows** | `56573` |
| **Post-Merge Rows** | `56578` |
| **Added Rows** | `5` |

### Merged Dataset Label Distribution

| Label | Count |
| :--- | :--- |
| **Real (0)** | `29261` |
| **Fake (1)** | `27317` |

## 3. Training Summary

- **Models Trained**: Logistic Regression, Linear SVM, Random Forest, XGBoost
- **Training Rows**: `45262`
- **Testing Rows**: `11316`
- **Feature Count**: `283`

### Per-Model Training Duration

| Model | Duration (s) |
| :--- | :--- |
| **logistic_regression** | `38.6045` |
| **svm** | `1.8626` |
| **random_forest** | `7.0741` |
| **xgboost** | `3.1402` |

## 4. Candidate Model Evaluation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **logistic_regression** | 0.9294 | 0.9223 | 0.9323 | 0.9273 | 0.9825 | 0.8587 |
| **svm** | 0.9965 | 0.9978 | 0.9949 | 0.9963 | 0.9998 | 0.9929 |
| **random_forest** | 0.9973 | 1.0000 | 0.9945 | 0.9972 | 0.9999 | 0.9947 |
| **xgboost** | 0.9983 | 0.9994 | 0.9971 | 0.9983 | 0.9999 | 0.9966 |

## 5. Generated Artifacts

| Artifact | Path |
| :--- | :--- |
| **deployment_decision** | `ml/retraining/reports/deployment_decision.json` |
| **registry** | `ml/retraining/registry/retraining_registry.json` |
| **retraining_statistics** | `ml/retraining/statistics/retraining_statistics.json` |
| **retraining_metadata** | `ml/retraining/metadata/retraining_metadata.json` |
