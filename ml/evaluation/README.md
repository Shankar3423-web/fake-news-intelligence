# Phase 7 — Model Evaluation

This module provides a scientific and reproducible evaluation framework for the Production-Grade Fake News Intelligence System.

## Features

- **Reproducible Splits:** Loads the dataset and splits it deterministically matching Phase 6 parameters to extract the identical test split.
- **Comprehensive Metrics:** Calculates Accuracy, Precision, Recall, F1 Score, ROC-AUC, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Cohen's Kappa, Log Loss, prediction speed (latency & throughput), and peak memory usage.
- **Structured Deliverables:**
  - Predictions saved in `predictions/`
  - Confusion Matrices (PNG, CSV, JSON) saved in `confusion_matrices/`
  - ROC Curves (PNG, JSON) saved in `roc_curves/`
  - Precision-Recall Curves (PNG, JSON) saved in `precision_recall_curves/`
  - Multi-model comparisons saved in `comparison/`
  - Ranked leaderboard saved in `leaderboard/`
- **Configurable Best Model Selection:** Uses a weighted strategy configured in `evaluation_config.yaml` to score and select the best overall candidate saved in `best_model.json`.
- **Integrity Auditing:** Calculates SHA-256 hashes of all deliverables and writes version histories into `evaluation_versions.json`.

## Command Line Usage

Run the complete evaluation pipeline:
```bash
python ml/evaluation/evaluation_pipeline.py
```

Run validation and verification:
```bash
python ml/evaluation/verify_evaluation.py
```

Run test suite:
```bash
pytest ml/evaluation/tests/
```
