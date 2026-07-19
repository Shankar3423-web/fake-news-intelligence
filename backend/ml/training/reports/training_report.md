# Phase 6 — Model Training Production Report

## 1. Overview
This report documents the training execution of multiple machine learning classifiers for the Fake News Detection pipeline. Training was conducted using the selected feature set generated in Phase 5, in a reproducible and stratified manner.

## 2. Dataset Information
- **Input Dataset Path**: `ml/feature_selection/processed/selected_feature_dataset_v1.csv`
- **Dataset File Size**: `162.59 MB` (170484084 bytes)
- **Feature Count**: `283` numeric features
- **Total Rows**: `56573` samples
- **Training Partition (80%)**: `45258` rows
- **Testing Partition (20%)**: `11315` rows

### Partition Class Distributions
| Partition | Class 0 (Real) | Class 1 (Fake) | Total |
| :--- | :--- | :--- | :--- |
| **Training** | 23405 | 21853 | 45258 |
| **Testing** | 5851 | 5464 | 11315 |

## 3. Model Information & Hyperparameters
Each classifier was trained using a custom set of hyperparameters configured for stability and performance:

### Logistic Regression
- **Algorithm Name**: Logistic Regression
- **Hyperparameters Used**:
```json
{}
```

### Linear SVM
- **Algorithm Name**: Linear SVM
- **Hyperparameters Used**:
```json
{}
```

### Random Forest
- **Algorithm Name**: Random Forest
- **Hyperparameters Used**:
```json
{}
```

### XGBoost
- **Algorithm Name**: XGBoost
- **Hyperparameters Used**:
```json
{}
```

## 4. Training Benchmark & Summary
Performance profiling was done to track the duration and memory overhead for each classifier:

| Model Algorithm | Training Duration (seconds) | RSS Memory Used (MB) | Samples Trained | Features Count |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 14.6533s | 2.26 MB | 45258 | 283 |
| **Linear SVM** | 2.3921s | 4.24 MB | 45258 | 283 |
| **Random Forest** | 2.1546s | 3.50 MB | 45258 | 283 |
| **XGBoost** | 2.3250s | 4.95 MB | 45258 | 283 |

- **Total Pipeline Duration**: `23.8903 seconds`

## 5. Generated Files Reference
The following files were created during Phase 6 and are saved in the production directory structure:

| Resource | Output Path |
| :--- | :--- |
| **logistic_regression_model** | `ml/training/models\logistic_regression\model.joblib` |
| **logistic_regression_config** | `ml/training/models\logistic_regression\training_config.json` |
| **logistic_regression_features** | `ml/training/models\logistic_regression\feature_order.json` |
| **logistic_regression_metadata** | `ml/training/models\logistic_regression\metadata.json` |
| **svm_model** | `ml/training/models\svm\model.joblib` |
| **svm_config** | `ml/training/models\svm\training_config.json` |
| **svm_features** | `ml/training/models\svm\feature_order.json` |
| **svm_metadata** | `ml/training/models\svm\metadata.json` |
| **random_forest_model** | `ml/training/models\random_forest\model.joblib` |
| **random_forest_config** | `ml/training/models\random_forest\training_config.json` |
| **random_forest_features** | `ml/training/models\random_forest\feature_order.json` |
| **random_forest_metadata** | `ml/training/models\random_forest\metadata.json` |
| **xgboost_model** | `ml/training/models\xgboost\model.joblib` |
| **xgboost_config** | `ml/training/models\xgboost\training_config.json` |
| **xgboost_features** | `ml/training/models\xgboost\feature_order.json` |
| **xgboost_metadata** | `ml/training/models\xgboost\metadata.json` |
| **registry_file** | `ml/training/registry.json` |
| **logistic_regression_central_metadata** | `ml/training/metadata\logistic_regression_metadata.json` |
| **linear_svm_central_metadata** | `ml/training/metadata\linear_svm_metadata.json` |
| **random_forest_central_metadata** | `ml/training/metadata\random_forest_metadata.json` |
| **xgboost_central_metadata** | `ml/training/metadata\xgboost_metadata.json` |
| **statistics_file** | `ml/training/statistics/training_statistics.json` |
| **logistic_regression_hashes** | `ml/training/hashes\logistic_regression_hashes.json` |
| **svm_hashes** | `ml/training/hashes\svm_hashes.json` |
| **random_forest_hashes** | `ml/training/hashes\random_forest_hashes.json` |
| **xgboost_hashes** | `ml/training/hashes\xgboost_hashes.json` |

## 6. Warnings
- No training execution warnings generated.

## 7. Recommendations
- 💡 Ensure prediction services parse and validate features in the exact order saved in 'feature_order.json'.
- 💡 Since training is complete and validated, proceed to Phase 7: Model Evaluation to run cross-validation, calculate metrics (accuracy, F1, ROC/AUC), and select the best model.
- 💡 Consider hyperparameter optimization if evaluation metrics in the next phase indicate overfitting (especially on high-capacity models like Random Forest or XGBoost).
