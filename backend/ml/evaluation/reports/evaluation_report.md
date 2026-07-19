# Phase 7: Model Evaluation Report
## Production-Grade Fake News Intelligence System

## 1. Overview
This report compiles the scientific evaluation results of every trained machine learning model produced during Phase 6. The primary objective of this phase is to evaluate and compare the classifiers under uniform testing conditions, using reproducible testing splits, to select the optimal model for production deployment.

## 2. Dataset Information
- **Dataset Path:** `ml/feature_selection/processed/selected_feature_dataset_v1.csv`
- **Dataset Size (bytes):** `170,484,084`
- **Total Samples:** `56,575`
- **Testing Split Size:** `11,315` (test_size=0.2)
- **Feature Count:** `283` features
- **Split Reproducibility:** Stratified split, random seed = `42`

## 3. Model & Pipeline Tracking Context
- **Training Run Version:** `run_2_20260719_085859`
- **Dataset Version Context:** `1.0.1`
- **Feature Selection Version Context:** `2`
- **Total Models Evaluated:** `4` classifiers

## 4. Model Comparison Matrix
The table below compares the performance, execution speed, and resource footprints of all evaluated classifiers:

|Model|Accuracy|Precision|Recall|F1 Score|ROC AUC|Pred Time (s)|Throughput (s/s)|Memory (MB)|Size (MB)|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Logistic Regression|0.9430|0.9391|0.9431|0.9411|0.9882|0.0131|861701.3|0.20|0.01|
|Linear SVM|0.9947|0.9951|0.9940|0.9945|0.9997|0.0131|863233.1|0.00|0.01|
|Random Forest|0.9981|0.9993|0.9967|0.9980|0.9999|0.0552|204899.9|3.28|5.63|
|XGBoost|0.9990|0.9996|0.9984|0.9990|1.0000|0.0216|522690.7|0.29|0.21|

## 5. Model Leaderboard
Rankings are based on the configurable weighted score selector:

|Rank|Model|Overall Score|F1 Score|ROC AUC|Throughput (samples/s)|Memory (MB)|
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
|1|**Linear SVM**|0.996363|0.9945|0.9997|863233.1|0.00|
|2|XGBoost|0.979649|0.9990|1.0000|522690.7|0.29|
|3|Random Forest|0.960607|0.9980|0.9999|204899.9|3.28|
|4|Logistic Regression|0.957982|0.9411|0.9882|861701.3|0.20|

## 6. Selected Production Model
🏆 The system has selected **Linear SVM** as the best production-ready candidate.

> [!IMPORTANT]
> **Model ID:** `svm`
> **Overall Weighted Score:** `0.996363`
> **Target Metric Selection Strategy:** `weighted_score`
> **Model Size:** `0.01 MB`
> **Inference Latency:** `0.001158 ms/sample`

## 7. Recommendations & Deployment Rationale
- **Linear SVM** delivers highly linear decision boundaries with quick decision times. However, since it lack probability scores, probability-based post-filtering or thresholding is disabled.
- **Resource footprint optimization:** Deserialization latency should be monitored during container startup. Larger tree-based models like Random Forest and XGBoost require slightly longer warm-up times compared to Logistic Regression.

## 8. Warnings & Operational Constraints
> [!WARNING]
> Linear SVM does not natively support probability estimates. Used decision_function instead.

## 9. Generated Evaluation Deliverables
Below is the listing of all structured outputs, visual plots, and version histories compiled by this execution:

| Deliverable Name | File Path |
| :--- | :--- |
| best_model_json | [best_model.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\best_model.json) |
| chart_metrics_comparison | [metrics_comparison.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\charts\metrics_comparison.png) |
| chart_model_size | [model_size_comparison.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\charts\model_size_comparison.png) |
| chart_prediction_time | [prediction_time_comparison.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\charts\prediction_time_comparison.png) |
| comparison_csv | [model_comparison.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\comparison\model_comparison.csv) |
| comparison_json | [model_comparison.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\comparison\model_comparison.json) |
| comparison_md | [model_comparison.md](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\comparison\model_comparison.md) |
| leaderboard_csv | [leaderboard.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\leaderboard\leaderboard.csv) |
| leaderboard_json | [leaderboard.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\leaderboard\leaderboard.json) |
| leaderboard_md | [leaderboard.md](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\leaderboard\leaderboard.md) |
| logistic_regression_central_evaluation_metadata | [logistic_regression_evaluation_metadata.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\metadata\logistic_regression_evaluation_metadata.json) |
| logistic_regression_classification_report_json | [classification_report_logistic_regression.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\classification_reports\classification_report_logistic_regression.json) |
| logistic_regression_classification_report_md | [classification_report_logistic_regression.md](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\classification_reports\classification_report_logistic_regression.md) |
| logistic_regression_confusion_matrix_csv | [confusion_matrix_logistic_regression.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_logistic_regression.csv) |
| logistic_regression_confusion_matrix_json | [confusion_matrix_logistic_regression.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_logistic_regression.json) |
| logistic_regression_confusion_matrix_png | [confusion_matrix_logistic_regression.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_logistic_regression.png) |
| logistic_regression_pr_curve_json | [precision_recall_curve_logistic_regression.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\precision_recall_curves\precision_recall_curve_logistic_regression.json) |
| logistic_regression_pr_curve_png | [precision_recall_curve_logistic_regression.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\precision_recall_curves\precision_recall_curve_logistic_regression.png) |
| logistic_regression_roc_curve_json | [roc_curve_logistic_regression.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\roc_curves\roc_curve_logistic_regression.json) |
| logistic_regression_roc_curve_png | [roc_curve_logistic_regression.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\roc_curves\roc_curve_logistic_regression.png) |
| predictions_logistic_regression | [predictions_logistic_regression.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\predictions\predictions_logistic_regression.csv) |
| predictions_random_forest | [predictions_random_forest.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\predictions\predictions_random_forest.csv) |
| predictions_svm | [predictions_svm.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\predictions\predictions_svm.csv) |
| predictions_xgboost | [predictions_xgboost.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\predictions\predictions_xgboost.csv) |
| random_forest_central_evaluation_metadata | [random_forest_evaluation_metadata.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\metadata\random_forest_evaluation_metadata.json) |
| random_forest_classification_report_json | [classification_report_random_forest.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\classification_reports\classification_report_random_forest.json) |
| random_forest_classification_report_md | [classification_report_random_forest.md](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\classification_reports\classification_report_random_forest.md) |
| random_forest_confusion_matrix_csv | [confusion_matrix_random_forest.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_random_forest.csv) |
| random_forest_confusion_matrix_json | [confusion_matrix_random_forest.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_random_forest.json) |
| random_forest_confusion_matrix_png | [confusion_matrix_random_forest.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_random_forest.png) |
| random_forest_pr_curve_json | [precision_recall_curve_random_forest.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\precision_recall_curves\precision_recall_curve_random_forest.json) |
| random_forest_pr_curve_png | [precision_recall_curve_random_forest.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\precision_recall_curves\precision_recall_curve_random_forest.png) |
| random_forest_roc_curve_json | [roc_curve_random_forest.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\roc_curves\roc_curve_random_forest.json) |
| random_forest_roc_curve_png | [roc_curve_random_forest.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\roc_curves\roc_curve_random_forest.png) |
| statistics_file | [evaluation_statistics.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\statistics\evaluation_statistics.json) |
| svm_central_evaluation_metadata | [svm_evaluation_metadata.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\metadata\svm_evaluation_metadata.json) |
| svm_classification_report_json | [classification_report_svm.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\classification_reports\classification_report_svm.json) |
| svm_classification_report_md | [classification_report_svm.md](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\classification_reports\classification_report_svm.md) |
| svm_confusion_matrix_csv | [confusion_matrix_svm.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_svm.csv) |
| svm_confusion_matrix_json | [confusion_matrix_svm.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_svm.json) |
| svm_confusion_matrix_png | [confusion_matrix_svm.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_svm.png) |
| svm_pr_curve_json | [precision_recall_curve_svm.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\precision_recall_curves\precision_recall_curve_svm.json) |
| svm_pr_curve_png | [precision_recall_curve_svm.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\precision_recall_curves\precision_recall_curve_svm.png) |
| svm_roc_curve_json | [roc_curve_svm.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\roc_curves\roc_curve_svm.json) |
| svm_roc_curve_png | [roc_curve_svm.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\roc_curves\roc_curve_svm.png) |
| xgboost_central_evaluation_metadata | [xgboost_evaluation_metadata.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\metadata\xgboost_evaluation_metadata.json) |
| xgboost_classification_report_json | [classification_report_xgboost.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\classification_reports\classification_report_xgboost.json) |
| xgboost_classification_report_md | [classification_report_xgboost.md](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\classification_reports\classification_report_xgboost.md) |
| xgboost_confusion_matrix_csv | [confusion_matrix_xgboost.csv](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_xgboost.csv) |
| xgboost_confusion_matrix_json | [confusion_matrix_xgboost.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_xgboost.json) |
| xgboost_confusion_matrix_png | [confusion_matrix_xgboost.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\confusion_matrices\confusion_matrix_xgboost.png) |
| xgboost_pr_curve_json | [precision_recall_curve_xgboost.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\precision_recall_curves\precision_recall_curve_xgboost.json) |
| xgboost_pr_curve_png | [precision_recall_curve_xgboost.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\precision_recall_curves\precision_recall_curve_xgboost.png) |
| xgboost_roc_curve_json | [roc_curve_xgboost.json](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\roc_curves\roc_curve_xgboost.json) |
| xgboost_roc_curve_png | [roc_curve_xgboost.png](file:///C:\Users\nered\Desktop\fake news detection\ml\evaluation\roc_curves\roc_curve_xgboost.png) |

---
*Report compiled automatically by Phase 7 Pipeline. Duration: 5.96 seconds.*