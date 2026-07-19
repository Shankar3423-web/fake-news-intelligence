import os
import sys
import json
from typing import Tuple, List

# Ensure project root is in path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.evaluation_validator import EvaluationValidator

def verify_evaluation_integrity(project_root_dir: str) -> Tuple[bool, List[str]]:
    """
    Runs evaluation verification checks on all generated artifacts.
    """
    failures = []
    
    # 1. Load config
    config_path = os.path.join(project_root_dir, "ml/evaluation/evaluation_config.yaml")
    if not os.path.exists(config_path):
        failures.append(f"Evaluation config file missing: {config_path}")
        return False, failures

    config = EvaluationConfig(config_path)
    
    # 2. Check Directory Structure
    dir_keys = [
        "reports_dir", "statistics_dir", "metadata_dir", "hashes_dir",
        "versions_dir", "leaderboard_dir", "comparison_dir", "logs_dir",
        "predictions_dir", "classification_reports_dir", "confusion_matrices_dir",
        "roc_curves_dir", "precision_recall_curves_dir", "charts_dir"
    ]
    for key in dir_keys:
        d = config.get_output_dir(key)
        abs_d = os.path.join(project_root_dir, d)
        if not os.path.exists(abs_d):
            failures.append(f"Directory missing: {d}")

    # 3. Check evaluation module files existence
    module_files = [
        "ml/evaluation/__init__.py",
        "ml/evaluation/evaluation_pipeline.py",
        "ml/evaluation/evaluation_config.py",
        "ml/evaluation/evaluation_config.yaml",
        "ml/evaluation/evaluation_logger.py",
        "ml/evaluation/evaluation_utils.py",
        "ml/evaluation/dataset_loader.py",
        "ml/evaluation/model_loader.py",
        "ml/evaluation/prediction_engine.py",
        "ml/evaluation/logistic_regression_evaluator.py",
        "ml/evaluation/svm_evaluator.py",
        "ml/evaluation/random_forest_evaluator.py",
        "ml/evaluation/xgboost_evaluator.py",
        "ml/evaluation/evaluator_factory.py",
        "ml/evaluation/metrics_calculator.py",
        "ml/evaluation/confusion_matrix_generator.py",
        "ml/evaluation/roc_auc_generator.py",
        "ml/evaluation/classification_report_generator.py",
        "ml/evaluation/comparison_generator.py",
        "ml/evaluation/leaderboard_generator.py",
        "ml/evaluation/best_model_selector.py",
        "ml/evaluation/evaluation_validator.py",
        "ml/evaluation/evaluation_statistics.py",
        "ml/evaluation/evaluation_profiler.py",
        "ml/evaluation/metadata_manager.py",
        "ml/evaluation/hash_generator.py",
        "ml/evaluation/version_manager.py",
        "ml/evaluation/report_generator.py",
        "ml/evaluation/verify_evaluation.py",
        "ml/evaluation/README.md"
    ]
    for f in module_files:
        abs_f = os.path.join(project_root_dir, f)
        if not os.path.exists(abs_f):
            failures.append(f"Code file missing: {f}")

    # 4. Delegate validation to EvaluationValidator
    validator = EvaluationValidator(config)
    success, validation_errors = validator.validate_all()
    if not success:
        failures.extend(validation_errors)

    # 5. Verify version records link correctly
    versions_file = os.path.join(project_root_dir, config.get_output_path("versions_file"))
    if os.path.exists(versions_file):
        try:
            with open(versions_file, "r", encoding="utf-8") as f:
                versions = json.load(f)
            if not isinstance(versions, list) or len(versions) == 0:
                failures.append("evaluation_versions.json is empty or not a list")
            else:
                last_run = versions[-1]
                if "version" not in last_run:
                    failures.append("Latest run version record is missing version number")
        except Exception as e:
            failures.append(f"Failed to read and parse evaluation_versions.json: {e}")

    # 6. Verify best model selection
    best_model_file = os.path.join(project_root_dir, config.get_output_path("best_model_file"))
    if os.path.exists(best_model_file):
        try:
            with open(best_model_file, "r", encoding="utf-8") as f:
                best = json.load(f)
            if not isinstance(best, dict) or "model_key" not in best:
                failures.append("best_model.json is malformed")
        except Exception as e:
            failures.append(f"Failed to read and parse best_model.json: {e}")

    return len(failures) == 0, failures

if __name__ == "__main__":
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    passed, errors = verify_evaluation_integrity(project_root_dir)
    
    if passed:
        print("\n" + "="*52)
        print("PHASE 7 MODEL EVALUATION VERIFICATION PASSED")
        print("="*52 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*52)
        print("PHASE 7 MODEL EVALUATION VERIFICATION FAILED")
        print("="*52)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        print("="*52 + "\n")
        sys.exit(1)
