import os
import sys
import json
import logging
from typing import Tuple, List

# Ensure project root is in path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.training.training_config import TrainingConfig
from ml.training.training_validator import TrainingValidator
from ml.training.training_utils import compute_file_sha256

def verify_training_integrity(project_root_dir: str) -> Tuple[bool, List[str]]:
    """
    Runs training verification checks on all generated artifacts.
    """
    failures = []
    
    # 1. Load config
    config_path = os.path.join(project_root_dir, "ml/training/training_config.yaml")
    if not os.path.exists(config_path):
        failures.append(f"Training config file missing: {config_path}")
        return False, failures

    config = TrainingConfig(config_path)
    
    # 2. Check Directory Structure
    directories = [
        config.get_output_dir("models_dir"),
        config.get_output_dir("metadata_dir"),
        config.get_output_dir("reports_dir"),
        config.get_output_dir("statistics_dir"),
        config.get_output_dir("versions_dir"),
        config.get_output_dir("hashes_dir"),
        config.get_output_dir("logs_dir")
    ]
    for d in directories:
        abs_d = os.path.join(project_root_dir, d)
        if not os.path.exists(abs_d):
            failures.append(f"Directory missing: {d}")

    # 3. Check training pipeline files existence
    pipeline_files = [
        "ml/training/__init__.py",
        "ml/training/training_pipeline.py",
        "ml/training/training_config.py",
        "ml/training/training_config.yaml",
        "ml/training/training_logger.py",
        "ml/training/training_utils.py",
        "ml/training/dataset_loader.py",
        "ml/training/dataset_splitter.py",
        "ml/training/logistic_regression_trainer.py",
        "ml/training/svm_trainer.py",
        "ml/training/random_forest_trainer.py",
        "ml/training/xgboost_trainer.py",
        "ml/training/trainer_factory.py",
        "ml/training/model_registry.py",
        "ml/training/training_validator.py",
        "ml/training/training_statistics.py",
        "ml/training/training_profiler.py",
        "ml/training/metadata_manager.py",
        "ml/training/hash_generator.py",
        "ml/training/version_manager.py",
        "ml/training/report_generator.py",
        "ml/training/verify_training.py",
        "ml/training/README.md"
    ]
    for f in pipeline_files:
        abs_f = os.path.join(project_root_dir, f)
        if not os.path.exists(abs_f):
            failures.append(f"Code file missing: {f}")

    # 4. Check Dataset Loader configuration & existence
    input_csv = os.path.join(project_root_dir, config.get_input_path("dataset_csv"))
    if not os.path.exists(input_csv):
        failures.append(f"Input dataset CSV file missing: {input_csv}")

    # 5. Delegate validation to TrainingValidator
    validator = TrainingValidator(config)
    success, validation_errors = validator.validate_all()
    if not success:
        failures.extend(validation_errors)

    # 6. Verify version records link correctly
    versions_file = os.path.join(project_root_dir, config.get_output_path("versions_file"))
    if os.path.exists(versions_file):
        try:
            with open(versions_file, "r", encoding="utf-8") as f:
                versions = json.load(f)
            if not isinstance(versions, list) or len(versions) == 0:
                failures.append("training_versions.json is empty or not a list")
            else:
                last_run = versions[-1]
                # Check version matches
                if "version" not in last_run:
                    failures.append("Latest run version record is missing version number")
        except Exception as e:
            failures.append(f"Failed to read and parse training_versions.json: {e}")

    # 7. Verify registry file contents
    registry_file = os.path.join(project_root_dir, config.get_output_path("registry_file"))
    if os.path.exists(registry_file):
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                registry = json.load(f)
            if not isinstance(registry, dict) or "models" not in registry:
                failures.append("registry.json is malformed")
        except Exception as e:
            failures.append(f"Failed to read and parse registry.json: {e}")

    return len(failures) == 0, failures

if __name__ == "__main__":
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    passed, errors = verify_training_integrity(project_root_dir)
    
    if passed:
        print("\n" + "="*52)
        print("PHASE 6 MODEL TRAINING VERIFICATION PASSED")
        print("="*52 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*52)
        print("PHASE 6 MODEL TRAINING VERIFICATION FAILED")
        print("="*52)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        print("="*52 + "\n")
        sys.exit(1)
