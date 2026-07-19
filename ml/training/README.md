# Phase 6 — Model Training Module

This package implements the **Model Training** phase (Phase 6) of the Fake News Detection pipeline. It loads the selected feature dataset produced in Phase 5, splits the dataset, trains multiple classifiers, serializes the models, tracks metadata, computes checksums, updates the registry, log versions, and generates a training report.

---

## 1. Directory Structure

```
ml/training/
├── __init__.py                  # Exposes pipeline and config interfaces
├── training_pipeline.py         # Main orchestrator workflow
├── training_config.py           # Configuration parser and manager
├── training_config.yaml         # Configuration YAML file
├── training_logger.py           # Structured logger configuration
├── training_utils.py            # Hashing and memory footprint benchmarks
├── dataset_loader.py            # Selected dataset schema & column validator
├── dataset_splitter.py          # Stratified train/test splitter
├── logistic_regression_trainer.py # Logistic Regression trainer
├── svm_trainer.py               # Linear SVM trainer
├── random_forest_trainer.py     # Random Forest trainer
├── xgboost_trainer.py           # XGBoost trainer
├── trainer_factory.py           # Factory pattern for trainer instantiation
├── model_registry.py            # Model registry (registry.json) manager
├── training_validator.py        # Post-training validator
├── training_statistics.py       # Aggregate run statistics generator
├── training_profiler.py         # Time and memory profiling context managers
├── metadata_manager.py          # Metadata generation and version tracing
├── hash_generator.py            # SHA-256 checksum manager
├── version_manager.py           # Incremental version manager
├── report_generator.py          # Training report markdown generator
├── verify_training.py           # Verification script
├── README.md                    # Documentation
├── models/                      # Serialized models (.joblib files)
├── metadata/                    # Model metadata copies
├── reports/                     # Training markdown report
├── statistics/                  # Pipeline execution statistics
├── versions/                    # Pipeline run version registries
├── hashes/                      # SHA-256 hash files
├── logs/                        # Training execution logs
└── tests/                       # Unit tests
```

---

## 2. Supported Models & Hyperparameters

The following classifiers are trained by the pipeline:

1. **Logistic Regression**
   - Configurable: `solver`, `max_iter`, `C`, `random_state`.
2. **Linear SVM (LinearSVC)**
   - Configurable: `C`, `max_iter`, `random_state`.
3. **Random Forest**
   - Configurable: `n_estimators`, `max_depth`, `random_state`, `n_jobs`.
4. **XGBoost**
   - Configurable: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `objective`, `random_state`.

All model configurations are defined in `training_config.yaml`.

---

## 3. Usage & Execution

To execute the training pipeline programmatically, load the training pipeline from python:

```python
from ml.training.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline("ml/training/training_config.yaml")
success = pipeline.run()
```

---

## 4. Verification

To verify that the model training pipeline succeeded and all outputs align with the verification suite:

```bash
python verify_training.py
```

This will run assertions on the generated file structure, hashes, metadata, version listings, and index entries.
