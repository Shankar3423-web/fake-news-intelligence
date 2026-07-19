# Phase 8 — Prediction (Inference) Engine

This folder contains the production-grade prediction (inference) engine for the Fake News Detection pipeline. The engine is designed to receive unseen news articles, validate them, execute the exact same NLP preprocessing, feature engineering, and feature selection pipelines defined in prior phases, load the best model dynamically, and produce standardized prediction responses.

## Architecture & Reusability

To guarantee mathematical parity between training-time transformations and prediction-time transformations, this engine does not duplicate any preprocessing, extraction, or selection logic. Instead, it reuses the original components:
1. **Phase 3 Preprocessing**: Applies the exact text cleaning, language checks, NLTK stopword removal, spaCy lemmatization, and token constraints.
2. **Phase 4 Feature Engineering**: Extracts dense statistical, readability, lexical, symbol, and linguistic features, and transforms cleaned text using the saved TF-IDF Vectorizer.
3. **Phase 5 Feature Selection**: Identifies and slices only the final selected features defined in `selected_feature_names.json`.
4. **Dynamic Model Loading**: Reads `best_model.json` to load the current production model, verifying feature count and ordering before inference.

---

## Folder Structure

```
ml/prediction/
│
├── README.md                     # Documentation
├── prediction_config.yaml        # Configurable limits, logging, and outputs
├── prediction_config.py          # Configuration manager
├── prediction_logger.py          # Structured logging utility
├── prediction_utils.py           # Memory checking and timestamp helpers
├── input_validator.py            # Sanitizes raw text inputs
├── model_loader.py               # Dynamic model registry loader
├── pipeline_executor.py          # Executes Phase 3 -> 4 -> 5 pipelines
├── confidence_calculator.py      # Normalizes probabilities & margins to [0.0, 1.0]
├── inference_engine.py           # Feeds features, measures latency & memory
├── response_builder.py           # Formats standardized output response
├── metadata_manager.py           # Generates prediction_metadata.json
├── prediction_statistics.py      # Tracks running average metrics
├── prediction_profiler.py        # Generates trend PNG charts
├── hash_generator.py             # Creates SHA-256 signatures for outputs
├── version_manager.py            # Registers prediction runs
├── report_generator.py           # Generates prediction_report.md
├── verify_prediction.py          # Validation script
│
├── logs/                         # Execution logs
├── reports/                      # Markdown summary reports
├── statistics/                   # Cumulative statistics JSON
├── metadata/                     # Runtime metadata JSON
├── hashes/                       # Integrity hash registries
├── versions/                     # Execution versions JSON
├── predictions/                  # Historical prediction CSV records
└── tests/                        # Pytest unit tests
```

---

## Execution

To verify the prediction engine and run a sample text inference:

```powershell
python ml/prediction/verify_prediction.py
```

This will run verification tests on the folder structure, config load, model compatibility, pipeline execution, and files exports. On success, it outputs:

```
====================================================
PHASE 8 PREDICTION ENGINE VERIFICATION PASSED
====================================================
```

To run unit tests:

```powershell
pytest ml/prediction/tests/
```

---

## Outputs

Every execution generates or appends details to the following files in the configured paths:
- **`predictions/prediction_history.csv`**: Logs every article prediction with timestamp, text hash, prediction class, confidence, latency, throughput, and version.
- **`statistics/prediction_statistics.json`**: Cumulative averages of memory footprint, throughput, confidence, and runtimes.
- **`reports/prediction_report.md`**: Beautiful markdown report summarizing model information, metrics, and verification warnings.
- **`metadata/prediction_metadata.json`**: Runtime versions log (training, evaluation, feature, and pipeline version).
- **`versions/prediction_versions.json`**: Version increment tracking for each run.
- **`hashes/prediction_hashes.json`**: SHA-256 signatures of all files generated in the run.
- **`charts/`**: Plot PNG charts of latency, throughput trends, and confidence distributions.

---

## Troubleshooting

- **SpaCy / NLTK errors**: Verify that NLTK stopwords and spaCy's `en_core_web_sm` model are installed. The executor will automatically attempt to install them on start.
- **Best Model Mismatch**: If `ml/evaluation/best_model.json` is missing or contains an invalid model path, ensure that the evaluation phase (Phase 7) has been successfully run to select a model.
