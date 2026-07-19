# Phase 11 - Admin Review System

The Admin Review System enables administrators to moderate collected user feedback and approve high-quality records for future model retraining.

## Modules

- **`admin_review_config.py` / `.yaml`**: Handles YAML configuration parsing.
- **`admin_review_logger.py`**: Configures static and session-based loggers.
- **`feedback_loader.py`**: Fetches pending feedback from Phase 10 feedback history.
- **`review_validator.py`**: Performs schema and status type checks before saving decisions.
- **`review_manager.py`**: Submits moderation decisions and coordinates storage workflows.
- **`history_manager.py`**: Logs all administrative decisions to `admin_review_history.csv`.
- **`approved_dataset_manager.py`**: Manages and synchronizes `approved_feedback.csv`.
- **`approval_manager.py`**: Determines if the quantity of approved feedback qualifies for retraining.
- **`statistics_manager.py`**: Saves counts and approval rates to `admin_review_statistics.json`.
- **`metadata_manager.py`**: Captures environment data in `admin_review_metadata.json`.
- **`report_generator.py`**: Formats reports in `admin_review_report.md`.
- **`hash_generator.py`**: Computes SHA-256 integrity signatures in `admin_review_hashes.json`.
- **`version_manager.py`**: Handles snapshot increment storage in `admin_review_versions.json`.
- **`visualization.py`**: Generates distribution charts (`approval_distribution.png`, `review_status_distribution.png`).
- **`validator.py`**: Assures that all output files conform to formatting/checksum schemas.
- **`admin_review_pipeline.py`**: The E2E orchestrator function.

## Execution and Verification

### Unit Tests
Execute the unit test suite:
```bash
pytest ml/admin_review/tests/
```

### Verification Runner
To run the full E2E validation script:
```bash
python verify_admin_review.py
```
This script will mock/load feedback history, process moderation approvals and rejections, check files and directories, render visualization plots, execute the unit test suite, and print the confirmation banner.
