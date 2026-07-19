# Phase 10 — Feedback Collection System

This directory implements a production-grade, configuration-driven **Feedback Collection System** for the Fake News Intelligence System.

The feedback collection system processes, validates, sanitizes, and records user-submitted feedback regarding ML predictions and verification results. It persists the transaction history, updates cumulative metrics, compiles metadata, writes markdown summaries, renders charts, signs file checksums, and handles sequential registry versioning.

---

## Folder Structure

```
ml/
feedback/
    __init__.py               # Public API exports
    feedback_pipeline.py      # Core pipeline orchestration
    feedback_config.py        # Config loader and validation
    feedback_config.yaml      # YAML configuration file
    feedback_logger.py        # Pipeline logging setup
    feedback_validator.py     # Schema, type, and bounds validation
    feedback_sanitizer.py     # Unicode normalization and comment cleaning
    feedback_manager.py       # ID generation and Feedback Record compilation
    feedback_history.py       # Appending to persistent CSV history
    feedback_statistics.py    # Running averages and distribution calculator
    feedback_metadata.py      # Metadata writer
    feedback_report.py        # Markdown report generator
    feedback_visualization.py # Matplotlib chart builder
    feedback_hashes.py        # SHA-256 integrity checksum builder
    feedback_versions.py      # Sequential versions registration manager
    response_builder.py       # Pipeline response formatter
    verify_feedback.py        # Validation and integrity script
    README.md                 # System documentation
    
    logs/                     # Log files directory
    metadata/                 # Run metadata JSON files
    reports/                  # Markdown report files
    statistics/               # Aggregated statistics JSON files
    history/                  # Persistent CSV history file
    hashes/                   # SHA-256 hashes registry JSON
    versions/                 # Registered executions JSON database
    charts/                   # Rendered visualization PNG charts
    tests/                    # Test suites directory
```

---

## Configuration (`feedback_config.yaml`)

You can control pipeline behaviors by modifying `feedback_config.yaml`:
- **Comment Limits**: Modify `max_comment_length` (default `500`), `min_comment_length` (default `3`), and `allow_empty_comments` (default `true`).
- **Allowed Feedback Values**: Restricted to `["Correct", "Incorrect", "Unsure"]`.
- **Toggles**: Individually enable/disable chart rendering, markdown reporting, statistics calculation, version registration, metadata, and SHA-256 hash creation.
- **Paths**: Define the folders and file destinations for the pipeline products.

---

## E2E Workflow

```
Prediction Output
↓
Verification Output
↓
Final Decision
↓
User Feedback Value
↓
Validation (FeedbackValidator)
↓
Input Sanitization (FeedbackSanitizer)
↓
Feedback Record Compilation (FeedbackManager)
↓
Persistent Storage (HistoryManager)
↓
Statistics Aggregation (StatisticsManager)
↓
Metadata Construction (MetadataManager)
↓
Charts Generation (FeedbackVisualizer)
↓
Markdown Report (ReportGenerator)
↓
Sequential Versioning (VersionManager)
↓
SHA-256 Checksums (HashGenerator)
↓
Unified JSON Output (ResponseBuilder)
```

---

## Execution

### Run Verification Script
To run the automated pipeline validation script:
```bash
python ml/feedback/verify_feedback.py
```
Upon success, the script prints:
```
====================================================
PHASE 10 FEEDBACK SYSTEM VERIFICATION PASSED
====================================================
```

### Run Unit Tests
To execute the pytest suite for Phase 10:
```bash
pytest ml/feedback/tests/test_feedback.py
```

---

## Integration Example

```python
from ml.feedback import run_feedback_pipeline

response = run_feedback_pipeline(
    prediction=1,
    prediction_confidence=0.88,
    verification_status="PARTIALLY VERIFIED",
    evidence_score=0.62,
    similarity_score=0.74,
    final_decision="SUSPECTED REAL",
    user_feedback="Correct",
    comment="Matches other reputable sources.",
    config_path="ml/feedback/feedback_config.yaml"
)

if response["status"] == "SUCCESS":
    print(f"Recorded feedback ID: {response['feedback_id']}")
    print(f"New history version: {response['version']}")
else:
    print(f"Failed: {response['error']}")
```

---

## Troubleshooting

- **Chart Generation Fails or is Skipped**: If `matplotlib` is not installed, the pipeline will log a warning and skip chart generation gracefully without crashing.
- **Validation Errors**: If inputs are rejected, inspect the `error` field in the response. Ensure that `prediction_confidence`, `evidence_score`, and `similarity_score` are floating point values between `0.0` and `1.0`, and that `user_feedback` is exactly `"Correct"`, `"Incorrect"`, or `"Unsure"`.
- **Locked File Errors**: On Windows, concurrent writes or active file viewers might lock the history CSV. The logger will output file operation failures to `feedback_pipeline.log`.
