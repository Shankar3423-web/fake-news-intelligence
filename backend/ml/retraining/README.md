# Phase 12 — Automatic Model Retraining System

## Overview

This module implements the **Automatic Model Retraining System** for the Fake News Intelligence System. It safely evolves the production model using only administrator-approved feedback, compares the candidate model against the production baseline using a configurable acceptance policy, and promotes only when the candidate satisfies all criteria.

---

## Folder Structure

```
ml/retraining/
├── __init__.py
├── retraining_pipeline.py          # Main orchestrator (18-step pipeline)
├── retraining_config.py            # Configuration manager
├── retraining_config.yaml          # Configuration file
├── retraining_logger.py            # Structured logger setup
├── approved_feedback_loader.py     # Phase 11 approved feedback reader
├── dataset_merge_manager.py        # Merges approved feedback with training data
├── dataset_validator.py            # Dataset integrity validation
├── preprocessing_executor.py       # Delegates to Phase 3 pipeline
├── feature_executor.py             # Delegates to Phase 4 pipeline
├── feature_selection_executor.py   # Delegates to Phase 5 pipeline
├── training_executor.py            # Trains candidate models (Phase 6 components)
├── evaluation_executor.py          # Evaluates candidates (Phase 7 components)
├── model_comparator.py             # Candidate vs. production comparison
├── deployment_manager.py           # Promotion / rejection logic
├── model_registry.py               # Retraining run registry
├── statistics_manager.py           # Pipeline statistics
├── metadata_manager.py             # Environment + run metadata
├── report_generator.py             # Markdown reports
├── visualization.py                # Charts (requires matplotlib)
├── hash_generator.py               # SHA-256 checksums
├── version_manager.py              # Version history
├── validator.py                    # Artifact validation
├── verify_retraining.py            # Phase 12 verification script (run from project root)
├── README.md
│
├── logs/                           # Pipeline logs
├── reports/                        # retraining_report.md, comparison_report.md
├── metadata/                       # retraining_metadata.json
├── statistics/                     # retraining_statistics.json
├── versions/                       # retraining_versions.json
├── hashes/                         # retraining_hashes.json
├── registry/                       # retraining_registry.json
├── candidate_models/               # Newly trained candidate model artifacts
├── production_models/              # Promoted production model artifacts
├── charts/                         # PNG visualization charts
└── tests/
    ├── __init__.py
    └── test_retraining.py
```

---

## Workflow

```
Approved Feedback (Phase 11)
        ↓
   Validation
        ↓
  Dataset Merge
        ↓
Dataset Integrity Check
        ↓
  Train Candidate (Phase 6 components)
        ↓
  Evaluate Candidate (Phase 7 components)
        ↓
Compare vs. Production (acceptance policy)
        ↓
  Deployment Decision (PROMOTE / REJECT)
        ↓
  Reports → Metadata → Statistics → Hashes → Versions
        ↓
   Verification
```

---

## Configuration

Edit `retraining_config.yaml` to control:

- **Approved feedback source** — path to `approved_feedback.csv` (Phase 11 output)
- **Training dataset** — path to Phase 5 feature-selected dataset
- **Acceptance policy** — primary metric, minimum delta, minimum thresholds, weighted comparison
- **Output paths** — all artifact directories

### Key Acceptance Policy Parameters

| Parameter | Description |
|:---|:---|
| `primary_metric` | Metric to use for primary comparison (default: `f1_score`) |
| `minimum_improvement_delta` | Minimum candidate must beat production by this delta |
| `minimum_thresholds` | Candidate must reach these scores regardless of comparison |
| `promote_on_tie` | Whether to promote when scores are exactly equal |
| `weighted_comparison.enabled` | Use weighted multi-metric scoring |
| `weighted_comparison.weights` | Per-metric weights (must sum to 1.0) |

---

## Generated Artifacts

| Artifact | Path |
|:---|:---|
| Main retraining report | `ml/retraining/reports/retraining_report.md` |
| Model comparison report | `ml/retraining/reports/comparison_report.md` |
| Deployment decision | `ml/retraining/reports/deployment_decision.json` |
| Statistics | `ml/retraining/statistics/retraining_statistics.json` |
| Metadata | `ml/retraining/metadata/retraining_metadata.json` |
| SHA-256 hashes | `ml/retraining/hashes/retraining_hashes.json` |
| Version history | `ml/retraining/versions/retraining_versions.json` |
| Registry | `ml/retraining/registry/retraining_registry.json` |
| Charts | `ml/retraining/charts/*.png` |

---

## Running the Pipeline

```bash
# From project root
python -m ml.retraining.retraining_pipeline
```

## Verifying Phase 12

```bash
# From project root
python verify_retraining.py
```

Expected output:
```
====================================================
PHASE 12 AUTOMATIC MODEL RETRAINING VERIFICATION PASSED
====================================================
```

## Running Unit Tests

```bash
pytest ml/retraining/tests/test_retraining.py -v
```

---

## Design Principles

- **SOLID**: Each class has a single responsibility; components are swappable.
- **DRY**: Phase 3–7 pipelines are delegated via adapter executors — no logic duplication.
- **KISS**: Configuration-driven; no magic constants in code.
- **PEP-8**: Full type hints, docstrings, and structured logging throughout.
