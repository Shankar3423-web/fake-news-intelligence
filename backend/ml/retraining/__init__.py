"""
Phase 12 – Automatic Model Retraining System
Fake News Intelligence System

This package orchestrates the full retraining lifecycle:
  - Load administrator-approved feedback
  - Merge with existing training data
  - Execute preprocessing → feature engineering → feature selection → training → evaluation
  - Compare candidate against production model
  - Promote if acceptance criteria met
  - Generate reports, metadata, statistics, hashes, versions, and visualization charts
"""

__version__ = "1.0.0"
__phase__ = 12
__description__ = "Automatic Model Retraining System"
