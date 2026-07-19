# Phase 12 — Model Comparison Report

*Generated: 2026-07-19 13:05:29 UTC | Run ID: `retrain_20260719_183430_f54e5499`*

---

## Decision

**❌ REJECTED**

> Candidate rejected: weighted_score 0.9986 ≤ production 0.9986 — production model retained.

## Weighted Score Comparison

| Model | Weighted Score |
| :--- | :---: |
| **Candidate** | `0.9986` |
| **Production** | `0.9986` |

## Per-Metric Comparison

| Metric | Candidate | Production | Delta | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **accuracy** | `0.9983` | `0.9983` | `0.0000` | — Tie |
| **balanced_accuracy** | `0.9983` | `0.9983` | `0.0000` | — Tie |
| **f1_score** | `0.9983` | `0.9983` | `0.0000` | — Tie |
| **inference_time_sec** | `0.0312` | `0.0301` | `0.0011` | 🥈 Production |
| **mcc** | `0.9966` | `0.9966` | `0.0000` | — Tie |
| **memory_used_mb** | `0.0000` | `0.0000` | `0.0000` | — Tie |
| **model_size_bytes** | `226444.0000` | `226444.0000` | `0.0000` | — Tie |
| **model_size_mb** | `0.2160` | `0.2160` | `0.0000` | — Tie |
| **precision** | `0.9994` | `0.9994` | `0.0000` | — Tie |
| **prediction_time_sec** | `0.0312` | `0.0301` | `0.0011` | 🥈 Production |
| **recall** | `0.9971` | `0.9971` | `0.0000` | — Tie |
| **roc_auc** | `0.9999` | `0.9999` | `0.0000` | — Tie |

## Minimum Threshold Check

| Metric | Required | Candidate Value | Passed |
| :--- | :---: | :---: | :---: |
| **f1_score** | `0.6` | `0.9983` | ✅ |
| **accuracy** | `0.6` | `0.9983` | ✅ |
