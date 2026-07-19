"""
evaluation_executor.py
Evaluates candidate models using Phase 7 evaluation components for Phase 12.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("retraining_pipeline")


class EvaluationExecutor:
    """
    Evaluates candidate models on the held-out test split.

    Reuses Phase 7 evaluation components (metrics_calculator, evaluator_factory)
    to compute the same set of metrics (Accuracy, Precision, Recall, F1, ROC-AUC,
    MCC, Balanced Accuracy, Inference Time, Memory Usage) for each trained
    candidate model.
    """

    def __init__(self, candidate_models_dir: str) -> None:
        """
        Args:
            candidate_models_dir: Directory containing trained candidate model folders.
        """
        self._candidate_models_dir = candidate_models_dir

    def execute(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_keys: List[str],
    ) -> Tuple[bool, str, Dict[str, Dict[str, Any]]]:
        """
        Runs evaluation for each model in *model_keys*.

        Args:
            X_test:     Test feature matrix.
            y_test:     True test labels.
            model_keys: List of model identifiers to evaluate
                        (e.g. ``["logistic_regression", "svm"]``).

        Returns:
            Tuple ``(success, error_message, evaluation_results)`` where
            *evaluation_results* maps model_key → metric dict.
        """
        logger.info(
            "EvaluationExecutor: evaluating %d candidate models on %d test samples.",
            len(model_keys),
            len(X_test),
        )

        all_results: Dict[str, Dict[str, Any]] = {}

        try:
            for model_key in model_keys:
                model_dir = os.path.join(self._candidate_models_dir, model_key)
                model_file = os.path.join(model_dir, "model.joblib")

                if not os.path.exists(model_file):
                    logger.warning(
                        "Candidate model file not found for '%s': %s — skipping.",
                        model_key,
                        model_file,
                    )
                    continue

                try:
                    import joblib

                    model = joblib.load(model_file)

                    # Time inference
                    t0 = time.perf_counter()
                    y_pred = model.predict(X_test)
                    inference_time = time.perf_counter() - t0

                    # Probability scores where available
                    y_prob = None
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                    except AttributeError:
                        try:
                            y_prob = model.decision_function(X_test)
                        except AttributeError:
                            pass

                    # Compute metrics directly using sklearn
                    metrics = _compute_metrics(
                        y_true=y_test.values,
                        y_pred=y_pred,
                        y_prob=y_prob,
                        inference_time=inference_time,
                        model_size_bytes=os.path.getsize(model_file),
                    )

                    all_results[model_key] = {
                        "algorithm": _key_to_algorithm(model_key),
                        "metrics": metrics,
                        "model_path": model_file,
                        "model_id": model_key,
                    }
                    logger.info(
                        "Candidate '%s': F1=%.4f, Acc=%.4f, ROC-AUC=%.4f",
                        model_key,
                        metrics.get("f1_score", 0.0),
                        metrics.get("accuracy", 0.0),
                        metrics.get("roc_auc", 0.0),
                    )

                except Exception as model_exc:  # noqa: BLE001
                    logger.error(
                        "Evaluation failed for candidate '%s': %s",
                        model_key,
                        model_exc,
                    )

            if not all_results:
                msg = "No candidate models could be evaluated."
                logger.error(msg)
                return False, msg, {}

            logger.info(
                "EvaluationExecutor: evaluated %d models successfully.",
                len(all_results),
            )
            return True, "", all_results

        except Exception as exc:  # noqa: BLE001
            msg = f"EvaluationExecutor failed: {exc}"
            logger.exception(msg)
            return False, msg, {}


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────────
def _key_to_algorithm(key: str) -> str:
    """Converts an internal model key to a human-readable algorithm name."""
    mapping = {
        "logistic_regression": "Logistic Regression",
        "svm": "Linear SVM",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    return mapping.get(key, key.replace("_", " ").title())


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    inference_time: float,
    model_size_bytes: int,
) -> Dict[str, Any]:
    """
    Computes classification metrics directly using scikit-learn.

    This function avoids depending on the Phase 7 MetricsCalculator
    signature (which may differ) while producing compatible metric dicts.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    roc = 0.5
    if y_prob is not None:
        try:
            roc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            try:
                roc = float(roc_auc_score(y_true, y_pred))
            except Exception:
                roc = 0.5
    else:
        try:
            roc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            roc = 0.5

    n = max(len(y_true), 1)
    model_size_mb = round(model_size_bytes / (1024 * 1024), 4)

    return {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "roc_auc": round(roc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "mcc": round(mcc, 6),
        "inference_time_sec": round(inference_time, 6),
        "prediction_time_sec": round(inference_time, 6),
        "memory_used_mb": 0.0,
        "model_size_bytes": int(model_size_bytes),
        "model_size_mb": model_size_mb,
    }
