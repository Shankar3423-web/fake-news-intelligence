import logging
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss
)

logger = logging.getLogger("model_evaluation_pipeline")

class MetricsCalculator:
    """
    Calculates all classification and profiling metrics for a given model:
    - Accuracy, Precision, Recall, F1 Score
    - ROC-AUC
    - Balanced Accuracy
    - Matthews Correlation Coefficient (MCC)
    - Cohen's Kappa
    - Log Loss (only for models supporting predict_proba)
    - Prediction Time, Inference Throughput, Memory Usage, Model Size
    """
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        y_prob_type: str,
        duration_sec: float,
        throughput_sps: float,
        memory_used_mb: float,
        model_size_bytes: int
    ) -> Dict[str, Any]:
        """
        Calculates and aggregates all performance and resource utilization metrics.
        """
        logger.info("Calculating standard evaluation metrics...")
        
        # 1. Standard classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        # 2. ROC-AUC calculation
        roc_auc = 0.5  # Default baseline
        if y_prob is not None:
            try:
                # If we have probabilities or decision scores, compute ROC AUC
                roc_auc = roc_auc_score(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Failed to calculate ROC-AUC: {e}. Defaulting to 0.5")
                roc_auc = 0.5
        else:
            # Fallback to class predictions if no scores are available
            try:
                roc_auc = roc_auc_score(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Failed to calculate ROC-AUC on predictions: {e}")

        # 3. Log Loss calculation
        loss_val = None
        if y_prob_type == "predict_proba" and y_prob is not None:
            try:
                # Log loss requires probability in bounds [0, 1]
                loss_val = log_loss(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Failed to calculate Log Loss: {e}")
        else:
            logger.info("Log Loss is skipped since probability scores are not available/supported.")

        model_size_mb = round(model_size_bytes / (1024 * 1024), 4)

        metrics = {
            "accuracy": round(float(accuracy), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1_score": round(float(f1), 6),
            "roc_auc": round(float(roc_auc), 6),
            "balanced_accuracy": round(float(balanced_accuracy), 6),
            "mcc": round(float(mcc), 6),
            "cohen_kappa": round(float(kappa), 6),
            "log_loss": round(float(loss_val), 6) if loss_val is not None else None,
            "prediction_time_sec": round(duration_sec, 4),
            "inference_throughput_sps": round(throughput_sps, 2),
            "inference_latency_ms": round(float((duration_sec * 1000.0) / len(y_true)), 6) if len(y_true) > 0 else 0.0,
            "memory_used_mb": round(memory_used_mb, 2),
            "model_size_bytes": int(model_size_bytes),
            "model_size_mb": model_size_mb
        }

        logger.info(
            f"Metrics: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}, "
            f"ROC_AUC={metrics['roc_auc']:.4f}, LogLoss={metrics['log_loss']}"
        )
        return metrics
