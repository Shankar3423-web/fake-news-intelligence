import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from ml.evaluation.evaluation_utils import get_memory_usage

logger = logging.getLogger("model_evaluation_pipeline")

class PredictionEngine:
    """
    Handles model inference and profiles execution:
    - Generates class predictions.
    - Generates probability scores (uses predict_proba when available, decision_function as fallback).
    - Measures peak memory consumption and prediction duration.
    - Calculates average inference latency per sample and throughput (samples/second).
    - Returns structured prediction outputs.
    """
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs prediction on the input dataframe and measures profiling metrics.
        
        Returns:
            Dict containing:
                y_pred: np.ndarray of predicted classes
                y_prob: Optional np.ndarray of positive class probabilities (or decision function values)
                y_prob_type: str, one of 'predict_proba', 'decision_function', or 'none'
                prediction_duration_sec: float
                inference_throughput_sps: float
                inference_latency_ms: float
                memory_used_mb: float
        """
        num_samples = len(X)
        logger.info(f"Running prediction engine for {num_samples} samples...")

        # 1. Warm-up (sanity prediction on first row to load dynamic libraries/modules if any)
        try:
            _ = self.model.predict(X.iloc[[0]])
        except Exception:
            pass

        # 2. Start profiling memory and time
        start_time = time.perf_counter()
        mem_before_rss, _ = get_memory_usage()

        # 3. Class predictions
        y_pred = self.model.predict(X)

        # 4. End profiling time and memory
        end_time = time.perf_counter()
        mem_after_rss, _ = get_memory_usage()

        prediction_duration = end_time - start_time
        memory_used = max(0.0, mem_after_rss - mem_before_rss)

        # 5. Class probabilities or decision function
        y_prob = None
        y_prob_type = "none"

        if hasattr(self.model, "predict_proba"):
            logger.info("Extracting probability scores via predict_proba...")
            probs = self.model.predict_proba(X)
            # Binary classification positive class probabilities
            if probs.ndim == 2 and probs.shape[1] == 2:
                y_prob = probs[:, 1]
            else:
                y_prob = probs
            y_prob_type = "predict_proba"
        elif hasattr(self.model, "decision_function"):
            logger.info("Extracting decision scores via decision_function...")
            y_prob = self.model.decision_function(X)
            y_prob_type = "decision_function"
        else:
            logger.warning("Model does not support probability predictions or decision function.")

        # Convert to numpy array safely
        if y_prob is not None:
            y_prob = np.array(y_prob)
            
        y_pred = np.array(y_pred)

        # 6. Calculate throughput and latency metrics
        # throughput = samples / second
        throughput = num_samples / prediction_duration if prediction_duration > 0 else 0.0
        # latency = total prediction time in milliseconds / number of samples
        latency_ms = (prediction_duration * 1000.0) / num_samples if num_samples > 0 else 0.0

        logger.info(
            f"Prediction completed. Duration: {prediction_duration:.4f}s, "
            f"Throughput: {throughput:.2f} samples/s, "
            f"Latency: {latency_ms:.4f} ms/sample, "
            f"Memory RSS Change: {memory_used:.2f} MB"
        )

        return {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_prob_type": y_prob_type,
            "prediction_duration_sec": round(prediction_duration, 4),
            "inference_throughput_sps": round(throughput, 2),
            "inference_latency_ms": round(latency_ms, 6),
            "memory_used_mb": round(memory_used, 2)
        }
