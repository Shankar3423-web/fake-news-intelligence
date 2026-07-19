import time
import logging
import pandas as pd
from typing import Any, Dict
from ml.prediction.confidence_calculator import ConfidenceCalculator
from ml.prediction.prediction_utils import get_memory_usage

logger = logging.getLogger("prediction_pipeline")

class InferenceEngine:
    """
    Feeds the feature vector into the classifier, captures the prediction label,
    invokes the confidence calculator, and measures performance metrics (latency, memory, throughput).
    """
    def __init__(self, confidence_calculator: ConfidenceCalculator) -> None:
        self.confidence_calculator = confidence_calculator

    def predict(self, model: Any, feature_vector: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs inference on the provided feature vector.
        
        Args:
            model: Ready-to-use scikit-learn or xgboost classifier
            feature_vector: Aligned feature DataFrame (single sample)
            
        Returns:
            Dict containing prediction label, confidence, and system metrics.
        """
        logger.info("Executing model inference...")
        
        # 1. Track resource usage before prediction
        mem_before, _ = get_memory_usage()
        start_time = time.perf_counter()
        
        # 2. Run prediction
        try:
            # Predict class
            prediction_array = model.predict(feature_vector)
            prediction_label = int(prediction_array[0])
        except Exception as e:
            logger.critical(f"Prediction failed at model level: {e}")
            raise RuntimeError(f"Prediction failed at model level: {e}")
            
        # 3. Calculate confidence
        confidence = self.confidence_calculator.calculate(model, feature_vector, prediction_label)
        
        # 4. Track resource usage after prediction
        end_time = time.perf_counter()
        mem_after, _ = get_memory_usage()
        
        # 5. Compute metrics
        duration_sec = end_time - start_time
        latency_ms = duration_sec * 1000.0
        
        # Avoid division by zero
        throughput_sps = 1.0 / duration_sec if duration_sec > 0 else 0.0
        memory_used_mb = max(0.0, mem_after - mem_before)
        
        logger.info(
            f"Inference complete | Prediction: {prediction_label} | "
            f"Confidence: {confidence:.4f} | Latency: {latency_ms:.2f}ms | "
            f"Throughput: {throughput_sps:.2f} sps | Memory delta: {memory_used_mb:.2f} MB"
        )
        
        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "latency_ms": float(round(latency_ms, 4)),
            "throughput_sps": float(round(throughput_sps, 2)),
            "memory_used_mb": float(round(memory_used_mb, 4)),
            "duration_sec": float(round(duration_sec, 6))
        }
