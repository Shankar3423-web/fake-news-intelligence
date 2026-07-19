import time
from datetime import datetime
from typing import Dict, Any

class ResponseBuilder:
    """
    Standardizes the prediction response format.
    """
    def __init__(self) -> None:
        self.label_mapping = {
            0: "REAL",
            1: "FAKE"
        }

    def build_response(
        self,
        prediction_label: int,
        confidence: float,
        inference_metrics: Dict[str, Any],
        model_metadata: Dict[str, Any],
        best_model_info: Dict[str, Any],
        evaluation_version: str
    ) -> Dict[str, Any]:
        """
        Builds a structured dictionary matching the required response schema.
        """
        timestamp = datetime.now().isoformat()
        
        label_str = self.label_mapping.get(prediction_label, "UNKNOWN")
        model_name = best_model_info.get("model_key", "unknown")
        model_version = model_metadata.get("dataset_version", "1.0.0")
        
        response = {
            "prediction": prediction_label,
            "label": label_str,
            "confidence": float(round(confidence, 4)),
            "model_name": model_name,
            "model_version": model_version,
            "evaluation_version": evaluation_version,
            "prediction_time_ms": inference_metrics.get("latency_ms", 0.0),
            "throughput": inference_metrics.get("throughput_sps", 0.0),
            "memory_usage": inference_metrics.get("memory_used_mb", 0.0),
            "timestamp": timestamp
        }
        
        return response
