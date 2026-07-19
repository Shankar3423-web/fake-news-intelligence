import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("prediction_pipeline")

class PredictionValidator:
    """
    Validation checklist to verify the integrity of the prediction inputs,
    model configuration, feature vector shapes, and all exported output files.
    """
    def __init__(self) -> None:
        pass

    def validate_prediction_output(self, response: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates prediction response dictionary structure and bounds."""
        errors = []
        
        required_keys = [
            "prediction", "label", "confidence", "model_name", 
            "model_version", "evaluation_version", "prediction_time_ms", 
            "throughput", "memory_usage", "timestamp"
        ]
        
        for k in required_keys:
            if k not in response:
                errors.append(f"Response missing required key: {k}")
                
        if not errors:
            pred = response["prediction"]
            if pred not in [0, 1]:
                errors.append(f"Invalid prediction class: {pred}. Expected 0 or 1.")
                
            label = response["label"]
            if label not in ["REAL", "FAKE"]:
                errors.append(f"Invalid prediction label: {label}. Expected 'REAL' or 'FAKE'.")
                
            conf = response["confidence"]
            if not (0.0 <= conf <= 1.0):
                errors.append(f"Confidence score {conf} out of bounds [0.0, 1.0].")
                
            latency = response["prediction_time_ms"]
            if latency < 0.0:
                errors.append(f"Negative prediction latency: {latency} ms.")

        return len(errors) == 0, errors

    def validate_exports(self, config: Any) -> Tuple[bool, List[str]]:
        """
        Validates that all configured exports exist, are non-empty, and contain
        valid structures.
        """
        errors = []
        
        # 1. Prediction History Export
        if config.enable_prediction_export:
            history_path = config.get_path("prediction_history_csv")
            if not os.path.exists(history_path):
                errors.append(f"Prediction history file does not exist: {history_path}")
            elif os.path.getsize(history_path) == 0:
                errors.append(f"Prediction history file is empty: {history_path}")
            else:
                try:
                    df = pd.read_csv(history_path)
                    expected_cols = ["Timestamp", "Input Hash", "Prediction", "Confidence", "Model", "Prediction Time", "Latency", "Version"]
                    for col in expected_cols:
                        if col not in df.columns:
                            errors.append(f"Prediction history missing expected column: {col}")
                except Exception as e:
                    errors.append(f"Failed to read prediction history CSV: {e}")

        # 2. Metadata Export
        if config.enable_metadata:
            metadata_path = config.get_path("prediction_metadata_json")
            if not os.path.exists(metadata_path):
                errors.append(f"Prediction metadata file does not exist: {metadata_path}")
            else:
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    expected = ["model_used", "training_version", "evaluation_version", "prediction_version", "prediction_timestamp"]
                    for key in expected:
                        if key not in data:
                            errors.append(f"Metadata file missing key: {key}")
                except Exception as e:
                    errors.append(f"Failed to parse metadata JSON: {e}")

        # 3. Statistics Export
        if config.enable_statistics:
            stats_path = config.get_path("prediction_statistics_json")
            if not os.path.exists(stats_path):
                errors.append(f"Statistics file does not exist: {stats_path}")
            else:
                try:
                    with open(stats_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    expected = ["total_predictions", "average_prediction_time_ms", "average_memory_usage_mb", "average_throughput_sps", "average_confidence", "model_used"]
                    for key in expected:
                        if key not in data:
                            errors.append(f"Statistics file missing key: {key}")
                except Exception as e:
                    errors.append(f"Failed to parse statistics JSON: {e}")

        # 4. Reports Export
        if config.enable_reports:
            report_path = config.get_path("prediction_report_md")
            if not os.path.exists(report_path):
                errors.append(f"Report file does not exist: {report_path}")
            elif os.path.getsize(report_path) == 0:
                errors.append(f"Report file is empty: {report_path}")

        # 5. Versions Export
        if config.enable_versions:
            versions_path = config.get_path("prediction_versions_json")
            if not os.path.exists(versions_path):
                errors.append(f"Versions database file does not exist: {versions_path}")
            else:
                try:
                    with open(versions_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        errors.append("Versions database must be a JSON list.")
                    elif len(data) > 0:
                        entry = data[-1]
                        expected = ["prediction_version", "training_version", "evaluation_version", "timestamp", "model_used", "hashes"]
                        for key in expected:
                            if key not in entry:
                                errors.append(f"Versions entry missing key: {key}")
                except Exception as e:
                    errors.append(f"Failed to parse versions JSON: {e}")

        # 6. Hashes Export
        if config.enable_hashing:
            hashes_path = config.get_path("prediction_hashes_json")
            if not os.path.exists(hashes_path):
                errors.append(f"Hashes registry does not exist: {hashes_path}")
            else:
                try:
                    with open(hashes_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        errors.append("Hashes registry must be a JSON dictionary.")
                except Exception as e:
                    errors.append(f"Failed to parse hashes JSON: {e}")

        return len(errors) == 0, errors
