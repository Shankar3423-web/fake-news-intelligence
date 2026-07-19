import os
import json
import time
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from ml.prediction.prediction_config import PredictionConfig
from ml.prediction.prediction_logger import setup_logger
from ml.prediction.input_validator import InputValidator
from ml.prediction.model_loader import ModelLoader
from ml.prediction.pipeline_executor import PipelineExecutor
from ml.prediction.confidence_calculator import ConfidenceCalculator
from ml.prediction.inference_engine import InferenceEngine
from ml.prediction.response_builder import ResponseBuilder
from ml.prediction.metadata_manager import MetadataManager
from ml.prediction.prediction_statistics import PredictionStatistics
from ml.prediction.prediction_profiler import PredictionProfiler
from ml.prediction.hash_generator import HashGenerator
from ml.prediction.version_manager import VersionManager
from ml.prediction.report_generator import ReportGenerator
from ml.prediction.prediction_validator import PredictionValidator

logger = logging.getLogger("prediction_pipeline")

def run_prediction_pipeline(
    raw_text: str,
    config_path: str = "ml/prediction/prediction_config.yaml"
) -> Dict[str, Any]:
    """
    Orchestrates the entire Phase 8 Prediction (Inference) Engine pipeline.
    """
    # 1. Load configuration and setup logger
    config = PredictionConfig(config_path)
    setup_logger(config.get_path("logs_dir"))
    
    logger.info("================================================================================")
    logger.info("STARTING PHASE 8 PREDICTION PIPELINE")
    logger.info("================================================================================")
    
    warnings = []
    
    # 2. Validate raw text input
    validator = InputValidator(min_length=config.min_text_length, max_length=config.max_text_length)
    try:
        validator.validate(raw_text)
    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        raise e
        
    # 3. Load model and feature order
    loader = ModelLoader(
        best_model_json_path=config.get_path("best_model_json"),
        models_root_dir=config.get_path("models_dir")
    )
    model, model_metadata, feature_order, best_model_info = loader.load_best_model()
    
    # Extract versions from evaluation history
    training_version = "unknown"
    evaluation_version = "unknown"
    feature_version = model_metadata.get("feature_selection_version", "unknown")
    
    eval_versions_path = "ml/evaluation/versions/evaluation_versions.json"
    if os.path.exists(eval_versions_path):
        try:
            with open(eval_versions_path, "r", encoding="utf-8") as f:
                eval_vers = json.load(f)
                if isinstance(eval_vers, list) and len(eval_vers) > 0:
                    last_entry = eval_vers[-1]
                    training_version = last_entry.get("training_version", "unknown")
                    evaluation_version = last_entry.get("evaluation_version", "unknown")
                    if feature_version == "unknown":
                        feature_version = last_entry.get("feature_selection_version", "unknown")
        except Exception as e:
            logger.warning(f"Could not load evaluation versions to sync: {e}")
            warnings.append(f"Could not load evaluation versions: {e}")
            
    # 4. Pipeline Execution: Preprocessing & Feature Engineering & Alignment
    executor = PipelineExecutor(
        preprocessing_config_path="config/preprocessing_config.yaml",
        feature_config_path="config/feature_config.yaml",
        tfidf_vectorizer_path=config.get_path("tfidf_vectorizer")
    )
    feature_vector, cleaned_text = executor.execute(raw_text, feature_order)
    
    # 5. Inference Engine Execution
    calculator = ConfidenceCalculator()
    engine = InferenceEngine(calculator)
    inference_metrics = engine.predict(model, feature_vector)
    
    # 6. Build response
    builder = ResponseBuilder()
    response = builder.build_response(
        prediction_label=inference_metrics["prediction"],
        confidence=inference_metrics["confidence"],
        inference_metrics=inference_metrics,
        model_metadata=model_metadata,
        best_model_info=best_model_info,
        evaluation_version=evaluation_version
    )
    
    # Calculate Input Hash
    input_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    
    # 7. Export / Save prediction history
    prediction_version = "unknown"
    history_csv = config.get_path("prediction_history_csv")
    if config.enable_prediction_export:
        logger.info(f"Saving prediction details to history CSV at {history_csv}...")
        
        # Determine next prediction version before saving so we can log it in the CSV
        ver_manager = VersionManager(versions_file_path=config.get_path("prediction_versions_json"))
        history_list = ver_manager.load_versions()
        prediction_version = f"prediction_v{len(history_list) + 1}"
        
        row_data = {
            "Timestamp": response["timestamp"],
            "Input Hash": input_hash,
            "Prediction": response["prediction"],
            "Confidence": response["confidence"],
            "Model": response["model_name"],
            "Prediction Time": inference_metrics["duration_sec"],
            "Latency": response["prediction_time_ms"],
            "Version": prediction_version
        }
        
        df_row = pd.DataFrame([row_data])
        header = not os.path.exists(history_csv)
        df_row.to_csv(history_csv, mode="a", index=False, header=header)
        logger.info("Prediction details successfully written.")

    # 8. Save Metadata
    if config.enable_metadata:
        meta_mgr = MetadataManager(metadata_dir=config.get_path("metadata_dir"))
        meta_mgr.save_metadata(
            file_path=config.get_path("prediction_metadata_json"),
            model_name=response["model_name"],
            training_version=training_version,
            evaluation_version=evaluation_version,
            feature_version=feature_version,
            prediction_version=prediction_version
        )

    # 9. Save Statistics
    stats_data = {}
    if config.enable_statistics:
        stats_mgr = PredictionStatistics(stats_file_path=config.get_path("prediction_statistics_json"))
        stats_data = stats_mgr.update_statistics(
            prediction_time_ms=response["prediction_time_ms"],
            memory_usage_mb=response["memory_usage"],
            throughput_sps=response["throughput"],
            confidence=response["confidence"],
            model_used=response["model_name"]
        )

    # 10. Generate Charts
    if config.enable_charts:
        profiler = PredictionProfiler(charts_dir=config.get_path("charts_dir"))
        profiler.generate_charts(history_csv)

    # 11. Generate Versions and Hashes (Hashes must come before version registry so registry gets hashes!)
    hashes_dict = {}
    if config.enable_hashing:
        hash_gen = HashGenerator(hash_file_path=config.get_path("prediction_hashes_json"))
        
        # Files to hash
        files_to_hash = {
            "prediction_history_csv": config.get_path("prediction_history_csv"),
            "prediction_statistics_json": config.get_path("prediction_statistics_json"),
            "prediction_report_md": config.get_path("prediction_report_md"),
            "prediction_metadata_json": config.get_path("prediction_metadata_json"),
            "prediction_versions_json": config.get_path("prediction_versions_json")
        }
        hashes_dict = hash_gen.generate_hashes(files_to_hash)

    # Register Version in Database
    if config.enable_versions:
        ver_manager = VersionManager(versions_file_path=config.get_path("prediction_versions_json"))
        prediction_version = ver_manager.register_version(
            training_version=training_version,
            evaluation_version=evaluation_version,
            model_used=response["model_name"],
            hashes=hashes_dict
        )
        # Update metadata again with finalized hashes in versions
        if config.enable_metadata:
            meta_mgr = MetadataManager(metadata_dir=config.get_path("metadata_dir"))
            meta_mgr.save_metadata(
                file_path=config.get_path("prediction_metadata_json"),
                model_name=response["model_name"],
                training_version=training_version,
                evaluation_version=evaluation_version,
                feature_version=feature_version,
                prediction_version=prediction_version
            )

    # 12. Save Markdown Report
    if config.enable_reports:
        rep_gen = ReportGenerator(report_path=config.get_path("prediction_report_md"))
        rep_gen.generate_report(
            latest_response=response,
            stats=stats_data,
            hashes=hashes_dict,
            warnings=warnings
        )
        # Re-compute hashes to include the report update
        if config.enable_hashing:
            hash_gen = HashGenerator(hash_file_path=config.get_path("prediction_hashes_json"))
            hashes_dict = hash_gen.generate_hashes({
                "prediction_history_csv": config.get_path("prediction_history_csv"),
                "prediction_statistics_json": config.get_path("prediction_statistics_json"),
                "prediction_report_md": config.get_path("prediction_report_md"),
                "prediction_metadata_json": config.get_path("prediction_metadata_json"),
                "prediction_versions_json": config.get_path("prediction_versions_json")
            })

    # 13. Pipeline outputs validation
    logger.info("Executing output validator checks...")
    output_validator = PredictionValidator()
    
    resp_valid, resp_errors = output_validator.validate_prediction_output(response)
    if not resp_valid:
        logger.error(f"Response validation failed: {resp_errors}")
        raise ValueError(f"Inference response does not comply with standards: {resp_errors}")
        
    export_valid, export_errors = output_validator.validate_exports(config)
    if not export_valid:
        logger.warning(f"Export outputs validation failed: {export_errors}")
        warnings.extend(export_errors)
        
    logger.info("================================================================================")
    logger.info("PHASE 8 PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("================================================================================")
    
    return response
