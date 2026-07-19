import os
import logging
from typing import Dict, Any, Optional, List

from ml.feedback.feedback_config import FeedbackConfig
from ml.feedback.feedback_logger import setup_logger
from ml.feedback.feedback_validator import FeedbackValidator
from ml.feedback.feedback_sanitizer import FeedbackSanitizer
from ml.feedback.feedback_manager import FeedbackManager
from ml.feedback.feedback_history import HistoryManager
from ml.feedback.feedback_statistics import StatisticsManager
from ml.feedback.feedback_metadata import MetadataManager
from ml.feedback.feedback_report import ReportGenerator
from ml.feedback.feedback_visualization import FeedbackVisualizer
from ml.feedback.feedback_hashes import HashGenerator
from ml.feedback.feedback_versions import VersionManager
from ml.feedback.response_builder import ResponseBuilder

def run_feedback_pipeline(
    prediction: Any,
    prediction_confidence: float,
    verification_status: str,
    evidence_score: float,
    similarity_score: float,
    final_decision: str,
    user_feedback: str,
    comment: Optional[str] = None,
    timestamp: Optional[str] = None,
    config_path: str = "ml/feedback/feedback_config.yaml"
) -> Dict[str, Any]:
    """
    Orchestrates the Phase 10 E2E Feedback Collection Pipeline.
    """
    config = FeedbackConfig(config_path)
    logger = setup_logger(config.get_path("logs_dir"))
    
    logger.info("Feedback collection pipeline started.")
    
    builder = ResponseBuilder()
    warnings: List[str] = []

    # 1. Input Validation
    logger.info("Starting input validation.")
    validator = FeedbackValidator(
        allowed_values=config.allowed_feedback_values,
        min_comment_len=config.min_comment_length,
        max_comment_len=config.max_comment_length,
        allow_empty_comments=config.allow_empty_comments
    )
    
    inputs = {
        "prediction": prediction,
        "prediction_confidence": prediction_confidence,
        "verification_status": verification_status,
        "evidence_score": evidence_score,
        "similarity_score": similarity_score,
        "final_decision": final_decision,
        "user_feedback": user_feedback,
        "comment": comment,
        "timestamp": timestamp
    }
    
    is_valid, validation_errors = validator.validate_inputs(inputs)
    if not is_valid:
        err_msg = f"Input validation failed: {'; '.join(validation_errors)}"
        logger.error(err_msg)
        return builder.build_response(
            status="FAILED",
            error=err_msg,
            warnings=warnings
        )
    logger.info("Input validation completed successfully.")

    # 2. Input Sanitization
    logger.info("Starting input sanitization.")
    sanitizer = FeedbackSanitizer()
    sanitized_comment = sanitizer.sanitize_comment(comment)
    logger.info("Input sanitization completed.")

    try:
        # 3. Create Feedback Record
        manager = FeedbackManager(system_version=config.system_version)
        record = manager.create_record(
            prediction=prediction,
            prediction_confidence=prediction_confidence,
            verification_status=verification_status,
            evidence_score=evidence_score,
            similarity_score=similarity_score,
            final_decision=final_decision,
            user_feedback=user_feedback,
            comment=sanitized_comment,
            timestamp=timestamp
        )
        logger.info(f"Feedback record generated: {record['feedback_id']}")

        # 4. Save to History CSV
        history_csv = config.get_path("feedback_history_csv")
        history_mgr = HistoryManager(history_csv_path=history_csv)
        history_mgr.save_history(
            feedback_id=record["feedback_id"],
            timestamp=record["timestamp"],
            prediction=record["prediction"],
            verification=record["verification_status"],
            decision=record["final_decision"],
            feedback=record["feedback_value"],
            comment=record["comment"]
        )
        logger.info("Feedback record saved to history.")

        # 5. Update Statistics
        stats_file = config.get_path("feedback_statistics_json")
        stats_mgr = StatisticsManager(stats_file_path=stats_file)
        if config.enable_statistics:
            stats = stats_mgr.update_statistics(
                feedback_value=record["feedback_value"],
                prediction_confidence=record["prediction_confidence"],
                similarity_score=record["similarity_score"],
                evidence_score=record["evidence_score"]
            )
            logger.info("Feedback statistics updated.")
        else:
            stats = stats_mgr.load_statistics()
            warnings.append("Statistics updates are disabled in configuration.")

        # 6. Save Metadata
        metadata_file = config.get_path("feedback_metadata_json")
        metadata_mgr = MetadataManager(metadata_dir=os.path.dirname(metadata_file))
        if config.enable_metadata:
            metadata = metadata_mgr.save_metadata(
                file_path=metadata_file,
                pipeline_version="1.0.0",
                feedback_version=config.system_version,
                configuration_version="1.0.0"
            )
            logger.info("Feedback metadata saved.")
        else:
            metadata = {}
            warnings.append("Metadata saving is disabled in configuration.")

        # 7. Generate Visualization Charts
        charts_dir = config.get_path("charts_dir")
        visualizer = FeedbackVisualizer(charts_dir=charts_dir, history_csv_path=history_csv)
        if config.enable_charts:
            visualizer.generate_charts(stats)
            logger.info("Charts generated.")
        else:
            warnings.append("Chart generation is disabled in configuration.")

        # 8. Hash Calculation and Version Registry
        hash_file = config.get_path("feedback_hashes_json")
        hash_generator = HashGenerator(hash_file_path=hash_file)
        
        versions_file = config.get_path("feedback_versions_json")
        version_mgr = VersionManager(versions_file_path=versions_file)
        
        report_file = config.get_path("feedback_report_md")
        report_generator = ReportGenerator(report_file_path=report_file, history_csv_path=history_csv)
        
        # Files to hash first (excluding versions and report initially)
        files_to_hash = {
            "history": history_csv,
            "metadata": metadata_file,
            "statistics": stats_file
        }
        
        initial_hashes = hash_generator.generate_hashes(files_to_hash)
        
        # Register execution version
        version_str = "N/A"
        if config.enable_versioning:
            version_str = version_mgr.register_version(hashes=initial_hashes)
            logger.info(f"Feedback version registered: {version_str}")
        else:
            warnings.append("Versioning is disabled in configuration.")
            
        # 9. Generate Report (uses hashes in layout)
        if config.enable_reports:
            report_generator.generate_report(
                record=record,
                stats=stats,
                hashes=initial_hashes,
                warnings=warnings
            )
            logger.info("Feedback report generated.")
        else:
            warnings.append("Report generation is disabled in configuration.")
            
        # Recompute final hashes registry to include report and version files
        final_files_to_hash = {
            "history": history_csv,
            "metadata": metadata_file,
            "statistics": stats_file,
            "report": report_file,
            "versions": versions_file
        }
        
        final_hashes = {}
        if config.enable_hash_generation:
            final_hashes = hash_generator.generate_hashes(final_files_to_hash)
            logger.info("SHA-256 hashes generated.")
        else:
            warnings.append("Hash generation is disabled in configuration.")

        # File paths mapping
        file_paths = {
            "history_csv": history_csv,
            "statistics_json": stats_file,
            "metadata_json": metadata_file,
            "report_md": report_file,
            "hashes_json": hash_file,
            "versions_json": versions_file
        }

        logger.info("Feedback pipeline completed successfully.")
        return builder.build_response(
            status="SUCCESS",
            record=record,
            warnings=warnings,
            version=version_str,
            file_paths=file_paths,
            hashes=final_hashes
        )
    except Exception as e:
        logger.exception(f"Unexpected error in feedback pipeline: {e}")
        return builder.build_response(
            status="FAILED",
            error=str(e),
            warnings=warnings
        )
