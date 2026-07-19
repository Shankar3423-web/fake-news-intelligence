import os
import logging
from typing import Dict, Any, List, Optional

from ml.admin_review.admin_review_config import AdminReviewConfig
from ml.admin_review.admin_review_logger import setup_logger
from ml.admin_review.feedback_loader import FeedbackLoader
from ml.admin_review.review_validator import ReviewValidator
from ml.admin_review.history_manager import HistoryManager
from ml.admin_review.approved_dataset_manager import ApprovedDatasetManager
from ml.admin_review.review_manager import ReviewManager
from ml.admin_review.approval_manager import ApprovalManager
from ml.admin_review.statistics_manager import StatisticsManager
from ml.admin_review.metadata_manager import MetadataManager
from ml.admin_review.report_generator import ReportGenerator
from ml.admin_review.visualization import AdminReviewVisualizer
from ml.admin_review.hash_generator import HashGenerator
from ml.admin_review.version_manager import VersionManager

def run_admin_review_pipeline(
    feedback_id: str,
    review_status: str,
    reviewer: str,
    notes: str,
    timestamp: Optional[str] = None,
    config_path: str = "ml/admin_review/admin_review_config.yaml"
) -> Dict[str, Any]:
    """
    Orchestrates the Phase 11 E2E Administrative Review Pipeline.
    Loads pending feedback, records review decisions, updates approved dataset,
    and updates stats, metadata, reports, hashes, versions, and charts.
    """
    config = AdminReviewConfig(config_path)
    logger = setup_logger(config.get_path("logs_dir"))
    
    logger.info("Admin review pipeline started.")
    warnings: List[str] = []

    # Initialize basic output structure
    response = {
        "feedback_id": feedback_id,
        "status": "FAILED",
        "review_record": None,
        "warnings": warnings,
        "error": None,
        "version": "N/A",
        "file_paths": {},
        "hashes": {},
        "retraining_eligible": False
    }

    # Gather configured paths
    feedback_csv = config.get_path("feedback_history_csv")
    history_csv = config.get_path("admin_review_history_csv")
    approved_csv = config.get_path("approved_feedback_csv")
    stats_json = config.get_path("admin_review_statistics_json")
    metadata_json = config.get_path("admin_review_metadata_json")
    report_md = config.get_path("admin_review_report_md")
    hash_json = config.get_path("admin_review_hashes_json")
    versions_json = config.get_path("admin_review_versions_json")

    # 1. Initialize component managers
    loader = FeedbackLoader(feedback_history_path=feedback_csv)
    validator = ReviewValidator(allowed_states=config.allowed_review_states)
    history_mgr = HistoryManager(history_csv_path=history_csv)
    approved_mgr = ApprovedDatasetManager(approved_csv_path=approved_csv)
    review_mgr = ReviewManager(
        loader=loader,
        validator=validator,
        history_mgr=history_mgr,
        approved_mgr=approved_mgr
    )
    approval_mgr = ApprovalManager(min_retraining_records=5)
    stats_mgr = StatisticsManager(stats_file_path=stats_json)
    metadata_mgr = MetadataManager(metadata_dir=os.path.dirname(metadata_json))
    visualizer = AdminReviewVisualizer(charts_dir=config.get_path("charts_dir"))
    hash_gen = HashGenerator(hash_file_path=hash_json)
    version_mgr = VersionManager(versions_file_path=versions_json)
    report_gen = ReportGenerator(report_file_path=report_md, history_csv_path=history_csv)

    # 2. Submit the review decision
    success, error_msg, review_record = review_mgr.submit_review(
        feedback_id=feedback_id,
        review_status=review_status,
        reviewer=reviewer,
        notes=notes,
        timestamp=timestamp
    )

    if not success:
        logger.error(f"Review submission failed: {error_msg}")
        response["error"] = error_msg
        return response

    response["review_record"] = review_record

    try:
        # 3. Calculate and Update Statistics
        review_history = history_mgr.load_history()
        if config.enable_statistics:
            stats = stats_mgr.calculate_and_save(review_history)
            logger.info("Admin review statistics updated.")
        else:
            stats = stats_mgr.load_statistics()
            warnings.append("Statistics updates are disabled in configuration.")

        # 4. Save Environment Metadata
        if config.enable_metadata:
            metadata = metadata_mgr.save_metadata(
                file_path=metadata_json,
                pipeline_version="1.0.0",
                system_version=config.system_version,
                configuration_version="1.0.0"
            )
            logger.info("Admin review metadata saved.")
        else:
            metadata = {}
            warnings.append("Metadata saving is disabled in configuration.")

        # 5. Check if retraining is eligible
        approved_records = approved_mgr.load_approved_dataset()
        retraining_eligible = approval_mgr.is_eligible_for_retraining(approved_records)
        response["retraining_eligible"] = retraining_eligible

        # 6. Generate Distribution Charts
        if config.enable_charts:
            visualizer.generate_charts(stats)
            logger.info("Admin review charts generated.")
        else:
            warnings.append("Chart generation is disabled in configuration.")

        # 7. Generate Initial Hashes for Version Snapshot
        files_to_hash = {
            "history": history_csv,
            "approved": approved_csv,
            "statistics": stats_json,
            "metadata": metadata_json
        }
        initial_hashes = hash_gen.generate_hashes(files_to_hash)

        # 8. Register Snapshot Version
        version_str = "N/A"
        if config.enable_versioning:
            version_str = version_mgr.register_version(
                hashes=initial_hashes,
                system_version=config.system_version
            )
            logger.info(f"Registered version: {version_str}")
        else:
            warnings.append("Versioning is disabled in configuration.")
        response["version"] = version_str

        # 9. Generate Markdown Audit Report
        if config.enable_reports:
            report_gen.generate_report(
                record=review_record,
                stats=stats,
                hashes=initial_hashes,
                warnings=warnings
            )
            logger.info("Admin review audit report generated.")
        else:
            warnings.append("Report generation is disabled in configuration.")

        # 10. Recompute Final Hashes Registry
        final_files_to_hash = {
            "history": history_csv,
            "approved": approved_csv,
            "statistics": stats_json,
            "metadata": metadata_json,
            "report": report_md,
            "versions": versions_json
        }
        
        final_hashes = {}
        if config.enable_hash_generation:
            final_hashes = hash_gen.generate_hashes(final_files_to_hash)
            logger.info("Final SHA-256 hashes generated.")
        else:
            warnings.append("Hash generation is disabled in configuration.")

        response["hashes"] = final_hashes
        response["status"] = "SUCCESS"
        response["file_paths"] = {
            "history_csv": history_csv,
            "approved_csv": approved_csv,
            "statistics_json": stats_json,
            "metadata_json": metadata_json,
            "report_md": report_md,
            "hashes_json": hash_json,
            "versions_json": versions_json
        }

        logger.info("Admin review pipeline completed successfully.")
        return response

    except Exception as e:
        logger.exception(f"Unexpected error during admin review pipeline execution: {e}")
        response["error"] = str(e)
        return response
