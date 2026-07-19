"""
metadata_manager.py
Generates and persists retraining run metadata for Phase 12.
"""
import json
import logging
import os
import platform
import sys
import time
from typing import Any, Dict

logger = logging.getLogger("retraining_pipeline")


class RetrainingMetadataManager:
    """
    Captures and persists environment and run metadata for Phase 12.

    Produced artifact: ``retraining_metadata.json``
    """

    def __init__(self, metadata_dir: str) -> None:
        self._metadata_dir = metadata_dir

    def generate_and_save(
        self,
        *,
        run_id: str,
        config_path: str,
        approved_csv_path: str,
        approved_records: int,
        merged_rows: int,
        feature_count: int,
        candidate_model_key: str,
        decision: str,
        pipeline_duration_sec: float,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Generates metadata and saves it to ``retraining_metadata.json``.

        Args:
            run_id:               Unique retraining run identifier.
            config_path:          Path to the retraining config YAML.
            approved_csv_path:    Path to the approved feedback CSV.
            approved_records:     Number of approved records merged.
            merged_rows:          Post-merge dataset row count.
            feature_count:        Feature dimension count.
            candidate_model_key:  Best candidate model key.
            decision:             ``"PROMOTED"`` or ``"REJECTED"``.
            pipeline_duration_sec: Wall-clock duration.
            extra:                Optional additional key-value pairs.

        Returns:
            The metadata dictionary (also saved to disk).
        """
        lib_versions = self._get_library_versions()

        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": 12,
            "description": "Automatic Model Retraining System",
            "configuration": {
                "config_path": config_path,
                "approved_feedback_csv": approved_csv_path,
            },
            "run_summary": {
                "approved_records_merged": approved_records,
                "merged_dataset_rows": merged_rows,
                "feature_count": feature_count,
                "best_candidate_model": candidate_model_key,
                "decision": decision,
                "pipeline_duration_sec": round(pipeline_duration_sec, 4),
            },
            "environment": {
                "python_version": platform.python_version(),
                "os": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "library_versions": lib_versions,
        }

        if extra:
            metadata["extra"] = extra

        os.makedirs(self._metadata_dir, exist_ok=True)
        output_path = os.path.join(self._metadata_dir, "retraining_metadata.json")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=4, default=str)

        logger.info("Retraining metadata saved to: %s", output_path)
        return metadata

    # ── Helpers ───────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_library_versions() -> Dict[str, str]:
        """Captures versions of key ML libraries."""
        versions: Dict[str, str] = {
            "python": platform.python_version(),
        }
        _libs = [
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("sklearn", "sklearn"),
            ("xgboost", "xgboost"),
            ("joblib", "joblib"),
            ("scipy", "scipy"),
        ]
        for display_name, module_name in _libs:
            try:
                import importlib

                mod = importlib.import_module(module_name)
                versions[display_name] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[display_name] = "not installed"
        return versions
