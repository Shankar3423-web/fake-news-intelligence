"""
validator.py
Post-run artifact validator for Phase 12 retraining system.
"""
import json
import logging
import os
from typing import Any, Dict, List, Tuple

from ml.retraining.retraining_config import RetrainingConfig

logger = logging.getLogger("retraining_pipeline")


class RetrainingArtifactValidator:
    """
    Validates all Phase 12 output artifacts after the retraining pipeline runs.

    Checks
    ------
    - Required output files exist and are non-empty.
    - Folder structure is complete.
    - JSON artifacts are well-formed.
    - Deployment decision contains required fields.
    - Registry contains at least one retraining run.
    - Comparison metrics are complete.
    """

    def __init__(self, config: RetrainingConfig) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Runs all validation checks.

        Returns:
            ``(is_valid, errors)`` — ``is_valid`` is False when any error is found.
        """
        errors: List[str] = []

        errors.extend(self._check_folder_structure())
        errors.extend(self._check_required_files())
        errors.extend(self._check_json_artifacts())
        errors.extend(self._check_deployment_decision())
        errors.extend(self._check_registry())
        errors.extend(self._check_comparison_report())

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("RetrainingArtifactValidator: All checks passed.")
        else:
            for err in errors:
                logger.error("Validation error: %s", err)

        return is_valid, errors

    # ── Checks ────────────────────────────────────────────────────────────────────────────────────
    def _check_folder_structure(self) -> List[str]:
        """Verifies that required output directories exist."""
        errors: List[str] = []
        required_dirs = [
            self._config.get_output_dir("logs_dir"),
            self._config.get_output_dir("reports_dir"),
            self._config.get_output_dir("metadata_dir"),
            self._config.get_output_dir("statistics_dir"),
            self._config.get_output_dir("versions_dir"),
            self._config.get_output_dir("hashes_dir"),
            self._config.get_output_dir("charts_dir"),
            self._config.get_output_dir("registry_dir"),
            self._config.get_candidate_models_dir(),
        ]
        for d in required_dirs:
            if d and not os.path.isdir(d):
                errors.append(f"Required directory missing: {d}")
        return errors

    def _check_required_files(self) -> List[str]:
        """Verifies that required output files exist and are non-empty."""
        errors: List[str] = []
        required_files = {
            "retraining_report": self._config.get_output_file("retraining_report"),
            "comparison_report": self._config.get_output_file("comparison_report"),
            "retraining_statistics": self._config.get_output_file("retraining_statistics"),
            "retraining_metadata": self._config.get_output_file("retraining_metadata"),
            "retraining_hashes": self._config.get_output_file("retraining_hashes"),
            "retraining_versions": self._config.get_output_file("retraining_versions"),
            "deployment_decision": self._config.get_output_file("deployment_decision"),
        }
        for name, path in required_files.items():
            if not path:
                errors.append(f"Output file path not configured for: {name}")
                continue
            if not os.path.isfile(path):
                errors.append(f"Required output file missing: {path} ({name})")
            elif os.path.getsize(path) == 0:
                errors.append(f"Required output file is empty: {path} ({name})")
        return errors

    def _check_json_artifacts(self) -> List[str]:
        """Validates that JSON artifacts are well-formed."""
        errors: List[str] = []
        json_files = [
            self._config.get_output_file("retraining_statistics"),
            self._config.get_output_file("retraining_metadata"),
            self._config.get_output_file("retraining_hashes"),
            self._config.get_output_file("retraining_versions"),
            self._config.get_output_file("deployment_decision"),
            self._config.get_registry_file(),
        ]
        for path in json_files:
            if not path or not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    json.load(fh)
            except json.JSONDecodeError as exc:
                errors.append(f"Malformed JSON in {path}: {exc}")
        return errors

    def _check_deployment_decision(self) -> List[str]:
        """Checks that the deployment decision has required fields."""
        errors: List[str] = []
        path = self._config.get_output_file("deployment_decision")
        if not path or not os.path.isfile(path):
            return errors  # Already caught by _check_required_files

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return errors  # Already caught by _check_json_artifacts

        required_fields = [
            "retraining_run_id",
            "decision",
            "reason",
            "candidate_model_key",
            "comparison_summary",
        ]
        for field in required_fields:
            if field not in data:
                errors.append(
                    f"Deployment decision missing field: '{field}'."
                )

        valid_decisions = {"PROMOTED", "REJECTED"}
        decision_val = data.get("decision", "")
        if decision_val not in valid_decisions:
            errors.append(
                f"Invalid deployment decision value: '{decision_val}'. "
                f"Expected one of {valid_decisions}."
            )

        return errors

    def _check_registry(self) -> List[str]:
        """Verifies that the retraining registry contains at least one run."""
        errors: List[str] = []
        path = self._config.get_registry_file()
        if not path or not os.path.isfile(path):
            errors.append(f"Retraining registry not found: {path}")
            return errors

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return errors

        runs = data.get("retraining_runs", [])
        if not runs:
            errors.append(
                "Retraining registry has no run entries (retraining_runs is empty)."
            )
        return errors

    def _check_comparison_report(self) -> List[str]:
        """Checks that the comparison report file is non-empty."""
        errors: List[str] = []
        path = self._config.get_output_file("comparison_report")
        if not path or not os.path.isfile(path):
            return errors  # Already caught

        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()

        required_sections = [
            "# Phase 12 — Model Comparison Report",
            "## Decision",
            "## Per-Metric Comparison",
        ]
        for section in required_sections:
            if section not in content:
                errors.append(
                    f"Comparison report missing section: '{section}'."
                )
        return errors
