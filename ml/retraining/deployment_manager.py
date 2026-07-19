"""
deployment_manager.py
Handles promotion or rejection of the candidate model for Phase 12.
"""
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, Optional

from ml.retraining.retraining_config import RetrainingConfig

logger = logging.getLogger("retraining_pipeline")


class DeploymentManager:
    """
    Promotes the candidate model to production or records the rejection.

    Promotion
    ---------
    - Copies the winning candidate model artifacts to the production_models
      directory.
    - Updates the retraining registry with the new production entry.
    - Records full deployment metadata.

    Rejection
    ---------
    - Keeps the existing production model unchanged.
    - Records the rejection reason in the deployment decision artifact.
    """

    def __init__(self, config: RetrainingConfig) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def execute(
        self,
        comparison_result: Dict[str, Any],
        candidate_model_key: str,
        candidate_metrics: Dict[str, Any],
        production_metrics: Dict[str, Any],
        retraining_run_id: str,
    ) -> Dict[str, Any]:
        """
        Executes the deployment decision.

        Args:
            comparison_result:   Output from :class:`ModelComparator.compare`.
            candidate_model_key: The best candidate model key (e.g. ``"svm"``).
            candidate_metrics:   Metrics of the best candidate model.
            production_metrics:  Metrics of the current production model.
            retraining_run_id:   Unique ID for this retraining run.

        Returns:
            Deployment decision dictionary (serialisable to JSON).
        """
        promote: bool = comparison_result.get("promote", False)
        reason: str = comparison_result.get("reason", "")
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        decision: Dict[str, Any] = {
            "retraining_run_id": retraining_run_id,
            "timestamp": timestamp,
            "decision": "PROMOTED" if promote else "REJECTED",
            "reason": reason,
            "candidate_model_key": candidate_model_key,
            "candidate_metrics": candidate_metrics,
            "production_metrics": production_metrics,
            "comparison_summary": {
                "primary_metric": comparison_result.get("primary_metric"),
                "delta": comparison_result.get("delta"),
                "candidate_score": comparison_result.get("candidate_score"),
                "production_score": comparison_result.get("production_score"),
                "thresholds_passed": comparison_result.get("thresholds_passed"),
            },
        }

        if promote:
            logger.info(
                "DeploymentManager: PROMOTING candidate model '%s'.",
                candidate_model_key,
            )
            promotion_result = self._promote(
                candidate_model_key=candidate_model_key,
                retraining_run_id=retraining_run_id,
                timestamp=timestamp,
            )
            decision["promoted_model_path"] = promotion_result.get("prod_model_path")
            decision["production_backup_path"] = promotion_result.get("backup_path")
        else:
            logger.info(
                "DeploymentManager: REJECTING candidate. Reason: %s", reason
            )
            decision["promoted_model_path"] = None
            decision["production_backup_path"] = None

        # Persist decision JSON
        decision_path = self._config.get_output_file("deployment_decision")
        self._save_json(decision, decision_path)
        logger.info("Deployment decision saved to: %s", decision_path)

        return decision

    # ── Internals ─────────────────────────────────────────────────────────────────────────────────
    def _promote(
        self,
        candidate_model_key: str,
        retraining_run_id: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """Copies candidate artifacts to the production directory."""
        candidate_dir = os.path.join(
            self._config.get_candidate_models_dir(), candidate_model_key
        )
        prod_dir = self._config.get_production_models_dir()
        prod_model_path = os.path.join(prod_dir, candidate_model_key)
        backup_path: Optional[str] = None

        os.makedirs(prod_dir, exist_ok=True)

        # Backup existing production model if present
        if os.path.exists(prod_model_path):
            backup_path = prod_model_path + f"_backup_{retraining_run_id}"
            shutil.copytree(prod_model_path, backup_path)
            shutil.rmtree(prod_model_path)
            logger.info("Backed up existing production model to: %s", backup_path)

        # Copy candidate → production
        shutil.copytree(candidate_dir, prod_model_path)
        logger.info("Promoted candidate to: %s", prod_model_path)

        return {
            "prod_model_path": prod_model_path,
            "backup_path": backup_path,
        }

    @staticmethod
    def _save_json(data: Dict[str, Any], path: str) -> None:
        """Saves *data* as a JSON file at *path*, creating parent directories."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, default=str)
