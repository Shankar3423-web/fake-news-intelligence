"""
model_registry.py
Retraining-specific model registry for Phase 12.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("retraining_pipeline")


class RetrainingModelRegistry:
    """
    Maintains a JSON registry of all retraining runs, candidate models,
    and production promotion history.

    Schema
    ------
    .. code-block:: json

        {
            "current_production": { ... },
            "retraining_runs": [ { ... }, ... ]
        }
    """

    def __init__(self, registry_path: str) -> None:
        self._registry_path = registry_path

    # ── Loading / Saving ──────────────────────────────────────────────────────────────────────────
    def load(self) -> Dict[str, Any]:
        """Loads the registry JSON, returning an empty structure on first use."""
        if not os.path.exists(self._registry_path):
            return {"current_production": {}, "retraining_runs": []}
        try:
            with open(self._registry_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning("Registry is malformed — re-initialising.")
                return {"current_production": {}, "retraining_runs": []}
            return data
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load retraining registry: %s", exc)
            return {"current_production": {}, "retraining_runs": []}

    def save(self, data: Dict[str, Any]) -> None:
        """Persists the registry dictionary to disk."""
        parent = os.path.dirname(self._registry_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._registry_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, default=str)
        logger.info("Retraining registry saved to: %s", self._registry_path)

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def register_run(
        self,
        run_id: str,
        decision: str,
        candidate_model_key: str,
        candidate_metrics: Dict[str, Any],
        production_metrics: Dict[str, Any],
        approved_records: int,
        merged_rows: int,
        promoted: bool,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Appends a retraining run entry to the registry.

        Args:
            run_id:               Unique run identifier.
            decision:             ``"PROMOTED"`` or ``"REJECTED"``.
            candidate_model_key:  Best candidate model key.
            candidate_metrics:    Evaluation metrics of the candidate.
            production_metrics:   Metrics of the production model at time of comparison.
            approved_records:     Number of approved feedback records used.
            merged_rows:          Total merged dataset rows.
            promoted:             Whether the candidate was promoted.
            reason:               Human-readable decision reason.

        Returns:
            The newly created run entry.
        """
        data = self.load()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        entry: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "decision": decision,
            "promoted": promoted,
            "reason": reason,
            "candidate_model_key": candidate_model_key,
            "approved_records_used": approved_records,
            "merged_dataset_rows": merged_rows,
            "candidate_metrics": candidate_metrics,
            "production_metrics_at_comparison": production_metrics,
        }

        data.setdefault("retraining_runs", []).append(entry)

        if promoted:
            data["current_production"] = {
                "model_key": candidate_model_key,
                "promoted_at": timestamp,
                "run_id": run_id,
                "metrics": candidate_metrics,
            }

        self.save(data)
        logger.info(
            "Registered retraining run '%s': decision=%s.", run_id, decision
        )
        return entry

    def get_production_metrics(self) -> Optional[Dict[str, Any]]:
        """Returns the metrics of the current production model, or ``None`` if none."""
        data = self.load()
        prod = data.get("current_production", {})
        return prod.get("metrics") if prod else None

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Returns all retraining run entries."""
        return self.load().get("retraining_runs", [])
