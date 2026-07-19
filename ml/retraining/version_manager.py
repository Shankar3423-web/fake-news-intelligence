"""
version_manager.py
Manages version history for Phase 12 retraining runs.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List

logger = logging.getLogger("retraining_pipeline")


class RetrainingVersionManager:
    """
    Records version history for each retraining run.

    Produced artifact: ``retraining_versions.json``
    Each entry captures the run ID, timestamp, decision, and file hashes.
    """

    def __init__(self, versions_dir: str) -> None:
        self._versions_dir = versions_dir
        self._versions_file = os.path.join(
            versions_dir, "retraining_versions.json"
        )

    # ── Loading ───────────────────────────────────────────────────────────────────────────────────
    def load(self) -> List[Dict[str, Any]]:
        """Loads existing version history (empty list if none)."""
        if not os.path.exists(self._versions_file):
            return []
        try:
            with open(self._versions_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load retraining versions: %s", exc)
            return []

    # ── Saving ────────────────────────────────────────────────────────────────────────────────────
    def _save(self, versions: List[Dict[str, Any]]) -> None:
        os.makedirs(self._versions_dir, exist_ok=True)
        with open(self._versions_file, "w", encoding="utf-8") as fh:
            json.dump(versions, fh, indent=4, default=str)

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def register(
        self,
        run_id: str,
        decision: str,
        promoted: bool,
        file_hashes: Dict[str, str],
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Appends a version entry for the current retraining run.

        Args:
            run_id:      Unique run identifier.
            decision:    ``"PROMOTED"`` or ``"REJECTED"``.
            promoted:    Whether the candidate was promoted.
            file_hashes: ``label → sha256_hash`` mapping from HashGenerator.
            extra:       Optional additional metadata.

        Returns:
            The newly created version entry.
        """
        versions = self.load()
        version_number = len(versions) + 1

        entry: Dict[str, Any] = {
            "version": version_number,
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "decision": decision,
            "promoted": promoted,
            "file_hashes": file_hashes,
        }
        if extra:
            entry["extra"] = extra

        versions.append(entry)
        self._save(versions)
        logger.info(
            "Retraining version %d registered: run_id=%s, decision=%s.",
            version_number,
            run_id,
            decision,
        )
        return entry

    @property
    def versions_file(self) -> str:
        """Absolute path to the versions JSON file."""
        return self._versions_file
