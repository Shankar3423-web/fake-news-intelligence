"""
model_comparator.py
Compares candidate and production model metrics using the configured acceptance policy.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from ml.retraining.retraining_config import RetrainingConfig

logger = logging.getLogger("retraining_pipeline")

# Metrics where LOWER is better (for directional comparison)
_LOWER_IS_BETTER = {"inference_time_sec", "prediction_time_sec", "memory_used_mb", "model_size_mb"}


class ModelComparator:
    """
    Compares candidate vs. production model metrics and decides whether to promote.

    Comparison Logic
    ----------------
    1. Weighted score comparison (configurable weights).
    2. Primary metric delta check (candidate must beat production by
       ``minimum_improvement_delta``).
    3. Minimum threshold enforcement for critical metrics.
    4. ``promote_on_tie`` flag for tie-breaking.

    All comparison parameters are driven by the ``acceptance_policy`` block
    in ``retraining_config.yaml`` — no hard-coded thresholds in this class.
    """

    def __init__(self, config: RetrainingConfig) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def compare(
        self,
        candidate_metrics: Dict[str, Any],
        production_metrics: Dict[str, Any],
        candidate_model_key: str = "candidate",
        production_model_key: str = "production",
    ) -> Dict[str, Any]:
        """
        Runs the full comparison and returns a detailed result dictionary.

        Args:
            candidate_metrics:   Metric dict for the best candidate model.
            production_metrics:  Metric dict for the current production model.
            candidate_model_key: Display name for the candidate model.
            production_model_key: Display name for the production model.

        Returns:
            Dictionary with keys:
            - ``promote``: bool — whether to promote the candidate.
            - ``reason``: str — human-readable promotion / rejection reason.
            - ``primary_metric``: str.
            - ``candidate_score``: float (weighted).
            - ``production_score``: float (weighted).
            - ``per_metric_comparison``: list of per-metric dicts.
            - ``minimum_threshold_results``: dict of threshold checks.
            - ``delta``: float (primary metric delta).
        """
        primary = self._config.primary_metric
        delta_required = self._config.minimum_improvement_delta
        promote_on_tie = self._config.promote_on_tie
        thresholds = self._config.minimum_thresholds
        use_weighted = self._config.weighted_comparison_enabled
        weights = self._config.weighted_comparison_weights

        # ── Per-metric comparison ─────────────────────────────────────────────────────────────────
        per_metric: List[Dict[str, Any]] = []
        all_metrics = set(candidate_metrics.keys()) | set(production_metrics.keys())

        for m in sorted(all_metrics):
            c_val = candidate_metrics.get(m)
            p_val = production_metrics.get(m)
            if c_val is None or p_val is None:
                continue
            try:
                c_val = float(c_val)
                p_val = float(p_val)
            except (TypeError, ValueError):
                continue

            lower_better = m in _LOWER_IS_BETTER
            delta = c_val - p_val
            if lower_better:
                winner = "candidate" if c_val < p_val else ("production" if p_val < c_val else "tie")
            else:
                winner = "candidate" if c_val > p_val else ("production" if p_val > c_val else "tie")

            per_metric.append(
                {
                    "metric": m,
                    "candidate": round(c_val, 6),
                    "production": round(p_val, 6),
                    "delta": round(delta, 6),
                    "lower_is_better": lower_better,
                    "winner": winner,
                }
            )

        # ── Weighted scores ───────────────────────────────────────────────────────────────────────
        candidate_score = self._weighted_score(candidate_metrics, weights)
        production_score = self._weighted_score(production_metrics, weights)

        # ── Primary metric delta ──────────────────────────────────────────────────────────────────
        c_primary = float(candidate_metrics.get(primary, 0.0))
        p_primary = float(production_metrics.get(primary, 0.0))
        primary_delta = c_primary - p_primary if primary not in _LOWER_IS_BETTER else p_primary - c_primary

        # ── Minimum thresholds ────────────────────────────────────────────────────────────────────
        threshold_results: Dict[str, Dict[str, Any]] = {}
        thresholds_passed = True
        for metric_name, min_val in thresholds.items():
            cand_val = float(candidate_metrics.get(metric_name, 0.0))
            passed = cand_val >= min_val
            if not passed:
                thresholds_passed = False
            threshold_results[metric_name] = {
                "required": min_val,
                "candidate_value": round(cand_val, 6),
                "passed": passed,
            }

        # ── Promotion decision ────────────────────────────────────────────────────────────────────
        promote, reason = self._decide(
            primary_delta=primary_delta,
            delta_required=delta_required,
            candidate_score=candidate_score,
            production_score=production_score,
            thresholds_passed=thresholds_passed,
            promote_on_tie=promote_on_tie,
            use_weighted=use_weighted,
            primary=primary,
        )

        result: Dict[str, Any] = {
            "promote": promote,
            "reason": reason,
            "primary_metric": primary,
            "candidate_score": round(candidate_score, 6),
            "production_score": round(production_score, 6),
            "candidate_model_key": candidate_model_key,
            "production_model_key": production_model_key,
            "delta": round(primary_delta, 6),
            "per_metric_comparison": per_metric,
            "minimum_threshold_results": threshold_results,
            "thresholds_passed": thresholds_passed,
        }

        logger.info(
            "Model comparison: promote=%s | reason='%s' | "
            "candidate_score=%.4f | production_score=%.4f | "
            "primary_metric=%s delta=%.4f",
            promote,
            reason,
            candidate_score,
            production_score,
            primary,
            primary_delta,
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────────────────────────────
    def _weighted_score(
        self, metrics: Dict[str, Any], weights: Dict[str, float]
    ) -> float:
        """Computes a weighted score from *metrics* using *weights*."""
        score = 0.0
        total_weight = 0.0
        for metric_name, weight in weights.items():
            val = metrics.get(metric_name)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue

            # Invert for lower-is-better metrics (so higher weighted score is still better)
            if metric_name in _LOWER_IS_BETTER:
                val = 1.0 / (1.0 + val) if val >= 0 else 0.0

            score += weight * val
            total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _decide(
        primary_delta: float,
        delta_required: float,
        candidate_score: float,
        production_score: float,
        thresholds_passed: bool,
        promote_on_tie: bool,
        use_weighted: bool,
        primary: str,
    ) -> Tuple[bool, str]:
        """Core promotion decision logic."""
        # Gate: minimum thresholds
        if not thresholds_passed:
            return (
                False,
                "Candidate failed minimum metric thresholds — production model retained.",
            )

        # Compare
        if use_weighted:
            if candidate_score > production_score + delta_required:
                return (
                    True,
                    f"Candidate promoted: weighted_score {candidate_score:.4f} > "
                    f"production {production_score:.4f} (required delta: {delta_required}).",
                )
            elif candidate_score == production_score and promote_on_tie:
                return (
                    True,
                    "Candidate promoted on tie (promote_on_tie=True).",
                )
            else:
                return (
                    False,
                    f"Candidate rejected: weighted_score {candidate_score:.4f} ≤ "
                    f"production {production_score:.4f} — production model retained.",
                )
        else:
            if primary_delta > delta_required:
                return (
                    True,
                    f"Candidate promoted: {primary} delta={primary_delta:.4f} "
                    f"> required {delta_required}.",
                )
            elif primary_delta == 0.0 and promote_on_tie:
                return (
                    True,
                    f"Candidate promoted on tie in '{primary}' (promote_on_tie=True).",
                )
            else:
                return (
                    False,
                    f"Candidate rejected: {primary} delta={primary_delta:.4f} "
                    f"≤ required {delta_required} — production model retained.",
                )
