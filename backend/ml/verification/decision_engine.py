import logging
from typing import Dict, Any
from ml.verification.verification_config import VerificationConfig

logger = logging.getLogger("verification_pipeline")

class DecisionEngine:
    """
    DecisionEngine makes the final verification determination based on the
    prediction context from Phase 8 and the synthesized evidence from Phase 9.

    Returns one of three definitive statuses:
      - VERIFIED_REAL:  Live APIs found strong matching evidence → the news IS being reported
      - VERIFIED_FAKE:  Live APIs found no meaningful matching evidence → the news is NOT being reported
      - INCONCLUSIVE:   Not enough API data to make any determination
    """
    def __init__(self, config: VerificationConfig) -> None:
        self.config = config

    def decide(self, prediction_response: Dict[str, Any], evidence_metrics: Dict[str, Any]) -> str:
        """
        Determines the verification status: VERIFIED_REAL, VERIFIED_FAKE, or INCONCLUSIVE.
        """
        total_articles = evidence_metrics.get("total_articles", 0)
        max_similarity = evidence_metrics.get("maximum_similarity", 0.0)
        avg_similarity = evidence_metrics.get("average_similarity", 0.0)
        evidence_strength = evidence_metrics.get("evidence_strength", 0.0)
        trusted_source_count = evidence_metrics.get("trusted_source_count", 0)

        # 1. INCONCLUSIVE — no articles retrieved from any API
        if total_articles == 0:
            logger.info("Decision Engine: INCONCLUSIVE — no live news articles retrieved from any provider.")
            return "INCONCLUSIVE"

        # Configured thresholds
        sim_threshold = self.config.similarity_threshold       # 0.25
        ev_threshold = self.config.evidence_threshold          # 0.2

        # 2. VERIFIED_REAL — live APIs found articles that match the input story
        #    Conditions: max similarity >= threshold OR evidence strength >= threshold
        #    This means at least some real news sources are covering this topic.
        if max_similarity >= sim_threshold or evidence_strength >= ev_threshold:
            logger.info(
                f"Decision Engine: VERIFIED_REAL — live news sources ARE covering this story "
                f"(max_sim={max_similarity:.4f} >= {sim_threshold}, "
                f"strength={evidence_strength:.4f} >= {ev_threshold}, "
                f"trusted_sources={trusted_source_count}, articles={total_articles})."
            )
            return "VERIFIED_REAL"

        # 3. VERIFIED_FAKE — articles were found but none meaningfully match the input story
        #    This means the live news world is NOT reporting on this specific story.
        logger.info(
            f"Decision Engine: VERIFIED_FAKE — live APIs returned articles but none meaningfully "
            f"match the input story (max_sim={max_similarity:.4f} < {sim_threshold}, "
            f"strength={evidence_strength:.4f} < {ev_threshold})."
        )
        return "VERIFIED_FAKE"
