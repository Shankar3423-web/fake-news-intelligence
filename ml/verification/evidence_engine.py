import logging
from typing import List, Dict, Any

logger = logging.getLogger("verification_pipeline")

class EvidenceEngine:
    """
    EvidenceEngine analyzes ranked article statistics to calculate evidence strength,
    average/max similarity, trusted source count, and evidence confidence.
    """
    def calculate_evidence(self, ranked_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates evidence metrics based on the ranked articles.
        """
        total_articles = len(ranked_articles)
        
        if total_articles == 0:
            logger.info("No matching articles found to evaluate as evidence.")
            return {
                "total_articles": 0,
                "average_similarity": 0.0,
                "maximum_similarity": 0.0,
                "trusted_source_count": 0,
                "evidence_strength": 0.0,
                "evidence_confidence": 0.0
            }

        # Extract composite similarity scores
        sim_scores = [art["similarity_scores"]["composite"] for art in ranked_articles]
        avg_similarity = sum(sim_scores) / total_articles
        max_similarity = max(sim_scores)
        
        trusted_source_count = sum(1 for art in ranked_articles if art.get("is_trusted_source", False))

        # 1. Evidence Strength: weighted average of similarity and trusted sources count
        # We cap trusted source factor at 3 sources
        trusted_factor = min(1.0, trusted_source_count / 3.0)
        evidence_strength = (0.5 * avg_similarity) + (0.3 * max_similarity) + (0.2 * trusted_factor)
        evidence_strength = round(max(0.0, min(1.0, evidence_strength)), 4)

        # 2. Evidence Confidence: scales with quantity of sources and maximum similarity
        # A higher number of matching articles increases confidence
        quantity_factor = min(1.0, total_articles / 4.0)
        evidence_confidence = quantity_factor * max_similarity
        evidence_confidence = round(max(0.0, min(1.0, evidence_confidence)), 4)

        logger.info(
            f"Evidence calculations complete: articles={total_articles}, "
            f"avg_sim={avg_similarity:.2f}, max_sim={max_similarity:.2f}, "
            f"strength={evidence_strength:.2f}, confidence={evidence_confidence:.2f}"
        )

        return {
            "total_articles": total_articles,
            "average_similarity": round(avg_similarity, 4),
            "maximum_similarity": round(max_similarity, 4),
            "trusted_source_count": trusted_source_count,
            "evidence_strength": evidence_strength,
            "evidence_confidence": evidence_confidence
        }
