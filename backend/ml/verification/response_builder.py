from datetime import datetime
from typing import Dict, Any, List

class ResponseBuilder:
    """
    ResponseBuilder standardizes the final output response for the Live News Verification Engine.
    """
    def build_response(
        self,
        prediction_result: Dict[str, Any],
        verification_status: str,
        evidence_metrics: Dict[str, Any],
        matched_articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds the structured verification response dictionary.
        """
        # Calculate provider summary (count of matched articles from each provider)
        provider_counts: Dict[str, int] = {}
        for art in matched_articles:
            provider = art.get("provider", "unknown")
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        # Format matches to include only required fields for display, or full article metadata
        formatted_articles = []
        for art in matched_articles:
            formatted_articles.append({
                "title": art.get("title", ""),
                "source": art.get("source", ""),
                "url": art.get("url", ""),
                "published_date": art.get("published_date", ""),
                "similarity_score": art.get("similarity_scores", {}).get("composite", 0.0),
                "ranking_score": art.get("ranking_score", 0.0),
                "provider": art.get("provider", ""),
                "is_trusted": art.get("is_trusted_source", False)
            })

        timestamp = datetime.now().isoformat()

        response = {
            "prediction_result": prediction_result,
            "verification_status": verification_status,
            "evidence_score": float(evidence_metrics.get("evidence_strength", 0.0)),
            "similarity_score": float(evidence_metrics.get("maximum_similarity", 0.0)),
            "trusted_source_count": int(evidence_metrics.get("trusted_source_count", 0)),
            "matched_articles": formatted_articles,
            "provider_summary": provider_counts,
            "verification_confidence": float(evidence_metrics.get("evidence_confidence", 0.0)),
            "timestamp": timestamp
        }

        return response
