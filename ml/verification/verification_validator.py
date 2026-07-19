import os
import json
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("verification_pipeline")

class VerificationValidator:
    """
    VerificationValidator performs schema, range, and type validation for
    inputs, normalized responses, similarity scores, and outputs of Phase 9.
    """
    def validate_api_keys(self, keys: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validates that API keys exist and warns if defaults/placeholders are used."""
        errors = []
        for name, key in keys.items():
            if not key:
                errors.append(f"API key for '{name}' is missing or empty.")
            elif "USER_WILL_PROVIDE" in key or key.startswith("<"):
                errors.append(f"API key for '{name}' is using a placeholder format.")
        return len(errors) == 0, errors

    def validate_normalized_article(self, article: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Asserts schema and formats of a normalized article."""
        errors = []
        required_keys = ["title", "description", "content", "url", "source", "author", "published_date", "language", "provider"]
        for key in required_keys:
            if key not in article:
                errors.append(f"Article missing required key: '{key}'")
            elif not isinstance(article[key], str):
                errors.append(f"Article field '{key}' must be a string, got {type(article[key])}")
        return len(errors) == 0, errors

    def validate_similarity_scores(self, scores: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Checks bounds and types of similarity calculations."""
        errors = []
        required_keys = ["cosine", "jaccard", "semantic", "composite"]
        for key in required_keys:
            if key not in scores:
                errors.append(f"Similarity score missing key: '{key}'")
            else:
                val = scores[key]
                if not isinstance(val, (int, float)):
                    errors.append(f"Similarity score '{key}' must be a float, got {type(val)}")
                elif val < 0.0 or val > 1.0:
                    errors.append(f"Similarity score '{key}' out of [0, 1] bounds: {val}")
        return len(errors) == 0, errors

    def validate_evidence_metrics(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates structure and ranges of compiled evidence metrics."""
        errors = []
        required_keys = [
            "total_articles", "average_similarity", "maximum_similarity", 
            "trusted_source_count", "evidence_strength", "evidence_confidence"
        ]
        for key in required_keys:
            if key not in metrics:
                errors.append(f"Evidence metrics missing key: '{key}'")
            
        if not errors:
            if not isinstance(metrics["total_articles"], int) or metrics["total_articles"] < 0:
                errors.append(f"total_articles must be non-negative integer: {metrics['total_articles']}")
            if not isinstance(metrics["trusted_source_count"], int) or metrics["trusted_source_count"] < 0:
                errors.append(f"trusted_source_count must be non-negative integer: {metrics['trusted_source_count']}")
                
            for float_key in ["average_similarity", "maximum_similarity", "evidence_strength", "evidence_confidence"]:
                val = metrics[float_key]
                if not isinstance(val, (int, float)):
                    errors.append(f"Evidence metric '{float_key}' must be numeric, got {type(val)}")
                elif val < 0.0 or val > 1.0:
                    errors.append(f"Evidence metric '{float_key}' out of [0, 1] bounds: {val}")
                    
        return len(errors) == 0, errors

    def validate_decision(self, status: str) -> Tuple[bool, List[str]]:
        """Ensures the decision is one of the allowed verification states."""
        errors = []
        valid_statuses = {"VERIFIED_REAL", "VERIFIED_FAKE", "INCONCLUSIVE"}
        if status not in valid_statuses:
            errors.append(f"Invalid verification decision: '{status}'. Valid statuses: {valid_statuses}")
        return len(errors) == 0, errors

    def validate_verification_response(self, response: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates the complete structured verification pipeline response."""
        errors = []
        required_keys = [
            "prediction_result", "verification_status", "evidence_score", "similarity_score",
            "trusted_source_count", "matched_articles", "provider_summary",
            "verification_confidence", "timestamp"
        ]
        
        for key in required_keys:
            if key not in response:
                errors.append(f"Response missing required key: '{key}'")
                
        if not errors:
            status_ok, status_errs = self.validate_decision(response["verification_status"])
            errors.extend(status_errs)
            
            # Type and range assertions
            for float_key in ["evidence_score", "similarity_score", "verification_confidence"]:
                val = response[float_key]
                if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
                    errors.append(f"Response '{float_key}' must be float in range [0, 1]: {val}")
                    
            if not isinstance(response["trusted_source_count"], int) or response["trusted_source_count"] < 0:
                errors.append(f"Response trusted_source_count must be non-negative integer: {response['trusted_source_count']}")
            if not isinstance(response["matched_articles"], list):
                errors.append(f"Response matched_articles must be a list, got {type(response['matched_articles'])}")
            if not isinstance(response["provider_summary"], dict):
                errors.append(f"Response provider_summary must be a dictionary, got {type(response['provider_summary'])}")
                
        return len(errors) == 0, errors
