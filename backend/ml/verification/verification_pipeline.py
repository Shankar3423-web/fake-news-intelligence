import os
import time
import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime

from ml.verification.verification_config import VerificationConfig
from ml.verification.verification_logger import setup_logger, shutdown_logger
from ml.verification.keyword_extractor import KeywordExtractor
from ml.verification.cache_manager import CacheManager
from ml.verification.rate_limiter import RateLimiter
from ml.verification.retry_manager import RetryManager
from ml.verification.provider_manager import ProviderManager
from ml.verification.article_normalizer import ArticleNormalizer
from ml.verification.duplicate_remover import DuplicateRemover
from ml.verification.similarity_engine import SimilarityEngine
from ml.verification.article_ranker import ArticleRanker
from ml.verification.evidence_engine import EvidenceEngine
from ml.verification.decision_engine import DecisionEngine
from ml.verification.response_builder import ResponseBuilder
from ml.verification.metadata_manager import MetadataManager
from ml.verification.statistics_manager import StatisticsManager
from ml.verification.history_manager import HistoryManager
from ml.verification.hash_generator import HashGenerator
from ml.verification.version_manager import VersionManager
from ml.verification.report_generator import ReportGenerator
from ml.verification.verification_validator import VerificationValidator

logger = logging.getLogger("verification_pipeline")

_cached_extractor = None
_cached_sim_engine = None

def run_verification_pipeline(
    article_text: str,
    prediction_response: Dict[str, Any],
    article_title: Optional[str] = None,
    config_path: str = "ml/verification/verification_config.yaml"
) -> Dict[str, Any]:
    """
    Executes the entire Phase 9 Live News Verification Engine pipeline.
    """
    # 1. Load config and setup logger
    config = VerificationConfig(config_path)
    setup_logger(config.get_path("logs_dir"))

    try:
        logger.info("================================================================================")
        logger.info("STARTING PHASE 9 LIVE NEWS VERIFICATION PIPELINE")
        logger.info("================================================================================")

        warnings: List[str] = []
        
        # 2. Extract title if not explicitly provided
        if not article_title:
            lines = article_text.strip().split("\n")
            first_line = lines[0] if lines else ""
            if len(first_line) > 100:
                article_title = first_line[:97] + "..."
            else:
                article_title = first_line or "Untitled News Article"
            logger.info(f"No title provided. Extracted title: '{article_title}'")

        # 3. Initialize pipeline components
        validator = VerificationValidator()
        
        # Validate API Keys in environment
        api_keys = {
            "NewsAPI": os.getenv("NEWS_API_KEY", ""),
            "GNews": os.getenv("GNEWS_API_KEY", ""),
            "NewsData": os.getenv("NEWSDATA_API_KEY", "")
        }
        keys_ok, key_warns = validator.validate_api_keys(api_keys)
        if not keys_ok:
            logger.warning(f"API key validation issues: {key_warns}")
            warnings.extend(key_warns)

        global _cached_extractor, _cached_sim_engine
        if _cached_extractor is None:
            _cached_extractor = KeywordExtractor()
        if _cached_sim_engine is None:
            _cached_sim_engine = SimilarityEngine()
            
        extractor = _cached_extractor
        cache_mgr = CacheManager(
            cache_dir=config.get_path("cache_dir"),
            ttl_seconds=config.cache_expiration
        )
        rate_limiter = RateLimiter(default_delay=config.rate_limit_delay)
        retry_mgr = RetryManager(max_retries=config.retry_count)
        
        prov_mgr = ProviderManager(
            config=config,
            cache_manager=cache_mgr,
            rate_limiter=rate_limiter,
            retry_manager=retry_mgr
        )
        normalizer = ArticleNormalizer()
        dupe_remover = DuplicateRemover(duplicate_threshold=config.duplicate_threshold)
        sim_engine = _cached_sim_engine
        ranker = ArticleRanker(similarity_engine=sim_engine)
        evidence_eng = EvidenceEngine()
        decision_eng = DecisionEngine(config=config)
        resp_builder = ResponseBuilder()

        # 4. Keyword Extraction
        logger.info("Extracting search keywords from article text...")
        keywords_info = extractor.extract(article_text)
        search_query = keywords_info["query"]
        extracted_keywords = [kw for kw in keywords_info["nouns"]] + [ent["text"] for ent in keywords_info["entities"]]

        # 5. Query News Providers & Calculate API execution time
        logger.info(f"Querying news providers with search query: '{search_query}'")
        api_start_time = time.time()
        
        raw_articles = prov_mgr.fetch_all(search_query)
        
        api_duration = time.time() - api_start_time
        logger.info(f"Provider retrieval complete in {api_duration:.2f}s. Received {len(raw_articles)} raw articles.")

        # 6. Normalize Responses
        logger.info("Normalizing articles schema...")
        normalized_articles = normalizer.normalize_batch(raw_articles)
        
        # Validate normalized articles
        for i, art in enumerate(normalized_articles):
            art_ok, art_errs = validator.validate_normalized_article(art)
            if not art_ok:
                logger.warning(f"Normalized article index {i} fails schema: {art_errs}")

        # 7. Remove Duplicates
        logger.info("Removing near-duplicates and overlapping articles...")
        unique_articles = dupe_remover.remove_duplicates(normalized_articles)

        # 8. Rank Retrieved Evidence
        logger.info("Ranking articles and measuring similarities...")
        ranked_articles = ranker.rank_articles(
            input_text=article_text,
            input_title=article_title,
            articles=unique_articles,
            keywords=extracted_keywords
        )

        # 9. Evidence Engine Calculation
        logger.info("Synthesizing evidence and confidence scores...")
        evidence_metrics = evidence_eng.calculate_evidence(ranked_articles)
        
        ev_ok, ev_errs = validator.validate_evidence_metrics(evidence_metrics)
        if not ev_ok:
            logger.error(f"Evidence metrics validation failed: {ev_errs}")
            raise ValueError(f"Invalid evidence metrics: {ev_errs}")

        # 10. Decision Engine Determination
        logger.info("Determining final verification decision...")
        verification_status = decision_eng.decide(prediction_response, evidence_metrics)

        # 11. Build Structured Verification Response
        response = resp_builder.build_response(
            prediction_result=prediction_response,
            verification_status=verification_status,
            evidence_metrics=evidence_metrics,
            matched_articles=ranked_articles
        )

        # 12. Save Registries and Generate Reports
        input_hash = hashlib.sha256(article_text.encode("utf-8")).hexdigest()
        
        # Save History
        if config.enable_history:
            history_mgr = HistoryManager(history_csv_path=config.get_path("verification_history_csv"))
            history_mgr.save_history(
                input_hash=input_hash,
                prediction=prediction_response.get("label", "UNKNOWN"),
                verification_status=verification_status,
                evidence_score=evidence_metrics["evidence_strength"],
                similarity=evidence_metrics["maximum_similarity"],
                providers=",".join(list(response["provider_summary"].keys()))
            )

        # Save Statistics
        stats_data = {}
        if config.enable_statistics:
            stats_mgr = StatisticsManager(stats_file_path=config.get_path("verification_statistics_json"))
            is_cache_hit = len(raw_articles) > 0 and api_duration < 0.1
            stats_data = stats_mgr.update_statistics(
                verification_status=verification_status,
                similarity_score=evidence_metrics["maximum_similarity"],
                api_time_sec=api_duration,
                response_count=len(raw_articles),
                cache_hit=is_cache_hit
            )

        # Save Metadata
        if config.enable_metadata:
            meta_mgr = MetadataManager(metadata_dir=config.get_path("metadata_dir"))
            ver_mgr = VersionManager(versions_file_path=config.get_path("verification_versions_json"))
            version_list = ver_mgr.load_versions()
            verification_version = f"verification_v{len(version_list) + 1}"
            
            meta_mgr.save_metadata(
                file_path=config.get_path("verification_metadata_json"),
                providers_used=list(response["provider_summary"].keys()),
                api_response_count=len(raw_articles),
                verification_version=verification_version,
                prediction_version=prediction_response.get("prediction_version", "unknown"),
                model_used=prediction_response.get("model_name", "unknown")
            )

        # Save Reports & Render Charts
        if config.enable_reports:
            rep_gen = ReportGenerator(
                report_file_path=config.get_path("verification_report_md"),
                charts_dir=config.get_path("charts_dir")
            )
            
            # Create a placeholder file to prevent HashGenerator from failing
            # This completely avoids rendering slow matplotlib charts twice
            with open(config.get_path("verification_report_md"), "w", encoding="utf-8") as f:
                f.write("Pending...")

        # Save Hashes Registry
        hashes_registry = {}
        if config.enable_hashing:
            hash_gen = HashGenerator(hash_file_path=config.get_path("verification_hashes_json"))
            files_to_hash = {}
            if config.enable_reports:
                files_to_hash["report"] = config.get_path("verification_report_md")
            if config.enable_metadata:
                files_to_hash["metadata"] = config.get_path("verification_metadata_json")
            if config.enable_statistics:
                files_to_hash["statistics"] = config.get_path("verification_statistics_json")
            if config.enable_history:
                files_to_hash["history"] = config.get_path("verification_history_csv")
                
            hashes_registry = hash_gen.generate_hashes(files_to_hash)
            
            if config.enable_reports:
                rep_gen.generate_report(response, stats_data, hashes_registry, warnings)
                hash_gen.generate_hashes(files_to_hash)

        # Save Versions Registry
        if config.enable_versions:
            ver_mgr = VersionManager(versions_file_path=config.get_path("verification_versions_json"))
            ver_mgr.register_version(
                prediction_version=prediction_response.get("prediction_version", "unknown"),
                model_used=prediction_response.get("model_name", "unknown"),
                hashes=hashes_registry
            )

        # 13. Validate Final Response Object
        resp_ok, resp_errs = validator.validate_verification_response(response)
        if not resp_ok:
            logger.error(f"Response validation failed: {resp_errs}")
            raise ValueError(f"Invalid verification response: {resp_errs}")

        logger.info("================================================================================")
        logger.info("LIVE NEWS VERIFICATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("================================================================================")

        return response
    finally:
        shutdown_logger()
