import os
import json
import tempfile
import pytest
import time
import pandas as pd
from datetime import datetime

from ml.verification.verification_config import VerificationConfig
from ml.verification.rate_limiter import RateLimiter
from ml.verification.retry_manager import RetryManager
from ml.verification.cache_manager import CacheManager
from ml.verification.newsapi_provider import NewsAPIProvider
from ml.verification.gnews_provider import GNewsProvider
from ml.verification.newsdata_provider import NewsDataProvider
from ml.verification.provider_manager import ProviderManager
from ml.verification.keyword_extractor import KeywordExtractor
from ml.verification.article_normalizer import ArticleNormalizer
from ml.verification.duplicate_remover import DuplicateRemover
from ml.verification.similarity_engine import SimilarityEngine
from ml.verification.article_ranker import ArticleRanker
from ml.verification.evidence_engine import EvidenceEngine
from ml.verification.decision_engine import DecisionEngine
from ml.verification.response_builder import ResponseBuilder
from ml.verification.verification_validator import VerificationValidator
from ml.verification.metadata_manager import MetadataManager
from ml.verification.statistics_manager import StatisticsManager
from ml.verification.history_manager import HistoryManager
from ml.verification.hash_generator import HashGenerator
from ml.verification.version_manager import VersionManager
from ml.verification.verification_pipeline import run_verification_pipeline

# 1. RateLimiter Test
def test_rate_limiter():
    limiter = RateLimiter(default_delay=0.1)
    
    start = time.time()
    limiter.wait("test_provider")
    # Immediate call (first call doesn't wait)
    elapsed = time.time() - start
    assert elapsed < 0.05
    
    start = time.time()
    limiter.wait("test_provider")
    # Second call should wait at least 0.1s
    elapsed = time.time() - start
    assert elapsed >= 0.09

# 2. RetryManager Test
def test_retry_manager():
    mgr = RetryManager(max_retries=2, initial_delay=0.01, backoff_factor=2)
    
    call_count = 0
    def failing_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Transient error")
        return "success"
        
    res = mgr.execute(failing_func)
    assert res == "success"
    assert call_count == 2
    
    # Absolute failure
    call_count2 = 0
    def absolute_failure():
        nonlocal call_count2
        call_count2 += 1
        raise ConnectionError("Persistent error")
        
    with pytest.raises(ConnectionError):
        mgr.execute(absolute_failure)
    assert call_count2 == 2

# 3. CacheManager Test
def test_cache_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CacheManager(cache_dir=tmpdir, ttl_seconds=1)
        
        # Cache miss
        assert mgr.get("provider1", "query1") is None
        
        # Cache set and get
        mgr.set("provider1", "query1", [{"title": "Cached Title"}])
        cached = mgr.get("provider1", "query1")
        assert cached is not None
        assert cached[0]["title"] == "Cached Title"
        
        # Expiry check
        time.sleep(1.1)
        assert mgr.get("provider1", "query1") is None

# 4. Providers Test
def test_providers():
    # Test stub mock returns when API keys are absent
    newsapi = NewsAPIProvider(api_key="", max_results=2)
    gnews = GNewsProvider(api_key="<USER_WILL_PROVIDE>", max_results=2)
    newsdata = NewsDataProvider(api_key=None, max_results=2)
    
    n_res = newsapi.fetch_news("economy")
    g_res = gnews.fetch_news("economy")
    nd_res = newsdata.fetch_news("economy")
    
    assert len(n_res) > 0
    assert len(g_res) > 0
    assert len(nd_res) > 0
    
    assert n_res[0]["provider"] == "newsapi"
    assert g_res[0]["provider"] == "gnews"
    assert nd_res[0]["provider"] == "newsdata"

# 5. ProviderManager Test
def test_provider_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = VerificationConfig()
        cache = CacheManager(cache_dir=tmpdir, ttl_seconds=10)
        limiter = RateLimiter(default_delay=0.01)
        retry = RetryManager(max_retries=1)
        
        pm = ProviderManager(config, cache, limiter, retry)
        # Verify it loads enabled providers
        assert len(pm.providers) > 0
        
        results = pm.fetch_all("climate change")
        assert len(results) > 0

# 6. KeywordExtractor Test
def test_keyword_extractor():
    extractor = KeywordExtractor()
    text = "The Federal Reserve and Jerome Powell in Washington D.C. announced new rules on inflation on July 19, 2026."
    res = extractor.extract(text)
    
    assert "query" in res
    assert len(res["query"]) > 0
    assert len(res["entities"]) > 0
    assert "Washington" in [e["text"] for e in res["entities"]] or "Jerome Powell" in [e["text"] for e in res["entities"]]

# 7. ArticleNormalizer Test
def test_article_normalizer():
    norm = ArticleNormalizer()
    raw = {
        "title": "<h1>Subsidies Announced</h1>",
        "description": "Government releases fund.",
        "content": "Official details <p>about the funds</p>.",
        "url": "http://example.com/art1",
        "source": "BBC",
        "author": "Jane",
        "published_date": "2026-07-19T08:00:00Z",
        "language": "EN",
        "provider": "NEWSAPI"
    }
    
    res = norm.normalize_article(raw)
    assert res["title"] == "Subsidies Announced"
    assert "<p>" not in res["content"]
    assert res["language"] == "en"
    assert res["provider"] == "newsapi"

# 8. DuplicateRemover Test
def test_duplicate_remover():
    remover = DuplicateRemover(duplicate_threshold=0.8)
    
    articles = [
        {"title": "Inflation rises in June", "url": "http://url1.com", "content": "Inflation details", "provider": "p1"},
        {"title": "Inflation rises in June", "url": "http://url1.com", "content": "Inflation details", "provider": "p2"}, # Dupe url
        {"title": "Inflation rises in June", "url": "http://url2.com", "content": "Inflation details", "provider": "p3"}, # Dupe title
        {"title": "Completely different report", "url": "http://url3.com", "content": "Something else", "provider": "p4"} # Unique
    ]
    
    uniques = remover.remove_duplicates(articles)
    assert len(uniques) == 2
    assert uniques[0]["url"] == "http://url1.com"
    assert uniques[1]["title"] == "Completely different report"

# 9. SimilarityEngine Test
def test_similarity_engine():
    se = SimilarityEngine()
    t1 = "Government announces tax cuts for medium earners today."
    t2 = "Government announces tax cuts for medium earners today."
    t3 = "A cat climbed up the oak tree in the backyard."
    
    sim1 = se.calculate_similarity(t1, t2)
    sim2 = se.calculate_similarity(t1, t3)
    
    assert sim1["composite"] > 0.8
    assert sim2["composite"] < 0.3
    
    assert 0.0 <= sim1["cosine"] <= 1.0
    assert 0.0 <= sim1["jaccard"] <= 1.0
    assert 0.0 <= sim1["semantic"] <= 1.0

# 10. ArticleRanker Test
def test_article_ranker():
    se = SimilarityEngine()
    ranker = ArticleRanker(similarity_engine=se)
    
    input_text = "Government passes new policies regarding carbon emissions."
    input_title = "Carbon Emission Policies"
    
    articles = [
        {
            "title": "Carbon Emission Policies Approved",
            "content": "The parliament passed the environmental carbon emission bill yesterday.",
            "source": "Reuters",
            "published_date": "2026-07-19T08:00:00Z",
            "url": "http://reuters.com",
            "provider": "newsapi"
        },
        {
            "title": "Local soccer match results",
            "content": "The match ended in a draw after full time.",
            "source": "Mock Tribune",
            "published_date": "2026-07-10T08:00:00Z",
            "url": "http://tribune.com",
            "provider": "gnews"
        }
    ]
    
    ranked = ranker.rank_articles(input_text, input_title, articles, ["carbon", "emission", "policies"])
    assert len(ranked) == 2
    # The relevant reuters article should rank first
    assert ranked[0]["title"] == "Carbon Emission Policies Approved"
    assert ranked[0]["is_trusted_source"] == True
    assert ranked[0]["ranking_score"] > ranked[1]["ranking_score"]

# 11. EvidenceEngine Test
def test_evidence_engine():
    ee = EvidenceEngine()
    ranked = [
        {
            "title": "Art 1", "source": "Reuters", "provider": "newsapi", "is_trusted_source": True,
            "similarity_scores": {"composite": 0.8}
        },
        {
            "title": "Art 2", "source": "Bloomberg", "provider": "gnews", "is_trusted_source": True,
            "similarity_scores": {"composite": 0.6}
        }
    ]
    
    metrics = ee.calculate_evidence(ranked)
    assert metrics["total_articles"] == 2
    assert metrics["trusted_source_count"] == 2
    assert metrics["maximum_similarity"] == 0.8
    assert metrics["average_similarity"] == 0.7
    assert metrics["evidence_strength"] > 0.5
    assert metrics["evidence_confidence"] > 0.3

# 12. DecisionEngine Test
def test_decision_engine():
    config = VerificationConfig()
    de = DecisionEngine(config)
    
    pred_real = {"label": "REAL"}
    pred_fake = {"label": "FAKE"}
    
    # 1. No evidence
    assert de.decide(pred_real, {"total_articles": 0}) == "NOT VERIFIED"
    
    # 2. Conflicting
    # ML predicted FAKE, but news corroborates it strongly
    stats_strong = {
        "total_articles": 3,
        "maximum_similarity": 0.9,
        "average_similarity": 0.8,
        "evidence_strength": 0.85,
        "trusted_source_count": 2
    }
    assert de.decide(pred_fake, stats_strong) == "CONFLICTING"
    
    # 3. Verified
    assert de.decide(pred_real, stats_strong) == "VERIFIED"
    
    # 4. Partially Verified
    stats_weak = {
        "total_articles": 1,
        "maximum_similarity": 0.35,
        "average_similarity": 0.3,
        "evidence_strength": 0.25,
        "trusted_source_count": 0
    }
    assert de.decide(pred_real, stats_weak) == "PARTIALLY VERIFIED"

# 13. ResponseBuilder Test
def test_response_builder():
    rb = ResponseBuilder()
    pred_res = {"label": "REAL", "confidence": 0.9}
    evidence = {"evidence_strength": 0.75, "maximum_similarity": 0.8, "trusted_source_count": 2, "evidence_confidence": 0.7}
    matches = [
        {"title": "Art 1", "source": "Reuters", "url": "http://u1", "published_date": "", "similarity_scores": {"composite": 0.8}, "ranking_score": 0.85, "provider": "newsapi", "is_trusted_source": True}
    ]
    
    resp = rb.build_response(pred_res, "VERIFIED", evidence, matches)
    assert resp["verification_status"] == "VERIFIED"
    assert resp["evidence_score"] == 0.75
    assert resp["similarity_score"] == 0.8
    assert resp["trusted_source_count"] == 2
    assert resp["verification_confidence"] == 0.7
    assert resp["provider_summary"] == {"newsapi": 1}
    assert len(resp["matched_articles"]) == 1

# 14. VerificationValidator Test
def test_verification_validator():
    validator = VerificationValidator()
    
    # API key check
    ok, warns = validator.validate_api_keys({"NewsAPI": "validkey", "GNews": "<placeholder>"})
    assert ok == False
    assert len(warns) == 1
    
    # Normalized article check
    ok2, errs2 = validator.validate_normalized_article({
        "title": "T", "description": "D", "content": "C", "url": "U",
        "source": "S", "author": "A", "published_date": "D", "language": "L",
        "provider": "P"
    })
    assert ok2 == True
    
    # Similarity check
    ok3, errs3 = validator.validate_similarity_scores({
        "cosine": 0.8, "jaccard": 0.5, "semantic": 0.7, "composite": 0.75
    })
    assert ok3 == True
    
    # Out of bounds similarity
    ok4, errs4 = validator.validate_similarity_scores({
        "cosine": 1.2, "jaccard": 0.5, "semantic": 0.7, "composite": 0.75
    })
    assert ok4 == False

# 15. Metadata, Statistics, History, Hashes, and Versions Managers Test
def test_storage_managers():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_file = os.path.join(tmpdir, "metadata.json")
        stats_file = os.path.join(tmpdir, "statistics.json")
        history_csv = os.path.join(tmpdir, "history.csv")
        hashes_file = os.path.join(tmpdir, "hashes.json")
        versions_file = os.path.join(tmpdir, "versions.json")
        
        # 1. Metadata Manager
        meta_mgr = MetadataManager(tmpdir)
        meta_mgr.save_metadata(
            file_path=meta_file,
            providers_used=["newsapi"],
            api_response_count=5,
            verification_version="verification_v1",
            prediction_version="pred_v1",
            model_used="svm"
        )
        assert os.path.exists(meta_file)
        
        # 2. Statistics Manager
        stats_mgr = StatisticsManager(stats_file)
        stats = stats_mgr.update_statistics(
            verification_status="VERIFIED",
            similarity_score=0.8,
            api_time_sec=1.5,
            response_count=5,
            cache_hit=False
        )
        assert stats["total_verifications"] == 1
        assert stats["verified_count"] == 1
        assert stats["average_similarity"] == 0.8
        
        # 3. History Manager
        hist_mgr = HistoryManager(history_csv)
        hist_mgr.save_history(
            input_hash="hash123",
            prediction="REAL",
            verification_status="VERIFIED",
            evidence_score=0.85,
            similarity=0.8,
            providers="newsapi"
        )
        assert os.path.exists(history_csv)
        df = pd.read_csv(history_csv)
        assert len(df) == 1
        assert df.iloc[0]["Input Hash"] == "hash123"
        
        # 4. Hash Generator
        hash_gen = HashGenerator(hashes_file)
        hashes = hash_gen.generate_hashes({
            "metadata": meta_file,
            "statistics": stats_file
        })
        assert "metadata" in hashes
        assert "statistics" in hashes
        assert os.path.exists(hashes_file)
        
        # 5. Version Manager
        ver_mgr = VersionManager(versions_file)
        ver_str = ver_mgr.register_version(
            prediction_version="pred_v1",
            model_used="svm",
            hashes=hashes
        )
        assert ver_str == "verification_v1"
        assert len(ver_mgr.load_versions()) == 1

# 16. Pipeline End-To-End Test
def test_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temp config YAML
        config_yaml_path = os.path.join(tmpdir, "verification_config.yaml")
        config_data = {
            "verification": {
                "timeout": 5,
                "retry_count": 1,
                "rate_limit_delay": 0.01,
                "max_results_per_provider": 2,
                "similarity_threshold": 0.4,
                "evidence_threshold": 0.5,
                "duplicate_threshold": 0.85,
                "cache_expiration": 3600
            },
            "providers": {
                "enable_newsapi": True,
                "enable_gnews": True,
                "enable_newsdata": True
            },
            "exports": {
                "enable_charts": True,
                "enable_reports": True,
                "enable_metadata": True,
                "enable_statistics": True,
                "enable_hashing": True,
                "enable_versions": True,
                "enable_history": True
            },
            "paths": {
                "output_dir": tmpdir,
                "logs_dir": os.path.join(tmpdir, "logs"),
                "reports_dir": os.path.join(tmpdir, "reports"),
                "statistics_dir": os.path.join(tmpdir, "statistics"),
                "metadata_dir": os.path.join(tmpdir, "metadata"),
                "hashes_dir": os.path.join(tmpdir, "hashes"),
                "versions_dir": os.path.join(tmpdir, "versions"),
                "cache_dir": os.path.join(tmpdir, "cache"),
                "history_dir": os.path.join(tmpdir, "history"),
                "charts_dir": os.path.join(tmpdir, "charts"),
                "verification_history_csv": os.path.join(tmpdir, "history", "history.csv"),
                "verification_statistics_json": os.path.join(tmpdir, "statistics", "stats.json"),
                "verification_report_md": os.path.join(tmpdir, "reports", "report.md"),
                "verification_metadata_json": os.path.join(tmpdir, "metadata", "meta.json"),
                "verification_hashes_json": os.path.join(tmpdir, "hashes", "hashes.json"),
                "verification_versions_json": os.path.join(tmpdir, "versions", "versions.json"),
                "verification_pipeline_log": os.path.join(tmpdir, "logs", "pipeline.log")
            }
        }
        
        with open(config_yaml_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(config_data, f)
            
        pred_response = {
            "prediction": 0,
            "label": "REAL",
            "confidence": 0.95,
            "model_name": "svm",
            "model_version": "1.0.0",
            "evaluation_version": "evaluation_v1",
            "prediction_time_ms": 10.0,
            "throughput": 1000.0,
            "memory_usage": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        article_text = "The President declared new measures to support green energy today."
        
        response = run_verification_pipeline(
            article_text=article_text,
            prediction_response=pred_response,
            article_title="Green Energy Declaration",
            config_path=config_yaml_path
        )
        
        assert response["verification_status"] in ["VERIFIED", "PARTIALLY VERIFIED", "NOT VERIFIED", "CONFLICTING"]
        assert len(response["matched_articles"]) > 0
        assert os.path.exists(config_data["paths"]["verification_report_md"])
        assert os.path.exists(config_data["paths"]["verification_statistics_json"])
        assert os.path.exists(config_data["paths"]["verification_metadata_json"])
        assert os.path.exists(config_data["paths"]["verification_hashes_json"])
        assert os.path.exists(config_data["paths"]["verification_versions_json"])
        assert os.path.exists(config_data["paths"]["verification_history_csv"])
