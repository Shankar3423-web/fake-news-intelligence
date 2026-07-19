# Phase 9: Live News Verification Engine

This sub-module implements the **Live News Verification Engine** for the Fake News Intelligence System. It cross-references ML classification predictions with trusted live news articles from global provider APIs to verify if the claims are backed by contemporary reporting.

---

## Folder Structure

```
ml/
└── verification/
    ├── __init__.py                  # Exposes pipeline and configurations
    ├── verification_pipeline.py     # Main workflow orchestrator
    ├── verification_config.py       # Configuration parser wrapper
    ├── verification_config.yaml     # YAML configuration settings
    ├── verification_logger.py       # Logger setup (standard & timestamped logs)
    ├── provider_base.py             # Base abstract class for news APIs
    ├── provider_manager.py          # Provider loading, execution, and error boundary manager
    ├── newsapi_provider.py          # NewsAPI.org implementation
    ├── gnews_provider.py            # GNews.io implementation
    ├── newsdata_provider.py         # NewsData.io implementation
    ├── keyword_extractor.py         # Named Entity Recognition and POS query generator
    ├── article_normalizer.py        # Text parser and schema normalizer
    ├── duplicate_remover.py         # Cosine TF-IDF duplicate filter
    ├── article_ranker.py            # Recency, overlap, and source trustworthiness ranker
    ├── similarity_engine.py         # Cosine, Jaccard, and spaCy similarity calculator
    ├── evidence_engine.py           # Synthesis of article stats and evidence strength
    ├── decision_engine.py           # Evaluates prediction context to assign status
    ├── response_builder.py          # Packages outputs into a standard response schema
    ├── cache_manager.py             # File-based JSON caching with TTL
    ├── retry_manager.py             # Exponential backoff retry utility
    ├── rate_limiter.py              # Provider request speed limiter
    ├── metadata_manager.py          # Compiles run metadata
    ├── statistics_manager.py        # Accumulates aggregate run statistics
    ├── history_manager.py           # CSV transaction history logger
    ├── hash_generator.py            # SHA-256 integrity checksum registry
    ├── version_manager.py           # Verification version database
    ├── report_generator.py          # Formats markdown reports and renders charts
    ├── verification_validator.py    # Range, schema, and type validator
    ├── verify_verification.py       # CLI validation script
    │
    ├── logs/                        # Verification execution log files
    ├── metadata/                    # Verification metadata outputs
    ├── reports/                     # Markdown report outputs
    ├── statistics/                  # Aggregate execution statistics
    ├── hashes/                      # Checksum verification registries
    ├── versions/                    # Run version records
    ├── cache/                       # Cached API responses
    ├── history/                     # CSV history database
    ├── charts/                      # Rendered PNG charts
    └── tests/                       # Pytest test scripts
```

---

## Workflow

1. **User Article Submission**: The input text is fed into the engine along with the prediction result from Phase 8.
2. **Keyword & NER Extraction**: spaCy extracts entity types (ORG, GPE, DATE, EVENT, LOC, PERSON) and prominent nouns to build an optimized query.
3. **Evidence Retrieval**: The engine queries NewsAPI, GNews, and NewsData.io using caching, rate limiting, and exponential retry policies. Stubs are used if API keys are missing or offline.
4. **Data Normalization**: Field formats, HTML tags, and timestamps are cleaned and mapped into a single unified schema.
5. **Deduplication**: Exact URL duplicates and high-similarity matches are removed.
6. **Relevance Ranking**: Articles are scored based on semantic similarity, keyword overlap, source quality weights, and date recency.
7. **Synthesis & Scoring**: Composite similarity, evidence strength, and confidence metrics are calculated.
8. **Decision Assignment**: Based on prediction context and evidence metrics, status is assigned: `VERIFIED`, `PARTIALLY VERIFIED`, `NOT VERIFIED`, or `CONFLICTING`.
9. **Registry Updates**: Run entries, stats, versions, markdown summaries, and checksum registries are written.
10. **Validation**: Outputs are asserted for correctness, types, and schema compliance.

---

## Provider Setup

API keys are read from the parent `.env` file. Do not hardcode keys. The variables are:

```ini
NEWS_API_KEY=your_news_api_key
GNEWS_API_KEY=your_gnews_api_key
NEWSDATA_API_KEY=your_newsdata_api_key
```

*If any API keys are omitted or invalid, the engine runs in mock fallback mode to prevent pipeline failure.*

---

## Configuration

Configuration values are defined in `verification_config.yaml`. The schema is as follows:

```yaml
verification:
  timeout: 10                       # Outbound API timeout (seconds)
  retry_count: 3                    # Number of retries on transient network errors
  rate_limit_delay: 1.0             # Sleep interval between provider calls (seconds)
  max_results_per_provider: 5       # Capping results per provider
  similarity_threshold: 0.4         # Minimum similarity to qualify as a match
  evidence_threshold: 0.5           # Minimum evidence strength for VERIFIED status
  duplicate_threshold: 0.85         # Similarity threshold for deduplication
  cache_expiration: 3600            # Cache TTL (seconds)

providers:
  enable_newsapi: true
  enable_gnews: true
  enable_newsdata: true

exports:
  enable_charts: true
  enable_reports: true
  enable_metadata: true
  enable_statistics: true
  enable_hashing: true
  enable_versions: true
  enable_history: true
```

---

## Execution

### Verification Script
To test configuration loading, pipeline flow, file generation, and integrity checks:
```bash
python ml/verification/verify_verification.py
```

### Run Unit Tests
To run pytests:
```bash
pytest ml/verification/tests/
```
