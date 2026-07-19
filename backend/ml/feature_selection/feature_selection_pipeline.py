import os
import sys
import time
import json
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.sparse
import joblib
from typing import List, Dict, Any, Tuple

from ml.feature_selection.selection_config import SelectionConfig
from ml.feature_selection.selection_logger import setup_logger
from ml.feature_selection.selection_utils import load_data, compute_file_sha256, get_memory_usage
from ml.feature_selection.variance_selector import VarianceSelector
from ml.feature_selection.correlation_selector import CorrelationSelector
from ml.feature_selection.mutual_information_selector import MutualInformationSelector
from ml.feature_selection.chi_square_selector import ChiSquareSelector
from ml.feature_selection.random_forest_selector import RandomForestSelector
from ml.feature_selection.rfe_selector import RFESelector
from ml.feature_selection.selector_merger import SelectorMerger
from ml.feature_selection.selection_validator import SelectionValidator
from ml.feature_selection.selection_statistics import SelectionStatistics
from ml.feature_selection.selection_profiler import SelectionProfiler

def generate_markdown_report(
    config: SelectionConfig,
    stats: Dict[str, Any],
    profile: Dict[str, Any],
    rankings: Dict[str, Dict[str, float]],
    selected_dense: List[str],
    selected_sparse: List[str],
    hashes: Dict[str, str],
    validation_passed: bool,
    validation_errors: List[str]
) -> str:
    """
    Generates a beautiful and comprehensive Markdown report for the Feature Selection phase.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = stats["summary"]
    by_type = stats["by_feature_type"]
    
    md = []
    md.append("# Phase 5 – Feature Selection Quality Report")
    md.append(f"\n*Generated on: {timestamp}*")
    md.append("\n## Executive Summary")
    md.append("\nThis report details the execution and results of the **Feature Selection** phase. The primary objective was to reduce the high-dimensional feature space (engineered dense metrics + TF-IDF textual features) to a highly predictive, low-redundancy subset for the subsequent model training phase.")
    
    # Selection stats table
    md.append("\n### Feature Space Compression")
    md.append("| Metric | Input Count | Selected Count | Reduction % |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Dense Engineered Features** | {by_type['dense']['input']} | {by_type['dense']['output']} | {by_type['dense']['reduction_percentage']}% |")
    md.append(f"| **TF-IDF Text Features** | {by_type['sparse']['input']} | {by_type['sparse']['output']} | {by_type['sparse']['reduction_percentage']}% |")
    md.append(f"| **Total Feature Space** | {summary['total_input_features']} | {summary['total_output_features']} | {summary['reduction_percentage']}% |")
    
    md.append(f"\n- **Total Time Elapsed:** {summary['total_execution_time_seconds']} seconds")
    md.append(f"- **Merger Strategy:** `{config.merger_strategy}` (threshold: {config.merger_voting_threshold if config.merger_strategy == 'voting' else 'N/A'})")
    
    # Validation results
    md.append("\n## Verification & Integrity Status")
    if validation_passed:
        md.append("\n> [!NOTE]\n> **VERIFICATION STATUS: PASSED**\n> All structural, alignment, and data-integrity checks completed successfully. Sample order and label mapping are verified as preserved row-for-row.")
    else:
        md.append("\n> [!CAUTION]\n> **VERIFICATION STATUS: FAILED**\n> Errors encountered during validation checks:")
        for err in validation_errors:
            md.append(f"\n- {err}")
            
    # Selected features list
    md.append("\n## Selected Features Overview")
    md.append(f"\n### Selected Dense Features ({len(selected_dense)})")
    if selected_dense:
        md.append(", ".join([f"`{f}`" for f in selected_dense]))
    else:
        md.append("*No dense features were selected.*")
        
    md.append(f"\n### Sample of Selected TF-IDF Features ({len(selected_sparse)} total)")
    if selected_sparse:
        sample_sparse = selected_sparse[:30]
        md.append(", ".join([f"`{f}`" for f in sample_sparse]))
        if len(selected_sparse) > 30:
            md.append(f" ... *and {len(selected_sparse) - 30} more.*")
    else:
        md.append("*No TF-IDF features were selected.*")

    # Selector-level contributions
    md.append("\n## Selector Contributions and Performance")
    md.append("\n| Selector | Enabled | Selected Count | Runtime (s) |")
    md.append("| :--- | :---: | :---: | :---: |")
    for name in ["variance", "correlation", "mutual_information", "chi_square", "random_forest", "rfe"]:
        enabled = config.is_selector_enabled(name)
        count = stats["selector_selected_counts"].get(name, 0)
        runtime = stats["selector_runtimes_seconds"].get(name, 0.0)
        md.append(f"| {name.replace('_', ' ').title()} | {enabled} | {count} | {runtime:.4f}s |")

    # Feature Importance Rankings
    md.append("\n## Top Selected Feature Rankings")
    for sel_name, feat_ranks in rankings.items():
        if not feat_ranks:
            continue
        md.append(f"\n### {sel_name.replace('_', ' ').title()} - Top 10 Features")
        md.append("| Rank | Feature Name | Score/Value |")
        md.append("| :---: | :--- | :---: |")
        
        # Depending on selector, sort order might be different
        # For RFE, smaller is better (1 is best). For others, higher is better.
        reverse_sort = True
        if sel_name == "rfe":
            reverse_sort = False
            
        sorted_feats = sorted(feat_ranks.keys(), key=lambda k: feat_ranks[k], reverse=reverse_sort)[:10]
        for rank_idx, f_name in enumerate(sorted_feats, 1):
            md.append(f"| {rank_idx} | `{f_name}` | {feat_ranks[f_name]:.6f} |")

    # Selected correlation matrix highlights
    md.append("\n## Selected Features Multicollinearity Analysis")
    if "correlation_matrix" in profile and profile["correlation_matrix"] and "error" not in profile["correlation_matrix"]:
        corr_matrix = profile["correlation_matrix"]
        high_corrs = []
        checked = set()
        
        # Look for pairs with correlation > 0.5 (excluding self)
        for f1 in corr_matrix:
            for f2 in corr_matrix[f1]:
                if f1 != f2 and (f2, f1) not in checked:
                    checked.add((f1, f2))
                    val = corr_matrix[f1][f2]
                    if abs(val) > 0.5:
                        high_corrs.append((f1, f2, val))
                        
        high_corrs = sorted(high_corrs, key=lambda x: abs(x[2]), reverse=True)[:15]
        
        if high_corrs:
            md.append("\nTop correlations among final selected features (absolute value > 0.5):")
            md.append("\n| Feature 1 | Feature 2 | Pearson Correlation |")
            md.append("| :--- | :--- | :---: |")
            for f1, f2, val in high_corrs:
                md.append(f"| `{f1}` | `{f2}` | {val:.4f} |")
        else:
            md.append("\n*No significant correlations (> 0.5) found among final selected features. Multicollinearity is minimized.*")
    else:
        md.append("\n*Correlation analysis skipped or unavailable.*")

    # Output file hashes
    md.append("\n## Artifact Signatures & Hash Registry")
    md.append("\n| File Path | SHA-256 Hash |")
    md.append("| :--- | :--- |")
    for fname, fhash in hashes.items():
        md.append(f"| `{fname}` | `{fhash}` |")
        
    return "\n".join(md)

def run_feature_selection_pipeline(config_path: str = "config/selection_config.yaml") -> bool:
    """
    Orchestrates the entire Phase 5 Feature Selection pipeline.
    """
    start_time = time.time()
    
    # 1. Load config and logger
    config = SelectionConfig(config_path)
    logger = setup_logger(config.get_path("logs_dir"))
    
    logger.info("==================================================")
    logger.info("STARTING PHASE 5 FEATURE SELECTION PIPELINE")
    logger.info("==================================================")
    
    initial_mem, _ = get_memory_usage()
    logger.info(f"Initial process memory usage: {initial_mem:.2f} MB")

    # 2. Ensure output directories exist
    for dir_key in ["processed_dir", "models_dir", "reports_dir", "statistics_dir", "versions_dir"]:
        os.makedirs(config.get_path(dir_key), exist_ok=True)

    # 3. Validate input dataset files
    validator = SelectionValidator()
    is_inputs_valid, input_errors = validator.validate_inputs(config)
    if not is_inputs_valid:
        logger.critical(f"Input validation failed: {input_errors}")
        return False
    logger.info("Input datasets and resources validated.")

    # 4. Load datasets
    try:
        df_base, df_dense, X_sparse, dense_names, sparse_names = load_data(config)
    except Exception as e:
        logger.exception(f"Failed to load datasets: {e}")
        return False

    y = df_base["label"].values
    
    # 5. Instantiate selectors and run fitting
    selections: Dict[str, List[str]] = {}
    rankings: Dict[str, Dict[str, float]] = {}
    runtimes: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    
    # variance selector
    if config.is_selector_enabled("variance"):
        sel_start = time.time()
        try:
            threshold = config.get_selector_setting("variance", "threshold", 0.001)
            selector = VarianceSelector(threshold=threshold)
            selector.fit(df_dense, X_sparse, y, dense_names, sparse_names)
            
            # Save selection results
            name = "variance"
            selections[name] = selector.get_selected_features()
            rankings[name] = selector.get_feature_rankings()
            counts[name] = len(selections[name])
            
            # Save selector model object
            model_path = os.path.join(config.get_path("models_dir"), "variance_selector.joblib")
            joblib.dump(selector, model_path)
            logger.info(f"Saved variance selector model to {model_path}")
        except Exception as e:
            logger.exception(f"Error running variance selector: {e}")
        runtimes["variance"] = time.time() - sel_start

    # correlation selector
    if config.is_selector_enabled("correlation"):
        sel_start = time.time()
        try:
            threshold = config.get_selector_setting("correlation", "threshold", 0.85)
            selector = CorrelationSelector(threshold=threshold)
            selector.fit(df_dense, X_sparse, y, dense_names, sparse_names)
            
            name = "correlation"
            selections[name] = selector.get_selected_features()
            rankings[name] = selector.get_feature_rankings()
            counts[name] = len(selections[name])
            
            model_path = os.path.join(config.get_path("models_dir"), "correlation_selector.joblib")
            joblib.dump(selector, model_path)
            logger.info(f"Saved correlation selector model to {model_path}")
        except Exception as e:
            logger.exception(f"Error running correlation selector: {e}")
        runtimes["correlation"] = time.time() - sel_start

    # mutual information selector
    if config.is_selector_enabled("mutual_information"):
        sel_start = time.time()
        try:
            top_k_dense = config.get_selector_setting("mutual_information", "top_k_dense", 15)
            top_k_sparse = config.get_selector_setting("mutual_information", "top_k_sparse", 150)
            sub_sample_size = config.get_selector_setting("mutual_information", "sub_sample_size", 10000)
            
            selector = MutualInformationSelector(
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse,
                sub_sample_size=sub_sample_size,
                random_state=config.random_state
            )
            selector.fit(df_dense, X_sparse, y, dense_names, sparse_names)
            
            name = "mutual_information"
            selections[name] = selector.get_selected_features()
            rankings[name] = selector.get_feature_rankings()
            counts[name] = len(selections[name])
            
            model_path = os.path.join(config.get_path("models_dir"), "mutual_information_selector.joblib")
            joblib.dump(selector, model_path)
            logger.info(f"Saved mutual information selector model to {model_path}")
        except Exception as e:
            logger.exception(f"Error running mutual information selector: {e}")
        runtimes["mutual_information"] = time.time() - sel_start

    # chi square selector
    if config.is_selector_enabled("chi_square"):
        sel_start = time.time()
        try:
            top_k_dense = config.get_selector_setting("chi_square", "top_k_dense", 15)
            top_k_sparse = config.get_selector_setting("chi_square", "top_k_sparse", 200)
            
            selector = ChiSquareSelector(
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse
            )
            selector.fit(df_dense, X_sparse, y, dense_names, sparse_names)
            
            name = "chi_square"
            selections[name] = selector.get_selected_features()
            rankings[name] = selector.get_feature_rankings()
            counts[name] = len(selections[name])
            
            model_path = os.path.join(config.get_path("models_dir"), "chi_square_selector.joblib")
            joblib.dump(selector, model_path)
            logger.info(f"Saved chi-square selector model to {model_path}")
        except Exception as e:
            logger.exception(f"Error running chi-square selector: {e}")
        runtimes["chi_square"] = time.time() - sel_start

    # random forest selector
    if config.is_selector_enabled("random_forest"):
        sel_start = time.time()
        try:
            top_k = config.get_selector_setting("random_forest", "top_k", 100)
            n_estimators = config.get_selector_setting("random_forest", "n_estimators", 50)
            max_depth = config.get_selector_setting("random_forest", "max_depth", 10)
            pre_selected_pool_size = config.get_selector_setting("random_forest", "pre_selected_pool_size", 300)
            
            selector = RandomForestSelector(
                top_k=top_k,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=config.random_state,
                pre_selected_pool_size=pre_selected_pool_size
            )
            selector.fit(df_dense, X_sparse, y, dense_names, sparse_names)
            
            name = "random_forest"
            selections[name] = selector.get_selected_features()
            rankings[name] = selector.get_feature_rankings()
            counts[name] = len(selections[name])
            
            model_path = os.path.join(config.get_path("models_dir"), "random_forest_selector.joblib")
            joblib.dump(selector, model_path)
            logger.info(f"Saved random forest selector model to {model_path}")
        except Exception as e:
            logger.exception(f"Error running random forest selector: {e}")
        runtimes["random_forest"] = time.time() - sel_start

    # RFE selector
    if config.is_selector_enabled("rfe"):
        sel_start = time.time()
        try:
            n_features_to_select = config.get_selector_setting("rfe", "n_features_to_select", 30)
            step = config.get_selector_setting("rfe", "step", 5)
            pre_selected_pool_size = config.get_selector_setting("rfe", "pre_selected_pool_size", 80)
            
            selector = RFESelector(
                n_features_to_select=n_features_to_select,
                step=step,
                random_state=config.random_state,
                pre_selected_pool_size=pre_selected_pool_size
            )
            selector.fit(df_dense, X_sparse, y, dense_names, sparse_names)
            
            name = "rfe"
            selections[name] = selector.get_selected_features()
            rankings[name] = selector.get_feature_rankings()
            counts[name] = len(selections[name])
            
            model_path = os.path.join(config.get_path("models_dir"), "rfe_selector.joblib")
            joblib.dump(selector, model_path)
            logger.info(f"Saved RFE selector model to {model_path}")
        except Exception as e:
            logger.exception(f"Error running RFE selector: {e}")
        runtimes["rfe"] = time.time() - sel_start

    # 6. Merge selections
    merger = SelectorMerger(
        strategy=config.merger_strategy,
        voting_threshold=config.merger_voting_threshold
    )
    final_selected_features = merger.merge(selections)
    logger.info(f"Final Selection Merger yielded {len(final_selected_features)} features.")
    
    if not final_selected_features:
        logger.critical("No features selected by the pipeline merger! Feature selection failed.")
        return False

    # Separate selected features by type
    selected_dense = [f for f in final_selected_features if f in dense_names]
    selected_sparse = [f for f in final_selected_features if f in sparse_names]
    
    logger.info(f"Final selected features split: Dense={len(selected_dense)}, Sparse/TF-IDF={len(selected_sparse)}")

    # 7. Construct selected dataset and save
    logger.info("Constructing final selected dataset DataFrame...")
    df_output = df_base.copy()
    
    # Add selected dense features
    for col in selected_dense:
        df_output[col] = df_dense[col]
        
    # Add selected TF-IDF features (converting only selected sparse columns to dense format)
    if selected_sparse:
        logger.info(f"Converting {len(selected_sparse)} selected sparse TF-IDF columns to dense format for CSV export...")
        # Get sparse indices
        sparse_name_to_idx = {name: idx for idx, name in enumerate(sparse_names)}
        selected_indices = [sparse_name_to_idx[name] for name in selected_sparse]
        
        # Retrieve columns
        X_selected_sparse = X_sparse[:, selected_indices].toarray()
        
        for idx, name in enumerate(selected_sparse):
            df_output[name] = X_selected_sparse[:, idx]

    output_csv = config.get_path("output_csv")
    logger.info(f"Saving final selected feature dataset to {output_csv}...")
    df_output.to_csv(output_csv, index=False)
    logger.info("Dataset saved successfully.")

    # 8. Save selected names
    selected_names_path = config.get_path("selected_names")
    logger.info(f"Saving selected feature names list to {selected_names_path}...")
    with open(selected_names_path, "w", encoding="utf-8") as f:
        json.dump(final_selected_features, f, indent=4)
        
    # 9. Save rankings
    rankings_path = config.get_path("feature_rankings")
    logger.info(f"Saving selector rankings metadata to {rankings_path}...")
    with open(rankings_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=4)

    # 10. Save summary file
    summary_path = config.get_path("summary_file")
    logger.info(f"Saving summary metadata to {summary_path}...")
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "merger_strategy": config.merger_strategy,
        "voting_threshold": config.merger_voting_threshold,
        "total_dense_features": len(dense_names),
        "total_sparse_features": len(sparse_names),
        "total_features_input": len(dense_names) + len(sparse_names),
        "total_selected_dense": len(selected_dense),
        "total_selected_sparse": len(selected_sparse),
        "total_selected_features": len(final_selected_features),
        "selected_dense_features": selected_dense,
        "selected_sparse_features": selected_sparse,
        "parameters": {
            sel_name: {
                key: config.get_selector_setting(sel_name, key)
                for key in ["enabled", "threshold", "top_k_dense", "top_k_sparse", "sub_sample_size", "top_k", "n_estimators", "max_depth", "n_features_to_select", "step", "pre_selected_pool_size"]
                if config.get_selector_setting(sel_name, key) is not None
            }
            for sel_name in ["variance", "correlation", "mutual_information", "chi_square", "random_forest", "rfe"]
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # 11. Run validator on saved CSV output
    logger.info("Running SelectionValidator on saved CSV output...")
    validation_passed, validation_errors = validator.validate_outputs(
        config.get_path("input_csv"),
        output_csv,
        final_selected_features
    )
    if validation_passed:
        logger.info("Validation checklist passed! Selected dataset is 100% compliant.")
    else:
        logger.error(f"Validation checklist FAILED: {validation_errors}")

    # 12. Run statistics and profiling
    total_time = time.time() - start_time
    stats_collector = SelectionStatistics()
    stats_data = stats_collector.calculate_statistics(
        total_dense_in=len(dense_names),
        total_sparse_in=len(sparse_names),
        selected_dense=selected_dense,
        selected_sparse=selected_sparse,
        selector_runtimes=runtimes,
        selector_counts=counts,
        total_execution_time=total_time
    )
    stats_collector.save(config.get_path("statistics_file"))

    profiler = SelectionProfiler()
    profile_data = profiler.profile_dataset(df_output, final_selected_features)
    # Save selection profile to a dedicated profile.json or append to stats?
    # Let's save profile to a separate file, say selection_profile.json
    profile_path = os.path.join(config.get_path("statistics_dir"), "selection_profile.json")
    profiler.save(profile_path)

    # 13. Calculate SHA-256 hashes of outputs
    logger.info("Registering SHA-256 hashes for output artifacts...")
    hashes = {}
    files_to_hash = {
        "selected_feature_dataset_v1.csv": output_csv,
        "selected_feature_names.json": selected_names_path,
        "feature_rankings.json": rankings_path,
        "feature_selection_summary.json": summary_path,
        "selection_statistics.json": config.get_path("statistics_file"),
        "selection_profile.json": profile_path,
    }
    for key, path in files_to_hash.items():
        hashes[key] = compute_file_sha256(path)
        
    hash_file_path = config.get_path("hash_file")
    with open(hash_file_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=4)
    logger.info(f"Artifact hash registry saved to {hash_file_path}")

    # 14. Save version metadata
    versions_file = config.get_path("versions_file")
    versions = []
    if os.path.exists(versions_file):
        try:
            with open(versions_file, "r", encoding="utf-8") as f:
                versions = json.load(f)
        except Exception:
            logger.warning("Could not read existing version history file, creating fresh.")

    current_version = {
        "version": len(versions) + 1,
        "timestamp": datetime.now().isoformat(),
        "total_selected_features": len(final_selected_features),
        "merger_strategy": config.merger_strategy,
        "reduction_rate_pct": stats_data["summary"]["reduction_percentage"],
        "files": hashes
    }
    versions.append(current_version)
    with open(versions_file, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=4)
        
    # Also save as a separate version stamp in versions/ directory
    version_stamp_path = os.path.join(
        config.get_path("versions_dir"), 
        f"selection_version_{len(versions)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(version_stamp_path, "w", encoding="utf-8") as f:
        json.dump(current_version, f, indent=4)
    logger.info(f"Version metadata logged to {versions_file} and {version_stamp_path}")

    # 15. Generate Quality Report
    logger.info("Generating Markdown Quality Report...")
    report_content = generate_markdown_report(
        config=config,
        stats=stats_data,
        profile=profile_data,
        rankings=rankings,
        selected_dense=selected_dense,
        selected_sparse=selected_sparse,
        hashes=hashes,
        validation_passed=validation_passed,
        validation_errors=validation_errors
    )
    
    report_file_path = config.get_path("report_file")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Markdown report generated successfully at {report_file_path}")

    final_mem, peak_mem = get_memory_usage()
    logger.info(f"Feature Selection completed successfully. Time: {total_time:.2f}s. Memory: {final_mem:.2f}MB (Peak: {peak_mem:.2f}MB)")
    logger.info("==================================================")
    
    return validation_passed

if __name__ == "__main__":
    success = run_feature_selection_pipeline()
    sys.exit(0 if success else 1)
