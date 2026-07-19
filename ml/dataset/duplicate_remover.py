import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Set, List
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

logger = logging.getLogger(__name__)

class DuplicateRemover:
    """
    Removes duplicate news articles from a DataFrame in a memory-safe manner.
    Detects duplicates based on:
    - Exact match of title
    - Exact match of text
    - Batch-wise near-duplicate detection using TF-IDF and Cosine Similarity (prevents MemoryError)
    """

    def __init__(self) -> None:
        pass

    def remove_duplicates(
        self, 
        df: pd.DataFrame, 
        similarity_threshold: float = 0.95,
        batch_size: int = 2000,
        max_features: int = 5000
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detects and removes duplicates in a scalable, memory-safe way.
        
        Args:
            df: Standardized DataFrame to clean.
            similarity_threshold: Cosine similarity threshold (0.0 to 1.0) above which articles
                                  are considered duplicates. Default is 0.95.
            batch_size: Size of batches to process for TF-IDF similarity (Option 2).
            max_features: Max features for TF-IDF vectorization.
            
        Returns:
            A tuple of (deduplicated DataFrame, deduplication report dict).
        """
        start_time = time.time()
        initial_count = len(df)
        logger.info(f"Starting duplicate detection. Initial rows: {initial_count}, Batch Size: {batch_size}")

        report: Dict[str, Any] = {
            "initial_rows": initial_count,
            "exact_title_duplicates_removed": 0,
            "exact_text_duplicates_removed": 0,
            "similarity_duplicates_removed": 0,
            "removed_records": [],  # List of dicts representing removed rows with reasons
            "duplicate_pairs_sampled": [],
            "final_rows": 0,
            "time_taken_seconds": 0.0,
            "semantic_deduplication_skipped": False
        }

        if df.empty:
            report["final_rows"] = 0
            report["time_taken_seconds"] = time.time() - start_time
            return df, report

        # Create working copies for matching
        df_work = df.copy()
        df_work["title_clean"] = df_work["title"].fillna("").astype(str).str.strip().str.lower()
        df_work["text_clean"] = df_work["text"].fillna("").astype(str).str.strip().str.lower()

        indices_to_drop: Set[int] = set()

        # --- Step 1: Remove Exact Title Duplicates ---
        # Keep first, drop others (only for non-empty titles)
        non_empty_title_mask = df_work["title_clean"] != ""
        title_dups = df_work[non_empty_title_mask].duplicated(subset=["title_clean"], keep="first")
        exact_title_dup_indices = df_work[non_empty_title_mask][title_dups].index
        
        for idx in exact_title_dup_indices:
            indices_to_drop.add(idx)
            # Find the original row it duplicated (for reason reporting)
            duplicate_val = df_work.loc[idx, "title_clean"]
            original_row = df_work[df_work["title_clean"] == duplicate_val].iloc[0]
            
            report["removed_records"].append({
                "id": df_work.loc[idx, "id"],
                "title": df_work.loc[idx, "title"],
                "text": df_work.loc[idx, "text"],
                "label": df_work.loc[idx, "label"],
                "source": df_work.loc[idx, "source"],
                "category": df_work.loc[idx, "category"],
                "published_date": df_work.loc[idx, "published_date"],
                "dataset_origin": df_work.loc[idx, "dataset_origin"],
                "removal_reason": f"Exact duplicate title of record ID {original_row['id']}"
            })
            
        report["exact_title_duplicates_removed"] = len(exact_title_dup_indices)
        logger.info(f"Identified {len(exact_title_dup_indices)} exact title duplicates.")

        # --- Step 2: Remove Exact Text Duplicates ---
        # Remaining rows not already marked for dropping
        remaining_indices = [idx for idx in df_work.index if idx not in indices_to_drop]
        df_temp = df_work.loc[remaining_indices]
        
        non_empty_text_mask = df_temp["text_clean"] != ""
        text_dups = df_temp[non_empty_text_mask].duplicated(subset=["text_clean"], keep="first")
        exact_text_dup_indices = df_temp[non_empty_text_mask][text_dups].index
        
        for idx in exact_text_dup_indices:
            indices_to_drop.add(idx)
            duplicate_val = df_work.loc[idx, "text_clean"]
            original_row = df_work[df_work["text_clean"] == duplicate_val].iloc[0]
            
            report["removed_records"].append({
                "id": df_work.loc[idx, "id"],
                "title": df_work.loc[idx, "title"],
                "text": df_work.loc[idx, "text"],
                "label": df_work.loc[idx, "label"],
                "source": df_work.loc[idx, "source"],
                "category": df_work.loc[idx, "category"],
                "published_date": df_work.loc[idx, "published_date"],
                "dataset_origin": df_work.loc[idx, "dataset_origin"],
                "removal_reason": f"Exact duplicate text of record ID {original_row['id']}"
            })
            
        report["exact_text_duplicates_removed"] = len(exact_text_dup_indices)
        logger.info(f"Identified {len(exact_text_dup_indices)} exact text duplicates.")

        # --- Step 3: Scalable Batch-wise Near-Duplicate Detection (Option 2 & Option 3 fallback) ---
        remaining_indices = [idx for idx in df_work.index if idx not in indices_to_drop]
        df_remaining = df_work.loc[remaining_indices].copy()
        
        # Combine title and text for similarity checking
        df_remaining["combined_text"] = df_remaining["title_clean"] + " " + df_remaining["text_clean"]
        valid_rows = df_remaining[df_remaining["combined_text"].str.strip().str.len() > 10].copy()

        similarity_dup_indices: Set[int] = set()

        if len(valid_rows) > 1:
            logger.info(f"Running memory-safe batch duplicate detection on {len(valid_rows)} articles.")
            try:
                # Divide valid_rows into chunks of batch_size
                num_chunks = int(np.ceil(len(valid_rows) / batch_size))
                sampled_count = 0
                
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * batch_size
                    chunk_end = min(chunk_start + batch_size, len(valid_rows))
                    
                    chunk_df = valid_rows.iloc[chunk_start:chunk_end]
                    if len(chunk_df) <= 1:
                        continue
                        
                    # Vectorize chunk
                    vectorizer = TfidfVectorizer(
                        max_features=max_features, 
                        stop_words="english",
                        lowercase=True
                    )
                    tfidf_matrix = vectorizer.fit_transform(chunk_df["combined_text"])
                    
                    # Compute Cosine Similarity for this chunk
                    similarity_sparse = tfidf_matrix @ tfidf_matrix.T
                    similarity_sparse = sp.triu(similarity_sparse, k=1)
                    
                    # Find coordinates
                    rows, cols = similarity_sparse.nonzero()
                    scores = similarity_sparse.data
                    
                    # Map position inside chunk to original dataframe index
                    chunk_indices = chunk_df.index.tolist()
                    
                    for r_pos, c_pos, score in zip(rows, cols, scores):
                        if score >= similarity_threshold:
                            r_idx = chunk_indices[r_pos]
                            c_idx = chunk_indices[c_pos]
                            
                            # Mark c_idx as duplicate of r_idx
                            if r_idx not in similarity_dup_indices and c_idx not in similarity_dup_indices:
                                similarity_dup_indices.add(c_idx)
                                indices_to_drop.add(c_idx)
                                
                                report["removed_records"].append({
                                    "id": df_work.loc[c_idx, "id"],
                                    "title": df_work.loc[c_idx, "title"],
                                    "text": df_work.loc[c_idx, "text"],
                                    "label": df_work.loc[c_idx, "label"],
                                    "source": df_work.loc[c_idx, "source"],
                                    "category": df_work.loc[c_idx, "category"],
                                    "published_date": df_work.loc[c_idx, "published_date"],
                                    "dataset_origin": df_work.loc[c_idx, "dataset_origin"],
                                    "removal_reason": f"Near-duplicate (similarity {round(score, 4)} >= {similarity_threshold}) of record ID {df_work.loc[r_idx, 'id']}"
                                })
                                
                                # Sample pairs for metadata reporting
                                if sampled_count < 10:
                                    report["duplicate_pairs_sampled"].append({
                                        "original_id": str(df_work.loc[r_idx, "id"]),
                                        "original_title": str(df_work.loc[r_idx, "title"])[:100],
                                        "duplicate_id": str(df_work.loc[c_idx, "id"]),
                                        "duplicate_title": str(df_work.loc[c_idx, "title"])[:100],
                                        "similarity_score": float(score)
                                    })
                                    sampled_count += 1
                
                report["similarity_duplicates_removed"] = len(similarity_dup_indices)
                logger.info(f"Identified {len(similarity_dup_indices)} near-duplicates in batches.")
                
            except Exception as e:
                # Option 3 fallback: Skip semantic duplicate detection, log warning, never crash
                logger.warning(
                    f"Semantic duplicate detection failed due to resource constraints/error: {str(e)}. "
                    "Skipping similarity deduplication and continuing execution."
                )
                report["semantic_deduplication_skipped"] = True
                
        # Drop all collected duplicates
        df_final = df.drop(index=list(indices_to_drop))
        
        final_count = len(df_final)
        report["final_rows"] = final_count
        report["time_taken_seconds"] = round(time.time() - start_time, 4)

        logger.info(
            f"Deduplication finished. Final rows: {final_count}. "
            f"Total removed: {initial_count - final_count} "
            f"({round((initial_count - final_count) / initial_count * 100, 2) if initial_count > 0 else 0}%)."
        )
        return df_final, report
