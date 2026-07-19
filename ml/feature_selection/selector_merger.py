import logging
from typing import List, Dict, Set, Any

logger = logging.getLogger("feature_selection_pipeline")

class SelectorMerger:
    """
    Merges selected features from different feature selection strategies.
    Supports strategies:
    - union: any feature selected by at least one selector
    - intersection: features selected by all selectors
    - voting: features selected by at least N selectors
    """
    def __init__(self, strategy: str = "voting", voting_threshold: int = 2) -> None:
        self.strategy = strategy
        self.voting_threshold = voting_threshold

    def merge(self, selections: Dict[str, List[str]]) -> List[str]:
        """
        Merges feature selections from different selectors.
        
        Args:
            selections: Dict mapping selector names to their selected feature lists.
            
        Returns:
            List of final selected feature names.
        """
        if not selections:
            logger.warning("No selections provided to merger. Returning empty list.")
            return []

        logger.info(f"Merging feature selections using strategy='{self.strategy}' (voting_threshold={self.voting_threshold if self.strategy == 'voting' else 'N/A'})...")
        
        for name, features in selections.items():
            logger.info(f"Selector '{name}' selected {len(features)} features.")

        if self.strategy == "union":
            merged_set: Set[str] = set()
            for features in selections.values():
                merged_set.update(features)
            merged_list = sorted(list(merged_set))
            logger.info(f"Union merger completed: selected {len(merged_list)} unique features.")
            return merged_list

        elif self.strategy == "intersection":
            # Initialize with first selector's features
            iterator = iter(selections.values())
            try:
                merged_set = set(next(iterator))
            except StopIteration:
                return []
                
            for features in iterator:
                merged_set.intersection_update(features)
            merged_list = sorted(list(merged_set))
            logger.info(f"Intersection merger completed: selected {len(merged_list)} features.")
            return merged_list

        elif self.strategy == "voting":
            # Count votes for each feature
            votes: Dict[str, int] = {}
            for features in selections.values():
                for feat in features:
                    votes[feat] = votes.get(feat, 0) + 1

            # Select features with votes >= threshold
            selected_features = [feat for feat, count in votes.items() if count >= self.voting_threshold]
            merged_list = sorted(selected_features)
            
            logger.info(f"Voting merger completed: selected {len(merged_list)} features with >= {self.voting_threshold} votes.")
            # Log some voting statistics
            vote_counts = {}
            for count in sorted(set(votes.values()), reverse=True):
                num_feats = sum(1 for feat, v in votes.items() if v == count)
                vote_counts[count] = num_feats
                logger.info(f"Features with exactly {count} votes: {num_feats}")
                
            return merged_list

        else:
            raise ValueError(f"Unknown merge strategy: {self.strategy}")
