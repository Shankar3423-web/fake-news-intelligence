import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("model_training_pipeline")

class DatasetSplitter:
    """
    Handles splitting the dataset into training and testing sets.
    Ensures reproducibility and stratification, and collects split metadata.
    """
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        shuffle: bool = True
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.shuffle = shuffle
        
        # Split statistics to be stored after split is performed
        self.split_info: Dict[str, Any] = {}

    def split(
        self,
        df: pd.DataFrame,
        features: List[str],
        label_col: str = "label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and testing sets.
        
        Args:
            df: Input DataFrame containing features and label
            features: List of feature names to split on
            label_col: Name of the label column
            
        Returns:
            X_train: training features DataFrame
            X_test: testing features DataFrame
            y_train: training labels Series
            y_test: testing labels Series
        """
        logger.info(f"Splitting dataset: test_size={self.test_size}, random_state={self.random_state}, stratify={self.stratify}")
        
        X = df[features]
        y = df[label_col]
        
        # Determine stratify parameter
        stratify_y = y if self.stratify else None
        
        # Perform split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_y,
            shuffle=self.shuffle
        )
        
        # Record stats
        timestamp = datetime.now(timezone.utc).isoformat()
        
        train_counts = y_train.value_counts().to_dict()
        test_counts = y_test.value_counts().to_dict()
        
        # Ensure standard python types for JSON serialization
        train_counts = {int(k): int(v) for k, v in train_counts.items()}
        test_counts = {int(k): int(v) for k, v in test_counts.items()}
        
        self.split_info = {
            "timestamp": timestamp,
            "config": {
                "test_size": self.test_size,
                "stratify": self.stratify,
                "shuffle": self.shuffle,
                "random_state": self.random_state
            },
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "label_distribution": {
                "train": train_counts,
                "test": test_counts
            }
        }
        
        logger.info(f"Split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Train label distribution: {train_counts}")
        logger.info(f"Test label distribution: {test_counts}")
        
        return X_train, X_test, y_train, y_test
        
    def get_split_info(self) -> Dict[str, Any]:
        """Returns metadata about the split. Raises ValueError if split hasn't been run."""
        if not self.split_info:
            raise ValueError("Dataset has not been split yet. Call split() first.")
        return self.split_info
