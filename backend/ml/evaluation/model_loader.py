import os
import json
import logging
import joblib
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("model_evaluation_pipeline")

class ModelLoader:
    """
    Loads each trained classifier produced in Phase 6 and performs validations:
    - Verifies file existence of model.joblib, metadata.json, feature_order.json.
    - Validates feature order compatibility with the current testing features.
    - Validates model compatibility.
    - Returns loaded model object along with its metadata.
    """
    def __init__(self, models_root_dir: str = "ml/training/models") -> None:
        self.models_root_dir = models_root_dir

    def load_model(self, model_key: str, expected_features: List[str]) -> Tuple[Any, Dict[str, Any], List[str]]:
        """
        Loads a trained model and its associated metadata.
        
        Args:
            model_key: Key of the model (e.g. 'logistic_regression', 'svm', 'random_forest', 'xgboost')
            expected_features: List of expected features from the current dataset split
            
        Returns:
            Tuple of:
                model: Ready-to-use scikit-learn or xgboost model object
                metadata: Model metadata dictionary
                feature_order: List of features expected by the model
                
        Raises:
            FileNotFoundError: If any of the required files are missing.
            ValueError: If metadata or feature orders are incompatible.
        """
        model_dir = os.path.join(self.models_root_dir, model_key)
        
        # 1. Validate files existence
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found for: {model_key} at {model_dir}")
            
        model_path = os.path.join(model_dir, "model.joblib")
        metadata_path = os.path.join(model_dir, "metadata.json")
        features_path = os.path.join(model_dir, "feature_order.json")
        
        for p in [model_path, metadata_path, features_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required model file not found: {p}")

        # 2. Load model
        logger.info(f"Loading joblib model object from {model_path}...")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load joblib model for {model_key}: {e}")
            
        if model is None:
            raise ValueError(f"Loaded model object is None for {model_key}")

        # 3. Load feature order
        logger.info(f"Loading feature order from {features_path}...")
        try:
            with open(features_path, "r", encoding="utf-8") as f:
                feature_order = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse feature_order.json for {model_key}: {e}")
            
        if not isinstance(feature_order, list) or len(feature_order) == 0:
            raise ValueError(f"Feature order for {model_key} must be a non-empty list of strings.")

        # 4. Load metadata
        logger.info(f"Loading metadata from {metadata_path}...")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse metadata.json for {model_key}: {e}")

        # 5. Validate feature order compatibility
        if len(feature_order) != len(expected_features):
            raise ValueError(
                f"Feature count mismatch for model '{model_key}'. "
                f"Model expects {len(feature_order)} features, but test dataset has {len(expected_features)}."
            )
            
        # Verify feature set match
        mismatched_features = [f for f in feature_order if f not in expected_features]
        if mismatched_features:
            raise ValueError(
                f"Feature names mismatch for model '{model_key}'. "
                f"Features expected by model but missing in test dataset: {mismatched_features[:10]}"
            )
            
        # Verify exact order match
        for idx, (f_model, f_data) in enumerate(zip(feature_order, expected_features)):
            if f_model != f_data:
                raise ValueError(
                    f"Feature ordering mismatch for model '{model_key}' at index {idx}. "
                    f"Model expects '{f_model}', but dataset has '{f_data}'."
                )

        logger.info(f"Model '{model_key}' successfully loaded and verified compatibility.")
        return model, metadata, feature_order
