import os
import json
import logging
import joblib
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("prediction_pipeline")

class ModelLoader:
    """
    Loads the best trained classifier automatically using ml/evaluation/best_model.json.
    Verifies that the model files exist and returns the model, metadata, and feature ordering.
    """
    def __init__(self, best_model_json_path: str = "ml/evaluation/best_model.json", models_root_dir: str = "ml/training/models") -> None:
        self.best_model_json_path = best_model_json_path
        self.models_root_dir = models_root_dir

    def load_best_model(self) -> Tuple[Any, Dict[str, Any], List[str], Dict[str, Any]]:
        """
        Loads the best model determined by evaluation.
        
        Returns:
            Tuple containing:
                - model: Ready-to-use scikit-learn or xgboost model object
                - metadata: Model training metadata dictionary
                - feature_order: List of features expected by the model
                - best_model_info: Contents of best_model.json
        """
        logger.info(f"Loading best model registry from {self.best_model_json_path}...")
        if not os.path.exists(self.best_model_json_path):
            raise FileNotFoundError(f"Best model descriptor file not found: {self.best_model_json_path}")

        try:
            with open(self.best_model_json_path, "r", encoding="utf-8") as f:
                best_model_info = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse best_model.json: {e}")

        model_key = best_model_info.get("model_key")
        if not model_key:
            raise ValueError("best_model.json does not specify 'model_key'.")

        model_dir = os.path.join(self.models_root_dir, model_key)
        if not os.path.exists(model_dir):
            # Fall back to evaluation metadata path if directory relative to root is not found
            model_dir = best_model_info.get("path", model_dir)

        model_path = os.path.join(model_dir, "model.joblib")
        metadata_path = os.path.join(model_dir, "metadata.json")
        features_path = os.path.join(model_dir, "feature_order.json")

        for path in [model_path, metadata_path, features_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required model file not found: {path}")

        logger.info(f"Loading model object from {model_path}...")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model file {model_path}: {e}")

        logger.info(f"Loading feature order from {features_path}...")
        try:
            with open(features_path, "r", encoding="utf-8") as f:
                feature_order = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse feature order file {features_path}: {e}")

        if not isinstance(feature_order, list) or len(feature_order) == 0:
            raise ValueError("Feature order must be a non-empty list of feature names.")

        logger.info(f"Loading model metadata from {metadata_path}...")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse model metadata file {metadata_path}: {e}")

        logger.info(f"Successfully loaded model '{model_key}' (Algorithm: {best_model_info.get('algorithm')}).")
        return model, metadata, feature_order, best_model_info

    def load_all_models(self) -> List[Dict[str, Any]]:
        """
        Loads ALL trained models from the models root directory.
        
        Returns:
            List of dicts, each containing:
                - model_key: Name of the model (e.g., 'svm', 'logistic_regression')
                - model: Ready-to-use classifier object
                - metadata: Model training metadata
                - feature_order: List of features expected by the model
        """
        logger.info(f"Loading all models from {self.models_root_dir}...")
        all_models = []
        
        if not os.path.exists(self.models_root_dir):
            logger.warning(f"Models root directory not found: {self.models_root_dir}")
            return all_models
        
        for model_key in sorted(os.listdir(self.models_root_dir)):
            model_dir = os.path.join(self.models_root_dir, model_key)
            if not os.path.isdir(model_dir):
                continue
            
            model_path = os.path.join(model_dir, "model.joblib")
            metadata_path = os.path.join(model_dir, "metadata.json")
            features_path = os.path.join(model_dir, "feature_order.json")
            
            # Skip if any required file is missing
            if not all(os.path.exists(p) for p in [model_path, metadata_path, features_path]):
                logger.warning(f"Skipping model '{model_key}': missing required files.")
                continue
            
            try:
                model = joblib.load(model_path)
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                with open(features_path, "r", encoding="utf-8") as f:
                    feature_order = json.load(f)
                
                all_models.append({
                    "model_key": model_key,
                    "model": model,
                    "metadata": metadata,
                    "feature_order": feature_order
                })
                logger.info(f"  Loaded model: '{model_key}' (Algorithm: {metadata.get('algorithm', 'unknown')})")
            except Exception as e:
                logger.warning(f"Failed to load model '{model_key}': {e}")
                continue
        
        logger.info(f"Successfully loaded {len(all_models)} models total.")
        return all_models
