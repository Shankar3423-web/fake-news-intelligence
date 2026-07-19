import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("model_training_pipeline")

class ModelRegistry:
    """
    Manages the model registry JSON file.
    Tracks model ID, algorithm, version, dataset/feature selection details,
    training dates, feature counts, and absolute or relative paths.
    """
    def __init__(self, registry_path: str = "ml/training/registry.json") -> None:
        self.registry_path = registry_path

    def load_registry(self) -> Dict[str, Any]:
        """
        Loads the registry JSON. If the file doesn't exist, returns a default empty structure.
        """
        if not os.path.exists(self.registry_path):
            logger.debug(f"Registry file not found at {self.registry_path}. Creating new registry structure.")
            return {
                "latest_model_id": "",
                "models": []
            }
        
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
                if not isinstance(registry, dict) or "models" not in registry:
                    logger.warning("Malformed registry JSON. Re-initializing.")
                    return {"latest_model_id": "", "models": []}
                return registry
        except Exception as e:
            logger.error(f"Error loading registry from {self.registry_path}: {e}. Returning empty registry.")
            return {"latest_model_id": "", "models": []}

    def register_model(
        self,
        model_id: str,
        algorithm: str,
        version: str,
        dataset_version: str,
        training_date: str,
        feature_count: int,
        training_samples: int,
        testing_samples: int,
        path: str
    ) -> Dict[str, Any]:
        """
        Appends a model registration entry and updates the 'latest_model_id' reference.
        
        Returns:
            The updated registry dictionary.
        """
        registry = self.load_registry()
        
        # Build new entry
        entry = {
            "id": model_id,
            "algorithm": algorithm,
            "version": version,
            "dataset_version": dataset_version,
            "training_date": training_date,
            "feature_count": feature_count,
            "training_samples": training_samples,
            "testing_samples": testing_samples,
            "path": path
        }
        
        # Check if model ID already exists, overwrite if it does
        registry["models"] = [m for m in registry["models"] if m.get("id") != model_id]
        registry["models"].append(entry)
        registry["latest_model_id"] = model_id
        
        self.save_registry(registry)
        logger.info(f"Registered model '{model_id}' ({algorithm}) under version {version} in registry.")
        return registry

    def save_registry(self, registry_data: Dict[str, Any]) -> None:
        """
        Writes the registry data back to the JSON file.
        """
        # Ensure parent directory exists
        parent_dir = os.path.dirname(self.registry_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write registry to {self.registry_path}: {e}")
            raise

    def get_model_entry(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a registered model entry by its ID."""
        registry = self.load_registry()
        for m in registry.get("models", []):
            if m.get("id") == model_id:
                return m
        return None
