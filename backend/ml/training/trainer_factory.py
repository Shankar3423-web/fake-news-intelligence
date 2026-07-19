import logging
from typing import Dict, Any
from ml.training.training_utils import BaseModelTrainer
from ml.training.logistic_regression_trainer import LogisticRegressionTrainer
from ml.training.svm_trainer import SvmTrainer
from ml.training.random_forest_trainer import RandomForestTrainer
from ml.training.xgboost_trainer import XgBoostTrainer

logger = logging.getLogger("model_training_pipeline")

class TrainerFactory:
    """
    Factory class to instantiate the appropriate trainer based on model type.
    """
    @staticmethod
    def create_trainer(model_type: str, hyperparameters: Dict[str, Any]) -> BaseModelTrainer:
        """
        Creates and returns a trainer instance for the specified model type.
        
        Args:
            model_type: Key for the trainer (logistic_regression, svm, random_forest, xgboost)
            hyperparameters: Dict of hyperparams for the model
            
        Returns:
            An instance of BaseModelTrainer
            
        Raises:
            ValueError: If the model_type is not supported.
        """
        model_type_clean = model_type.strip().lower()
        logger.debug(f"Creating trainer for model type: '{model_type_clean}'")
        
        if model_type_clean == "logistic_regression":
            return LogisticRegressionTrainer(hyperparameters)
        elif model_type_clean == "svm":
            return SvmTrainer(hyperparameters)
        elif model_type_clean == "random_forest":
            return RandomForestTrainer(hyperparameters)
        elif model_type_clean == "xgboost":
            return XgBoostTrainer(hyperparameters)
        else:
            supported = ["logistic_regression", "svm", "random_forest", "xgboost"]
            raise ValueError(
                f"Unsupported model type: '{model_type}'. Supported models: {supported}"
            )
