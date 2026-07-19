import logging
from typing import Dict, Type
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.evaluation_utils import BaseModelEvaluator
from ml.evaluation.logistic_regression_evaluator import LogisticRegressionEvaluator
from ml.evaluation.svm_evaluator import SvmEvaluator
from ml.evaluation.random_forest_evaluator import RandomForestEvaluator
from ml.evaluation.xgboost_evaluator import XgboostEvaluator

logger = logging.getLogger("model_evaluation_pipeline")

class EvaluatorFactory:
    """
    Factory pattern class to instantiate the appropriate BaseModelEvaluator subclass.
    """
    _evaluators: Dict[str, Type[BaseModelEvaluator]] = {
        "logistic_regression": LogisticRegressionEvaluator,
        "svm": SvmEvaluator,
        "random_forest": RandomForestEvaluator,
        "xgboost": XgboostEvaluator
    }

    @classmethod
    def create_evaluator(cls, model_key: str, config: EvaluationConfig) -> BaseModelEvaluator:
        """
        Creates and returns an evaluator instance based on the model_key.
        """
        key = model_key.lower().strip()
        if key not in cls._evaluators:
            logger.error(f"Unsupported model evaluator key: '{model_key}'")
            raise ValueError(f"Unsupported model evaluator key: '{model_key}'. Allowed: {list(cls._evaluators.keys())}")
            
        evaluator_cls = cls._evaluators[key]
        logger.debug(f"Instantiating evaluator for model key '{key}' using {evaluator_cls.__name__}...")
        return evaluator_cls(config)
