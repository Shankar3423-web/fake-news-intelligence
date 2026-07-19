import os
import logging
import pandas as pd
from typing import Dict, Any
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.evaluation_utils import BaseModelEvaluator
from ml.evaluation.model_loader import ModelLoader
from ml.evaluation.prediction_engine import PredictionEngine
from ml.evaluation.metrics_calculator import MetricsCalculator

logger = logging.getLogger("model_evaluation_pipeline")

class LogisticRegressionEvaluator(BaseModelEvaluator):
    """
    Evaluator implementation for Logistic Regression.
    """
    def __init__(self, config: EvaluationConfig) -> None:
        super().__init__(
            model_key="logistic_regression",
            algorithm_name="Logistic Regression",
            config=config
        )

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        logger.info(f"Starting evaluation for {self.algorithm_name}...")
        
        # 1. Load Model
        loader = ModelLoader()
        self.model, self.metadata, self.feature_order = loader.load_model(
            self.model_key, list(X_test.columns)
        )
        
        # 2. Get Model Size on disk
        model_path = os.path.join("ml/training/models", self.model_key, "model.joblib")
        model_size_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        
        # 3. Predict & Profile
        engine = PredictionEngine(self.model)
        pred_outputs = engine.predict(X_test)
        
        # 4. Calculate metrics
        calc = MetricsCalculator()
        metrics = calc.calculate_metrics(
            y_true=y_test,
            y_pred=pred_outputs["y_pred"],
            y_prob=pred_outputs["y_prob"],
            y_prob_type=pred_outputs["y_prob_type"],
            duration_sec=pred_outputs["prediction_duration_sec"],
            throughput_sps=pred_outputs["inference_throughput_sps"],
            memory_used_mb=pred_outputs["memory_used_mb"],
            model_size_bytes=model_size_bytes
        )
        
        logger.info(f"Evaluation complete for {self.algorithm_name}. F1-Score: {metrics.get('f1_score'):.4f}")
        
        return {
            "model_key": self.model_key,
            "model_id": self.metadata.get("model_name", self.model_key),  # wait! Registry lists model ID, let's also support custom model ID if found, or default
            "algorithm": self.algorithm_name,
            "metadata": self.metadata,
            "model_size_bytes": model_size_bytes,
            "predictions": pred_outputs,
            "metrics": metrics
        }
