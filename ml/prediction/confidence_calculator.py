import numpy as np
import logging
from typing import Any

logger = logging.getLogger("prediction_pipeline")

class ConfidenceCalculator:
    """
    Calculates prediction confidence scores in the range [0.0, 1.0].
    Handles both probability-based and margin-based classifiers.
    """
    def __init__(self) -> None:
        pass

    def calculate(self, model: Any, feature_vector: Any, prediction: int) -> float:
        """
        Calculates the confidence score of a prediction.
        
        Args:
            model: The classifier model object.
            feature_vector: The feature vector (DataFrame or array) used for prediction.
            prediction: The predicted class label (0 or 1).
            
        Returns:
            A float confidence score between 0.0 and 1.0.
        """
        # 1. Try predict_proba
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(feature_vector)
                # For binary classification, proba[0] has two elements: [p_class_0, p_class_1]
                # Ensure index safety
                if len(proba) > 0 and len(proba[0]) > prediction:
                    p_class = proba[0][prediction]
                    return float(np.clip(p_class, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Failed to calculate confidence using predict_proba: {e}. Falling back.")

        # 2. Try decision_function
        if hasattr(model, "decision_function"):
            try:
                margin = model.decision_function(feature_vector)
                # For binary classification, decision_function returns a single float (margin for class 1)
                val = margin[0] if hasattr(margin, "__len__") else margin
                
                # Apply sigmoid function to map margin to [0, 1] probability of class 1
                p_class_1 = 1.0 / (1.0 + np.exp(-val))
                p_class_0 = 1.0 - p_class_1
                
                p_class = p_class_1 if prediction == 1 else p_class_0
                return float(np.clip(p_class, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Failed to calculate confidence using decision_function: {e}. Falling back.")

        # 3. Fallback: Default to 1.0 if no confidence calculation is possible
        logger.warning("Model does not support predict_proba or decision_function. Defaulting confidence to 1.0.")
        return 1.0
