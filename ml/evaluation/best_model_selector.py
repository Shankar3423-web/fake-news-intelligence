import os
import json
import logging
from typing import Dict, Any, Tuple
from ml.evaluation.evaluation_config import EvaluationConfig

logger = logging.getLogger("model_evaluation_pipeline")

class BestModelSelector:
    """
    Selects the best production-ready model based on a configurable scoring strategy.
    Supports either a single metric selector or a multi-metric weighted score selector.
    Saves outputs to best_model.json.
    """
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def calculate_overall_scores(self, model_eval_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates the overall evaluation score for every model.
        Supports weighted average scaling (normalized throughput is used for speed).
        """
        metric_name = self.config.selection_metric
        logger.info(f"Computing overall scores using selection metric: '{metric_name}'")
        
        scores: Dict[str, float] = {}

        if metric_name == "weighted_score":
            weights = self.config.selection_weights
            logger.info(f"Using weights: {weights}")
            
            # Extract throughputs to normalize prediction speed
            throughputs: Dict[str, float] = {}
            for key, res in model_eval_results.items():
                throughputs[key] = float(res["metrics"]["inference_throughput_sps"])
                
            max_throughput = max(throughputs.values()) if throughputs else 1.0
            if max_throughput <= 0:
                max_throughput = 1.0
                
            for key, res in model_eval_results.items():
                metrics = res["metrics"]
                
                f1 = metrics["f1_score"]
                roc = metrics["roc_auc"]
                prec = metrics["precision"]
                rec = metrics["recall"]
                
                # Speed is normalized throughput (higher is better)
                norm_speed = throughputs[key] / max_throughput
                
                weighted_score = (
                    f1 * weights.get("f1_score", 0.40) +
                    roc * weights.get("roc_auc", 0.30) +
                    prec * weights.get("precision", 0.15) +
                    rec * weights.get("recall", 0.10) +
                    norm_speed * weights.get("prediction_speed", 0.05)
                )
                
                scores[key] = round(weighted_score, 6)
                logger.info(
                    f"Model '{key}' weighted score: {scores[key]:.6f} "
                    f"(F1={f1:.4f}, ROC={roc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Speed_Norm={norm_speed:.4f})"
                )
        else:
            # Select by a single metric directly
            # e.g., "f1_score", "accuracy", "roc_auc"
            for key, res in model_eval_results.items():
                metrics = res["metrics"]
                # Safe fallback if metric key doesn't exist
                if metric_name in metrics:
                    scores[key] = float(metrics[metric_name] if metrics[metric_name] is not None else 0.0)
                elif f"{metric_name}_score" in metrics:
                    scores[key] = float(metrics[f"{metric_name}_score"])
                else:
                    logger.warning(f"Metric '{metric_name}' not found. Defaulting to F1 Score.")
                    scores[key] = float(metrics["f1_score"])
                    
                logger.info(f"Model '{key}' score: {scores[key]:.6f} (direct metric '{metric_name}')")
                
        return scores

    def select_best_model(
        self,
        model_eval_results: Dict[str, Dict[str, Any]],
        overall_scores: Dict[str, float]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Identifies the model with the highest overall score, saves metadata to best_model.json,
        and returns the best model key and summary dictionary.
        """
        if not overall_scores:
            raise ValueError("No model scores calculated.")
            
        best_model_key = max(overall_scores, key=lambda k: overall_scores[k])
        best_score = overall_scores[best_model_key]
        
        best_res = model_eval_results[best_model_key]
        
        logger.info(f"Selected best model: '{best_model_key}' with score {best_score:.6f}")
        
        # Build structure for best_model.json
        best_model_data = {
            "model_key": best_model_key,
            "model_id": best_res["model_id"],
            "algorithm": best_res["algorithm"],
            "selection_metric_used": self.config.selection_metric,
            "overall_score": best_score,
            "metrics": best_res["metrics"],
            "path": best_res["metadata"].get("path", f"ml/training/models/{best_model_key}"),
            "feature_count": len(best_res["feature_order"]) if "feature_order" in best_res else 0,
            "testing_samples": len(best_res["predictions"]["y_pred"]) if "predictions" in best_res else 0
        }
        
        best_model_path = self.config.get_output_path("best_model_file")
        parent_dir = os.path.dirname(best_model_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        try:
            with open(best_model_path, "w", encoding="utf-8") as f:
                json.dump(best_model_data, f, indent=4)
            logger.info(f"Saved best model details to {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to save best model details: {e}")
            raise
            
        return best_model_key, best_model_data
