import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Dict, Any, Optional

logger = logging.getLogger("model_evaluation_pipeline")

class RocAucGenerator:
    """
    Generates ROC-AUC curves and saves data files for models.
    Saves outputs as JSON (fpr, tpr, thresholds, AUC) and PNG visualization.
    """
    def __init__(self, output_dir: str = "ml/evaluation/roc_curves") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, model_key: str, y_true: np.ndarray, y_prob: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Computes ROC curve values, saves JSON/PNG, and returns AUC score and file paths.
        If y_prob is None, skips and returns None.
        """
        if y_prob is None:
            logger.warning(f"Probability scores or decision function not available for model '{model_key}'. Skipping ROC generation.")
            return None
            
        logger.info(f"Generating ROC Curve for model '{model_key}'...")
        
        # 1. Compute ROC curve metrics
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            logger.error(f"Failed to calculate ROC curve for {model_key}: {e}")
            return None
            
        # 2. Save JSON values (cast arrays to lists for serialization)
        # To avoid massive JSON files (e.g., if there are 10,000 threshold steps),
        # we can serialize them directly.
        json_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "roc_auc": float(roc_auc)
        }
        
        json_path = os.path.join(self.output_dir, f"roc_curve_{model_key}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)
            logger.info(f"Saved ROC curve JSON data to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save ROC curve JSON data: {e}")
            
        # 3. Save PNG Plot
        png_path = os.path.join(self.output_dir, f"roc_curve_{model_key}.png")
        try:
            self._plot_roc_curve(fpr, tpr, roc_auc, model_key, png_path)
            logger.info(f"Saved ROC curve PNG plot to {png_path}")
        except Exception as e:
            logger.error(f"Failed to save ROC curve PNG plot: {e}")
            
        return {
            "roc_auc": roc_auc,
            "json_path": json_path,
            "png_path": png_path
        }

    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, model_key: str, save_path: str) -> None:
        """Renders and saves a polished ROC curve plot."""
        plt.figure(figsize=(6, 5))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        # Plot baseline random classifier
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess (AUC = 0.5000)")
        
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        
        plt.title(f"ROC Curve — {model_key.replace('_', ' ').title()}", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("False Positive Rate", fontsize=11, fontweight='bold', labelpad=10)
        plt.ylabel("True Positive Rate", fontsize=11, fontweight='bold', labelpad=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc="lower right", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
