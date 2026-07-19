import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Dict, Any

logger = logging.getLogger("model_evaluation_pipeline")

class ConfusionMatrixGenerator:
    """
    Generates confusion matrices for evaluated models.
    Saves outputs as CSV, JSON, and high-quality PNG visualization.
    """
    def __init__(self, output_dir: str = "ml/evaluation/confusion_matrices") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, model_key: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Computes confusion matrix, saves CSV/JSON/PNG, and returns raw numbers.
        """
        logger.info(f"Generating confusion matrix for model '{model_key}'...")
        
        # 1. Compute confusion matrix
        # binary classification: 0 (Real/Negative), 1 (Fake/Positive)
        # Confusion matrix:
        # [[TN, FP],
        #  [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
        
        matrix_dict = {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        }
        
        # 2. Save JSON
        json_path = os.path.join(self.output_dir, f"confusion_matrix_{model_key}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(matrix_dict, f, indent=4)
            logger.info(f"Saved confusion matrix JSON to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix JSON: {e}")
            
        # 3. Save CSV
        csv_path = os.path.join(self.output_dir, f"confusion_matrix_{model_key}.csv")
        try:
            df_cm = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual Real (0)", "Actual Fake (1)"],
                columns=["Predicted Real (0)", "Predicted Fake (1)"]
            )
            df_cm.to_csv(csv_path)
            logger.info(f"Saved confusion matrix CSV to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix CSV: {e}")
            
        # 4. Generate & Save PNG Visualization
        png_path = os.path.join(self.output_dir, f"confusion_matrix_{model_key}.png")
        try:
            self._plot_confusion_matrix(tn, fp, fn, tp, model_key, png_path)
            logger.info(f"Saved confusion matrix PNG plot to {png_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix PNG plot: {e}")
            
        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "json_path": json_path,
            "csv_path": csv_path,
            "png_path": png_path
        }

    def _plot_confusion_matrix(self, tn: int, fp: int, fn: int, tp: int, model_key: str, save_path: str) -> None:
        """Renders and saves a polished confusion matrix heatmap."""
        plt.figure(figsize=(6, 5))
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Create heatmap using matplotlib imshow
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im)
        
        # Formatting labels and ticks
        classes = ["Real (0)", "Fake (1)"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=10)
        plt.yticks(tick_marks, classes, fontsize=10)
        
        plt.title(f"Confusion Matrix — {model_key.replace('_', ' ').title()}", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("Predicted Label", fontsize=11, fontweight='bold', labelpad=10)
        plt.ylabel("Actual Label", fontsize=11, fontweight='bold', labelpad=10)
        
        # Add labels to cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                # Add text label with raw value
                # And cell type annotation
                cell_type = ""
                if i == 0 and j == 0:
                    cell_type = "TN"
                elif i == 0 and j == 1:
                    cell_type = "FP"
                elif i == 1 and j == 0:
                    cell_type = "FN"
                elif i == 1 and j == 1:
                    cell_type = "TP"
                    
                text_color = "white" if val > thresh else "black"
                plt.text(
                    j, i, f"{cell_type}\n{val:,}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=text_color,
                    fontsize=12,
                    fontweight='bold'
                )
                
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
