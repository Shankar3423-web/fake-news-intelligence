import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from typing import Dict, Any, Optional

logger = logging.getLogger("model_evaluation_pipeline")

class ClassificationReportGenerator:
    """
    Generates classification reports (JSON/Markdown) and Precision-Recall Curves (PNG/JSON).
    """
    def __init__(
        self,
        reports_dir: str = "ml/evaluation/classification_reports",
        pr_curves_dir: str = "ml/evaluation/precision_recall_curves"
    ) -> None:
        self.reports_dir = reports_dir
        self.pr_curves_dir = pr_curves_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.pr_curves_dir, exist_ok=True)

    def generate_report(self, model_key: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Generates sklearn classification report, saving it as JSON and Markdown.
        """
        logger.info(f"Generating classification report for '{model_key}'...")
        
        # 1. Generate report dictionary and text
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_text = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
        
        # 2. Save JSON report
        json_path = os.path.join(self.reports_dir, f"classification_report_{model_key}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=4)
            logger.info(f"Saved classification report JSON to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save classification report JSON: {e}")
            
        # 3. Format & Save Markdown report
        md_path = os.path.join(self.reports_dir, f"classification_report_{model_key}.md")
        try:
            md_content = self._format_report_to_markdown(report_dict, model_key, report_text)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Saved classification report Markdown to {md_path}")
        except Exception as e:
            logger.error(f"Failed to save classification report Markdown: {e}")
            
        return {
            "report_dict": report_dict,
            "json_path": json_path,
            "md_path": md_path
        }

    def generate_pr_curve(self, model_key: str, y_true: np.ndarray, y_prob: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Generates Precision-Recall curve, saving it as JSON and PNG.
        """
        if y_prob is None:
            logger.warning(f"Probability scores or decision function not available for model '{model_key}'. Skipping PR curve generation.")
            return None
            
        logger.info(f"Generating Precision-Recall Curve for '{model_key}'...")
        
        # 1. Compute PR curve metrics
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
        except Exception as e:
            logger.error(f"Failed to calculate PR curve for {model_key}: {e}")
            return None
            
        # 2. Save JSON values
        json_data = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist() if thresholds is not None else [],
            "average_precision": float(avg_precision)
        }
        
        json_path = os.path.join(self.pr_curves_dir, f"precision_recall_curve_{model_key}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)
            logger.info(f"Saved PR curve JSON data to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save PR curve JSON data: {e}")
            
        # 3. Save PNG Plot
        png_path = os.path.join(self.pr_curves_dir, f"precision_recall_curve_{model_key}.png")
        try:
            self._plot_pr_curve(precision, recall, avg_precision, model_key, png_path)
            logger.info(f"Saved PR curve PNG plot to {png_path}")
        except Exception as e:
            logger.error(f"Failed to save PR curve PNG plot: {e}")
            
        return {
            "average_precision": avg_precision,
            "json_path": json_path,
            "png_path": png_path
        }

    def _format_report_to_markdown(self, report: Dict[str, Any], model_key: str, raw_text: str) -> str:
        """Formats the classification report dict into a clean Markdown document."""
        title = model_key.replace("_", " ").title()
        
        md = f"# Classification Report — {title}\n\n"
        md += "## Performance Metrics Table\n\n"
        md += "| Metric / Class | Precision | Recall | F1-Score | Support |\n"
        md += "| :--- | :---: | :---: | :---: | :---: |\n"
        
        # Add classes
        for key, val in report.items():
            if key in ["accuracy", "macro avg", "weighted avg"]:
                continue
            # Format class label
            class_name = f"Class {key}"
            if key == "0":
                class_name = "Class 0 (Real)"
            elif key == "1":
                class_name = "Class 1 (Fake)"
                
            md += f"| {class_name} | {val['precision']:.4f} | {val['recall']:.4f} | {val['f1-score']:.4f} | {val['support']:,} |\n"
            
        md += "| | | | | |\n" # Empty row separation
        
        # Add accuracy
        acc = report.get("accuracy", 0.0)
        support = report.get("weighted avg", {}).get("support", 0)
        md += f"| **Accuracy** | | | **{acc:.4f}** | **{support:,}** |\n"
        
        # Add macro & weighted averages
        for key in ["macro avg", "weighted avg"]:
            val = report.get(key, {})
            label = "**Macro Average**" if key == "macro avg" else "**Weighted Average**"
            md += f"| {label} | {val['precision']:.4f} | {val['recall']:.4f} | {val['f1-score']:.4f} | {val['support']:,} |\n"
            
        md += "\n## Text Formatting Output\n\n"
        md += "```text\n"
        md += raw_text
        md += "\n```\n"
        
        return md

    def _plot_pr_curve(self, precision: np.ndarray, recall: np.ndarray, avg_precision: float, model_key: str, save_path: str) -> None:
        """Renders and saves a polished Precision-Recall curve plot."""
        plt.figure(figsize=(6, 5))
        
        # Plot PR curve
        plt.plot(recall, precision, color='forestgreen', lw=2, label=f"PR Curve (AP = {avg_precision:.4f})")
        
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        
        plt.title(f"Precision-Recall Curve — {model_key.replace('_', ' ').title()}", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("Recall", fontsize=11, fontweight='bold', labelpad=10)
        plt.ylabel("Precision", fontsize=11, fontweight='bold', labelpad=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc="lower left", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
