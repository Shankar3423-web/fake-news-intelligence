import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger("model_evaluation_pipeline")

class ReportGenerator:
    """
    Compiles a comprehensive markdown report (evaluation_report.md) for the evaluation run.
    Integrates results, metrics, comparisons, leaderboards, best model details, and recommendations.
    """
    def __init__(self, output_path: str = "ml/evaluation/reports/evaluation_report.md") -> None:
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def generate(
        self,
        dataset_info: Dict[str, Any],
        training_info: Dict[str, Any],
        model_results: Dict[str, Dict[str, Any]],
        overall_scores: Dict[str, float],
        leaderboard: List[Dict[str, Any]],
        best_model: Dict[str, Any],
        pipeline_duration: float,
        generated_files: Dict[str, str],
        warnings: List[str]
    ) -> None:
        """
        Generates and saves the evaluation report.
        """
        logger.info(f"Compiling evaluation report at {self.output_path}...")
        
        md = "# Phase 7: Model Evaluation Report\n"
        md += "## Production-Grade Fake News Intelligence System\n\n"
        
        # 1. Overview
        md += "## 1. Overview\n"
        md += "This report compiles the scientific evaluation results of every trained machine learning model produced during Phase 6. "
        md += "The primary objective of this phase is to evaluate and compare the classifiers under uniform testing conditions, "
        md += "using reproducible testing splits, to select the optimal model for production deployment.\n\n"
        
        # 2. Dataset Information
        md += "## 2. Dataset Information\n"
        md += f"- **Dataset Path:** `{dataset_info.get('path', 'N/A')}`\n"
        md += f"- **Dataset Size (bytes):** `{dataset_info.get('size_bytes', 0):,}`\n"
        md += f"- **Total Samples:** `{dataset_info.get('total_rows', 0):,}`\n"
        md += f"- **Testing Split Size:** `{dataset_info.get('test_rows', 0):,}` (test_size={dataset_info.get('test_size', 0.2)})\n"
        md += f"- **Feature Count:** `{dataset_info.get('feature_count', 0)}` features\n"
        md += f"- **Split Reproducibility:** Stratified split, random seed = `{dataset_info.get('random_state', 42)}`\n\n"

        # 3. Model Information
        md += "## 3. Model & Pipeline Tracking Context\n"
        md += f"- **Training Run Version:** `{training_info.get('training_version', 'N/A')}`\n"
        md += f"- **Dataset Version Context:** `{training_info.get('dataset_version', 'N/A')}`\n"
        md += f"- **Feature Selection Version Context:** `{training_info.get('feature_selection_version', 'N/A')}`\n"
        md += f"- **Total Models Evaluated:** `{len(model_results)}` classifiers\n\n"

        # 4. Evaluation Metrics & Comparison Matrix
        md += "## 4. Model Comparison Matrix\n"
        md += "The table below compares the performance, execution speed, and resource footprints of all evaluated classifiers:\n\n"
        
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Pred Time (s)", "Throughput (s/s)", "Memory (MB)", "Size (MB)"]
        md += "|" + "|".join(headers) + "|\n"
        md += "|" + "|".join([":---"] + [":---:"] * (len(headers) - 1)) + "|\n"
        
        for key, res in model_results.items():
            metrics = res["metrics"]
            row = [
                res["algorithm"],
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['prediction_time_sec']:.4f}",
                f"{metrics['inference_throughput_sps']:.1f}",
                f"{metrics['memory_used_mb']:.2f}",
                f"{metrics['model_size_mb']:.2f}"
            ]
            md += "|" + "|".join(row) + "|\n"
        md += "\n"

        # 5. Leaderboards
        md += "## 5. Model Leaderboard\n"
        md += "Rankings are based on the configurable weighted score selector:\n\n"
        
        l_headers = ["Rank", "Model", "Overall Score", "F1 Score", "ROC AUC", "Throughput (samples/s)", "Memory (MB)"]
        md += "|" + "|".join(l_headers) + "|\n"
        md += "|" + "|".join([":---:"] + [":---"] + [":---:"] * (len(l_headers) - 2)) + "|\n"
        
        for entry in leaderboard:
            l_row = [
                str(entry["Rank"]),
                f"**{entry['Model']}**" if entry["Rank"] == 1 else entry["Model"],
                f"{entry['Overall Score']:.6f}",
                f"{entry['F1']:.4f}",
                f"{entry['ROC']:.4f}",
                f"{entry['Prediction Speed']:.1f}",
                f"{entry['Memory']:.2f}"
            ]
            md += "|" + "|".join(l_row) + "|\n"
        md += "\n"

        # 6. Best Model Selection
        md += "## 6. Selected Production Model\n"
        md += f"🏆 The system has selected **{best_model['algorithm']}** as the best production-ready candidate.\n\n"
        md += "> [!IMPORTANT]\n"
        md += f"> **Model ID:** `{best_model['model_id']}`\n"
        md += f"> **Overall Weighted Score:** `{best_model['overall_score']:.6f}`\n"
        md += f"> **Target Metric Selection Strategy:** `{best_model['selection_metric_used']}`\n"
        md += f"> **Model Size:** `{best_model['metrics']['model_size_mb']:.2f} MB`\n"
        md += f"> **Inference Latency:** `{best_model['metrics']['inference_latency_ms']:.6f} ms/sample`\n\n"

        # 7. Recommendations
        md += "## 7. Recommendations & Deployment Rationale\n"
        best_key = best_model["model_key"]
        
        if best_key == "xgboost":
            md += "- **XGBoost** provides the highest classification power (F1 and ROC AUC) with competitive inference latency. "
            md += "It is highly recommended for standard production serving unless ultra-low latency or minimal memory footprint is required.\n"
        elif best_key == "random_forest":
            md += "- **Random Forest** offers excellent classification capacity and robustness, but exhibits a larger file footprint and potentially slower tree traversal. "
            md += "Monitor disk usage and deserialization latency in serverless environments.\n"
        elif best_key == "logistic_regression":
            md += "- **Logistic Regression** is extremely lightweight, executes inference in micro-seconds, and has minimal memory overhead. "
            md += "It serves as an excellent low-complexity baseline or fall-back model for resource-constrained edge-servers.\n"
        elif best_key == "svm":
            md += "- **Linear SVM** delivers highly linear decision boundaries with quick decision times. However, since it lack probability scores, probability-based post-filtering or thresholding is disabled.\n"
            
        md += "- **Resource footprint optimization:** Deserialization latency should be monitored during container startup. Larger tree-based models like Random Forest and XGBoost require slightly longer warm-up times compared to Logistic Regression.\n\n"

        # 8. Warnings & Limitations
        md += "## 8. Warnings & Operational Constraints\n"
        if warnings:
            for w in warnings:
                md += f"> [!WARNING]\n> {w}\n\n"
        else:
            md += "*No structural or distribution warnings were triggered during this evaluation run.*\n\n"

        # 9. Generated Deliverables
        md += "## 9. Generated Evaluation Deliverables\n"
        md += "Below is the listing of all structured outputs, visual plots, and version histories compiled by this execution:\n\n"
        
        md += "| Deliverable Name | File Path |\n"
        md += "| :--- | :--- |\n"
        for name, path in sorted(generated_files.items()):
            # Format clean link
            rel_path = path.replace("\\", "/")
            md += f"| {name} | [{os.path.basename(path)}](file:///{os.path.abspath(rel_path)}) |\n"
        md += "\n"
        
        md += "---\n"
        md += f"*Report compiled automatically by Phase 7 Pipeline. Duration: {pipeline_duration:.2f} seconds.*"

        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(md)
            logger.info(f"Report generated successfully at {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to write evaluation report: {e}")
            raise
