import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("model_training_pipeline")

class ReportGenerator:
    """
    Generates a markdown report (training_report.md) for the training run.
    Contains model algorithms, hyperparameters, run stats, benchmarks,
    warnings, recommendations, and list of produced files.
    """
    def __init__(self, report_path: str = "ml/training/training_report.md") -> None:
        self.report_path = report_path

    def generate(
        self,
        dataset_info: Dict[str, Any],
        split_info: Dict[str, Any],
        model_summaries: List[Dict[str, Any]],
        pipeline_duration: float,
        generated_files: Dict[str, str],
        warnings: List[str]
    ) -> str:
        """
        Compiles execution info and writes the markdown report.
        
        Returns:
            The raw markdown content generated.
        """
        # Build recommendations dynamically
        recommendations = [
            "Ensure prediction services parse and validate features in the exact order saved in 'feature_order.json'.",
            "Since training is complete and validated, proceed to Phase 7: Model Evaluation to run cross-validation, calculate metrics (accuracy, F1, ROC/AUC), and select the best model.",
            "Consider hyperparameter optimization if evaluation metrics in the next phase indicate overfitting (especially on high-capacity models like Random Forest or XGBoost)."
        ]
        
        # Build markdown sections
        md = []
        md.append("# Phase 6 — Model Training Production Report")
        md.append("")
        md.append("## 1. Overview")
        md.append("This report documents the training execution of multiple machine learning classifiers for the Fake News Detection pipeline. Training was conducted using the selected feature set generated in Phase 5, in a reproducible and stratified manner.")
        md.append("")
        
        md.append("## 2. Dataset Information")
        md.append(f"- **Input Dataset Path**: `{dataset_info.get('path', 'N/A')}`")
        md.append(f"- **Dataset File Size**: `{dataset_info.get('size_bytes', 0) / (1024*1024):.2f} MB` ({dataset_info.get('size_bytes', 0)} bytes)")
        md.append(f"- **Feature Count**: `{dataset_info.get('feature_count', 0)}` numeric features")
        md.append(f"- **Total Rows**: `{split_info.get('training_samples', 0) + split_info.get('testing_samples', 0)}` samples")
        md.append(f"- **Training Partition (80%)**: `{split_info.get('training_samples', 0)}` rows")
        md.append(f"- **Testing Partition (20%)**: `{split_info.get('testing_samples', 0)}` rows")
        
        # Class distribution
        train_dist = split_info.get("label_distribution", {}).get("train", {})
        test_dist = split_info.get("label_distribution", {}).get("test", {})
        md.append("")
        md.append("### Partition Class Distributions")
        md.append("| Partition | Class 0 (Real) | Class 1 (Fake) | Total |")
        md.append("| :--- | :--- | :--- | :--- |")
        md.append(f"| **Training** | {train_dist.get(0, 0)} | {train_dist.get(1, 0)} | {split_info.get('training_samples', 0)} |")
        md.append(f"| **Testing** | {test_dist.get(0, 0)} | {test_dist.get(1, 0)} | {split_info.get('testing_samples', 0)} |")
        md.append("")
        
        md.append("## 3. Model Information & Hyperparameters")
        md.append("Each classifier was trained using a custom set of hyperparameters configured for stability and performance:")
        md.append("")
        
        for summary in model_summaries:
            algo = summary.get("algorithm", "Unknown")
            hyperparams = summary.get("split_info", {}).get("hyperparameters", {}) if "split_info" in summary else {}
            if not hyperparams:
                # Try getting from other keys
                hyperparams = summary.get("hyperparameters", {})
                
            md.append(f"### {algo}")
            md.append("- **Algorithm Name**: " + algo)
            md.append("- **Hyperparameters Used**:")
            md.append("```json")
            md.append(json.dumps(hyperparams, indent=4))
            md.append("```")
            md.append("")

        md.append("## 4. Training Benchmark & Summary")
        md.append("Performance profiling was done to track the duration and memory overhead for each classifier:")
        md.append("")
        md.append("| Model Algorithm | Training Duration (seconds) | RSS Memory Used (MB) | Samples Trained | Features Count |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for summary in model_summaries:
            algo = summary.get("algorithm", "Unknown")
            dur = summary.get("training_duration_sec", 0.0)
            mem = summary.get("memory_used_mb", 0.0)
            samples = summary.get("samples_trained", 0)
            features = summary.get("feature_count", 0)
            md.append(f"| **{algo}** | {dur:.4f}s | {mem:.2f} MB | {samples} | {features} |")
        md.append("")
        
        md.append(f"- **Total Pipeline Duration**: `{pipeline_duration:.4f} seconds`")
        md.append("")

        md.append("## 5. Generated Files Reference")
        md.append("The following files were created during Phase 6 and are saved in the production directory structure:")
        md.append("")
        md.append("| Resource | Output Path |")
        md.append("| :--- | :--- |")
        for ref_name, path in generated_files.items():
            md.append(f"| **{ref_name}** | `{path}` |")
        md.append("")

        md.append("## 6. Warnings")
        if warnings:
            for w in warnings:
                md.append(f"- ⚠️ {w}")
        else:
            md.append("- No training execution warnings generated.")
        md.append("")

        md.append("## 7. Recommendations")
        for rec in recommendations:
            md.append(f"- 💡 {rec}")
        md.append("")

        md_content = "\n".join(md)
        
        # Write report to path
        parent_dir = os.path.dirname(self.report_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Generated training report at {self.report_path}")
        except Exception as e:
            logger.error(f"Failed to save training report to {self.report_path}: {e}")
            raise
            
        return md_content
