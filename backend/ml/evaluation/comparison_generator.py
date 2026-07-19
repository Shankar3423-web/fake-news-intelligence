import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

logger = logging.getLogger("model_evaluation_pipeline")

class ComparisonGenerator:
    """
    Assembles evaluation metrics from all models into comparison structures.
    Saves outputs as CSV, JSON, Markdown, and generates comparative bar charts.
    """
    def __init__(
        self,
        comparison_dir: str = "ml/evaluation/comparison",
        charts_dir: str = "ml/evaluation/charts"
    ) -> None:
        self.comparison_dir = comparison_dir
        self.charts_dir = charts_dir
        os.makedirs(self.comparison_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)

    def generate(self, model_eval_results: Dict[str, Dict[str, Any]], overall_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generates comparison table, saves as CSV/JSON/Markdown, and creates comparison charts.
        
        Args:
            model_eval_results: Dictionary mapping model keys to their evaluator result dicts.
            overall_scores: Dictionary mapping model keys to their computed overall scores.
        """
        logger.info("Generating multi-model comparison deliverables...")
        
        rows = []
        for key, res in model_eval_results.items():
            metrics = res["metrics"]
            overall_score = overall_scores.get(key, 0.0)
            
            rows.append({
                "Model": res["algorithm"],
                "Model Key": key,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "ROC AUC": metrics["roc_auc"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "MCC": metrics["mcc"],
                "Kappa": metrics["cohen_kappa"],
                "Log Loss": metrics["log_loss"] if metrics["log_loss"] is not None else float('nan'),
                "Prediction Time (s)": metrics["prediction_time_sec"],
                "Throughput (samples/s)": metrics["inference_throughput_sps"],
                "Memory Usage (MB)": metrics["memory_used_mb"],
                "Model Size (MB)": metrics["model_size_mb"],
                "Overall Score": round(overall_score, 6)
            })
            
        df = pd.DataFrame(rows)
        
        # 1. Save CSV
        csv_path = os.path.join(self.comparison_dir, "model_comparison.csv")
        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved model comparison CSV to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save model comparison CSV: {e}")
            
        # 2. Save JSON
        json_path = os.path.join(self.comparison_dir, "model_comparison.json")
        try:
            # Convert nan to None for valid JSON serialization
            df_clean = df.copy()
            df_clean = df_clean.replace({np.nan: None})
            json_list = df_clean.to_dict(orient="records")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_list, f, indent=4)
            logger.info(f"Saved model comparison JSON to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save model comparison JSON: {e}")
            
        # 3. Save Markdown
        md_path = os.path.join(self.comparison_dir, "model_comparison.md")
        try:
            md_content = self._format_comparison_markdown(df)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Saved model comparison Markdown to {md_path}")
        except Exception as e:
            logger.error(f"Failed to save model comparison Markdown: {e}")

        # 4. Generate visual comparison charts
        try:
            self._create_comparison_charts(df)
            logger.info("Successfully generated visual comparison charts in charts/")
        except Exception as e:
            logger.error(f"Failed to generate visual comparison charts: {e}")

        return {
            "csv_path": csv_path,
            "json_path": json_path,
            "md_path": md_path
        }

    def _format_comparison_markdown(self, df: pd.DataFrame) -> str:
        """Formats the comparison dataframe into a markdown document."""
        md = "# Model Comparison Matrix\n\n"
        md += "This table aggregates the performance, resource footprint, and speed of all evaluated models.\n\n"
        
        headers = [
            "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", 
            "MCC", "Pred Time (s)", "Memory (MB)", "Size (MB)", "Overall Score"
        ]
        
        md += "|" + "|".join(headers) + "|\n"
        md += "|" + "|".join([":---"] + [":---:"] * (len(headers) - 1)) + "|\n"
        
        for _, row in df.iterrows():
            md_row = [
                row["Model"],
                f"{row['Accuracy']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1 Score']:.4f}",
                f"{row['ROC AUC']:.4f}",
                f"{row['MCC']:.4f}",
                f"{row['Prediction Time (s)']:.4f}",
                f"{row['Memory Usage (MB)']:.2f}",
                f"{row['Model Size (MB)']:.2f}",
                f"{row['Overall Score']:.4f}"
            ]
            md += "|" + "|".join(md_row) + "|\n"
            
        return md

    def _create_comparison_charts(self, df: pd.DataFrame) -> None:
        """Creates professional visual comparisons and saves them as PNG."""
        # Setup modern plotting style
        plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.linestyle': ':', 'grid.alpha': 0.6})

        # Chart 1: Key Performance Metrics Comparison (F1, Accuracy, Precision, Recall, ROC AUC)
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_keys = ["F1 Score", "Accuracy", "Precision", "Recall", "ROC AUC"]
        x = np.arange(len(df))
        width = 0.15
        
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']
        
        for idx, m_key in enumerate(metrics_keys):
            ax.bar(x + (idx - 2) * width, df[m_key], width, label=m_key, color=colors[idx])
            
        ax.set_title("Performance Metrics Comparison", fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Model"], fontweight='bold')
        ax.set_ylabel("Score", fontsize=11, fontweight='bold')
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right", shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, "metrics_comparison.png"), dpi=300)
        plt.close()

        # Chart 2: Prediction Time Comparison (lower is better)
        plt.figure(figsize=(7, 5))
        bars = plt.bar(df["Model"], df["Prediction Time (s)"], color='#d62728', alpha=0.85, width=0.5)
        plt.title("Prediction Time Comparison (Lower is Better)", fontsize=12, fontweight='bold', pad=15)
        plt.ylabel("Prediction Time (seconds)", fontsize=11, fontweight='bold')
        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f"{height:.4f}s",
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, "prediction_time_comparison.png"), dpi=300)
        plt.close()

        # Chart 3: Model Size Comparison (lower is better)
        plt.figure(figsize=(7, 5))
        bars = plt.bar(df["Model"], df["Model Size (MB)"], color='#9467bd', alpha=0.85, width=0.5)
        plt.title("Model Size Comparison (Lower is Better)", fontsize=12, fontweight='bold', pad=15)
        plt.ylabel("Model Size (MB)", fontsize=11, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f"{height:.2f} MB",
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, "model_size_comparison.png"), dpi=300)
        plt.close()
