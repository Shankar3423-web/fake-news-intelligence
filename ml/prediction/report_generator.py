import os
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger("prediction_pipeline")

class ReportGenerator:
    """
    Generates a beautiful prediction report in markdown summarizing the latest prediction.
    """
    def __init__(self, report_path: str = "ml/prediction/reports/prediction_report.md") -> None:
        self.report_path = report_path
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

    def generate_report(
        self,
        latest_response: Dict[str, Any],
        stats: Dict[str, Any],
        hashes: Dict[str, str],
        warnings: List[str]
    ) -> None:
        """
        Creates the markdown report.
        """
        logger.info(f"Generating markdown prediction report at {self.report_path}...")
        
        md = []
        md.append("# Production Prediction Engine Report")
        md.append(f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        md.append("\n## Prediction Summary")
        md.append(f"\n- **Timestamp:** `{latest_response['timestamp']}`")
        md.append(f"- **Final Prediction:** `{latest_response['label']}` (Class {latest_response['prediction']})")
        md.append(f"- **Confidence Score:** `{latest_response['confidence'] * 100:.2f}%`")
        
        md.append("\n## Model Information")
        md.append(f"\n- **Model Selected:** `{latest_response['model_name']}`")
        md.append(f"- **Model/Dataset Version:** `{latest_response['model_version']}`")
        md.append(f"- **Evaluation Version Reference:** `{latest_response['evaluation_version']}`")

        md.append("\n## Performance Metrics")
        md.append(f"\n- **Prediction Latency:** `{latest_response['prediction_time_ms']:.4f} ms`")
        md.append(f"- **Throughput:** `{latest_response['throughput']:.2f} samples/second`")
        md.append(f"- **Process Memory Increase:** `{latest_response['memory_usage']:.4f} MB`")

        md.append("\n## Cumulative Pipeline Statistics")
        md.append("\n| Statistic Metric | Value |")
        md.append("| :--- | :---: |")
        md.append(f"| **Total Predictions Run** | {stats.get('total_predictions', 0)} |")
        md.append(f"| **Average Latency** | {stats.get('average_prediction_time_ms', 0.0):.4f} ms |")
        md.append(f"| **Average Memory footprint** | {stats.get('average_memory_usage_mb', 0.0):.4f} MB |")
        md.append(f"| **Average Throughput** | {stats.get('average_throughput_sps', 0.0):.2f} sps |")
        md.append(f"| **Average Confidence** | {stats.get('average_confidence', 0.0) * 100:.2f}% |")

        md.append("\n## Verification & Warnings")
        if warnings:
            md.append("\n> [!WARNING]")
            md.append("> **Inference Warnings Detected:**")
            for warn in warnings:
                md.append(f"> - {warn}")
        else:
            md.append("\n> [!NOTE]\n> **Inference Verification: Passed**\n> No anomalies or structure violations were detected during preprocessing or model execution.")

        md.append("\n## File Integrity Registry (Hashes)")
        md.append("\n| Artifact | SHA-256 Signature |")
        md.append("| :--- | :--- |")
        for key, sha in hashes.items():
            md.append(f"| `{key}` | `{sha}` |")

        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md))
            logger.info("Markdown prediction report written successfully.")
        except Exception as e:
            logger.error(f"Failed to write prediction report: {e}")
            raise e
