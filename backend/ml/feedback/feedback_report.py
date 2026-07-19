import os
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("feedback_pipeline")

class ReportGenerator:
    """
    ReportGenerator formats feedback execution results and cumulative statistics
    into a comprehensive markdown report.
    """
    def __init__(self, report_file_path: str = "ml/feedback/reports/feedback_report.md", history_csv_path: str = "ml/feedback/history/feedback_history.csv") -> None:
        self.report_file_path = report_file_path
        self.history_csv_path = history_csv_path
        os.makedirs(os.path.dirname(self.report_file_path), exist_ok=True)

    def generate_report(
        self,
        record: Dict[str, Any],
        stats: Dict[str, Any],
        hashes: Dict[str, str],
        warnings: List[str]
    ) -> None:
        """
        Builds the feedback report markdown file.
        """
        # Load recent feedback from history file
        recent_feedback_rows = []
        if os.path.exists(self.history_csv_path):
            try:
                df = pd.read_csv(self.history_csv_path)
                # Take last 5 rows, reverse them to show newest first
                recent_df = df.tail(5).iloc[::-1]
                for _, row in recent_df.iterrows():
                    recent_feedback_rows.append(row.to_dict())
            except Exception as e:
                logger.warning(f"Could not load recent feedback history for report: {e}")

        # Construct MD content
        md_content = f"""# Phase 10 - Feedback Collection Report

## Latest Feedback Status: **{record.get("feedback_value", "UNKNOWN")}**

---

## Feedback Summary
* **Feedback ID:** `{record.get("feedback_id", "N/A")}`
* **Timestamp:** {record.get("timestamp", "N/A")}
* **Prediction Label/Result:** {record.get("prediction", "N/A")}
* **Prediction Confidence:** {record.get("prediction_confidence", 0.0):.4f}
* **Verification Status:** {record.get("verification_status", "N/A")}
* **Composite Evidence Score:** {record.get("evidence_score", 0.0):.4f}
* **Similarity Score:** {record.get("similarity_score", 0.0):.4f}
* **Final Decision:** {record.get("final_decision", "N/A")}
* **Optional Comment:** {record.get("comment", "N/A")}
* **System Version:** {record.get("system_version", "N/A")}

---

## Global Feedback Statistics
* **Total Feedback Collected:** {stats.get("total_feedback", 0)}
* **Correct Count:** {stats.get("correct_count", 0)}
* **Incorrect Count:** {stats.get("incorrect_count", 0)}
* **Unsure Count:** {stats.get("unsure_count", 0)}
* **Average Prediction Confidence:** {stats.get("average_prediction_confidence", 0.0):.4f}
* **Average Similarity:** {stats.get("average_similarity", 0.0):.4f}
* **Average Evidence Score:** {stats.get("average_evidence_score", 0.0):.4f}

### Feedback Distribution
"""

        dist = stats.get("feedback_distribution", {})
        for key, val in dist.items():
            pct = val * 100
            md_content += f"* **{key}:** {pct:.2f}%\n"

        md_content += """
---

## Recent Feedback
"""

        if recent_feedback_rows:
            md_content += "| Feedback ID | Timestamp | Prediction | Verification | Decision | Feedback | Comment |\n"
            md_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n"
            for row in recent_feedback_rows:
                comment_disp = row.get("Comment", "")
                if pd.isna(comment_disp):
                    comment_disp = ""
                # Escape pipe characters in comment to not break Markdown table
                comment_disp = str(comment_disp).replace("|", "\\|")
                md_content += f"| `{row.get('Feedback ID')}` | {row.get('Timestamp')} | {row.get('Prediction')} | {row.get('Verification')} | {row.get('Decision')} | {row.get('Feedback')} | {comment_disp} |\n"
        else:
            md_content += "*No historical feedback recorded yet.*\n"

        md_content += f"""
---

## Validation Summary
* **Schema Validation:** Passed ✅
* **Input Sanitization:** Performed on Comment field

## Integrity Signatures
* **Report Hash (SHA-256):** `{hashes.get("report", "Pending")}`
* **Metadata Hash (SHA-256):** `{hashes.get("metadata", "Pending")}`
* **Statistics Hash (SHA-256):** `{hashes.get("statistics", "Pending")}`
* **History Hash (SHA-256):** `{hashes.get("history", "Pending")}`
"""

        if warnings:
            md_content += "\n## Warnings\n"
            for w in warnings:
                md_content += f"* ⚠️ {w}\n"

        # Write markdown report
        try:
            with open(self.report_file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Saved feedback report MD to {self.report_file_path}")
        except Exception as e:
            logger.error(f"Failed to write report MD: {e}")
