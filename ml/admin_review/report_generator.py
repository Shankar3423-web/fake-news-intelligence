import os
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("admin_review_pipeline")

class ReportGenerator:
    """
    ReportGenerator compiles review statistics and recent audit histories
    into a comprehensive markdown report file.
    """
    def __init__(self, report_file_path: str, history_csv_path: str) -> None:
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
        Creates and writes the admin review report in Markdown format.
        """
        recent_reviews = []
        if os.path.exists(self.history_csv_path):
            try:
                df = pd.read_csv(self.history_csv_path)
                df = df.where(pd.notnull(df), None)
                # Take last 5 entries, reversed to show the latest first
                recent_df = df.tail(5).iloc[::-1]
                for _, row in recent_df.iterrows():
                    recent_reviews.append(row.to_dict())
            except Exception as e:
                logger.warning(f"Could not load recent review history for report: {e}")

        # Construct markdown content
        md_content = f"""# Phase 11 - Admin Review Report

## Latest Administrative Action Summary
* **Feedback ID:** `{record.get("feedback_id", "N/A")}`
* **Review Decision:** **{record.get("review_status", "UNKNOWN")}**
* **Reviewer:** {record.get("reviewer", "N/A")}
* **Timestamp:** {record.get("timestamp", "N/A")}

---

## Global Review Statistics
* **Total Reviews Conducted:** {stats.get("Total Reviews", 0)}
* **Approved Count:** {stats.get("Approved Count", 0)}
* **Rejected Count:** {stats.get("Rejected Count", 0)}
* **Pending Count:** {stats.get("Pending Count", 0)}
* **Approval Rate:** {stats.get("Approval Rate", 0.0) * 100:.2f}%

---

## Recent Administrator Decisions
"""

        if recent_reviews:
            md_content += "| Feedback ID | Review Status | Reviewer | Review Notes | Timestamp |\n"
            md_content += "| :--- | :---: | :--- | :--- | :--- |\n"
            for row in recent_reviews:
                notes_disp = row.get("Review Notes", "")
                if not notes_disp:
                    notes_disp = "*No notes provided*"
                else:
                    notes_disp = str(notes_disp).replace("|", "\\|")
                md_content += f"| `{row.get('Feedback ID')}` | **{row.get('Review Status')}** | {row.get('Reviewer')} | {notes_disp} | {row.get('Timestamp')} |\n"
        else:
            md_content += "*No reviews have been recorded yet.*\n"

        md_content += f"""
---

## Integrity & Verification Signatures
* **Report Hash (SHA-256):** `{hashes.get("report", "Pending")}`
* **History Hash (SHA-256):** `{hashes.get("history", "Pending")}`
* **Approved Dataset Hash (SHA-256):** `{hashes.get("approved", "Pending")}`
* **Statistics Hash (SHA-256):** `{hashes.get("statistics", "Pending")}`
* **Metadata Hash (SHA-256):** `{hashes.get("metadata", "Pending")}`
* **Versions Registry Hash (SHA-256):** `{hashes.get("versions", "Pending")}`
"""

        if warnings:
            md_content += "\n## System Warnings\n"
            for w in warnings:
                md_content += f"* ⚠️ {w}\n"

        # Write to report file
        try:
            with open(self.report_file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Admin review report generated and saved to {self.report_file_path}")
        except Exception as e:
            logger.error(f"Failed to write admin review report: {e}")
            raise
