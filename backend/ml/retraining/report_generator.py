"""
report_generator.py
Generates Markdown reports for Phase 12 retraining runs.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("retraining_pipeline")

# ── Helper ────────────────────────────────────────────────────────────────────────────────────────
def _fmt(value: Any, decimals: int = 4) -> str:
    """Formats a value for Markdown table display."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    if value is None:
        return "N/A"
    return str(value)


class RetrainingReportGenerator:
    """
    Generates ``retraining_report.md`` and ``comparison_report.md`` for Phase 12.
    """

    def __init__(self, reports_dir: str) -> None:
        self._reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def generate_retraining_report(
        self,
        *,
        run_id: str,
        approved_records: int,
        merge_stats: Dict[str, Any],
        training_summary: Dict[str, Any],
        candidate_results: Dict[str, Dict[str, Any]],
        decision: str,
        reason: str,
        pipeline_duration_sec: float,
        generated_files: Dict[str, str],
    ) -> str:
        """
        Generates the main retraining report.

        Returns:
            Path to the generated ``.md`` file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        md: List[str] = []

        md.append("# Phase 12 — Automatic Model Retraining Report")
        md.append("")
        md.append(f"*Generated: {timestamp}*")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 1. Run Overview")
        md.append("")
        md.append(f"| Field | Value |")
        md.append(f"| :--- | :--- |")
        md.append(f"| **Run ID** | `{run_id}` |")
        md.append(f"| **Decision** | **{decision}** |")
        md.append(f"| **Reason** | {reason} |")
        md.append(f"| **Pipeline Duration** | `{pipeline_duration_sec:.2f}s` |")
        md.append("")

        md.append("## 2. Approved Feedback Summary")
        md.append("")
        md.append(f"| Metric | Value |")
        md.append(f"| :--- | :--- |")
        md.append(f"| **Approved Records Used** | `{approved_records}` |")
        md.append(f"| **Pre-Merge Rows** | `{merge_stats.get('existing_rows', 0)}` |")
        md.append(f"| **Post-Merge Rows** | `{merge_stats.get('merged_rows', 0)}` |")
        md.append(f"| **Added Rows** | `{merge_stats.get('added_rows', 0)}` |")
        md.append("")

        # Label distribution
        label_dist = merge_stats.get("label_distribution_merged", {})
        if label_dist:
            md.append("### Merged Dataset Label Distribution")
            md.append("")
            md.append("| Label | Count |")
            md.append("| :--- | :--- |")
            for label, count in label_dist.items():
                label_name = "Real (0)" if str(label) == "0" else "Fake (1)"
                md.append(f"| **{label_name}** | `{count}` |")
            md.append("")

        md.append("## 3. Training Summary")
        md.append("")
        models_trained = training_summary.get("models_trained", [])
        md.append(f"- **Models Trained**: {', '.join(models_trained) or 'None'}")
        md.append(f"- **Training Rows**: `{training_summary.get('training_rows', 0)}`")
        md.append(f"- **Testing Rows**: `{training_summary.get('testing_rows', 0)}`")
        md.append(f"- **Feature Count**: `{training_summary.get('feature_count', 0)}`")
        md.append("")

        durations = training_summary.get("model_durations", {})
        if durations:
            md.append("### Per-Model Training Duration")
            md.append("")
            md.append("| Model | Duration (s) |")
            md.append("| :--- | :--- |")
            for m_key, dur in durations.items():
                md.append(f"| **{m_key}** | `{dur:.4f}` |")
            md.append("")

        md.append("## 4. Candidate Model Evaluation")
        md.append("")
        if candidate_results:
            md.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MCC |")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            for m_key, res in candidate_results.items():
                metrics = res.get("metrics", {})
                md.append(
                    f"| **{m_key}** "
                    f"| {_fmt(metrics.get('accuracy'))} "
                    f"| {_fmt(metrics.get('precision'))} "
                    f"| {_fmt(metrics.get('recall'))} "
                    f"| {_fmt(metrics.get('f1_score'))} "
                    f"| {_fmt(metrics.get('roc_auc'))} "
                    f"| {_fmt(metrics.get('mcc'))} |"
                )
            md.append("")

        md.append("## 5. Generated Artifacts")
        md.append("")
        md.append("| Artifact | Path |")
        md.append("| :--- | :--- |")
        for name, path in generated_files.items():
            md.append(f"| **{name}** | `{path}` |")
        md.append("")

        content = "\n".join(md)
        path = os.path.join(self._reports_dir, "retraining_report.md")
        _write(content, path)
        logger.info("Retraining report saved to: %s", path)
        return path

    def generate_comparison_report(
        self,
        *,
        run_id: str,
        comparison_result: Dict[str, Any],
        candidate_metrics: Dict[str, Any],
        production_metrics: Dict[str, Any],
    ) -> str:
        """
        Generates the model comparison report.

        Returns:
            Path to the generated ``.md`` file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        md: List[str] = []

        promote = comparison_result.get("promote", False)
        decision = "✅ PROMOTED" if promote else "❌ REJECTED"

        md.append("# Phase 12 — Model Comparison Report")
        md.append("")
        md.append(f"*Generated: {timestamp} | Run ID: `{run_id}`*")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## Decision")
        md.append("")
        md.append(f"**{decision}**")
        md.append("")
        md.append(f"> {comparison_result.get('reason', '')}")
        md.append("")

        md.append("## Weighted Score Comparison")
        md.append("")
        md.append("| Model | Weighted Score |")
        md.append("| :--- | :---: |")
        md.append(f"| **Candidate** | `{_fmt(comparison_result.get('candidate_score'))}` |")
        md.append(f"| **Production** | `{_fmt(comparison_result.get('production_score'))}` |")
        md.append("")

        md.append("## Per-Metric Comparison")
        md.append("")
        md.append("| Metric | Candidate | Production | Delta | Winner |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for entry in comparison_result.get("per_metric_comparison", []):
            winner_icon = "🏆 Candidate" if entry["winner"] == "candidate" else (
                "🥈 Production" if entry["winner"] == "production" else "— Tie"
            )
            md.append(
                f"| **{entry['metric']}** "
                f"| `{_fmt(entry['candidate'])}` "
                f"| `{_fmt(entry['production'])}` "
                f"| `{_fmt(entry['delta'])}` "
                f"| {winner_icon} |"
            )
        md.append("")

        md.append("## Minimum Threshold Check")
        md.append("")
        md.append("| Metric | Required | Candidate Value | Passed |")
        md.append("| :--- | :---: | :---: | :---: |")
        for metric, result in comparison_result.get("minimum_threshold_results", {}).items():
            icon = "✅" if result["passed"] else "❌"
            md.append(
                f"| **{metric}** "
                f"| `{result['required']}` "
                f"| `{_fmt(result['candidate_value'])}` "
                f"| {icon} |"
            )
        md.append("")

        content = "\n".join(md)
        path = os.path.join(self._reports_dir, "comparison_report.md")
        _write(content, path)
        logger.info("Comparison report saved to: %s", path)
        return path


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────────
def _write(content: str, path: str) -> None:
    """Writes *content* to *path*, creating parent directories if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
