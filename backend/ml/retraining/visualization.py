"""
visualization.py
Generates charts for Phase 12 retraining pipeline results.
"""
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("retraining_pipeline")


class RetrainingVisualizer:
    """
    Generates charts comparing candidate vs. production metrics and
    visualising dataset changes for Phase 12.

    Produced artifacts (in ``charts_dir``):
    - ``metrics_comparison.png``: Bar chart of key metrics side-by-side.
    - ``training_duration.png``: Bar chart of per-model training durations.
    - ``label_distribution.png``: Pie chart of merged dataset label distribution.
    """

    def __init__(self, charts_dir: str) -> None:
        self._charts_dir = charts_dir
        os.makedirs(charts_dir, exist_ok=True)

    def generate_all(
        self,
        candidate_metrics: Dict[str, Any],
        production_metrics: Dict[str, Any],
        model_durations: Dict[str, float],
        label_distribution: Dict[str, int],
    ) -> Dict[str, str]:
        """
        Generates all charts.

        Args:
            candidate_metrics:  Best candidate model metrics.
            production_metrics: Production model metrics.
            model_durations:    Per-model training durations (seconds).
            label_distribution: Merged dataset label distribution.

        Returns:
            Mapping of chart name → file path.
        """
        chart_paths: Dict[str, str] = {}

        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning(
                "matplotlib not available — skipping chart generation."
            )
            return chart_paths

        # 1. Metrics comparison
        path = self._metrics_comparison_chart(
            candidate_metrics, production_metrics, plt, np
        )
        if path:
            chart_paths["metrics_comparison"] = path

        # 2. Training durations
        if model_durations:
            path = self._training_duration_chart(model_durations, plt)
            if path:
                chart_paths["training_duration"] = path

        # 3. Label distribution
        if label_distribution:
            path = self._label_distribution_chart(label_distribution, plt)
            if path:
                chart_paths["label_distribution"] = path

        logger.info(
            "RetrainingVisualizer: generated %d chart(s).", len(chart_paths)
        )
        return chart_paths

    # ── Internals ─────────────────────────────────────────────────────────────────────────────────
    def _metrics_comparison_chart(
        self,
        candidate: Dict[str, Any],
        production: Dict[str, Any],
        plt: Any,
        np: Any,
    ) -> Optional[str]:
        """Generates a grouped bar chart comparing key metrics."""
        key_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "mcc"]
        c_vals = [float(candidate.get(m, 0.0) or 0.0) for m in key_metrics]
        p_vals = [float(production.get(m, 0.0) or 0.0) for m in key_metrics]

        x = np.arange(len(key_metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        rects1 = ax.bar(x - width / 2, c_vals, width, label="Candidate", color="#4C9BE8")
        rects2 = ax.bar(x + width / 2, p_vals, width, label="Production", color="#F26522")

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("Candidate vs. Production — Key Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(key_metrics, rotation=20, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.4)

        for rect, val in zip(list(rects1) + list(rects2), c_vals + p_vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        plt.tight_layout()
        path = os.path.join(self._charts_dir, "metrics_comparison.png")
        plt.savefig(path, dpi=120)
        plt.close()
        return path

    def _training_duration_chart(
        self, durations: Dict[str, float], plt: Any
    ) -> Optional[str]:
        """Generates a horizontal bar chart of per-model training durations."""
        labels = list(durations.keys())
        values = [durations[k] for k in labels]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(labels, values, color="#56C78A")
        ax.set_xlabel("Duration (seconds)")
        ax.set_title("Candidate Model Training Duration")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s",
                va="center",
                fontsize=9,
            )
        ax.grid(axis="x", alpha=0.4)
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "training_duration.png")
        plt.savefig(path, dpi=120)
        plt.close()
        return path

    def _label_distribution_chart(
        self, distribution: Dict[str, int], plt: Any
    ) -> Optional[str]:
        """Generates a pie chart of the merged dataset label distribution."""
        labels = [
            f"Real (0): {distribution.get('0', distribution.get(0, 0))}",
            f"Fake (1): {distribution.get('1', distribution.get(1, 0))}",
        ]
        values = [
            distribution.get("0", distribution.get(0, 0)),
            distribution.get("1", distribution.get(1, 0)),
        ]
        colors = ["#4ECDC4", "#FF6B6B"]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.set_title("Merged Dataset Label Distribution")
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "label_distribution.png")
        plt.savefig(path, dpi=120)
        plt.close()
        return path
