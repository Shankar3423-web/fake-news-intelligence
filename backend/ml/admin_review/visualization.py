import os
import logging
from typing import Dict, Any

logger = logging.getLogger("admin_review_pipeline")

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/script execution
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib is not installed. Visualization chart generation will be skipped.")

class AdminReviewVisualizer:
    """
    AdminReviewVisualizer generates visual charts showing statistics and distributions
    of admin reviews (Approval rate, Review Status distribution, etc.).
    """
    def __init__(self, charts_dir: str) -> None:
        self.charts_dir = charts_dir
        os.makedirs(self.charts_dir, exist_ok=True)

    def generate_charts(self, stats: Dict[str, Any]) -> None:
        """
        Renders and saves visualization charts based on the calculated review statistics.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.info("Skipping chart generation: matplotlib is not available.")
            return

        approved_count = stats.get("approved_count", 0)
        rejected_count = stats.get("rejected_count", 0)
        pending_count = stats.get("pending_count", 0)
        total_reviews = stats.get("total_reviews", 0)

        # Handle empty dataset charting gracefully
        if total_reviews == 0:
            logger.warning("No review statistics to plot. Skipping chart generation.")
            return

        try:
            # 1. Approval Distribution (Bar Chart: Approved vs Rejected vs Pending)
            plt.figure(figsize=(6, 5))
            labels = ["Approved", "Rejected", "Pending"]
            values = [approved_count, rejected_count, pending_count]
            colors = ["#2ecc71", "#e74c3c", "#f39c12"]  # Emerald green, Red, Orange

            plt.bar(labels, values, color=colors, edgecolor="black", width=0.5)
            plt.title("Administrator Decisions Distribution", fontsize=12, fontweight="bold", pad=15)
            plt.ylabel("Number of Records", fontsize=10)
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            
            # Force integer ticks on Y-axis
            max_val = max(values)
            plt.yticks(range(0, max_val + 2, max(1, max_val // 5)))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, "approval_distribution.png"), dpi=150)
            plt.close()

            # 2. Review Status Distribution (Pie Chart)
            plt.figure(figsize=(6, 6))
            pie_labels = []
            pie_values = []
            pie_colors = []
            
            if approved_count > 0:
                pie_labels.append("APPROVED")
                pie_values.append(approved_count)
                pie_colors.append("#2ecc71")
            if rejected_count > 0:
                pie_labels.append("REJECTED")
                pie_values.append(rejected_count)
                pie_colors.append("#e74c3c")
            if pending_count > 0:
                pie_labels.append("PENDING")
                pie_values.append(pending_count)
                pie_colors.append("#f39c12")

            if pie_values:
                plt.pie(
                    pie_values,
                    labels=pie_labels,
                    autopct="%1.1f%%",
                    startangle=140,
                    colors=pie_colors,
                    wedgeprops={"edgecolor": "black", "linewidth": 1, "antialiased": True}
                )
            plt.title("Review Status Share", fontsize=12, fontweight="bold", pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, "review_status_distribution.png"), dpi=150)
            plt.close()

            logger.info("Successfully generated admin review visualization charts.")
        except Exception as e:
            logger.error(f"Could not generate charts via matplotlib: {e}")
