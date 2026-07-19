import os
import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger("feedback_pipeline")

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib is not installed. Visualization chart generation will be skipped.")

class FeedbackVisualizer:
    """
    FeedbackVisualizer handles chart generation for metrics and histories of feedback.
    """
    def __init__(self, charts_dir: str = "ml/feedback/charts", history_csv_path: str = "ml/feedback/history/feedback_history.csv") -> None:
        self.charts_dir = charts_dir
        self.history_csv_path = history_csv_path
        os.makedirs(self.charts_dir, exist_ok=True)

    def generate_charts(self, stats: Dict[str, Any]) -> None:
        """
        Generates feedback visualizations if matplotlib is available and history is populated.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.info("Skipping chart generation: matplotlib is not available.")
            return

        if not os.path.exists(self.history_csv_path):
            logger.warning(f"History CSV not found at {self.history_csv_path}. Skipping chart generation.")
            return

        try:
            df = pd.read_csv(self.history_csv_path)
        except Exception as e:
            logger.error(f"Failed to read history CSV for charting: {e}")
            return

        if len(df) == 0:
            logger.warning("History CSV is empty. Skipping chart generation.")
            return

        try:
            # 1. Feedback Distribution (Pie chart of Feedback options)
            feedback_counts = df["Feedback"].value_counts()
            plt.figure(figsize=(6, 6))
            plt.pie(
                feedback_counts.values,
                labels=feedback_counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=["lightgreen", "lightcoral", "skyblue"][:len(feedback_counts)]
            )
            plt.title("User Feedback Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, "feedback_distribution.png"))
            plt.close()

            # 2. Correct vs Incorrect (Bar chart)
            correct_val = stats.get("correct_count", 0)
            incorrect_val = stats.get("incorrect_count", 0)
            plt.figure(figsize=(6, 5))
            plt.bar(["Correct", "Incorrect"], [correct_val, incorrect_val], color=["green", "red"], edgecolor="black", width=0.5)
            plt.title("Correct vs Incorrect Feedback")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, "correct_vs_incorrect.png"))
            plt.close()

            # 3. Prediction Distribution (Bar chart)
            pred_counts = df["Prediction"].value_counts()
            plt.figure(figsize=(6, 5))
            plt.bar([str(x) for x in pred_counts.index], pred_counts.values, color="skyblue", edgecolor="black", width=0.5)
            plt.title("Prediction Value Distribution")
            plt.xlabel("Prediction Label")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, "prediction_distribution.png"))
            plt.close()

            # 4. Verification Distribution (Pie chart)
            ver_counts = df["Verification"].value_counts()
            plt.figure(figsize=(6, 6))
            plt.pie(
                ver_counts.values,
                labels=ver_counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=["gold", "lightcoral", "lightgreen", "skyblue"][:len(ver_counts)]
            )
            plt.title("Verification Status Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, "verification_distribution.png"))
            plt.close()

            # 5. Decision Distribution (Bar chart of Final Decisions)
            dec_counts = df["Decision"].value_counts()
            plt.figure(figsize=(8, 5))
            plt.bar(dec_counts.index, dec_counts.values, color="plum", edgecolor="black", width=0.5)
            plt.title("Final Decision Distribution")
            plt.ylabel("Count")
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, "decision_distribution.png"))
            plt.close()

            logger.info("Successfully generated feedback visualization charts.")
        except Exception as e:
            logger.warning(f"Could not generate charts via matplotlib: {e}")
