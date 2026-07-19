import os
import logging
import pandas as pd

logger = logging.getLogger("prediction_pipeline")

class PredictionProfiler:
    """
    Profiles execution metrics and generates visualization charts
    for latency, throughput, and confidence.
    """
    def __init__(self, charts_dir: str = "ml/prediction/charts") -> None:
        self.charts_dir = charts_dir
        os.makedirs(self.charts_dir, exist_ok=True)

    def generate_charts(self, history_csv_path: str) -> None:
        """
        Reads the prediction history CSV and plots metric distributions.
        Saves PNG charts if matplotlib is installed.
        """
        if not os.path.exists(history_csv_path):
            logger.warning(f"History CSV not found at {history_csv_path}. Skipping charts generation.")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed. Skipping prediction visualizations.")
            return

        try:
            df = pd.read_csv(history_csv_path)
            if df.empty:
                logger.info("Prediction history is empty. Skipping chart generation.")
                return

            # Plot Latency Line Chart
            if "Latency" in df.columns and len(df) >= 1:
                plt.figure(figsize=(8, 4))
                plt.plot(df.index + 1, df["Latency"], marker='o', color='#3b82f6', linewidth=2)
                plt.title("Prediction Latency Trend")
                plt.xlabel("Prediction Index")
                plt.ylabel("Latency (ms)")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                latency_path = os.path.join(self.charts_dir, "prediction_latency.png")
                plt.savefig(latency_path, dpi=150)
                plt.close()

            # Plot Throughput Line Chart
            if "Throughput" in df.columns and len(df) >= 1:
                plt.figure(figsize=(8, 4))
                plt.plot(df.index + 1, df["Throughput"], marker='s', color='#10b981', linewidth=2)
                plt.title("Prediction Throughput Trend")
                plt.xlabel("Prediction Index")
                plt.ylabel("Throughput (samples/sec)")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                tp_path = os.path.join(self.charts_dir, "prediction_throughput.png")
                plt.savefig(tp_path, dpi=150)
                plt.close()

            # Plot Confidence Distribution
            if "Confidence" in df.columns and len(df) >= 1:
                plt.figure(figsize=(8, 4))
                # If there is only 1 point, bins=10 might cause warning or look bad, but it works
                plt.hist(df["Confidence"], bins=min(10, max(1, len(df))), color='#8b5cf6', edgecolor='white', alpha=0.8)
                plt.title("Prediction Confidence Distribution")
                plt.xlabel("Confidence Score")
                plt.ylabel("Frequency")
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                conf_path = os.path.join(self.charts_dir, "prediction_confidence.png")
                plt.savefig(conf_path, dpi=150)
                plt.close()

            logger.info(f"Successfully generated prediction profile charts in {self.charts_dir}.")
        except Exception as e:
            logger.error(f"Failed to generate profiler charts: {e}")
            # Do not re-raise as this is optional visualization
