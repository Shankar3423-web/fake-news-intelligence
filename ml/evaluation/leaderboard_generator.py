import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("model_evaluation_pipeline")

class LeaderboardGenerator:
    """
    Ranks evaluated models based on their overall score.
    Generates leaderboard.json, leaderboard.csv, and leaderboard.md.
    """
    def __init__(self, output_dir: str = "ml/evaluation/leaderboard") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, model_eval_results: Dict[str, Dict[str, Any]], overall_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Ranks models, exports leaderboard data to CSV/JSON/MD, and returns the list of ranked models.
        """
        logger.info("Generating evaluation leaderboard...")
        
        # 1. Collect and rank
        entries = []
        for key, res in model_eval_results.items():
            metrics = res["metrics"]
            overall_score = overall_scores.get(key, 0.0)
            
            entries.append({
                "Model": res["algorithm"],
                "Model Key": key,
                "Overall Score": overall_score,
                "F1": metrics["f1_score"],
                "ROC": metrics["roc_auc"],
                "Prediction Speed": metrics["inference_throughput_sps"], # samples/s
                "Memory": metrics["memory_used_mb"] # MB
            })
            
        # Sort in descending order of Overall Score
        entries.sort(key=lambda x: x["Overall Score"], reverse=True)
        
        # Add rank
        ranked_entries = []
        for rank, entry in enumerate(entries, 1):
            ranked_entries.append({
                "Rank": rank,
                **entry
            })
            
        df = pd.DataFrame(ranked_entries)
        
        # 2. Save CSV
        csv_path = os.path.join(self.output_dir, "leaderboard.csv")
        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved leaderboard CSV to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save leaderboard CSV: {e}")
            
        # 3. Save JSON
        json_path = os.path.join(self.output_dir, "leaderboard.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(ranked_entries, f, indent=4)
            logger.info(f"Saved leaderboard JSON to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save leaderboard JSON: {e}")
            
        # 4. Save Markdown
        md_path = os.path.join(self.output_dir, "leaderboard.md")
        try:
            md_content = self._format_leaderboard_markdown(ranked_entries)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Saved leaderboard Markdown to {md_path}")
        except Exception as e:
            logger.error(f"Failed to save leaderboard Markdown: {e}")
            
        return ranked_entries

    def _format_leaderboard_markdown(self, entries: List[Dict[str, Any]]) -> str:
        """Formats the leaderboard into a clean Markdown table."""
        md = "# Evaluation Leaderboard\n\n"
        md += "Model rankings based on overall evaluation score.\n\n"
        
        headers = ["Rank", "Model", "Overall Score", "F1", "ROC", "Prediction Speed (samples/s)", "Memory (MB)"]
        md += "|" + "|".join(headers) + "|\n"
        md += "|" + "|".join([":---:"] + [":---"] + [":---:"] * (len(headers) - 2)) + "|\n"
        
        for entry in entries:
            row = [
                str(entry["Rank"]),
                f"**{entry['Model']}**" if entry["Rank"] == 1 else entry["Model"],
                f"{entry['Overall Score']:.6f}",
                f"{entry['F1']:.4f}",
                f"{entry['ROC']:.4f}",
                f"{entry['Prediction Speed']:.2f}",
                f"{entry['Memory']:.2f}"
            ]
            md += "|" + "|".join(row) + "|\n"
            
        md += "\n"
        if entries:
            md += f"🏆 **Best Model:** {entries[0]['Model']} (Score: {entries[0]['Overall Score']:.6f})\n"
            
        return md
