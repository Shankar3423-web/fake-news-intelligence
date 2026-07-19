import os
from ml.evaluation.leaderboard_generator import LeaderboardGenerator

def test_leaderboard_generator(tmp_path):
    output_dir = tmp_path / "leaderboard"
    output_dir.mkdir()
    
    model_eval_results = {
        "xgboost": {
            "algorithm": "XGBoost",
            "metrics": {
                "f1_score": 0.95,
                "roc_auc": 0.97,
                "inference_throughput_sps": 200.0,
                "memory_used_mb": 4.5
            }
        },
        "logistic_regression": {
            "algorithm": "Logistic Regression",
            "metrics": {
                "f1_score": 0.90,
                "roc_auc": 0.92,
                "inference_throughput_sps": 1000.0,
                "memory_used_mb": 0.5
            }
        }
    }
    
    overall_scores = {
        "xgboost": 0.95,
        "logistic_regression": 0.91
    }
    
    gen = LeaderboardGenerator(output_dir=str(output_dir))
    ranked = gen.generate(model_eval_results, overall_scores)
    
    assert len(ranked) == 2
    assert ranked[0]["Model Key"] == "xgboost"
    assert ranked[0]["Rank"] == 1
    assert ranked[1]["Model Key"] == "logistic_regression"
    assert ranked[1]["Rank"] == 2
    
    assert os.path.exists(os.path.join(output_dir, "leaderboard.json"))
    assert os.path.exists(os.path.join(output_dir, "leaderboard.csv"))
    assert os.path.exists(os.path.join(output_dir, "leaderboard.md"))
