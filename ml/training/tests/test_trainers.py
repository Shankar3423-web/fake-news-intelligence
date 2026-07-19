import os
import pytest
import pandas as pd
import numpy as np
from ml.training.logistic_regression_trainer import LogisticRegressionTrainer
from ml.training.svm_trainer import SvmTrainer
from ml.training.random_forest_trainer import RandomForestTrainer
from ml.training.xgboost_trainer import XgBoostTrainer
from ml.training.trainer_factory import TrainerFactory

@pytest.fixture
def dummy_data():
    X = pd.DataFrame({
        "feat1": np.random.rand(50),
        "feat2": np.random.rand(50)
    })
    y = pd.Series(np.random.choice([0, 1], size=50))
    split_info = {
        "timestamp": "2026-07-19T00:00:00",
        "config": {"test_size": 0.2, "random_state": 42},
        "training_samples": 40,
        "testing_samples": 10
    }
    return X, y, split_info

def test_logistic_regression_trainer(dummy_data, tmp_path):
    X, y, split_info = dummy_data
    hyperparams = {"solver": "lbfgs", "max_iter": 100, "C": 1.0, "random_state": 42}
    
    trainer = LogisticRegressionTrainer(hyperparams)
    assert trainer.algorithm_name == "Logistic Regression"
    
    summary = trainer.train(X, y, split_info)
    assert summary["algorithm"] == "Logistic Regression"
    assert "training_duration_sec" in summary
    assert "memory_used_mb" in summary
    assert summary["feature_count"] == 2
    assert summary["samples_trained"] == 50
    
    saved = trainer.save(str(tmp_path))
    assert os.path.exists(saved["model"])
    assert os.path.exists(saved["config"])
    assert os.path.exists(saved["features"])
    assert os.path.exists(saved["metadata"])

def test_svm_trainer(dummy_data, tmp_path):
    X, y, split_info = dummy_data
    hyperparams = {"C": 1.0, "max_iter": 100, "random_state": 42}
    
    trainer = SvmTrainer(hyperparams)
    assert trainer.algorithm_name == "Linear SVM"
    
    summary = trainer.train(X, y, split_info)
    assert summary["algorithm"] == "Linear SVM"
    
    saved = trainer.save(str(tmp_path))
    assert os.path.exists(saved["model"])

def test_random_forest_trainer(dummy_data, tmp_path):
    X, y, split_info = dummy_data
    hyperparams = {"n_estimators": 5, "max_depth": 3, "random_state": 42, "n_jobs": 1}
    
    trainer = RandomForestTrainer(hyperparams)
    assert trainer.algorithm_name == "Random Forest"
    
    summary = trainer.train(X, y, split_info)
    assert summary["algorithm"] == "Random Forest"
    
    saved = trainer.save(str(tmp_path))
    assert os.path.exists(saved["model"])

def test_xgboost_trainer(dummy_data, tmp_path):
    X, y, split_info = dummy_data
    hyperparams = {
        "n_estimators": 5,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
        "objective": "binary:logistic",
        "eval_metric": "logloss"
    }
    
    trainer = XgBoostTrainer(hyperparams)
    assert trainer.algorithm_name == "XGBoost"
    
    summary = trainer.train(X, y, split_info)
    assert summary["algorithm"] == "XGBoost"
    
    saved = trainer.save(str(tmp_path))
    assert os.path.exists(saved["model"])

def test_trainer_factory():
    hyperparams = {"random_state": 42}
    
    lr_trainer = TrainerFactory.create_trainer("logistic_regression", hyperparams)
    assert isinstance(lr_trainer, LogisticRegressionTrainer)
    
    svm_trainer = TrainerFactory.create_trainer("svm", hyperparams)
    assert isinstance(svm_trainer, SvmTrainer)
    
    rf_trainer = TrainerFactory.create_trainer("random_forest", hyperparams)
    assert isinstance(rf_trainer, RandomForestTrainer)
    
    xgb_trainer = TrainerFactory.create_trainer("xgboost", hyperparams)
    assert isinstance(xgb_trainer, XgBoostTrainer)
    
    with pytest.raises(ValueError, match="Unsupported model type"):
        TrainerFactory.create_trainer("invalid_classifier", hyperparams)
