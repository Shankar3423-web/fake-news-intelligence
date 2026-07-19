import os
import tempfile
import yaml
from ml.feature_selection.selection_config import SelectionConfig

def test_config_defaults() -> None:
    # Test fallback to DEFAULT_CONFIG when config file is missing
    config = SelectionConfig("config/non_existent_path.yaml")
    
    assert config.is_selector_enabled("variance") is True
    assert config.is_selector_enabled("rfe") is True
    assert config.merger_strategy == "voting"
    assert config.merger_voting_threshold == 2
    assert config.random_state == 42
    assert config.get_path("input_csv") == "ml/feature_engineering/processed/feature_dataset_v1.csv"

def test_config_custom() -> None:
    # Test loading custom config via YAML file
    custom_data = {
        "selectors": {
            "variance": {
                "enabled": False,
                "threshold": 0.05
            },
            "correlation": {
                "enabled": True,
                "threshold": 0.90
            }
        },
        "merger": {
            "strategy": "union"
        },
        "paths": {
            "input_csv": "custom_input.csv"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom_data, f)
        temp_path = f.name
        
    try:
        config = SelectionConfig(temp_path)
        assert config.is_selector_enabled("variance") is False
        assert config.get_selector_setting("variance", "threshold") == 0.05
        assert config.is_selector_enabled("correlation") is True
        assert config.get_selector_setting("correlation", "threshold") == 0.90
        # RFE should fall back to default
        assert config.is_selector_enabled("rfe") is True
        assert config.merger_strategy == "union"
        assert config.get_path("input_csv") == "custom_input.csv"
    finally:
        os.remove(temp_path)
