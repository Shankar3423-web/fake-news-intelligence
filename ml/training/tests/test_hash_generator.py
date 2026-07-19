import os
import json
import pytest
from ml.training.training_config import TrainingConfig
from ml.training.hash_generator import HashGenerator

def test_hash_generator_lifecycle(tmp_path):
    config_yaml = tmp_path / "training_config.yaml"
    with open(config_yaml, "w") as f:
        f.write(f"""
outputs:
  hashes_dir: "{tmp_path}/hashes"
""")
        
    config = TrainingConfig(str(config_yaml))
    hash_gen = HashGenerator(config)
    
    # Create mock file to hash
    target_file = tmp_path / "model.joblib"
    with open(target_file, "w") as f:
        f.write("mock binary contents")
        
    paths_dict = {"model": str(target_file)}
    hash_file = hash_gen.generate_model_hashes("logistic_regression", paths_dict)
    
    assert os.path.exists(hash_file)
    with open(hash_file, "r") as f:
        data = json.load(f)
        
    # Check that a hash exists as key
    assert len(data) == 1
    # Verify that the value is the target file path
    assert list(data.values())[0] == str(target_file)
