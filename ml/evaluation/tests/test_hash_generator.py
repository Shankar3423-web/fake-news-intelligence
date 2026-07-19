import os
import json
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.hash_generator import HashGenerator

def test_hash_generator(tmp_path):
    config_yaml = tmp_path / "evaluation_config.yaml"
    hashes_file = tmp_path / "evaluation_hashes.json"
    
    with open(config_yaml, "w") as f:
        f.write(f"""
outputs:
  hashes_dir: {str(tmp_path).replace("\\", "/")}
  hashes_file: {str(hashes_file).replace("\\", "/")}
""")
        
    config = EvaluationConfig(str(config_yaml))
    hash_gen = HashGenerator(config)
    
    # Create mock file to hash
    test_file = tmp_path / "test.txt"
    with open(test_file, "w") as f:
        f.write("hello world")
        
    files_map = {"test_file": str(test_file)}
    
    path = hash_gen.generate_hashes(files_map)
    assert os.path.exists(path)
    assert os.path.normpath(path) == os.path.normpath(str(hashes_file))
    
    with open(hashes_file, "r") as f:
        data = json.load(f)
    assert len(data) == 1
    # Check that key is a hash string and value is the path
    for k, v in data.items():
        assert len(k) == 64 # SHA-256 length
        assert v == str(test_file)
