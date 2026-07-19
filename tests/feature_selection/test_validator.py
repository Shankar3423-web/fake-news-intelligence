import os
import tempfile
import pandas as pd
from ml.feature_selection.selection_validator import SelectionValidator

def test_validator_compliance() -> None:
    validator = SelectionValidator()
    
    # Create temp input and output files
    df_in = pd.DataFrame({
        "id": [1, 2, 3],
        "label": [0, 1, 0],
        "cleaned_text": ["text1", "text2", "text3"],
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [4.0, 5.0, 6.0]
    })
    
    df_out_valid = pd.DataFrame({
        "id": [1, 2, 3],
        "label": [0, 1, 0],
        "cleaned_text": ["text1", "text2", "text3"],
        "feat1": [1.0, 2.0, 3.0]
    })
    
    # Output with null values
    df_out_nulls = pd.DataFrame({
        "id": [1, 2, 3],
        "label": [0, 1, 0],
        "cleaned_text": ["text1", "text2", "text3"],
        "feat1": [1.0, None, 3.0]
    })
    
    # Output with mismatched order
    df_out_mismatch = pd.DataFrame({
        "id": [2, 1, 3],
        "label": [1, 0, 0],
        "cleaned_text": ["text2", "text1", "text3"],
        "feat1": [2.0, 1.0, 3.0]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in.csv")
        out_valid_path = os.path.join(tmpdir, "out_valid.csv")
        out_nulls_path = os.path.join(tmpdir, "out_nulls.csv")
        out_mismatch_path = os.path.join(tmpdir, "out_mismatch.csv")
        
        df_in.to_csv(in_path, index=False)
        df_out_valid.to_csv(out_valid_path, index=False)
        df_out_nulls.to_csv(out_nulls_path, index=False)
        df_out_mismatch.to_csv(out_mismatch_path, index=False)
        
        # Test valid output
        passed, errors = validator.validate_outputs(in_path, out_valid_path, ["feat1"])
        assert passed is True
        assert len(errors) == 0
        
        # Test output with nulls
        passed, errors = validator.validate_outputs(in_path, out_nulls_path, ["feat1"])
        assert passed is False
        assert any("contains 1 null/NaN values" in err for err in errors)
        
        # Test output with sample order mismatch
        passed, errors = validator.validate_outputs(in_path, out_mismatch_path, ["feat1"])
        assert passed is False
        assert any("Sample order mismatch" in err for err in errors)
