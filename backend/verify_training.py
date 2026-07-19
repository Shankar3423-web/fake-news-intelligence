import os
import sys

# Ensure root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import verification logic from the training package
from ml.training.verify_training import verify_training_integrity

if __name__ == "__main__":
    passed, errors = verify_training_integrity(project_root)
    
    if passed:
        print("\n" + "="*52)
        print("PHASE 6 MODEL TRAINING VERIFICATION PASSED")
        print("="*52 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*52)
        print("PHASE 6 MODEL TRAINING VERIFICATION FAILED")
        print("="*52)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        print("="*52 + "\n")
        sys.exit(1)
