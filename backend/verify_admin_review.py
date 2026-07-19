import os
import sys

# Ensure root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import verification logic from the admin_review package
from ml.admin_review.verify_admin_review import verify

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
