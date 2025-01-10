# test_imports.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/utilities'))


# Import shared module
try:
    import globi_db_queries
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")