"""Script de teste para isolar problemas de importação."""

import sys
print(f"Python version: {sys.version}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except Exception as e:
    print(f"Error importing pandas: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except Exception as e:
    print(f"Error importing numpy: {e}")

try:
    from sklearn import __version__ as sklearn_version
    print(f"Scikit-learn version: {sklearn_version}")
except Exception as e:
    print(f"Error importing scikit-learn: {e}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except Exception as e:
    print(f"Error importing matplotlib: {e}")

print("All basic imports successful!")