#!/usr/bin/env python
# test_imports.py - Simple test to verify imports work

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing basic imports...")

try:
    import pandas as pd
    import numpy as np
    print('✅ Basic libraries imported')
except Exception as e:
    print(f'❌ Basic libraries error: {e}')

try:
    from indicators.technical_indicators import TechnicalIndicators
    print('✅ TechnicalIndicators imported')
except Exception as e:
    print(f'❌ TechnicalIndicators error: {e}')

try:
    import stable_baselines3
    print('✅ stable-baselines3 imported')
except Exception as e:
    print(f'❌ stable-baselines3 error: {e}')

try:
    import gymnasium
    print('✅ gymnasium imported')
except Exception as e:
    print(f'❌ gymnasium error: {e}')

print("\nImport test complete!")
