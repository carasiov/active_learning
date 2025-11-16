"""Pytest configuration for service tests."""

import os
import sys

# IMPORTANT: Configure JAX to use CPU BEFORE any imports
# This prevents segfaults when running tests in environments without GPU
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "False"

# Add paths for imports
sys.path.insert(0, '/home/user/active_learning')
sys.path.insert(0, '/home/user/active_learning/src')
