"""Pytest configuration for experiment management tests.

This conftest.py ensures that the project root is in sys.path so that
all project modules (src/, use_cases/) can be imported correctly.
"""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add src/ for editable installs without packaging step
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
