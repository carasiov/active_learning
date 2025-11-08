"""
Run safety tests against the REFACTORED version.

This verifies the refactored implementation passes all baseline tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Temporarily replace the import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Monkey-patch the import to use refactored version
import ssvae
from ssvae.models_refactored import SSVAE as SSVAERefactored

# Replace SSVAE with refactored version for testing
ssvae.SSVAE = SSVAERefactored
ssvae.models.SSVAE = SSVAERefactored

# Now import the safety tests (they'll use the refactored version)
from test_refactor_safety import *

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
