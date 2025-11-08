"""
Run safety tests against the current SSVAE implementation.

This verifies the current implementation passes all baseline tests.
Note: Since the refactored version is now the default, this just runs
the standard safety tests without any monkey-patching.
"""
from __future__ import annotations

# Just re-export all safety tests to run against current implementation
from test_refactor_safety import *

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
