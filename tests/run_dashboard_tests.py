#!/usr/bin/env python3
"""Standalone test runner for dashboard integration tests.

Run without pytest: python tests/run_dashboard_tests.py
"""

import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.commands import (
    CreateModelCommand,
    LoadModelCommand,
    DeleteModelCommand,
)
from use_cases.dashboard.core.model_manager import ModelManager


class TestRunner:
    """Simple test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.temp_dir = None
    
    def setup(self):
        """Setup temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_models = Path(self.temp_dir) / "models"
        self.temp_models.mkdir()
        
        # Monkey patch MODELS_DIR
        import use_cases.dashboard.core.model_manager as mm
        mm.MODELS_DIR = self.temp_models
        
        # Reset state
        dashboard_state.app_state = None
        dashboard_state._initialized = False
    
    def teardown(self):
        """Cleanup."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset state
        dashboard_state.app_state = None
        dashboard_state._initialized = False
    
    def run_test(self, test_func, test_name):
        """Run a single test."""
        try:
            self.setup()
            print(f"\n▶ {test_name}...", end=" ")
            test_func(self.temp_models)
            print("✓ PASS")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ FAIL")
            print(f"  AssertionError: {e}")
            self.failed += 1
            self.errors.append((test_name, str(e)))
        except Exception as e:
            print(f"✗ ERROR")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            self.failed += 1
            self.errors.append((test_name, f"{type(e).__name__}: {e}"))
        finally:
            self.teardown()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("="*70)
        
        if self.errors:
            print("\nFAILURES:")
            for test_name, error in self.errors:
                print(f"\n{test_name}:")
                print(f"  {error}")
        
        return self.failed == 0


# Test functions
def test_create_model_with_simple_name(temp_models_dir):
    """Test creating a model with a simple alphanumeric name."""
    dashboard_state.initialize_app_state()
    
    cmd = CreateModelCommand(name="test_model", config_preset="default")
    success, model_id = dashboard_state.dispatcher.execute(cmd)
    
    assert success, "Model creation should succeed"
    assert model_id == "test_model", f"Expected 'test_model', got '{model_id}'"
    
    model_dir = temp_models_dir / "test_model"
    assert model_dir.exists(), "Model directory should exist"
    assert (model_dir / "metadata.json").exists(), "Metadata file should exist"


def test_create_model_with_spaces_and_special_chars(temp_models_dir):
    """Test that model names are properly sanitized."""
    dashboard_state.initialize_app_state()
    
    cmd = CreateModelCommand(name="My Test Model #1!", config_preset="default")
    success, model_id = dashboard_state.dispatcher.execute(cmd)
    
    assert success
    assert model_id == "my_test_model_1", f"Expected 'my_test_model_1', got '{model_id}'"
    
    model_dir = temp_models_dir / "my_test_model_1"
    assert model_dir.exists(), "Sanitized model directory should exist"


def test_create_model_with_duplicate_name(temp_models_dir):
    """Test that duplicate names get unique IDs."""
    dashboard_state.initialize_app_state()
    
    cmd1 = CreateModelCommand(name="duplicate", config_preset="default")
    success1, model_id1 = dashboard_state.dispatcher.execute(cmd1)
    
    cmd2 = CreateModelCommand(name="duplicate", config_preset="default")
    success2, model_id2 = dashboard_state.dispatcher.execute(cmd2)
    
    assert success1 and success2
    assert model_id1 == "duplicate"
    assert model_id2 == "duplicate_1", f"Expected 'duplicate_1', got '{model_id2}'"
    
    assert (temp_models_dir / "duplicate").exists()
    assert (temp_models_dir / "duplicate_1").exists()


def test_delete_specific_model(temp_models_dir):
    """CRITICAL: Test that deleting a model deletes the CORRECT model."""
    dashboard_state.initialize_app_state()
    
    # Create three models
    _, id_a = dashboard_state.dispatcher.execute(
        CreateModelCommand(name="model_a", config_preset="default")
    )
    _, id_b = dashboard_state.dispatcher.execute(
        CreateModelCommand(name="model_b", config_preset="default")
    )
    _, id_c = dashboard_state.dispatcher.execute(
        CreateModelCommand(name="model_c", config_preset="default")
    )
    
    # Verify all exist
    assert (temp_models_dir / "model_a").exists()
    assert (temp_models_dir / "model_b").exists()
    assert (temp_models_dir / "model_c").exists()
    
    # Delete model_b
    delete_cmd = DeleteModelCommand(model_id="model_b")
    success, msg = dashboard_state.dispatcher.execute(delete_cmd)
    
    assert success, f"Delete should succeed: {msg}"
    
    # Verify ONLY model_b was deleted
    assert (temp_models_dir / "model_a").exists(), "model_a should still exist"
    assert not (temp_models_dir / "model_b").exists(), "model_b should be deleted"
    assert (temp_models_dir / "model_c").exists(), "model_c should still exist"


def test_delete_first_model_in_list(temp_models_dir):
    """REGRESSION TEST: delete first model should not delete wrong one."""
    dashboard_state.initialize_app_state()
    
    cmd1 = CreateModelCommand(name="mnist_baseline", config_preset="default")
    cmd2 = CreateModelCommand(name="experiment_001", config_preset="default")
    
    _, id1 = dashboard_state.dispatcher.execute(cmd1)
    _, id2 = dashboard_state.dispatcher.execute(cmd2)
    
    # Try to delete mnist_baseline
    delete_cmd = DeleteModelCommand(model_id="mnist_baseline")
    success, msg = dashboard_state.dispatcher.execute(delete_cmd)
    
    assert success
    
    # CRITICAL: mnist_baseline should be gone, experiment_001 should remain
    assert not (temp_models_dir / "mnist_baseline").exists(), \
        "mnist_baseline should be deleted"
    assert (temp_models_dir / "experiment_001").exists(), \
        "experiment_001 should still exist"


def test_cannot_delete_active_model(temp_models_dir):
    """Test that active model cannot be deleted."""
    dashboard_state.initialize_app_state()
    
    create_cmd = CreateModelCommand(name="active_model", config_preset="default")
    _, model_id = dashboard_state.dispatcher.execute(create_cmd)
    
    load_cmd = LoadModelCommand(model_id=model_id)
    dashboard_state.dispatcher.execute(load_cmd)
    
    # Try to delete active model
    delete_cmd = DeleteModelCommand(model_id=model_id)
    success, msg = dashboard_state.dispatcher.execute(delete_cmd)
    
    assert not success, "Should not allow deleting active model"
    assert "Cannot delete active model" in msg


def test_load_model_basic(temp_models_dir):
    """Test basic model loading."""
    dashboard_state.initialize_app_state()
    
    create_cmd = CreateModelCommand(name="test_load", config_preset="default")
    _, model_id = dashboard_state.dispatcher.execute(create_cmd)
    
    load_cmd = LoadModelCommand(model_id=model_id)
    success, msg = dashboard_state.dispatcher.execute(load_cmd)
    
    assert success, f"Load should succeed: {msg}"
    
    with dashboard_state.state_lock:
        assert dashboard_state.app_state.active_model is not None
        assert dashboard_state.app_state.active_model.model_id == model_id


def test_state_matches_filesystem(temp_models_dir):
    """Test that state registry matches filesystem."""
    dashboard_state.initialize_app_state()
    
    _, id1 = dashboard_state.dispatcher.execute(
        CreateModelCommand(name="fs_test_1", config_preset="default")
    )
    _, id2 = dashboard_state.dispatcher.execute(
        CreateModelCommand(name="fs_test_2", config_preset="default")
    )
    
    with dashboard_state.state_lock:
        state_model_ids = set(dashboard_state.app_state.models.keys())
    
    fs_model_ids = {d.name for d in temp_models_dir.iterdir() if d.is_dir()}
    
    assert state_model_ids == fs_model_ids, \
        f"State and filesystem mismatch: state={state_model_ids}, fs={fs_model_ids}"


def main():
    """Run all tests."""
    print("="*70)
    print("DASHBOARD INTEGRATION TESTS")
    print("="*70)
    
    runner = TestRunner()
    
    # Core functionality tests
    print("\n📦 MODEL CREATION TESTS")
    runner.run_test(test_create_model_with_simple_name, "Create model with simple name")
    runner.run_test(test_create_model_with_spaces_and_special_chars, "Create model with spaces/special chars")
    runner.run_test(test_create_model_with_duplicate_name, "Create models with duplicate names")
    
    # Deletion tests (the critical bug)
    print("\n🗑️  MODEL DELETION TESTS")
    runner.run_test(test_delete_specific_model, "Delete specific model (not others)")
    runner.run_test(test_delete_first_model_in_list, "Delete first model (regression test)")
    runner.run_test(test_cannot_delete_active_model, "Cannot delete active model")
    
    # Loading tests
    print("\n📂 MODEL LOADING TESTS")
    runner.run_test(test_load_model_basic, "Load model basic")
    
    # Consistency tests
    print("\n🔄 STATE CONSISTENCY TESTS")
    runner.run_test(test_state_matches_filesystem, "State matches filesystem")
    
    # Print summary
    success = runner.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
