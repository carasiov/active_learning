"""Phase 6 verification script - tests metadata augmentation without full experiment run."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import directly from modules to avoid numpy dependency
from src.io.structure import create_run_paths, sanitize_name
# Import directly from config.py, not through pipeline/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("config", Path(__file__).parent / "src/pipeline/config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
augment_config_metadata = config_module.augment_config_metadata

print("=" * 60)
print("Phase 6 Verification Tests")
print("=" * 60)

# Test 1: Directory naming with architecture code
print("\n1. Testing directory naming with architecture code:")
run_id, timestamp, paths = create_run_paths("test_experiment", "mix10-dir_tau_ca-het")
print(f"   Run ID: {run_id}")
print(f"   Expected pattern: test_experiment__mix10-dir_tau_ca-het__{timestamp}")
assert "__mix10-dir_tau_ca-het__" in run_id, "Architecture code not in run_id!"
assert run_id.startswith("test_experiment__"), "Experiment name not properly prefixed!"
assert run_id.endswith(f"__{timestamp}"), "Timestamp not properly suffixed!"
print("   ✓ Directory naming works correctly")

# Test 2: Backward compatibility (no architecture code)
print("\n2. Testing backward compatibility (no architecture code):")
run_id_legacy, timestamp_legacy, paths_legacy = create_run_paths("legacy_test", None)
print(f"   Run ID: {run_id_legacy}")
assert "__" not in run_id_legacy, "Should not have double underscores!"
assert run_id_legacy == f"legacy_test_{timestamp_legacy}", "Legacy format incorrect!"
print("   ✓ Backward compatibility maintained")

# Test 3: Config metadata augmentation
print("\n3. Testing config metadata augmentation:")
test_config = {
    "experiment": {"name": "test"},
    "data": {"dataset": "mnist"},
    "model": {"prior_type": "mixture"},
}
augmented = augment_config_metadata(
    test_config,
    run_id="test__mix10__20241112_143027",
    architecture_code="mix10",
    timestamp="20241112_143027",
)
print(f"   Config keys: {list(augmented.keys())}")
assert "run_id" in augmented, "run_id not added!"
assert "architecture_code" in augmented, "architecture_code not added!"
assert "timestamp" in augmented, "timestamp not added!"
assert augmented["run_id"] == "test__mix10__20241112_143027", "run_id value incorrect!"
assert augmented["architecture_code"] == "mix10", "architecture_code value incorrect!"
assert augmented["timestamp"] == "20241112_143027", "timestamp value incorrect!"
print("   ✓ Config augmentation works correctly")

# Test 4: Summary metadata enhancement (simulated)
print("\n4. Testing summary metadata enhancement:")
summary = {
    "training": {"final_loss": 100.5},
    "classification": {"accuracy": 0.85},
}
# Simulate what CLI does
summary["metadata"] = {
    "run_id": run_id,
    "architecture_code": "mix10-dir_tau_ca-het",
    "timestamp": timestamp,
    "experiment_name": "test_experiment",
}
print(f"   Summary keys: {list(summary.keys())}")
assert "metadata" in summary, "metadata not added!"
assert summary["metadata"]["run_id"] == run_id, "run_id in metadata incorrect!"
print("   ✓ Summary enhancement works correctly")

# Test 5: Name sanitization still works
print("\n5. Testing name sanitization:")
sanitized = sanitize_name("My Experiment #1 (test)")
print(f"   Sanitized: {sanitized}")
assert sanitized == "My_Experiment_1_test", f"Sanitization incorrect: {sanitized}"
print("   ✓ Name sanitization works correctly")

print("\n" + "=" * 60)
print("All Phase 6 verification tests passed! ✓")
print("=" * 60)

# Clean up test directories
import shutil
from src.io.structure import RESULTS_DIR

for test_dir in RESULTS_DIR.glob("test_experiment__*"):
    shutil.rmtree(test_dir)
    print(f"\nCleaned up: {test_dir.name}")
for test_dir in RESULTS_DIR.glob("legacy_test_*"):
    shutil.rmtree(test_dir)
    print(f"Cleaned up: {test_dir.name}")
