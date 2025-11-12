"""Test ComponentResult.to_dict() fix for status reporting bug."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import directly from the status.py file to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location("status", Path(__file__).parent / "src/metrics/status.py")
status_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status_module)
ComponentResult = status_module.ComponentResult
ComponentStatus = status_module.ComponentStatus

print("=" * 60)
print("ComponentResult.to_dict() Fix Verification")
print("=" * 60)

# Test 1: Success with data
print("\n1. Success with data:")
result = ComponentResult.success(data={"K_eff": 7.3, "active_components": 8})
d = result.to_dict()
print(f"   Result: {d}")
assert "status" in d, "Missing 'status' key!"
assert d["status"] == "success", f"Wrong status: {d['status']}"
assert d["K_eff"] == 7.3, "Data not included!"
print("   ✓ Status included with data")

# Test 2: Success with empty data (THE BUG CASE)
print("\n2. Success with empty data (previously returned {}):")
result = ComponentResult.success(data={})
d = result.to_dict()
print(f"   Result: {d}")
assert "status" in d, "BUG: Missing 'status' key!"
assert d["status"] == "success", f"Wrong status: {d['status']}"
assert len(d) == 1, f"Should only have status key, got: {d}"
print("   ✓ Status included even with empty data")

# Test 3: Disabled
print("\n3. Disabled status:")
result = ComponentResult.disabled(reason="Requires mixture prior")
d = result.to_dict()
print(f"   Result: {d}")
assert "status" in d, "Missing 'status' key!"
assert d["status"] == "disabled", f"Wrong status: {d['status']}"
assert d["reason"] == "Requires mixture prior", "Reason not included!"
print("   ✓ Disabled status works correctly")

# Test 4: Skipped
print("\n4. Skipped status:")
result = ComponentResult.skipped(reason="No training data available")
d = result.to_dict()
print(f"   Result: {d}")
assert "status" in d, "Missing 'status' key!"
assert d["status"] == "skipped", f"Wrong status: {d['status']}"
assert d["reason"] == "No training data available", "Reason not included!"
print("   ✓ Skipped status works correctly")

# Test 5: Failed
print("\n5. Failed status:")
try:
    raise ValueError("Test error")
except ValueError as e:
    result = ComponentResult.failed(reason="Computation failed", error=e)
    d = result.to_dict()
    print(f"   Result: {d}")
    assert "status" in d, "Missing 'status' key!"
    assert d["status"] == "failed", f"Wrong status: {d['status']}"
    assert d["reason"] == "Computation failed", "Reason not included!"
    print("   ✓ Failed status works correctly")

# Test 6: Backward compatibility - metrics registry expects data to be merged
print("\n6. Backward compatibility - metrics with data:")
result = ComponentResult.success(data={
    "final_loss": 76.0602,
    "epochs_completed": 100,
    "training_time_sec": 25.28
})
d = result.to_dict()
print(f"   Result keys: {list(d.keys())}")
assert d["status"] == "success", "Status missing!"
assert d["final_loss"] == 76.0602, "Metrics not merged!"
assert d["epochs_completed"] == 100, "Metrics not merged!"
print("   ✓ Metrics merged correctly with status")

print("\n" + "=" * 60)
print("All ComponentResult.to_dict() tests passed! ✓")
print("=" * 60)
print("\nThe fix ensures:")
print("  - Status is ALWAYS included in to_dict() output")
print("  - Empty data {} no longer causes 'unknown' status")
print("  - Data is merged with status for metrics")
print("  - Backward compatibility maintained")
