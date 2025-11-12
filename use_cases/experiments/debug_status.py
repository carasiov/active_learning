"""Debug script to check ComponentResult.to_dict() behavior."""

import sys
from pathlib import Path

print("=" * 60)
print("ComponentResult Status Debug")
print("=" * 60)

# Check if we can import directly
try:
    # Try importing from the source file directly
    spec_path = Path(__file__).parent / "src" / "metrics" / "status.py"
    print(f"\n1. Reading status.py from: {spec_path}")
    print(f"   File exists: {spec_path.exists()}")

    if spec_path.exists():
        with open(spec_path) as f:
            content = f.read()
            # Check for the fix
            if 'result = {"status": self.status.value}' in content:
                print("   ✓ Fix is present in source file")
            else:
                print("   ✗ Fix NOT present in source file!")
                print("   → You may need to pull the latest code")

    print("\n2. Checking for Python bytecode cache:")
    pycache = Path(__file__).parent / "src" / "metrics" / "__pycache__"
    if pycache.exists():
        cache_files = list(pycache.glob("status*.pyc"))
        if cache_files:
            print(f"   Found {len(cache_files)} cached file(s)")
            for cf in cache_files:
                print(f"   - {cf.name} (modified: {cf.stat().st_mtime})")
            print("   → Consider deleting __pycache__ directories")
        else:
            print("   No status.pyc cache found")
    else:
        print("   No __pycache__ directory")

    print("\n3. Testing ComponentResult behavior:")
    print("   (This will fail if there are import issues)")

    # Create a minimal test without full imports
    exec_globals = {}
    exec(compile(open(spec_path).read(), spec_path, 'exec'), exec_globals)
    ComponentResult = exec_globals['ComponentResult']

    # Test the fix
    result = ComponentResult.success(data={})
    d = result.to_dict()

    print(f"   ComponentResult.success(data={{}}).to_dict() = {d}")

    if "status" in d and d["status"] == "success":
        print("   ✓ Fix is WORKING! Status is present")
    else:
        print(f"   ✗ Fix NOT working! Missing status key")
        print(f"   → Dict keys: {list(d.keys())}")
        print(f"   → This should not happen if source file has the fix")

    # Test with data
    result_with_data = ComponentResult.success(data={"test": 123})
    d2 = result_with_data.to_dict()
    print(f"\n   ComponentResult.success(data={{'test': 123}}).to_dict() = {d2}")

    if "status" in d2 and "test" in d2:
        print("   ✓ Data merging works correctly")
    else:
        print("   ✗ Data merging issue")

except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Action Items:")
print("=" * 60)
print("1. If fix is NOT present: git pull origin <branch-name>")
print("2. If bytecode cache exists: rm -rf use_cases/experiments/src/**/__pycache__")
print("3. If fix present but not working: check Python version compatibility")
print("4. Re-run experiment after addressing issues")
