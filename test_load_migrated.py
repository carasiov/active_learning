#!/usr/bin/env python3
"""Test loading the migrated baseline model."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from use_cases.dashboard import state as dashboard_state
from use_cases.dashboard.commands import LoadModelCommand

# Initialize
dashboard_state.initialize_app_state()

print("Models in registry:")
with dashboard_state.state_lock:
    for model_id, metadata in dashboard_state.app_state.models.items():
        print(f"  - {model_id}: {metadata.name}")

print("\nTrying to load model_migrated_baseline_001...")
load_cmd = LoadModelCommand(model_id="model_migrated_baseline_001")

# Check validation first
with dashboard_state.state_lock:
    error = load_cmd.validate(dashboard_state.app_state)
    if error:
        print(f"Validation failed: {error}")
        sys.exit(1)

print("Validation passed, executing...")
success, message = dashboard_state.dispatcher.execute(load_cmd)

print(f"Success: {success}")
print(f"Message: {message}")

if success:
    with dashboard_state.state_lock:
        if dashboard_state.app_state.active_model:
            print(f"✓ Model loaded: {dashboard_state.app_state.active_model.model_id}")
        else:
            print("✗ No active model despite success")
else:
    print(f"✗ Load failed: {message}")
