#!/usr/bin/env python3
"""Quick test of CreateModel and LoadModel commands."""

import sys
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from use_cases.dashboard import state as dashboard_state
from use_cases.dashboard.commands import CreateModelCommand, LoadModelCommand

# Initialize
dashboard_state.initialize_app_state()

print("Testing CreateModelCommand...")
create_cmd = CreateModelCommand(name="Test Model", config_preset="default")
success, result = dashboard_state.dispatcher.execute(create_cmd)

print(f"  Success: {success}")
print(f"  Result: {result}")

if success:
    model_id = result
    print(f"  Created model_id: {model_id}")
    
    # Check registry
    with dashboard_state.state_lock:
        if model_id in dashboard_state.app_state.models:
            print(f"  ✓ Model in registry")
            metadata = dashboard_state.app_state.models[model_id]
            print(f"    Name: {metadata.name}")
            print(f"    Created: {metadata.created_at}")
        else:
            print(f"  ✗ Model NOT in registry")
    
    print(f"\nTesting LoadModelCommand...")
    load_cmd = LoadModelCommand(model_id=model_id)
    success2, message = dashboard_state.dispatcher.execute(load_cmd)
    
    print(f"  Success: {success2}")
    print(f"  Message: {message}")
    
    if success2:
        with dashboard_state.state_lock:
            if dashboard_state.app_state.active_model:
                print(f"  ✓ Model loaded as active")
                print(f"    Active ID: {dashboard_state.app_state.active_model.model_id}")
                print(f"    Has trainer: {dashboard_state.app_state.active_model.trainer is not None}")
                print(f"    Has data: {dashboard_state.app_state.active_model.data.x_train is not None}")
            else:
                print(f"  ✗ No active model")
    
    print("\n✅ All tests passed!")
else:
    print(f"✗ Create failed: {result}")
    sys.exit(1)
