"""Tests for dashboard configuration page."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_config_page_imports():
    """Test that config page can be imported."""
    from use_cases.dashboard.pages_training import build_training_config_page
    
    layout = build_training_config_page()
    assert layout is not None


def test_config_serialization():
    """Test that SSVAEConfig can be serialized to dict."""
    import dataclasses
    from src.ssvae.config import SSVAEConfig
    
    config = SSVAEConfig(batch_size=512, encoder_type='conv', latent_dim=10)
    config_dict = dataclasses.asdict(config)
    
    assert config_dict['batch_size'] == 512
    assert config_dict['encoder_type'] == 'conv'
    assert config_dict['latent_dim'] == 10
    assert 'hidden_dims' in config_dict
    assert 'learning_rate' in config_dict


def test_config_callbacks_import():
    """Test that config callbacks can be imported."""
    from use_cases.dashboard.callbacks.config_callbacks import register_config_callbacks
    
    assert callable(register_config_callbacks)


if __name__ == "__main__":
    print("Running config page tests...")
    test_config_page_imports()
    print("✓ Config page imports successfully")
    
    test_config_serialization()
    print("✓ Config serialization works")
    
    test_config_callbacks_import()
    print("✓ Config callbacks import successfully")
    
    print("\n✅ All tests passed!")
