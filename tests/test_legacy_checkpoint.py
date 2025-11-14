"""Test loading legacy checkpoint without mixture fields."""
import json
from pathlib import Path

from model.ssvae import SSVAEConfig


def test_load_legacy_checkpoint_config():
    """Verify legacy config (without mixture fields) loads with defaults."""
    config_path = Path("/workspaces/active_learning_showcase/use_cases/artifacts/models/test1/config.json")
    
    if not config_path.exists():
        print(f"Skipping: {config_path} not found")
        return
    
    with open(config_path) as f:
        legacy_dict = json.load(f)
    
    # Verify legacy config doesn't have mixture fields
    assert "prior_type" not in legacy_dict
    assert "num_components" not in legacy_dict
    assert "component_kl_weight" not in legacy_dict
    
    # Create config from legacy dict - should use defaults for missing fields
    config = SSVAEConfig(**legacy_dict)
    
    # Verify defaults applied
    assert config.prior_type == "standard"
    assert config.num_components == 10
    assert config.kl_c_weight == 1.0
    
    # Verify other fields preserved
    assert config.latent_dim == 2
    assert config.encoder_type == "dense"
    assert config.learning_rate == 0.001


if __name__ == "__main__":
    test_load_legacy_checkpoint_config()
    print("✓ Legacy config loads successfully with mixture field defaults")
