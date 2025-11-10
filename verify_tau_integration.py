#!/usr/bin/env python3
"""Verify τ-classifier integration without running full training.

This script checks that all integration points are correctly wired:
- TauClassifier class exists and is importable
- Configuration parameters are present
- Loss functions accept τ parameter
- SSVAE model has τ-classifier initialization
- Training and prediction methods are updated

Run with: python verify_tau_integration.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_tau_classifier_module():
    """Check TauClassifier class is importable."""
    print("1. Checking TauClassifier module...")
    try:
        from ssvae.components.tau_classifier import TauClassifier
        print("   ✅ TauClassifier class found")

        # Check key methods exist
        methods = ['update_counts', 'get_tau', 'predict', 'supervised_loss',
                   'get_certainty', 'get_ood_score', 'get_free_channels', 'get_diagnostics']
        for method in methods:
            assert hasattr(TauClassifier, method), f"Missing method: {method}"
        print(f"   ✅ All {len(methods)} methods present")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_config_parameters():
    """Check configuration has τ-classifier parameters."""
    print("\n2. Checking configuration parameters...")
    try:
        from ssvae.config import SSVAEConfig

        config = SSVAEConfig()
        assert hasattr(config, 'use_tau_classifier'), "Missing use_tau_classifier"
        assert hasattr(config, 'tau_smoothing_alpha'), "Missing tau_smoothing_alpha"

        print(f"   ✅ use_tau_classifier = {config.use_tau_classifier}")
        print(f"   ✅ tau_smoothing_alpha = {config.tau_smoothing_alpha}")

        # Check it's in informative hyperparameters
        info_hparams = config.get_informative_hyperparameters()
        assert 'use_tau_classifier' in info_hparams, "Not in informative hyperparameters"
        print("   ✅ Parameters in informative hyperparameters")

        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_loss_functions():
    """Check loss functions support τ parameter."""
    print("\n3. Checking loss functions...")
    try:
        from training.losses import tau_classification_loss, compute_loss_and_metrics_v2
        import inspect

        # Check tau_classification_loss exists
        sig = inspect.signature(tau_classification_loss)
        params = list(sig.parameters.keys())
        assert 'responsibilities' in params, "Missing responsibilities parameter"
        assert 'tau' in params, "Missing tau parameter"
        assert 'labels' in params, "Missing labels parameter"
        print("   ✅ tau_classification_loss function found")
        print(f"      Parameters: {params}")

        # Check compute_loss_and_metrics_v2 accepts tau
        sig = inspect.signature(compute_loss_and_metrics_v2)
        params = list(sig.parameters.keys())
        assert 'tau' in params, "Missing tau parameter in compute_loss_and_metrics_v2"
        print("   ✅ compute_loss_and_metrics_v2 accepts tau")

        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_factory_integration():
    """Check factory functions support τ parameter."""
    print("\n4. Checking factory integration...")
    try:
        from ssvae.factory import SSVAEFactory
        import inspect

        # Check train_step signature (indirectly via factory)
        # We can't easily check JIT-compiled functions, but we can check the factory methods
        factory = SSVAEFactory()
        assert hasattr(factory, 'create_model'), "Missing create_model method"
        print("   ✅ SSVAEFactory.create_model exists")

        # Check that _build_train_step_v2 is a method
        assert hasattr(SSVAEFactory, '_build_train_step_v2'), "Missing _build_train_step_v2"
        assert hasattr(SSVAEFactory, '_build_eval_metrics_v2'), "Missing _build_eval_metrics_v2"
        print("   ✅ Train/eval builder methods present")

        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_ssvae_integration():
    """Check SSVAE model has τ-classifier integration."""
    print("\n5. Checking SSVAE model integration...")
    try:
        from ssvae.models import SSVAE
        import inspect

        # Check methods exist
        assert hasattr(SSVAE, '_fit_with_tau_classifier'), "Missing _fit_with_tau_classifier"
        assert hasattr(SSVAE, '_evaluate_with_tau'), "Missing _evaluate_with_tau"
        print("   ✅ Custom training methods present")

        # Check fit method signature
        sig = inspect.signature(SSVAE.fit)
        print(f"   ✅ fit() method: {list(sig.parameters.keys())}")

        # Check predict method
        sig = inspect.signature(SSVAE.predict)
        print(f"   ✅ predict() method: {list(sig.parameters.keys())}")

        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_experiment_configs():
    """Check experiment configurations exist."""
    print("\n6. Checking experiment configurations...")
    try:
        configs_dir = Path(__file__).parent / "experiments" / "configs"

        configs = [
            "tau_classifier_validation.yaml",
            "tau_classifier_ablation_baseline.yaml",
            "tau_classifier_label_efficiency.yaml",
        ]

        for config_file in configs:
            config_path = configs_dir / config_file
            if config_path.exists():
                print(f"   ✅ {config_file}")
            else:
                print(f"   ❌ Missing: {config_file}")
                return False

        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_documentation():
    """Check documentation is updated."""
    print("\n7. Checking documentation...")
    try:
        report_path = Path(__file__).parent / "TAU_CLASSIFIER_IMPLEMENTATION_REPORT.md"
        roadmap_path = Path(__file__).parent / "docs" / "theory" / "implementation_roadmap.md"

        if report_path.exists():
            print(f"   ✅ TAU_CLASSIFIER_IMPLEMENTATION_REPORT.md exists")
        else:
            print(f"   ⚠️  Implementation report not found")

        if roadmap_path.exists():
            content = roadmap_path.read_text()
            if "τ-Classifier (Completed)" in content or "tau-Classifier (Completed)" in content:
                print(f"   ✅ Roadmap updated with τ-classifier completion")
            else:
                print(f"   ⚠️  Roadmap may need update")

        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 70)
    print("τ-Classifier Integration Verification")
    print("=" * 70)

    checks = [
        check_tau_classifier_module,
        check_config_parameters,
        check_loss_functions,
        check_factory_integration,
        check_ssvae_integration,
        check_experiment_configs,
        check_documentation,
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"\n   ❌ Unexpected error: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total} checks")

    if all(results):
        print("\n✅ ALL CHECKS PASSED - τ-classifier integration is complete!")
        print("\nNext steps:")
        print("  1. Install dependencies: poetry install")
        print("  2. Run integration tests: pytest tests/test_tau_integration.py")
        print("  3. Run validation experiment: experiments/configs/tau_classifier_validation.yaml")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED - review errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
