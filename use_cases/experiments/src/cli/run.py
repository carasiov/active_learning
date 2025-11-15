"""Legacy CLI entrypoint that delegates to use_cases.experiments.run_experiment."""
from __future__ import annotations

from use_cases.experiments.run_experiment import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - convenience for direct invocation
    raise SystemExit(main())
