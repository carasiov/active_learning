# SSVAE Dashboard

Interactive dashboard for semi-supervised active learning.

## Quick Start
```bash
poetry run python use_cases/dashboard/app.py
```

Open http://localhost:8050

## Features

- Interactive 60k-point latent space visualization
- Color-by toggles for user labels, predicted class, true label, or certainty
- Click-to-label workflow (updates CSV immediately)
- Configure training parameters (coming soon)
- Background training with live progress (coming soon)
- Real-time loss curve visualization (coming soon)
- Dataset statistics tracking (coming soon)

## Workflow

1. Browse latent space, click uncertain points
2. Label them (0-9)
3. Adjust training parameters if desired (upcoming)
4. Click "Start Training" (upcoming)
5. Watch metrics update (upcoming)
6. Repeat

## Architecture

- Single-user localhost deployment
- Background threading for training (planned)
- State preserved across training sessions
- Integrates with existing CLI tools (train.py, infer.py still work)
