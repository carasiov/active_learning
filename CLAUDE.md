# Project: RCM-VAE Research

## Workflow
- Three-tier agent model: see `docs/workflow/README.md`
- Current project: `docs/projects/decentralized_latents/channel_curriculum/`
- Project briefing: `docs/projects/decentralized_latents/channel_curriculum/CONTEXT.md`

## Rules
- Be concise in plans (sacrifice grammar if needed)
- Always end plans with "Unresolved Questions" section
- Tag questions by tier: `[user]`, `[high-level]`, `[codebase]`
- Use `poetry run` for all Python commands
- Prefer Explore agent for "does code do X?" verification questions
- Commit between implementation phases

## On session start
1. Read `docs/workflow/README.md` (your role)
2. Check for `docs/workflow/SESSION_*.md` files (recent state)
3. Read current project CONTEXT.md if working on that project

## Environment
- 2x CUDA GPUs available
- Can run experiments directly: `poetry run python use_cases/experiments/run_experiment.py`
- Results go to `use_cases/experiments/results/`
