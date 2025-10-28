# SSVAE Dashboard - Multi-Model Architecture

Interactive dashboard for semi-supervised learning with multiple independent model experiments.

## Quick Start
```bash
cd /workspaces/active_learning
poetry run python use_cases/dashboard/app.py
```
Open http://localhost:8050

## Key Features

- **Multi-Model Management:** Create, switch, and delete models with isolated state
- **Interactive Labeling:** 60k-point WebGL visualization with click-to-label
- **Background Training:** Live progress updates with graceful stop
- **Configuration:** 17+ hyperparameters with presets

## Project Structure
```
use_cases/dashboard/
├── app.py                 # Entry point
├── core/                  # Infrastructure (state, commands, I/O)
├── pages/                 # Page layouts
├── callbacks/             # Event handlers
├── utils/                 # Helpers (visualization, logging)
├── assets/                # Static files
└── docs/                  # Documentation

artifacts/models/{model_id}/
├── checkpoint.ckpt    # Weights
├── labels.csv         # Labels
├── history.json       # Loss curves
└── metadata.json      # Stats
```

## Routes
- `/` - Model list
- `/model/{id}` - Dashboard
- `/model/{id}/training-hub` - Training
- `/model/{id}/configure-training` - Config

## Development
```bash
# Logs
tail -f /tmp/ssvae_dashboard.log

# Tests
poetry run python tests/run_dashboard_tests.py

# Debug mode (app.py line 23)
DashboardLogger.setup(console_level=logging.DEBUG)
```

---

## 📚 Documentation Guide

This dashboard has comprehensive documentation organized by use case and abstraction level. **AI agents and developers should consult these documents in the following order based on their task:**

### For Understanding the System
**Start here if:** You're new to the codebase or need to understand how it works internally.

📖 **[Developer Guide](docs/DEVELOPER_GUIDE.md)**
- **Purpose:** Explains internal architecture, state management, and debugging
- **Key topics:** Immutable state architecture, thread safety, command pattern, training system, performance considerations
- **Use when:** Understanding existing code, debugging issues, tracing data flow, investigating bugs
- **Abstraction level:** Low-level implementation details

### For Adding Features
**Start here if:** You need to extend the dashboard with new functionality.

🤖 **[Agent Extension Guide](docs/AGENT_GUIDE.md)**
- **Purpose:** Patterns and templates for adding features safely and consistently
- **Key topics:** Command pattern templates, UI component patterns, callback organization, testing workflows
- **Use when:** Implementing new commands, adding UI components, creating callbacks, modifying state
- **Abstraction level:** Mid-level patterns and practical templates
- **Special note:** Written specifically for AI coding agents with emphasis on consistency and safety

### Behavioral Relationship

```
User Request (Feature/Bug)
         ↓
    [Agent Guide] ← Start here for implementation
         ↓
    Understand patterns & templates
         ↓
    [Developer Guide] ← Consult when you need implementation details
         ↓
    Make changes following established patterns
         ↓
    Test & verify
```

### Quick Reference by Task

| Task | Primary Doc | Secondary Doc |
|------|-------------|---------------|
| Add new command | [Agent Guide](docs/AGENT_GUIDE.md) § Pattern 1 | [Developer Guide](docs/DEVELOPER_GUIDE.md) § Command Pattern |
| Add UI component | [Agent Guide](docs/AGENT_GUIDE.md) § Pattern 3 | - |
| Debug state issue | [Developer Guide](docs/DEVELOPER_GUIDE.md) § Debugging State | - |
| Add visualization | [Agent Guide](docs/AGENT_GUIDE.md) § Pattern 4 | [Developer Guide](docs/DEVELOPER_GUIDE.md) § Callback Organization |
| Fix training bug | [Developer Guide](docs/DEVELOPER_GUIDE.md) § Training System | [Agent Guide](docs/AGENT_GUIDE.md) § Pattern 2 |
| Understand architecture | [Developer Guide](docs/DEVELOPER_GUIDE.md) § State Management | - |
| Add new page | [Agent Guide](docs/AGENT_GUIDE.md) § FAQ | [Developer Guide](docs/DEVELOPER_GUIDE.md) § Page Architecture |
| Performance issue | [Developer Guide](docs/DEVELOPER_GUIDE.md) § Performance | - |

### For AI Agents: Heuristic Selection Rules

**Use this decision tree to fetch the right documentation:**

1. **Is this a NEW feature request?**
   - YES → Read [Agent Guide](docs/AGENT_GUIDE.md) first
   - NO → Continue to step 2

2. **Is this a BUG or unexpected behavior?**
   - YES → Read [Developer Guide](docs/DEVELOPER_GUIDE.md) § Common Issues
   - NO → Continue to step 3

3. **Do you need to understand HOW something works?**
   - YES → Read [Developer Guide](docs/DEVELOPER_GUIDE.md) relevant section
   - NO → Continue to step 4

4. **Do you need a PATTERN or TEMPLATE?**
   - YES → Read [Agent Guide](docs/AGENT_GUIDE.md) relevant pattern
   - NO → Read both guides

**Pattern matching heuristics:**
- Keywords like "add", "create", "new", "implement" → [Agent Guide](docs/AGENT_GUIDE.md)
- Keywords like "why", "how", "understand", "architecture" → [Developer Guide](docs/DEVELOPER_GUIDE.md)
- Keywords like "broken", "not working", "debug", "error" → [Developer Guide](docs/DEVELOPER_GUIDE.md) § Common Issues
- Keywords like "command", "callback", "UI component" → [Agent Guide](docs/AGENT_GUIDE.md) relevant pattern

---

## System Architecture Overview

**State Management:** Immutable dataclasses with command pattern for all modifications  
**Threading:** Background training worker with queue-based progress updates  
**UI Framework:** Dash/Plotly with page-based routing  
**Performance:** Aggressive caching for 60k-point scatter plots

For details, see [Developer Guide](docs/DEVELOPER_GUIDE.md).

---

## Contributing

When extending this dashboard:

1. ✅ **Follow established patterns** from the [Agent Guide](docs/AGENT_GUIDE.md)
2. ✅ **Use commands for state changes** (never mutate directly)
3. ✅ **Test your changes** (both automated and manual)
4. ✅ **Consult the guides** when unsure

The architecture is designed to be AI-agent-friendly with clear separation of concerns and consistent patterns throughout.
