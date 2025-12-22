# Agent Workflow

> **Purpose:** Describes the multi-agent coordination model used for this project.

---

## Three-Tier Agent Model

This project uses three agent layers operating at different abstraction levels:

### High-Level Agent (Conceptual/Mathematical)

**Role:** Synthesizes supervisor input, provides mathematical direction, reasons about research choices.

**Context:** Meeting transcripts, conceptual history, mathematical specification.

**Outputs:**
- Conceptual synthesis from supervisor conversations
- Mathematical characterization of the system
- Diagnosis when results don't match theory
- Alternative formulations and experiment suggestions

**Current instance:** GPT-5.2 Thinking (ChatGPT Plus project)

**Uploaded files:**
- `docs/projects/decentralized_latents/channel_curriculum/CONTEXT.md`
- `docs/workflow/README.md`
- `docs/theory/conceptual_model.md`
- `docs/theory/mathematical_specification.md`
- `docs/projects/decentralized_latents/channel_curriculum/design_contract.md`
- `docs/projects/decentralized_latents/channel_curriculum/high_level_context.md`

**Update policy:** Re-upload `CONTEXT.md` when implementation changes significantly. Other files are stable.

**System instructions:**
```
You are the high-level agent in a three-tier workflow for an RCM-VAE research project.

## Your Role
- Conceptual and mathematical synthesis
- Interpret supervisor decisions and translate them into design direction
- Diagnose when results don't match theory
- Propose alternative formulations and experiments

## What You Receive
The user relays information from the intermediate agent (Claude Code), which has direct codebase access. Expect:
- Implementation verification reports (current factorization, objective structure, how components work)
- Experimental results and unexpected behaviors
- Specific questions requiring mathematical/conceptual direction

## What You Provide
- Mathematical characterization and interpretation
- Diagnosis of why something isn't working (conceptually)
- What to monitor/measure to distinguish hypotheses
- Alternative approaches if current design has fundamental issues

## When You Need Implementation Details
Ask testable questions the intermediate agent can answer by inspection:
- "Is q(c|x) amortized or computed via Bayes rule?"
- "Is KL weighted by responsibilities or summed over all k?"
- "Does the decoder condition on sampled c or weighted embedding?"

## Communication Protocol
- Treat older transcripts as historical intent; ground reasoning in the latest verification report
- When the user shares results, ask for the current-state packet if not provided
- Report your reasoning so decisions can be logged in the repository

## Project Context
This is a semi-supervised VAE with decentralized latent channels ("pots"). The current focus is a channel-unlocking curriculum. Key files in the project provide stable context:
- CONTEXT.md: Comprehensive briefing (supervisor decisions, math, priorities, gotchas)
- conceptual_model.md: Stable design vision
- mathematical_specification.md: Formal specification
- design_contract.md: Curriculum invariants

Refer to these when reasoning about whether implementation matches intent.
```

### Intermediate Agent (Coordination)

**Role:** Translates conceptual direction into implementation, verifies codebase state, coordinates low-level tasks.

**Context:** Full codebase access, project documentation, conversation history.

**Environment capabilities:**
- **GPU access**: 2x CUDA GPUs available for training
- **Direct experiment execution**: Can run `poetry run python use_cases/experiments/run_experiment.py`
- **Result analysis**: Can read summaries, configs, figures, and `.npy` diagnostics directly

This enables fast iteration: implement → run → analyze → adjust, all within a single session without user copy-pasting results.

**Outputs:**
- Implementation verification reports
- Task delegation specs
- Decision tracking
- Diagnosis of unexpected behavior

**Current instance:** Claude Code (Opus 4.5)

### Low-Level Agent (Implementation)

**Role:** Executes well-scoped implementation tasks, debugging, code changes.

**Context:** Task spec from intermediate agent, relevant code sections.

**Outputs:**
- Code changes
- Test results
- Error reports for escalation

**Current instance:** Claude Code or similar, receiving delegation docs

---

## Communication Protocol

### When Intermediate Asks High-Level for Direction

Include in the request:
1. **Current factorization** — what are the random variables, conditioning structure
2. **Exact objective** — ELBO terms, losses, weights, schedules
3. **What "component" means** — VampPrior pseudo-input, mixture component, decoder channel, etc.
4. **How labels influence training** — where supervision enters
5. **Concrete symptom or decision point** — what's broken or needs deciding

### When High-Level Needs Implementation Verification

Phrase as testable questions the intermediate agent can answer by inspection:
- "Is q(c|x) amortized logits or computed by Bayes rule from densities?"
- "Is KL computed per-component or over the mixture?"
- "Does the decoder condition on sampled c, weighted embedding, or not at all?"

### Drift Management

Report changes as **diffs against previous state**, not prose:
- "Changed: logit-MoG now gated when k_active ≤ 1"
- "Changed: kick window adds +10 logit bias to newly unlocked channel"

---

## Artifacts

### Project Context Docs

Each major project has a `CONTEXT.md` in its docs folder containing:
- Supervisor decisions (authoritative)
- Mathematical characterization (verified)
- Open questions
- Priorities
- Gotchas

Example: `docs/projects/decentralized_latents/channel_curriculum/CONTEXT.md`

### Delegation Docs

For low-level handoffs, create a `DELEGATION*.md` with:
- Goal
- Current state
- Required changes (ordered)
- Acceptance criteria
- Verification plan

Example: `docs/projects/decentralized_latents/channel_curriculum/DELEGATION_V2.md`

---

## When to Use Each Layer

| Situation | Layer |
|-----------|-------|
| "Why does the math say this should work?" | High-level |
| "Does the code actually do X?" | Intermediate |
| "Implement this specific change" | Low-level |
| Results don't match expectations | Intermediate diagnoses, escalates to high-level if conceptual |
| Need supervisor input | High-level formulates question for next meeting |

---

## Fast Iteration Loop

Because the intermediate agent has GPU access and can run experiments directly:

```
┌──────────────────────────────────────────────────────────────┐
│  Implement change                                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Run experiment (GPU, ~10 min for curriculum)                │
│  poetry run python use_cases/experiments/run_experiment.py   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Analyze results (summary.json, figures, .npy files)         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Diagnose and adjust                                         │
│  → If implementation bug: fix and re-run                     │
│  → If conceptual issue: escalate to high-level agent         │
└──────────────────────────────────────────────────────────────┘
```

No user copy-paste required. The intermediate agent can complete multiple iterations autonomously within a single session.

---

## Updating This Workflow

This workflow emerged from practice. If coordination patterns change, update this doc. The goal is lightweight documentation that prevents re-learning, not bureaucratic process.
