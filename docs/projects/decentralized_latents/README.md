# Decentralized Latent Spaces Project

This directory contains the complete specification for implementing **decentralized latent spaces** (mixture-of-VAEs architecture), where each component maintains its own latent space instead of sharing a single global latent.

**Key Infrastructure**: The modular decoder refactor is a critical enabler for this architecture, but it's a means to the end goal of decentralized latents.

## Documents

### 1. [design_context.md](./design_context.md)
**Purpose**: High-level architectural vision from supervisor specification.

**Contents**:
- Core concept: K separate latent channels
- Architecture specification (encoder, sampling, decoder)
- Loss function (all-channel KL)
- Curriculum training strategy
- Visualization requirements

**For**: Understanding the "why" and theoretical foundation.

---

### 2. [implementation_spec.md](./implementation_spec.md)
**Purpose**: Complete implementation specification for AI agents.

**Contents**:
- Current state audit (existing code, issues)
- Architectural transition (shared → decentralized latents)
- Decoder refactor (modular composition to support all features)
- Implementation tasks (extract modules, build decoders, migrate factory)
- Validation experiments
- Success criteria
- File references with line numbers

**For**: Executing the full implementation from scratch.

---

## How to Use

**For AI Agents**:
1. Read `design_context.md` to understand the decentralized latents architecture
2. Read `implementation_spec.md` for complete implementation instructions
3. Note: The decoder refactor is **infrastructure** to enable decentralized latents
4. Execute tasks in order (decoder modules → modular decoders → validation)
5. Run validation experiments to confirm decentralization works
6. Update documentation as specified

**For Humans**:
- `design_context.md` explains the decentralized latent vision (K channels, curriculum training)
- `implementation_spec.md` shows the current state and implementation plan
- The modular decoder refactor is a necessary step, but not the end goal

---

## Related Documentation

**Theory**:
- [`docs/theory/conceptual_model.md`](../../theory/conceptual_model.md) - Will be updated after implementation
- [`docs/theory/mathematical_specification.md`](../../theory/mathematical_specification.md) - Will be updated with Gumbel-Softmax formulation

**Development**:
- [`docs/development/architecture.md`](../../development/architecture.md) - Will be updated with modular decoder pattern
- [`docs/development/implementation.md`](../../development/implementation.md) - Will document new modules
- [`docs/development/extending.md`](../../development/extending.md) - Will add tutorials

**Code**:
- [`src/rcmvae/domain/components/decoders.py`](../../../src/rcmvae/domain/components/decoders.py) - Current decoder implementations
- [`src/rcmvae/domain/components/factory.py`](../../../src/rcmvae/domain/components/factory.py) - Decoder factory (contains silent overrides)
- [`src/rcmvae/domain/network.py`](../../../src/rcmvae/domain/network.py) - Network forward pass
