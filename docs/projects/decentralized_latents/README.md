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
