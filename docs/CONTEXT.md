# Context and Motivation

This document explains **why** the repository looks the way it does. The root `README.md` tells you **what** is in the project and how to run it; this file tells you **what we are ultimately building toward**.

The short version is:

- the real target is **interactive, active-learning-based classification for high-dimensional text/notification embeddings**;
- MNIST is only the **first, visual proof of concept**;
- the model is intentionally **more general** than MNIST requires, because we want to reuse it for the text/notification case later.

---

## 1. Target Application Scenario

The primary application is to classify **device notifications / error messages** that are available as **precomputed text embeddings** (≈1,500 dimensions). In this setting:

- only a **small subset** of messages is labeled, because labels come from an **expert**;
- new messages appear over time, often as **slight rephrasings** of existing ones;
- the system must therefore rely on **semantic proximity in embedding space**, not exact string matching;
- it should be possible to **reuse** the solution for other, similar "notifications / errors / recurring text tasks."

This kind of data is **too high-dimensional** to cluster or visualize directly. We need an intermediate, **learned latent space** that is smaller (≈12–64 dimensions), structured, and stable enough to project to 2D for interactive labeling.

---

## 2. Design Principles

To support that application, the project follows a few principles:

1. **Unsupervised first, supervised second**  
   Most data are unlabeled → learn structure with a VAE first, then let labels "nudge" that structure.

2. **Structure the latent, don't just compress**  
   A plain VAE with a single `N(0, I)` prior tends to squash everything near 0. We want **clusterable** and **semantically meaningful** latents → hence support for mixture/structured priors and later, data-shaped priors.

3. **Label signals should reshape the space**  
   When an expert labels 3–5 points, that information should not be thrown away in a separate classifier only. It should also influence the latent (e.g. via contrastive / label-aware terms).

4. **Interactivity is a first-class use case**  
   Labeling happens in short bursts; the model must tolerate **incremental training** and **immediate re-visualization** (the dashboard mentioned in the README).

5. **Keep the implementation modality-friendly**  
   Today it's MNIST (images), tomorrow it's 1.5k-dim text embeddings. Architecturally, the pipeline should work for both.

---

## 3. Why MNIST First?

MNIST is **not** the final target. It is a **convenient stand-in** because it lets us:

- see the latent space immediately (2D scatter),
- demonstrate "few labels → many colored points,"
- test interactive training (1–5 epochs at a time),
- test how well mixture/structured priors improve clusterability,
- test an active-learning style "show me the most uncertain samples" view.

Some parts of the MNIST code may therefore look **over-engineered** for digits. That is intentional: we are **validating the workflow**, not beating MNIST benchmarks.

---

## 4. Current Repository State

Right now the repository provides:

- a **semi-supervised VAE** (JAX/Flax) that can train on many unlabeled and few labeled samples,
- support for **standard and mixture-of-Gaussians** priors,
- **training infrastructure** for incremental / interactive runs,
- **experiment scripts** to compare configurations,
- and a **dashboard scaffold** (see `use_cases/dashboard/`) that will become the main interactive interface.

This matches the design principles above: VAE-first, modular priors, active-learning hooks.

For the actual project structure and how to run things, see the root `README.md`.

---

## 5. Planned Direction

The next steps that follow naturally from this context are:

1. **Swap MNIST input → text/notification embeddings**  
   Change the data loader, increase latent dimensionality, keep the rest of the pipeline.

2. **Expose/activate uncertainty and "label suggestions"**  
   Rank points by how well the current latent structure and current labels can explain them; surface the hardest ones to the expert.

3. **Add label-aware shaping**  
   For labeled points only, add contrastive / pull-push terms so the latent becomes more aligned with expert classes.

4. **Refine priors**  
   Once reconstruction is stable, enable more structured/data-shaped priors to make later class boundaries simpler.

---

## 6. How to Read This Alongside the README

- **Start** with `README.md` if you want to run code, train models, or see the project layout.
- **Read** this file (`docs/CONTEXT.md`) if you want to understand why the model and repo are built more generally than MNIST needs.
- **Check** `use_cases/dashboard/` if you are interested in the interactive labeling part.

This separation lets the README stay a navigation hub, while this document keeps the long-term motivation in one place.
