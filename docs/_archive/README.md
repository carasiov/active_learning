# Archived Documentation

This directory contains documentation files that have been superseded by the restructured documentation.

## November 2025 Restructuring

The documentation was reorganized to improve clarity and reduce redundancy. The following files have been archived:

### `implementation.md` → Replaced by multiple focused docs

**Reason for archiving:** This 1000+ line document mixed detailed API reference with conceptual information, creating redundancy with `architecture.md`.

**Replaced by:**
- **[api_reference.md](../development/api_reference.md)** - Concise module-by-module reference
- **[OVERVIEW.md](../development/OVERVIEW.md)** - Quick intro to codebase
- **[STATUS.md](../development/STATUS.md)** - Implementation status details

**What was preserved:**
- All component APIs → `api_reference.md`
- Training infrastructure overview → `api_reference.md`
- Code examples → Moved to appropriate sections

### `implementation_roadmap.md` → Split into three focused docs

**Reason for archiving:** This document mixed stable theory (vision vs. reality), changing status (feature progress), and design decisions. These have different update frequencies and audiences.

**Replaced by:**
- **[vision_gap.md](../theory/vision_gap.md)** - High-level comparison of vision vs. current state (stable, in theory/)
- **[STATUS.md](../development/STATUS.md)** - Detailed implementation status (changes frequently, in development/)
- **[DECISIONS.md](../development/DECISIONS.md)** - Architectural choices and rationale (changes occasionally, in development/)

**What was preserved:**
- Feature status tables → `vision_gap.md` and `STATUS.md`
- Component-aware decoder validation → `STATUS.md`
- Mode collapse prevention → `DECISIONS.md`
- Configuration guides → `STATUS.md`
- Key findings → `DECISIONS.md`

## Accessing Old Content

All archived content remains available in this directory and in git history:

```bash
# View archived files
ls docs/_archive/

# See git history
git log --all -- docs/_archive/
```

## Migration Guide

If you have bookmarks or links to the old documentation:

| Old Link | New Link |
|----------|----------|
| `docs/development/implementation.md` | `docs/development/api_reference.md` (API) or `docs/development/OVERVIEW.md` (intro) |
| `docs/theory/implementation_roadmap.md` | `docs/theory/vision_gap.md` (high-level) or `docs/development/STATUS.md` (details) |

## Questions?

See the main [README.md](../../README.md) for the current documentation map.
