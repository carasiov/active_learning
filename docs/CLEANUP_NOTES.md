Codebase Cleanup Summary

Actions applied
- Moved  →  and wrapped with a main guard.
- Added  to document current layout and roles.
- Appended  to  (temporary artifacts like ).

Suggested follow-ups (non-breaking)
- Create a light  at repo root with quickstart and links to  and  specs.
- Normalize script output paths (e.g.,  via a CLI flag) to avoid proliferating filenames.

Suggested follow-ups (with backward-compatible shims)
- Rename  to a neutral package (e.g., ) and keep  as a thin shim re-exporting symbols for legacy imports.
  - Rationale:  currently houses JAX code; the name is misleading.
  - Approach: Move  to  and re-export from .
  - Validate: Run  and  (legacy imports), then adopt  in new code.

Low-priority items
- Consolidate  entries (there are duplicates and broad patterns like  that may hide docs like ). Consider scoping ignores to data/model artifacts only.
- Move binary assets like  under  (manual move recommended to avoid binary diffs in patches).

Notes
- All changes so far preserve the existing script entry points and checkpoint format.
