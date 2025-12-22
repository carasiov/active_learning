---
description: Generate SESSION_*.md file capturing current session state before /clear
argument-hint: [optional notes]
allowed-tools: Read, Write, Bash, Glob
---

## Your Task

Generate a session capture document before the user runs /clear. This preserves state for the next session to bootstrap from.

**Additional notes from user:** $ARGUMENTS

## Instructions

1. **Gather information:**
   - Run `git diff --name-only HEAD~5` to see recently modified files
   - Run `git log --oneline -10` to see recent commits
   - Review the conversation history for what was accomplished

2. **Generate the session file** with this structure (follow `docs/workflow/SESSION_2025-12-22.md` format):

```markdown
# Session Summary: <TODAY'S DATE>

> **Purpose:** Capture session state for "Document & Clear" restart.
> **Delete after:** Next session successfully bootstraps from this.

---

## What Was Accomplished

[Bullet points or numbered list of main accomplishments this session]

---

## Key Decisions Made

[Numbered list of decisions that affect future work]

---

## Current State

- **[Area 1]:** [Status]
- **[Area 2]:** [Status]

---

## Next Steps (for next session)

1. [Priority task]
2. [Secondary task]
3. [If needed: what to escalate to high-level agent]

---

## Files Modified This Session

```
Created:
  [list of new files]

Modified:
  [list of changed files]
```

---

## Restart Command

After `/clear`, tell the new session:

```
Read these files to bootstrap:
1. docs/workflow/README.md (your role and capabilities)
2. docs/workflow/SESSION_<DATE>.md (what happened last session)
3. docs/projects/decentralized_latents/channel_curriculum/CONTEXT.md (project briefing)

Then proceed with: [current priority]
```
```

3. **Write the file** to `docs/workflow/SESSION_<YYYY-MM-DD>.md` using today's date.

4. **Output confirmation** showing the file path and a brief summary of what was captured.

## Important

- Be concise — this is for quick bootstrapping, not exhaustive documentation
- Focus on actionable state: what's done, what's next, what files matter
- Include the restart command so the user can copy-paste it after /clear
