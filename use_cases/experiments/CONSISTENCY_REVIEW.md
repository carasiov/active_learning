# Consistency Review - Terminal Output & Report Generation

**Review Date:** 2025-11-12
**Scope:** Warning messages, terminal output, REPORT.md generation

## Executive Summary

Reviewed all warning expressions, terminal output formatting, and report generation for consistency. Found **6 minor inconsistencies** that should be polished for professional coherence.

**Overall Assessment:** ✓ System is well-structured, issues are cosmetic only

---

## Issues Found

### 1. ⚠️ Inconsistent Python/YAML Syntax in Warnings

**Location:** `src/ssvae/config.py` vs `use_cases/experiments/src/core/validation.py`

**Current State:**
- SSVAEConfig: `"use_tau_classifier=True"` (Python syntax)
- Validation: `"use_tau_classifier=true"` (YAML syntax)

**Issue:** Mixed conventions make messages less professional.

**Recommendation:** Use YAML syntax in all user-facing messages (since users edit YAML configs).

```python
# SSVAEConfig (config.py:230)
-warnings.warn("use_tau_classifier=True only applies to...")
+warnings.warn("use_tau_classifier: true only applies to...")

# Similar for:
- use_component_aware_decoder=True → use_component_aware_decoder: true
- learnable_pi=True → learnable_pi: true
```

---

### 2. ⚠️ Duplicate Geometric MoG Warnings with Different Wording

**Location:** `src/ssvae/config.py:298` and `use_cases/experiments/src/core/validation.py:218`

**Current State:**

**SSVAEConfig:**
```python
"WARNING: geometric_mog prior induces artificial topology on latent space. "
"Use only for diagnostic/curriculum purposes, not production models."
```

**Validation:**
```python
"Geometric MoG prior induces topology in latent space. "
"This is a diagnostic tool only. For production use, prefer "
"mixture or VampPrior."
```

**Issue:** Two different warnings for the same thing → user sees duplicate messages with inconsistent wording.

**Recommendation:** Remove warning from validation.py (since SSVAEConfig already warns), OR consolidate into one canonical message.

**Preferred Fix:** Keep in SSVAEConfig only, remove from validation.py:219-223

---

### 3. ⚠️ Inconsistent Capitalization in Warning Prefixes

**Location:** Various files

**Current State:**
- Some warnings start with "WARNING:" (config.py:298)
- Most warnings have no prefix
- Terminal output uses "⚠" symbol (cli/run.py:82)

**Issue:** Mixed styles reduce professional appearance.

**Recommendation:**
- Remove "WARNING:" prefix from config.py:298 (it's redundant with warnings.warn)
- Terminal already formats warnings nicely with "⚠" symbol

```python
# config.py:298
-"WARNING: geometric_mog prior induces artificial topology..."
+"geometric_mog prior induces artificial topology..."
```

---

### 4. ⚠️ Inconsistent Wording: "doesn't use" vs "only applies to"

**Location:** Various warning messages

**Current State:**
```python
# Pattern A: "only applies to"
"use_tau_classifier=True only applies to mixture-based priors"

# Pattern B: "doesn't use"
"learnable_pi=True but prior_type='...' doesn't use mixture weights"

# Pattern C: "requires"
"Component-aware decoder requires mixture-based prior"
```

**Issue:** Three different phrasings for similar constraint violations.

**Recommendation:** Standardize to "requires" for hard constraints, "only applies to" for graceful fallbacks.

**Suggested Rewording:**
```python
# τ-classifier (graceful fallback)
"use_tau_classifier: true only applies to mixture-based priors {priors}. "
"Falling back to standard classifier."

# learnable_pi (ignored setting)
"learnable_pi: true only applies to mixture and geometric_mog priors. "
"This setting will be ignored for prior_type '{prior_type}'."

# Component-aware decoder (graceful fallback)
"use_component_aware_decoder: true only applies to mixture-based priors {priors}. "
"Falling back to standard decoder."
```

---

### 5. ✓ Terminal Output Symbols - GOOD

**Location:** `use_cases/experiments/src/cli/run.py` and `io/reporting.py`

**Current State:**
- Configuration warnings: `⚠` (cli/run.py:82)
- Configuration errors: `✗` (cli/run.py:71)
- Report status: `✓ ○ ⊘ ✗` (reporting.py:121-130)

**Assessment:** ✓ **Consistent and professional**

---

### 6. ⚠️ Minor: Inconsistent Separators

**Location:** Terminal output formatting

**Current State:**
- Most use `"=" * 80`
- Some use `"=" * 60` (cli/run.py:154)

**Issue:** Minor visual inconsistency.

**Recommendation:** Standardize to 80 characters (matches formatters.py).

```python
# cli/run.py:154
-print("\n" + "=" * 60)
+print("\n" + "=" * 80)
```

---

## Non-Issues (Already Consistent) ✓

### ✓ Report.md Generation
- Clean structure with embedded images
- Consistent status indicators
- Handles missing plots gracefully

### ✓ Config Validation Error Messages
- Clear, actionable error messages
- Good examples of valid values
- Proper use of ConfigValidationError

### ✓ Terminal Header Formatting
- Professional layout
- Consistent indentation (2 spaces)
- Extensible design (formatters.py)

### ✓ Warning Display System
- Warnings captured and displayed cleanly in a box
- Separated from training output
- Filtered after display to avoid duplicates

---

## Recommendations Summary

### Priority 1 (High Impact, Low Effort)
1. **Standardize YAML syntax** in all warnings (`use_tau_classifier: true` not `=True`)
2. **Remove duplicate geometric_mog warning** from validation.py
3. **Standardize separator length** to 80 characters

### Priority 2 (Polish)
4. **Remove "WARNING:" prefix** from geometric_mog message
5. **Standardize constraint phrasing** ("only applies to" pattern)

### Priority 3 (Nice to have)
6. Consider adding color to terminal output (optional, requires checking terminal capabilities)

---

## Testing Checklist

After implementing fixes, verify:

- [ ] Run with standard prior + τ-classifier → warning displays cleanly
- [ ] Run with geometric_mog → single warning (not duplicate)
- [ ] Run with component-aware decoder + standard prior → warning displays cleanly
- [ ] Check REPORT.md generation for all prior types
- [ ] Verify no regression in existing experiments

---

## Impact Assessment

**Breaking Changes:** None
**User-Facing Changes:** Slightly improved warning messages
**Risk Level:** Very Low (cosmetic changes only)

**Estimated Effort:** 30 minutes

---

## Files to Modify

1. `src/ssvae/config.py` - 4 warning messages (lines ~230, 255, 265, 299)
2. `use_cases/experiments/src/core/validation.py` - Remove duplicate warning (lines 218-223)
3. `use_cases/experiments/src/cli/run.py` - Separator consistency (line 154)

---

## Additional Notes

### Ambiguities Resolved ✓

1. **τ-classifier warnings are clear:**
   - Config level: Wrong prior type → fallback + warning
   - Training level: Insufficient labels → warning but continues
   - Validation level: Hard error for architectural constraint violation

2. **Component-aware decoder behavior is consistent:**
   - Warning in SSVAEConfig (graceful degradation)
   - No hard error in validation (removed per recent commit 2bff261)

3. **Report generation is robust:**
   - Handles missing plots gracefully
   - Status indicators are consistent
   - Subdirectory organization is clear

### Design Patterns Observed ✓

1. **Progressive validation:**
   - SSVAEConfig.__post_init__() → warnings for graceful fallbacks
   - validate_config() → errors for hard constraints
   - Clean separation of concerns

2. **Warning capture system:**
   - Warnings caught during config creation
   - Displayed once in clean format
   - Filtered for rest of execution

3. **Extensible formatting:**
   - formatters.py handles all prior/classifier/decoder types
   - Easy to add new types
   - Consistent structure throughout

---

## Conclusion

The system is **fundamentally sound** with excellent architecture. The identified issues are **purely cosmetic** and affect message clarity rather than functionality.

Implementing the Priority 1 recommendations will achieve the goal of "keeping it clean" with no added complexity.
