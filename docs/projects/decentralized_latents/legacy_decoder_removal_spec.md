# Legacy Decoder Removal Specification

## Summary
Remove all deprecated legacy decoder classes to simplify the codebase. The factory already uses modular decoders exclusively, so these classes are dead code.

---

## Classes to Remove (10 total)

From `src/rcmvae/domain/components/decoders.py`:

1. **ConvDecoder** (line ~337)
2. **DenseDecoder** (line ~319)
3. **HeteroscedasticConvDecoder** (line ~429)
4. **HeteroscedasticDenseDecoder** (line ~364)
5. **ComponentAwareConvDecoder** (line ~139)
6. **ComponentAwareDenseDecoder** (line ~81)
7. **ComponentAwareHeteroscedasticConvDecoder** (line ~617)
8. **ComponentAwareHeteroscedasticDenseDecoder** (line ~535)
9. **FiLMConvDecoder** (line ~250)
10. **FiLMDenseDecoder** (line ~223)

**Keep**: `ModularConvDecoder`, `ModularDenseDecoder`

---

## Files to Update

### 1. **src/rcmvae/domain/components/decoders.py**
**Action**: Delete the 10 legacy class definitions
**Keep**: 
- `ModularConvDecoder` (lines 48-59)
- `ModularDenseDecoder` (lines 62-76)
- `deprecated` decorator function (can also be removed after cleanup)

### 2. **src/rcmvae/domain/components/__init__.py**
**Current**:
```python
from .decoders import ConvDecoder, DenseDecoder
```

**New**:
```python
from .decoders import ModularConvDecoder, ModularDenseDecoder
```

**Why**: The public API should only expose the modular decoders

---

## Removal Checklist

- [ ] Remove 10 legacy decoder class definitions from `decoders.py`
- [ ] Update `components/__init__.py` to export modular decoders
- [ ] Remove `deprecated` decorator function (no longer needed)
- [ ] Quick smoke test: `python -m compileall src/rcmvae/domain/components/`

---

## Expected File Sizes (After Cleanup)

- **decoders.py**: ~80 lines (down from ~700 lines) - **90% reduction!**
- **__init__.py**: 1-line change

---

## Risk Assessment

**Low Risk** ✅
- Factory doesn't use legacy classes
- No imports found in `src/` (except `__init__.py`)
- Modular decoders are drop-in replacements
- Tests use new modular architecture

**Note**: If any external scripts/notebooks import legacy decoders, they will break. Since this is research code, that's acceptable.
