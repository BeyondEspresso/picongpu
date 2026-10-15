# Expose built-in charge and energy density field outputs

## Problem and behavior

PIConGPU already has built-in charge and energy density deposition attributes,
but PICMI `DerivedFieldDump` required a custom particle functor. Accept the
selectors `"ChargeDensity"` and `"EnergyDensity"`, convert them to the
existing C++ attributes and emit each required field definition once even
when several output files request it.

The field still uses its selected species and particle filter. Built-in
selectors do not request extra functor attributes; any enclosing particle
filter still registers the attributes it needs. Existing custom functors and
inherited particle shape conversion remain supported.

These string selectors are PIConGPU extensions, not PICMI-standard names.
The PR leaves the typed openPMD backend configuration and spatial range
handling as they are on `dev`.

## Validation

From the repository root with Python test dependencies installed:

```bash
PIC_RC=/tmp/picmi-rebundle-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-rebundle-matplotlib \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/lib/python" \
python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/diagnostics/test_field_dump.py \
  lib/python/test/picongpu/quick/picmi/diagnostics/test_openpmd_backend_config.py \
  --tb=short
```

Result: **41 passed**. The generated-input test covers charge/energy fields,
deduplication across output files and a filter requiring `momentumPrev1`.
Ordinary functor and backend-schema regressions also pass. No PIConGPU
compilation or simulation was performed.
