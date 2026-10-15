# Expose PML thickness and coefficients through PICMI

## Problem and behavior

The PICMI grid accepted `pml_cells` but the PIConGPU backend rejected it, so
the generated setup could not reproduce a custom absorber. Convert per-axis
`pml_cells` and PIConGPU-specific `picongpu_pml` coefficients into a validated
backend PML model and render `fieldAbsorber.param`. The run configuration
selects the PML absorber explicitly.

Validation checks periodic axes, the absent z axis in 2D, local slab sizes and
the clearance between configured absorber cells and incident-field Huygens
surfaces. Omitted settings retain the existing defaults. `pml_cells` is a PICMI
grid input; `picongpu_pml` is a PIConGPU extension for absorber coefficients.

## Validation

From the repository root with Python test dependencies installed:

```bash
PIC_RC=/tmp/picmi-rebundle-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-rebundle-matplotlib \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/lib/python" \
python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_pml.py \
  lib/python/test/picongpu/quick/picmi/test_grid.py \
  lib/python/test/picongpu/quick/picmi/test_simulation.py --tb=short
```

Result: **144 passed**. These include 2D/3D conversion, periodic axes,
thickness limits, laser clearance and generated parameter/configuration files.
The TWTS fixture uses values accepted by base `dev`; the test measures PML
behavior without relying on a separate TWTS fix. No C++ compilation or
simulation was performed.
