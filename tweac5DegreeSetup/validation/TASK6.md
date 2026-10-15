# Task 6: custom PML thickness and coefficients

Task 6 is complete. Add these arguments to the reference's Cartesian grid:

```python
pml_cells=[110, 110, 110],
picongpu_pml=picmi.PML(
    sigma_kappa_grading_order=2.5,
    sigma_opt_multiplier=1.25,
    kappa_max=[4, 1, 4],
    alpha_grading_order=1.0,
    alpha_max_si=[0, 0, 0],
),
```

`pml_cells` is an existing standard PICMI parameter: one thickness per axis,
used on both faces. `picongpu_pml` and `picmi.PML` are new PIConGPU extensions,
not definitions adopted by the PICMI standard. A complete grid example is in
the [public documentation](../../docs/source/python_package/selected_topics/grids_and_solvers.rst).

## Implementation and validation

- The [PICMI grid](../../lib/python/picongpu/picmi/grid.py) converts thickness
  and the [public coefficient object](../../lib/python/picongpu/picmi/pml.py)
  into a [typed backend PML](../../lib/python/picongpu/pypicongpu/pml.py).
  Frontend and backend share coefficient validation and defaults.
- Thicknesses are strict integers in `[0, 2**32 - 1]`; booleans, floats,
  strings and wrong-length lists are rejected. Zero disables absorption.
  Explicit thickness must be zero on periodic axes. Coefficients must be
  finite and nonnegative, except kappa components, which must be at least one.
  Unknown coefficient names are rejected.
- Omitting both options retains the existing C++ defaults: 12 cells,
  sigma/kappa grading 4, conductivity multiplier 1, kappa `(1, 1, 1)`,
  alpha grading 1 and alpha maxima `(0.2, 0.2, 0.2)` S/m. Periodic faces
  are disabled at runtime. With only coefficients supplied, thickness defaults
  to 12 on open axes and zero on periodic axes. With only thickness supplied,
  coefficients retain their defaults.
- Coefficient vectors always have three components. Explicitly configured
  2D PML has zero z thickness; `to_2d()` retains coefficients and x/y thickness.
- A new [absorber template](../../lib/python/picongpu/templates/include/picongpu/param/fieldAbsorber.param.mustache)
  renders all six thicknesses and the coefficients. Both conductivity formulas
  remain C++ expressions using `sim.si.getZ0()` and the appropriate cell size.
  [N.cfg](../../lib/python/picongpu/templates/etc/picongpu/N.cfg.mustache)
  explicitly selects `--fieldAbsorber pml` and includes it in program arguments.
  As before, PIConGPU disables absorption for an entirely periodic domain.
- Explicit PML is checked against the smallest local domain on each axis,
  including nonuniform GPU partitions. On an axis with one rank, the sum of
  opposing layers must fit. These rules follow the slab-allocation check in
  [Field.tpp](../../include/picongpu/fields/absorber/pml/Field.tpp) and the active
  layer check in [Pml.hpp](../../include/picongpu/fields/absorber/pml/Pml.hpp).
- The backend simulation checks absorber clearance on incident-field faces
  enabled by the current template: YMin, plus the appropriate Z face for each
  TWTS pulse in 3D. Required offset is `thickness + margin - 1`, matching
  [Solver.hpp](../../include/picongpu/fields/incidentField/Solver.hpp).
  The margin follows [DerivativeCoefficients.hpp](../../include/picongpu/fields/incidentField/DerivativeCoefficients.hpp):
  one for Yee/CKC, two along the default Lehe Cherenkov-free y direction,
  and the neighbor count for arbitrary-order FDTD. A None solver with incident
  fields and configured PML is rejected. No laser-class changes are needed.

Geometry checks apply when PML is explicitly configured. Existing implicit
defaults retain their behavior and C++ runtime checks. The new Python check
covers absorber clearance, not all incident-field stencil placement constraints
at local GPU interfaces. It never moves or clamps Huygens surfaces.

## Acceptance results

**274 Python tests and 12 subtests passed**, including **40 PML tests** and
existing grid, species, deposition, interpolation, pusher and laser regressions.
Final template inspection found and corrected an extra namespace declaration
in the footer. All 40 PML tests then passed again, including default/custom
setup generation and new namespace-closure checks.

The custom generation combines the active reference grid `3584 x 3584 x 1792`,
devices `8 x 16 x 8`, supercells `8 x 8 x 8`, timestep `5.15136e-17 s`, CKC,
binomial smoothing, moving-window point zero and **1000 steps**. It includes
both TWTS pulses at the reference angles and PQS/EmZ electrons with the
Free/HigueraCary composite switch at step 6720. Input generation uses
`-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'`.

A separate Python comparison stripped comments, compared numeric constants
and normalized whitespace in conductivity expressions against the original
[fieldAbsorber.param](../include/picongpu/param/fieldAbsorber.param).
It confirmed:

| Parameter | Generated and reference |
| --- | --- |
| `NUM_CELLS` | 110 on all six faces |
| `SIGMA_KAPPA_GRADING_ORDER` | 2.5 |
| `SIGMA_OPT_MULTIPLIER` | 1.25 |
| `KAPPA_MAX` | `(4, 1, 4)` |
| `ALPHA_GRADING_ORDER` | 1 |
| `ALPHA_MAX_SI` | `(0, 0, 0)` |
| `SIGMA_OPT_SI[d]` | `0.8 * (2.5 + 1) / (Z0 * cell_size[d])` |
| `SIGMA_MAX_SI[d]` | `1.25 * SIGMA_OPT_SI[d]` |
| Runtime absorber | `--fieldAbsorber pml` |

The actual cell sizes remain `(6.177353e-8, 3.088676e-8, 6.177353e-8)` m.
Consequently y conductivity is approximately twice x/z conductivity; no
isotropic-cell assumption is introduced. Both conductivity expressions match
the reference exactly after whitespace removal. The default generated case
also matches the repository's default absorber constants and expressions.

The comparison additionally confirmed the unchanged timestep and full Huygens
`POSITION = [[116, -116], [116, 100000000], [116, -116]]`. Both laser delays
are zero. The intentionally distant, inactive YMax remains untouched.
Generated namespace closure and balanced braces were checked as text.

The unused exponential absorber retains the repository's default strength
`1.0e-3` on every face; the reference defines `0.3e-3`. This difference does
not affect the explicitly selected PML. Exponential-absorber migration and
arbitrary conductivity formulas are outside this task.

Latest artifacts:

- `/tmp/pytest-of-debus/pytest-8/test_pml_setup_generation_Fals0/setup`
  (default absorber).
- `/tmp/pytest-of-debus/pytest-8/test_pml_setup_generation_True0/setup`
  (reference PML).

These are temporary paths, reproducible with the tests below. The combined
case uses a uniform placeholder electron density and does not yet reproduce
the reference plasma profile or diagnostics. It is not the final migration;
tasks 7–9 remain pending. No PIConGPU compilation, job submission or simulation
was performed, and no simulation grid was allocated. Runtime absorption and
physical equivalence have not been tested.

## Reproduction

Use the isolated `uv` environment described in [task 3](TASK3.md). From the
repository root, run the following Python suite:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_pml.py \
  lib/python/test/picongpu/quick/picmi/test_grid.py \
  lib/python/test/picongpu/quick/picmi/test_grid_2d.py \
  lib/python/test/picongpu/quick/picmi/test_current_deposition.py \
  lib/python/test/picongpu/quick/picmi/test_species_interpolation.py \
  lib/python/test/picongpu/quick/picmi/test_composite_pusher.py \
  lib/python/test/picongpu/quick/picmi/test_species.py \
  lib/python/test/picongpu/quick/pypicongpu/test_species_constants.py \
  lib/python/test/picongpu/quick/picmi/test_twts_laser.py \
  lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py
```

Run just `test_pml.py` for the focused 40-test check. The environment uses
Python 3.13.5, pytest 9.1.1 and pydantic 2.13.5. `PIC_RC` names a nonexistent
file to avoid loading a user toolchain profile.
