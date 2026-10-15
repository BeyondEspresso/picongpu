# Task 5: per-species current deposition

Task 5 is complete. Use the PIConGPU-specific option:

```python
electrons = picmi.Species(
    name="e",
    particle_type="electron",
    particle_shape="cubic",
    picongpu_current_deposition="EmZ",
    picongpu_composite_pusher=picmi.CompositePusher(
        first="free-streaming", second="Higuera-Cary", switch_step=6720,
    ),
)
```

The species now uses PQS consistently for its shape flag, field interpolation
and current deposition. Its deposition type resolves to
`currentSolver::EmZ<particles::shapes::PQS>`, matching the reference
[species.param](../include/picongpu/param/species.param).

## Implementation and supported choices

- [PICMI species](../../lib/python/picongpu/picmi/species.py) accepts exactly
  `"Esirkepov"`, `"EmZ"` and its alias `"EZ"` through `picongpu_current_deposition`.
  Both EmZ spellings normalize to `"EmZ"` in the backend and generate identical
  types. Omission selects Esirkepov. `None`, other spellings and arbitrary
  C++ expressions fail validation. This is a new PIConGPU extension, not a new
  definition adopted by the PICMI standard.
- The [backend species](../../lib/python/picongpu/pypicongpu/species/species.py)
  stores a `CurrentDeposition` enum. The
  [species template](../../lib/python/picongpu/templates/include/picongpu/param/speciesDefinition.param.mustache)
  uses `currentSolver::<algorithm><ParticleShape_species_<name>>`. This shares
  task 4's shape alias with interpolation and the species shape flag.
- Default inputs retain `Esirkepov<TSC>`. Cubic shape without a deposition
  override now produces `Esirkepov<PQS>` instead of the previous global TSC
  deposition. Different species can select different algorithms and shapes.
- The C++ [Esirkepov declaration](../../include/picongpu/fields/currentDeposition/Esirkepov/Esirkepov.def)
  and [EmZ declaration](../../include/picongpu/fields/currentDeposition/EmZ/EmZ.def)
  provide the strategy default. The generated type omits that argument, exactly
  as in the reference. Custom strategies are not exposed by this option.
- Particle pushing and solver-level `source_smoother` remain independent of
  deposition. No C++ algorithm or smoothing implementation changed.

| PICMI particle shape | Esirkepov | EmZ |
| --- | --- | --- |
| `NGP` | Accepted | Rejected |
| `linear` / CIC | Accepted | Accepted |
| `quadratic` / TSC | Accepted | Accepted |
| `cubic` / PQS | Accepted | Accepted |
| `other:quartic` / PCS | Accepted | Accepted |
| `other:counter` | Rejected as a species shape | Rejected as a species shape |

The EmZ/NGP restriction follows from source inspection:
[NGP](../../include/picongpu/particles/shapes/NGP.hpp) has support one.
[EmZ](../../include/picongpu/fields/currentDeposition/EmZ/EmZ.hpp) sets
`end = begin + support`, and
[DepositCurrent](../../include/picongpu/fields/currentDeposition/EmZ/DepositCurrent.hpp)
loops along a current component from `begin` to strictly below `end - 1`.
These loops have no iterations for NGP: all three current components in 3D,
and Jx/Jy in 2D. Esirkepov uses the broader assignment function and its
cell-crossing logic; the same restriction is not imposed on it. Counter's
existing task-4 rejection remains in place.

## Acceptance results

**181 Python tests and 12 subtests passed**, including 30 new
[deposition tests](../../lib/python/test/picongpu/quick/picmi/test_current_deposition.py).
Coverage includes the allowed algorithm/shape pairs, invalid combinations and
names, frontend/backend defaults, JSON/schema round trips, and independent
choices for species with the same or different shapes. Earlier interpolation,
composite-pusher, species and TWTS/Gaussian tests also passed.

The subsequent EZ-alias change passed all **40 deposition tests**, including
four setup generations (EmZ/EZ, each with smoothing enabled and disabled).
It also checks identical rendering contexts and C++ output for both spellings,
direct backend normalization and the same NGP/Counter rejections. Run the first
test file in the command below to reproduce this focused check. No C++
compilation or simulation was performed.

Two complete input-directory generations use the active reference grid
`3584 x 3584 x 1792`, devices `8 x 16 x 8`, cell sizes, timestep, CKC,
moving-window point zero, 1000 steps and incident-field/componentwise CMake
options. The runs differ only in whether binomial smoothing is configured.
The generated runtime option tracks that choice, while deposition remains
EmZ/PQS for `e`, Esirkepov/CIC for `linear_e`, and Esirkepov/TSC for `default_e`.

A direct Python comparison with the original `species.param` resolved the
shape aliases and removed whitespace. In both generated setups, the electron
shape, interpolation, deposition and composite-pusher types matched the active
reference exactly. The enabled-smoothing case reproduces the reference choice.

Artifacts from this validation were:

- `/tmp/pytest-of-debus/pytest-3/test_current_deposition_setup_0/setup`
  (smoothing disabled).
- `/tmp/pytest-of-debus/pytest-3/test_current_deposition_setup_1/setup`
  (smoothing enabled).

These temporary paths are not permanent artifacts. The setups include extra
species for comparison, use uniform density and omit lasers and diagnostics.
They are not the final migration; PML, minimum weighting and tasks 8–9 remain
pending. No PIConGPU compilation, job submission or simulation was performed,
and no simulation grid was allocated. Numerical current deposition, charge
conservation and physical equivalence were not tested by execution.

## Reproduction

Use the isolated `uv` environment described in [task 3](TASK3.md), then run
from the repository root:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_current_deposition.py \
  lib/python/test/picongpu/quick/picmi/test_species_interpolation.py \
  lib/python/test/picongpu/quick/picmi/test_composite_pusher.py \
  lib/python/test/picongpu/quick/picmi/test_species.py \
  lib/python/test/picongpu/quick/pypicongpu/test_species_constants.py \
  lib/python/test/picongpu/quick/picmi/test_twts_laser.py \
  lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py
```

The run used Python 3.13.5, pytest 9.1.1 and pydantic 2.13.5. `PIC_RC` names a
nonexistent file so no user toolchain profile is loaded.
