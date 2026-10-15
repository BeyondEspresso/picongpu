# Task 7: minimum macroparticle weighting

**Rebase update:** Current `dev` already supplies minimum weighting. The rebased implementation
retains its omitted/`None` default of 10.0, adds strict numeric and float32
representability checks, and preserves the reference's explicit 0.1.
The original report below describes the pre-rebase default and test results.
See `resolvedConflicts.md` in the repository root for current validation.

Task 7 is complete. The reference value is now available as a simulation-level
PIConGPU extension:

```python
simulation = picmi.Simulation(
    # existing solver, timestep and other settings
    picongpu_min_weighting=0.1,
)
```

This is not a standard PICMI parameter. It sets the global
`particles::MIN_WEIGHTING` used by all species. Omission preserves `1.0`.
The [frontend](../../lib/python/picongpu/picmi/simulation.py) passes the value to
the [backend simulation](../../lib/python/picongpu/pypicongpu/simulation.py),
which supplies the [particle template](../../lib/python/picongpu/templates/include/picongpu/param/particle.param.mustache).
No C++ implementation changes are needed. The public option is documented in
[simulation settings](../../docs/source/python_package/selected_topics/simulation_settings.rst).

## Allowed values and source justification

Both Python models require a finite, strictly positive number. Zero, negative
values, NaN, infinity, `None`, booleans and numeric strings are rejected.
Backend validation also rejects values that round to zero or overflow in
the selected 32-bit core precision; values such as `1e-50` and `1e40` remain
available with 64-bit precision.

The C++ [WeightMacroParticles.hpp](../../include/picongpu/particles/startPosition/detail/WeightMacroParticles.hpp)
explicitly asserts `MIN_WEIGHTING > float_X(0.0)` and divides by it. For
physical-particle count `Nr` and requested maximum `N` macroparticles per cell,
the algorithm is:

```text
count = floor(min(N, Nr / MIN_WEIGHTING))
weight = Nr / count if count > 0, otherwise 0
```

[RandomImpl](../../include/picongpu/particles/startPosition/RandomImpl.hpp)
calls this algorithm. The generated `PseudoRandomLayout` continues to select
`RandomImpl` with `numParticlesPerCell = 1`. The
[creation kernel](../../include/picongpu/particles/ParticlesInit.kernel)
forms `Nr` from density and cell volume; it creates nothing when `Nr <= 0`.
[FreeFormulaImpl](../../include/picongpu/particles/densityProfiles/FreeFormulaImpl.hpp)
evaluates the density at cell centers, including moving-window offsets.

Thus, at one macroparticle per cell the cutoff is on the total physical-particle
count, not on normalized density or typical particles per cell. Equality at
the threshold creates one macroparticle, subject to floating-point rounding.
For larger requested counts, the algorithm can reduce the number of
macroparticles before dropping the cell entirely.

## Reference comparison and acceptance scope

**337 Python tests and 12 subtests passed**, including **30 new weighting tests**,
the existing simulation tests and the task-6 regression suite.

Two Python input generations compare the default and `0.1` thresholds using
the active grid `3584 x 3584 x 1792`, devices `8 x 16 x 8`, supercells
`8 x 8 x 8`, timestep `5.15136e-17 s`, CKC, moving-window point zero and
**1000 steps**. The test includes a cold electron species, one macroparticle
per cell and the reference's longitudinal linear ramp after `6720 * dy`.
Generation passes the incident-field/componentwise options
`-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'`.

Direct Python comparison with the original
[particle.param](../include/picongpu/param/particle.param) and
[simulation.param](../include/picongpu/param/simulation.param) confirms:

| Constant | Generated and reference |
| --- | --- |
| `MIN_WEIGHTING` | 0.1 |
| `BASE_DENSITY_SI` | `1e25 m^-3` |
| `TYPICAL_PARTICLES_PER_CELL` | 1 |

The renderer may spell `0.1` as `0.10000000000000001` to preserve a binary64
round trip; these are the same Python binary64 value and yield the same
`float_X` constant. Comparing both generated directories confirms that changing
this option changes only the minimum-weighting constant among the generated
parameter headers. The density formula, normalization, layout and initialization
pipeline are unchanged.

For cell sizes `(6.177353e-8, 3.088676e-8, 6.177353e-8)` m, cell volume is
`1.1786291893794713e-22 m^3`. The density cutoff falls from approximately
`8.48443e21` to `8.48443e20 m^-3`. On the transverse plateau, the first four
cell centers after the reference's longitudinal vacuum boundary give:

| Cell after boundary | Physical particles per cell, approximately | Count with 1.0 | Count with 0.1 |
| --- | --- | --- | --- |
| 0 | 0.182020 | 0 | 1 |
| 1 | 0.546061 | 0 | 1 |
| 2 | 0.910101 | 0 | 1 |
| 3 | 1.274141 | 1 | 1 |

These are representative moving-window ramp cells; they need not be present
in the initial domain or reached during the 1000-step run. A dilute transverse
ramp product at `1e-5` of peak density remains below even the custom cutoff.
The tests check these cases, plus values immediately below, exactly at and
immediately above the threshold, using a Python transcription of the inspected
C++ algorithm in both float32 and float64. They do not execute C++ or claim
bitwise reproduction of every normalized-density operation on a device.

The focused generation omits the transverse ramps, lasers, custom PML and
reference outputs. Full integration is task 9. No PIConGPU compilation,
job submission or simulation is performed, and no simulation grid is allocated.
Numerical evolution and physical equivalence remain untested.

Latest generated artifacts are under
`/tmp/pytest-of-debus/pytest-10/test_setup_generation_preserve0/`, in `default/`
and `reference_weighting/`. These temporary paths are reproducible with the
tests below. Tasks 8–9 remain pending.

## Reproduction

Use the isolated `uv` environment described in [task 3](TASK3.md). Run the
focused tests from the repository root:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_min_weighting.py
```

The acceptance suite additionally runs `test_simulation.py` and every test file
listed in [task 6's reproduction command](TASK6.md#reproduction). `PIC_RC`
names a nonexistent file so no user toolchain profile is loaded.
