# Task 4: species shape and field interpolation

Task 4 is complete. The generated electron interpolation resolves to the type
used in the reference [species.param](../include/picongpu/param/species.param):

```cpp
FieldToParticleInterpolation<particles::shapes::PQS, AssignedTrilinearInterpolation>
```

The existing PICMI input `particle_shape="cubic"` selects it. This change adds
no public parameter or new interpolation algorithm.

## Implementation

The [species template](../../lib/python/picongpu/templates/include/picongpu/param/speciesDefinition.param.mustache)
previously used the global `UsedField2Particle` alias. The default
[species.param](../../include/picongpu/param/species.param) defines that alias
with TSC, regardless of the species' selected shape.

Each generated species now defines its own shape alias, for example:

```cpp
using ParticleShape_species_e = particles::shapes::PQS;
// Within ParticleFlags_species_e:
shape<ParticleShape_species_e>,
interpolation<FieldToParticleInterpolation<ParticleShape_species_e, AssignedTrilinearInterpolation>>,
```

Both flags share the same shape source. PICMI/backend shape mapping already
existed, so no extra serialized interpolation field is necessary. Default and
explicit `None` PICMI shapes still select TSC. Global aliases are unchanged.
Task 5 can use this species shape alias for current deposition as well.

The C++ [FieldToParticleInterpolation](../../include/picongpu/algorithms/FieldToParticleInterpolation.hpp)
uses the shape's `ChargeAssignmentOnSupport` and cached assignment functions.
[Counter](../../include/picongpu/particles/shapes/Counter.hpp) is explicitly a
diagnostic counting shape, with zero support and no `shapeArray` implementation
required by [Cached](../../include/picongpu/particles/shapes/ShapeSelector.hpp).
The backend species therefore rejects Counter with a clear validation error.
This also rejects PICMI `other:counter` during conversion; the enum itself is
retained for diagnostic uses. Supported interpolation shapes are NGP, CIC,
TSC, PQS and PCS. Public behavior and the remaining deposition limitation are
documented in the [species guide](../../docs/source/python_package/selected_topics/species_distributions_layouts.rst).

## Acceptance results

- The new mixed-species test failed before the template fix, showing cubic
  particles with `interpolation<UsedField2Particle>`.
- **151 Python tests and 12 subtests passed** after the fix, including 12 new
  [interpolation tests](../../lib/python/test/picongpu/quick/picmi/test_species_interpolation.py)
  and existing species, composite-pusher and TWTS/Gaussian regressions.
- A complete input-directory generation includes cubic electrons, linear
  electrons and electrons with the default shape. Each species' flags are
  checked independently: PQS, CIC and TSC interpolation respectively.
- Generation retains the active `3584 x 3584 x 1792` grid, `8 x 16 x 8`
  devices, cell sizes, timestep, CKC, binomial smoothing, moving-window point
  zero, 1000 steps and incident-field/componentwise CMake options.
- A direct Python comparison resolved the generated electron shape alias and
  the reference's `UsedParticleShape` alias. Their interpolation types matched
  exactly after whitespace removal. The reference Free/HigueraCary composite
  switching at step 6720 also matched.

The generated case was at
`/tmp/pytest-of-debus/pytest-2/test_species_interpolation_set0/setup`.
Pytest temporary paths are not permanent artifacts.

This is an isolated acceptance setup with extra species to verify independence
of their shapes. It uses uniform density and omits lasers and diagnostics;
it is not the final migrated setup. Current deposition still uses the global
`UsedParticleCurrentSolver` (default `Esirkepov<TSC>`), pending task 5. PML,
minimum weighting and the remaining migration tasks are also pending.

Only Python input generation and C++ source inspection were performed. No
PIConGPU compilation, job submission or simulation occurred; no simulation
grid was allocated. Runtime interpolation and physical equivalence were not
tested.

## Reproduction

Use the isolated `uv` environment described in [task 3](TASK3.md), then run from
the repository root:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_species_interpolation.py \
  lib/python/test/picongpu/quick/picmi/test_composite_pusher.py \
  lib/python/test/picongpu/quick/picmi/test_species.py \
  lib/python/test/picongpu/quick/pypicongpu/test_species_constants.py \
  lib/python/test/picongpu/quick/picmi/test_twts_laser.py \
  lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py
```

The run used Python 3.13.5, pytest 9.1.1 and pydantic 2.13.5. `PIC_RC` names a
nonexistent file so no user toolchain profile is loaded.
