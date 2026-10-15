# Task 3: composite particle pusher

Task 3 is complete. The generated electron pusher matches the active
`UsedParticlePusher` alias in [species.param](../include/picongpu/param/species.param):

```cpp
particles::pusher::Composite<
    particles::pusher::Free,
    particles::pusher::HigueraCary,
    particles::pusher::CompositeBinarySwitchActivationFunctor<6720>>
```

The PICMI declaration is:

```python
from picongpu import picmi

electrons = picmi.Species(
    name="e",
    particle_type="electron",
    picongpu_composite_pusher=picmi.CompositePusher(
        first="free-streaming",
        second="Higuera-Cary",
        switch_step=6720,
    ),
)
```

Omit `method` or set it to `None`. Explicit methods, including `"Boris"`,
conflict with the composite. Without a composite, the default remains Boris.
The complete public API and supported names are documented in
[species documentation](../../docs/source/python_package/selected_topics/species_distributions_layouts.rst).

## Implementation and source audit

- [PICMI species](../../lib/python/picongpu/picmi/species.py) validates the
  two method names and converts the composite into a structured backend model.
- [Backend species](../../lib/python/picongpu/pypicongpu/species/species.py)
  accepts either a single enum or a composite. Its computed `pusher_cpp` type is
  derived from the validated selection and used in the
  [species template](../../lib/python/picongpu/templates/include/picongpu/param/speciesDefinition.param.mustache).
- The switch must be an integer in `[0, 2**32 - 1]`, matching the C++ template's
  `uint32_t` argument. Floats, booleans, strings and out-of-range values fail.
  Extra options, nested composites and unsupported methods fail as well.
- The [C++ composite](../../include/picongpu/particles/pusher/particlePusherComposite.hpp)
  inherits from both pusher types, so duplicate types are rejected. Its
  activation condition is `currentStep < T_switchTimeStep ? 1 : 2`: Free is
  selected through step 6719, and HigueraCary from 6720. A switch at zero
  selects the second pusher immediately. This was established by source
  inspection, not C++ execution.
- Audited enum values against actual declarations in
  [pusher.unitless](../../include/picongpu/unitless/pusher.unitless).
  Corrected the backend value `Higuera-Cary` to `HigueraCary`, preserving the
  PICMI spelling `"Higuera-Cary"`. Removed `Pusher.Axel` and the
  `"other:Axel"` mapping because no C++ implementation exists in this checkout.
  These are intentional compatibility corrections to previously invalid output.

## Validation

**139 tests and 12 subtests passed** using Python 3.13.5, pytest 9.1.1 and
pydantic 2.13.5 in an isolated `uv` environment under `/tmp`.
The [52 new tests](../../lib/python/test/picongpu/quick/picmi/test_composite_pusher.py)
cover conversion, schema-validated rendering, exact C++ types, all supported
single pushers, defaults, conflicts and invalid inputs. Existing species tests
and all 62 TWTS/Gaussian regressions also passed.

Two complete input-directory generations use switches at 0 and 6720, with
the reference grid `3584 x 3584 x 1792`, device decomposition `8 x 16 x 8`,
cell sizes, timestep, CKC, binomial smoothing, moving-window point zero,
1000 steps and incident-field/componentwise CMake options. The generated
electron species uses a cubic shape and one random macroparticle per cell.
A direct Python comparison of the generated 6720 case with the active alias
in the original `species.param` passed after removing whitespace. The switch
is retained even though it lies beyond this run's 1000 steps.

The generated 6720 case was at
`/tmp/pytest-of-debus/pytest-1/test_composite_setup_generatio1/setup`.
Pytest temporary paths are not permanent artifacts.

These are isolated pusher acceptance cases, not the final migrated setup.
They use a uniform density and omit lasers and diagnostics; interpolation,
deposition, PML and minimum weighting remain pending tasks. The TWTS regressions
separately generate their laser acceptance cases. No PIConGPU C++ compilation,
job submission or simulation execution was performed, and no simulation grid
was allocated. Runtime switching and physical equivalence were not tested.

## Reproduction

From the repository root, create the Python environment if it is absent:

```bash
UV_CACHE_DIR=/tmp/picmi-uv-cache uv venv --python python3 /tmp/picmi-task1-venv
UV_CACHE_DIR=/tmp/picmi-uv-cache uv pip install \
  --python /tmp/picmi-task1-venv/bin/python -e './lib/python[test]'
```

Run only the Python checks:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_composite_pusher.py \
  lib/python/test/picongpu/quick/picmi/test_species.py \
  lib/python/test/picongpu/quick/pypicongpu/test_species_constants.py \
  lib/python/test/picongpu/quick/picmi/test_twts_laser.py \
  lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py
```

`PIC_RC` deliberately names a nonexistent file so no user toolchain profile is
loaded. Dependency installation builds Python packages only; it does not build
the PIConGPU simulation executable.
