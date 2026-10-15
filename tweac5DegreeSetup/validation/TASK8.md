# Task 8: phase-space output extension

Task 8 is complete. The phase-space diagnostic now accepts the PIConGPU-specific option
`picongpu_extension="bp"`. For example:

```python
from picongpu.picmi import constants
from picongpu.picmi.diagnostics import PhaseSpace, TS

phase_space = PhaseSpace(
    species=electrons,
    period=TS[::812],
    spatial_coordinate="y",
    momentum_coordinate="py",
    min_momentum=-50 * constants.m_e * constants.c,
    max_momentum=2000 * constants.m_e * constants.c,
    picongpu_extension="bp",
)
simulation.add_diagnostic(phase_space)
```

The frontend momentum bounds remain SI momenta in kg m/s. Conversion divides
by the species mass times `c`, producing the C++ plugin's dimensionless bounds
`-50` and `2000` in this example. This conversion already existed and is unchanged.

## Implementation and supported values

- [PICMI PhaseSpace](../../lib/python/picongpu/picmi/diagnostics/phase_space.py)
  adds `picongpu_extension`; default conversion maps it to `extension` in
  [backend PhaseSpace](../../lib/python/picongpu/pypicongpu/output/phase_space.py).
  This is a PIConGPU extension, not a new parameter adopted by the PICMI standard.
- Both models share validation for a bare extension token: an ASCII letter or
  digit followed by ASCII letters, digits, underscores or hyphens. For example,
  `bp`, `bp4`, `bp5` and `h5` are accepted. Leading dots, empty strings,
  whitespace, paths, shell syntax and non-string values are rejected.
  Token validation does not determine whether the eventual openPMD installation
  supports the requested backend.
- Omission or `None` emits no extension argument and preserves the existing
  C++ default. The [runtime template](../../lib/python/picongpu/templates/etc/picongpu/N.cfg.mustache)
  emits one `--<species>_phaseSpace.ext <extension>` for each explicit value.
  No C++ code or default-extension selection changed.

## Per-instance semantics and omitted values

In [PhaseSpace.x.cpp](../../include/picongpu/plugins/PhaseSpace/PhaseSpace.x.cpp),
`file_name_extension` is a `plugins::multi::Option<std::string>`.
Each instance passes `file_name_extension.get(m_id)` to its writer. Its
registration prefix is per species. Consequently different diagnostics of one
species can select different formats; these are not conflicting global settings.

[multi::Option](../../include/picongpu/plugins/multi/Option.hpp) collects values
into a vector independently for each option. `get(id)` uses the stored element
when present and the default only when the index is beyond that vector.
There is no CLI placeholder for an omitted value inside it. The
[backend simulation](../../lib/python/picongpu/pypicongpu/simulation.py) therefore
validates the following rule for each species, including all its filters:

| Extensions in diagnostic order | Result |
| --- | --- |
| `None, None` | Both use the C++ default; no extension options |
| `"bp", "h5"` | Distinct explicit formats, accepted |
| `"bp", None` | First uses bp, second uses the C++ default |
| `None, "bp"` | Rejected; emitting just bp would apply it to the first diagnostic |
| `"bp", None, "h5"` | Rejected for the same positional mismatch |

Different species have independent vectors. Filtered and unfiltered diagnostics
of the same species share one vector. Validation also applies to direct backend
construction. To resolve a rejected combination, specify the preceding extensions
explicitly or place diagnostics with explicit extensions first. The implementation
does not reorder diagnostics or guess the C++ default.

## Reference comparison

**405 Python tests and 3501 subtests passed**, including 41 new extension tests,
the existing quick diagnostic tests and the task-7 regression suite.

Python setup generation includes all seven active electron phase spaces from
[config.cfg](../etc/picongpu/config.cfg). Their common settings are filter `all`,
period 812 and extension `bp`:

| Spatial axis | Momentum axis | Minimum, in `m_e*c` | Maximum, in `m_e*c` |
| --- | --- | --- | --- |
| x | px | -20 | 20 |
| x | pz | -20 | 20 |
| y | px | -20 | 20 |
| y | pz | -20 | 20 |
| y | py | -50 | 2000 |
| x | py | -50 | 2000 |
| z | py | -50 | 2000 |

The test multiplies each table bound by `m_e*c` before constructing its PICMI
diagnostic. A separate Python comparison reads the original configuration and
the generated `N.cfg` as data, without executing either file, and confirms every
axis, bound, filter and extension. Momentum round-trip comparisons allow normal
floating-point rounding. The period renders as `0:-1:812`, the existing PICMI
representation of an open-ended 812-step interval. Over the active 1000 steps,
the scheduled outputs remain steps 0 and 812.

The reference repeats `--e_phaseSpace.ext bp` once for each diagnostic and once
again in the aggregate variable. C++ creates seven instances from the seven
period entries, so the eighth extension is unused. Generation emits exactly
seven explicit bp entries, one for each active instance.

The generated setup retains the active grid `3584 x 3584 x 1792`, devices
`8 x 16 x 8`, supercells `8 x 8 x 8`, reference cell sizes and timestep,
CKC, moving-window point zero and **1000 steps**. It uses one electron
macroparticle per cell and `picongpu_min_weighting=0.1`. Generation passes
`-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'`.

This diagnostic-focused generation uses uniform density and omits lasers,
custom PML and other reference outputs. Full integration remains task 9.
No PIConGPU compilation, job submission or simulation was performed, and no
simulation grid was allocated. No phase-space files were written by a running
plugin; runtime openPMD backend availability and file contents remain untested.

The latest generated artifact is
`/tmp/pytest-of-debus/pytest-13/test_seven_reference_phase_spa0/setup`.
This is a temporary path, reproducible by running the tests below.

## Reproduction

Use the isolated `uv` environment described in [task 3](TASK3.md). From the
repository root, run:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/diagnostics/test_phase_space.py \
  lib/python/test/picongpu/quick/picmi/diagnostics/test_phase_space_extensions.py
```

These 45 tests include 41 new extension tests and four existing momentum
conversion tests. Coverage includes invalid tokens, defaults, serialization,
distinct formats, trailing defaults, invalid ordering in both frontend conversion
and direct backend construction, separate species, shared filtered-species indices
and generation of the seven reference diagnostics. The broader acceptance suite
adds the other quick diagnostic tests and the task-7 regression suite.
