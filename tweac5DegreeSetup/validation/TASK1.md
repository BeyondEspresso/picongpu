# Task 1: TWTS centroid and timing validation

This report records the original task-1 validation. [Task 2](TASK2.md) later
corrected polarization metadata, replaced the temporary ±45° test angles with
the actual reference angles, and reran the generation tests successfully.

Task 1 is complete at the Python generation level: **35 tests passed**, including
five cases that call `Simulation.write_input_file()` to generate complete setup
directories. The application implementation needed no further changes.

Only Python dependencies were installed, in `/tmp/picmi-task1-venv`, using `uv`.
No PIConGPU C++ compilation, job submission or simulation was performed. Generated
workflow/build scripts were written as setup artifacts and were not executed.

## Scope

The tests are in
[test_twts_laser.py](../../lib/python/test/picongpu/quick/picmi/test_twts_laser.py)
and [test_gaussian_laser.py](../../lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py).

Each new setup-generation case contains two TWTS lasers at incidence angles
`+5°` and `-5°`, with focal y `68.5 µm`, `beta0=1`, and the reference switching
window. The following centroid positions exercise the manual delay formula:

| Centroid y (µm) | Delay of each beam (fs) |
| --- | --- |
| -10 | 261.847814730549 |
| 0 | 228.491405210734 |
| 30 | 128.422176651289 |
| 68.5 | 0 |
| 100 | -105.072689987418 |

Assertions cover PICMI-to-backend conversion, generated metadata, rendered
`TDELAY`, `AUTO_TDELAY=false`, focus, lateral offsets, incidence angles, switching
window, and beam selection on YMin/ZMin/ZMax. TWTS supplies `pulse_init=0.0` for
backend compatibility; neither `PULSE_INIT` nor `RAMP_INIT` appears in its generated
incident-field configuration.

The other tests verify that Gaussian, dispersive and plane-wave lasers still
reject positive centroid y, and that TWTS still validates propagation direction
and polarization normalization. Existing Gaussian/dispersive duration checks
also pass.

## Comparison with the reference

The zero-delay generation was additionally compared directly with the active
reference files, rather than only the constants in the tests:

| Reference | Checked result |
| --- | --- |
| `include/picongpu/param/TwtsBackgroundLaser.param` | Zero delay, automatic delay disabled, window `0 / 10000000 / 10000`; wavelength, waist, duration, focal y, signed incidence/lateral offsets and beta0 agree |
| `include/picongpu/param/incidentField.param` | Huygens positions `[[116,-116],[116,100000000],[116,-116]]` agree |
| `include/picongpu/param/simulation.param` | All three cell dimensions and `dt=5.15136e-17 s` agree |
| `etc/picongpu/config.cfg` | Grid `3584 × 3584 × 1792`, devices `8 × 16 × 8`, and 1000 steps agree |
| `cmakeFlagsSetup` | Incident-field/componentwise options agree verbatim with `build_cmake` in the generated workflow input |

The reference options recorded in each generated setup are:

```text
-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'
```

CKC, binomial current smoothing and moving-window start `0.0` are configured in
these generation cases. The complete reference grid is represented as metadata;
Python generation does not allocate its field or particle arrays.

## Reproduction

From the repository root, create the isolated environment once:

```bash
UV_CACHE_DIR=/tmp/picmi-uv-cache uv venv --python python3 /tmp/picmi-task1-venv
UV_CACHE_DIR=/tmp/picmi-uv-cache uv pip install --python /tmp/picmi-task1-venv/bin/python -e './lib/python[test]'
```

Run only the two Python test files:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_twts_laser.py \
  lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py
```

`PIC_RC` points to a nonexistent file to avoid loading a personal runtime profile.
Pytest retains generated setups in its temporary directories. In this validation
run, the zero-delay artifact was:

```text
/tmp/pytest-of-debus/pytest-0/test_twts_setup_generation_6_80/setup
```

That temporary path may change or be cleaned on later runs; the tests are the
reproducible artifact. The successful run reported `35 passed in 24.69s` with
Python 3.13.5, pytest 9.1.1, Pydantic 2.13.5 and NumPy 2.5.3. The editable PIConGPU
package used the local checkout. Its Git dependencies resolved to:

- picmistandard: `c1d662238fc6642ca683826df688ccca12ed3765`
- moosetash: `295c54ab5f33aebaf1272aad6eb2dc0f42681c1a`
- rocrate: `057a2b60b63c58cc26d091c7503bc7878c2b414e`

## Limits

These cases deliberately use polarization angles `±45°` instead of the setup's
`±45.10716816968629°`, because the latter encountered the independent
polarization-vector bug at the time of task 1. Task 2 has removed this
substitution from the current tests.

The generation cases contain no particle species or diagnostics and do not yet
reproduce the custom PML. Those are later migration tasks. No claim is made about
compilation or physical field evolution. Positive centroid support configures
boundary injection; it does not initialize an analytic field in the interior at
time zero.
