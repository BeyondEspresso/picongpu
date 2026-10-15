# PICMI migration of tweac5DegreeSetup

[picmi_setup.py](picmi_setup.py) migrates the active reference, including the
setup-local TWTS header's 64-bit internal arithmetic and envelope cutoff of 8.
These settings now use the existing `precision.param` through
`picmi.PrecisionConfig(twts=64, twts_num_sigmas=8)`; the simulation core remains
float32. Both pulses share these settings, with incident-field, componentwise
injection. No setup-local implementation header or additional param file is needed.
It defaults to the active **1000 steps**, grid **3584 × 3584 × 1792** and
**8 × 16 × 8** devices. It includes the reference density, cold electrons,
numerical methods, PML, memory settings and two TWTS pulses.

It includes field/electron energy, seven PhaseSpace outputs, full fields, three
field slices and checkpoints. Fields include E, B and the built-in electron
charge/energy densities, with reference compression. The separate Binning
`eSpec` diagnostic records charge versus energy and pointing every 500 steps,
with the reference overflow bins and backend settings. Grid adjustment is
explicitly disabled and progress output is set to 1 percent, matching the reference.
See [TASK9D.md](validation/TASK9D.md) for the configuration comparison and limitations,
and [TASK9E.md](validation/TASK9E.md) for the TWTS fix and expanded final audit;
[TASK9A.md](validation/TASK9A.md), [TASK9B.md](validation/TASK9B.md), and
[TASK9C.md](validation/TASK9C.md) record the earlier subtasks.

## Generate with Python only

The reference's Binning backend key `adios2.dataset.chunks="auto"` is omitted:
openPMD ignores it for ADIOS2, and the new typed backend model rejects it.
The `.bp` output uses the equivalent ADIOS2 defaults. See the root
`resolvedConflicts.md` for source evidence and the rebase compatibility changes.

Use this modified repository's Python package. From the repository root, an
isolated environment can be prepared with `uv` (no PIConGPU toolchain needed):

```bash
UV_CACHE_DIR=/tmp/picmi-uv-cache uv venv --python python3.13 /tmp/picmi-task1-venv
UV_CACHE_DIR=/tmp/picmi-uv-cache uv pip install \
  --python /tmp/picmi-task1-venv/bin/python -e './lib/python[test]'
```

The environment above already exists in this workspace; reuse it without
recreating it. Acceptance used Python 3.13.5. The package declares Python >=3.11,
but this generation path uses newer `tempfile` APIs; use the tested 3.13 environment.
The editable installation includes this repository's local PICMI extensions;
an unrelated released package may not. Installation can require network access;
generation and checks with the existing environment ran entirely locally.

Generate into a **new directory outside `tweac5DegreeSetup`**:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python tweac5DegreeSetup/picmi_setup.py \
  /tmp/tweac-picmi-input
```

`PIC_RC` here deliberately names a nonexistent file, avoiding a user toolchain
profile. The output directory must not already exist. Optional `--steps N`
changes the run length; it does not resize the grid or target. Omission selects
1000, not the commented reference alternative of 230000.

The script calls only `write_input_file(...)`. Importing it or calling
`build_simulation()` constructs no files and does not build or run PIConGPU.
Generated files include parameter headers, `etc/picongpu/N.cfg`, metadata and
the library's normal workflow helper files. Those helper files are not executed.
Incident-field/componentwise flags are recorded in `workflow/input.yaml`.
The reference `.build` directory and binaries are not read or used.

## Check the migration

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  tweac5DegreeSetup/validation/test_core.py \
  tweac5DegreeSetup/validation/test_outputs.py \
  tweac5DegreeSetup/validation/test_binning.py \
  tweac5DegreeSetup/validation/test_audit.py
```

This generates a temporary input directory and compares it with the reference
sources. It checks density values, including positions reached after window
slides, and outputs at 1000, 230000 and zero steps. Binning checks cover units,
weighting, pointing quadrants, bin boundaries, accumulation and generated code.
It does not compile PIConGPU, allocate a simulation grid, submit a job, or evolve
fields and particles.

## Acceptance scope

The default 1000-step moving window remains wholly before the plasma start,
matching the reference's active vacuum run. For longer runs, the analytic
density branches and parameters agree, but generated expression evaluation and
the reference's mixed-precision intermediates are not guaranteed bitwise equal.
Other documented differences are inactive settings, equivalent defaults or
floating-point conversion roundoff. Compilation, device execution and actual
diagnostic-file validation have deliberately not been performed.
