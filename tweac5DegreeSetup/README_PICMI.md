# PICMI migration of tweac5DegreeSetup

[picmi_setup.py](picmi_setup.py) implements the core setup and runtime outputs (tasks 9a/9b).
It defaults to the active **1000 steps**, grid **3584 × 3584 × 1792** and
**8 × 16 × 8** devices. It includes the reference density, cold electrons,
numerical methods, PML, memory settings and two TWTS pulses.

It includes field/electron energy, seven PhaseSpace outputs, full fields, three
field slices and checkpoints. Fields include E, B and the built-in electron
charge/energy densities, with reference compression. The separate Binning
`eSpec` diagnostic remains pending **9c**; **9d** audits the complete migration.
See [TASK9A.md](validation/TASK9A.md) and [TASK9B.md](validation/TASK9B.md) for
comparisons and known differences.

## Generate with Python only

Use this modified repository's Python package. From the repository root, an
isolated environment can be prepared with `uv` (no PIConGPU toolchain needed):

```bash
UV_CACHE_DIR=/tmp/picmi-uv-cache uv venv --python python3 /tmp/picmi-task1-venv
UV_CACHE_DIR=/tmp/picmi-uv-cache uv pip install \
  --python /tmp/picmi-task1-venv/bin/python -e './lib/python[test]'
```

The environment above already exists in this workspace; reuse it without
recreating it. Python 3.11 or newer is required. The editable installation
includes the local PICMI extensions; an unrelated released package may not.

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
  tweac5DegreeSetup/validation/test_outputs.py
```

This generates a temporary input directory and compares it with the reference
sources. It checks density values, including positions reached after window
slides, and outputs at 1000, 230000 and zero steps. It does not compile PIConGPU, allocate a simulation grid, submit a job,
or evolve fields and particles.
