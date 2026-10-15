# Task 9b — active runtime diagnostics

Implemented in [picmi_setup.py](../picmi_setup.py). All outputs assigned to 9b
match the active reference configuration in Python generation and source-based
comparisons. The separate Binning `eSpec` definition is reserved for 9c; the
complete migration audit is 9d. No PIConGPU compilation, job submission,
simulation or diagnostic data-file creation was performed.

## Reference comparison

The acceptance test reads the quoted assignments in `etc/picongpu/config.cfg`
as data, expands the active `TBG_plugins` references, and compares them with
generated `N.cfg` options and openPMD TOML files. It does not execute the shell
configuration or read the reference `.build` directory or binaries.

| Output | Preserved settings (default 1000 steps) |
| --- | --- |
| Field energy | Every 812 steps |
| Electron energy | Every 812 steps, `all` filter |
| PhaseSpace | Seven instances, every 812 steps, `all` filter, `.bp` |
| Full fields | `0:1000:8120`, file `simOutput`, range `:,:,:` |
| X slice | `0:1000:812`, file `simOutputX`, range `1780:1804,:,:` |
| Y slice | `0:1000:812`, file `simOutputY`, range `:,1780:1804,:` |
| Z slice | `0:1000:812`, file `simOutputZ`, range `:,:,884:908` |
| Checkpoint | `0:1000:50000`, infix `_%T`, `mappedMemory` |

Phase-space ordering and momentum limits in electron `m*c` units are
`x/px`, `x/pz`, `y/px`, `y/pz` with [-20,20], then `y/py`, `x/py`, `z/py`
with [-50,2000]. The Python definition supplies SI momentum as required by the
frontend. Conversion back to `m*c` renders -50 as -49.999999999999993, a binary64
rounding difference below 1e-15 relative; both round to the same float32 value.
The reference has an unused eighth `.ext bp` option; the script emits one per
actual instance, as established in task 8.

The energy and phase-space periods render as `0:-1:812`, equivalent to the
reference's shorthand `812`. Both include step zero. Bounded field/checkpoint
schedules retain their explicit inclusive endpoint, including when the period
exceeds the run length. At 1000 steps, full fields and checkpoints are scheduled
only at zero, slices and energy/phase-space outputs at 0 and 812. An explicit
`--steps N` changes the bounded endpoints, leaving intervals and grid unchanged.
Acceptance covers 1000, 230000 and zero steps.

All four field outputs preserve `.bp`, `_%T`, `mappedMemory`, and only the active
ADIOS2 compression: Blosc, level `"1"`, compressor `"zstd"`, shuffle
`"BLOSC_BITSHUFFLE"`. The unused BP5 engine/buffer/profile configuration is not
selected. Compression is embedded in four generated TOML files with reproducible
content hashes; no external configuration file needs copying. The existing C++
TOML reader forwards `backend_config` tables to openPMD.

Checkpoint extension, backend configuration, directory and filename remain
unspecified, preserving the reference's C++ defaults. No restart option is
enabled. Checkpoint fields/particles use the independent `FileCheckpointFields`
and `FileCheckpointParticles` definitions.

## Field inventory and necessary interface changes

The reference inherits repository `include/picongpu/param/fileOutput.param`:
`fields_all` selects E, B, `e_all_chargeDensity`, and `e_all_energyDensity`.
`CreateEligible_t` defaults to the `All` filter. Additional defined reference
particle filters do not add further fields. The script requests these four
sources explicitly in each output group, so the Python template's available J
field is not written. No particle dumps are enabled.

`DerivedFieldDump.functor` now also accepts `"ChargeDensity"` and
`"EnergyDensity"`. These are PIConGPU extensions, not new PICMI-standard
definitions. They select the existing C++ attributes rather than translating
their arithmetic into custom functors. This retains species shape/deposition,
macroparticle charge/energy weighting, cell-volume normalization, and openPMD
unit scaling and dimensions. Charge density is C/m³ with dimensions
(-3,0,1,1,0,0,0); kinetic-energy density is J/m³ with dimensions
(-1,1,-2,0,0,0,0). Built-in solvers are restricted to the requested species and
filter and registered once across all output files. Existing custom functors
remain available and unchanged.

`OpenPMDConfig.backend_config` now accepts nested dictionaries and inline strings
in addition to existing `PathLike` values. Two pre-existing range defects were
also fixed: the validator now accepts the documented `RangeSpec` objects, and
unrestricted dimensions serialize to `:`. The latter matters because
`pmacc::pluginSystem::toRangeSlice` skips empty components: `,,884:908` would
restrict X, whereas `:,:,884:908` correctly restricts Z. The slice end is
exclusive, so each reference slice is 24 cells thick.

The effective generated option inventory contains exactly the task 9b outputs.
Unused PNG, energy histogram, pinhole, field-slice-plugin, macro-count and
resource-log definitions stay inactive. The Python backend adds its required
radiation template placeholder with empty species and an empty schedule; this
does not enable a radiation output. Binning `eSpec` is not yet migrated.

## Acceptance

Using the existing isolated Python environment, with a nonexistent `PIC_RC`
and without installing a compiler/toolchain:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/diagnostics \
  lib/python/test/picongpu/quick/picmi/test_simulation.py \
  tweac5DegreeSetup/validation/test_core.py \
  tweac5DegreeSetup/validation/test_outputs.py
```

The selected tests passed in two runs: **131 tests and 3489 subtests**,
including 18 migration acceptance tests. Regression coverage includes range
objects/tuples on all axes, inline backend configuration, invalid built-in
selectors, and multiple species/filters/files without duplicate registration.
Core density, laser, grid, PML and initialization comparisons still pass.

The CLI also generated `/tmp/tweac-task9b-final` with default 1000 steps.
Generated configuration files are portable and use content-derived filenames.
These checks validate input generation and its agreement with the checked-out
C++ source. They do not validate compiled execution, actual openPMD files,
compression support in a future installation, or numerical simulation results.
Task 9a's density intermediate-rounding caveat and the remaining 9d runtime
option audit still apply.
