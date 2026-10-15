# Task 9d — final migration audit

**Historical correction, resolved by [task 9e](TASK9E.md).** The original audit missed
`include/picongpu/fields/background/templates/twtstight/TWTSTight.hpp` in the
reference setup. It overrides TWTS internal arithmetic to `float_64` and the
envelope cutoff to `numSigmas=8`; generated input at that point inherited `float_X`
(32-bit here) and 6 from the repository header. The parameter-file inventory
below did not cover this implementation-header override. The 471 passing tests
therefore do not establish complete agreement with the reference.

This affects laser evaluation even in vacuum; the density-rounding argument
does not resolve it. Task 9e adds shared TWTS controls to the existing
`precision.param`, preserves 64-bit/8, and expands the inventory to every file
under the reference `include/`. Its report records the updated acceptance results.
Whether recent numerical changes make the override unnecessary has not been
established. Compilation and simulation remain unrun. The results below describe
the earlier task 9d run unless explicitly marked as updated by 9e.

This audit used checkout `a352391e4` plus the task 9d working-tree changes.
The reference is `etc/picongpu/config.cfg` and the active parameter files under
`tweac5DegreeSetup/include/picongpu/param`. The old build, binaries and alternate
`*Old.param`/`binningSetupNew.param` files are not migration inputs.

## Complete configuration comparison

| Reference source | Active settings and generated equivalent | Evidence |
| --- | --- | --- |
| `simulation.param` | Cells `(3584,3584,1792)`, cell sizes `(6.177353e-8,3.088676e-8,6.177353e-8)` m, `dt=5.15136e-17` s, base density `1e25 m^-3`, typical PPC 1 | `test_core.py` |
| `config.cfg` grid/run | Devices `(8,16,8)`, 1000 steps, 2 h walltime, PML, moving window at point 0, binomial current smoothing, version output | Core tests and complete runtime-option comparison |
| `config.cfg` remaining controls | `--autoAdjustGrid false`, `--percent 1` now explicitly generated; nonperiodic boundaries explicitly match the reference's C++ default | `test_audit.py`, runtime-control regressions |
| `fieldSolver.param` | CKC | Core tests |
| `species.param` | PQS shape, matching AssignedTrilinearInterpolation, EmZ/PQS current deposition, Free → HigueraCary at step 6720 | Core tests and feature regressions |
| `speciesDefinition.param` | Electrons `e` only; charge ratio 1, mass ratio rounds to 1 in float32; position, momentum and weighting attributes | Core tests; inactive ions excluded |
| `particle.param` | Random layout, one macroparticle/cell, minimum weighting 0.1 | Core and minimum-weighting tests |
| `speciesInitialization.param` | One density creation operation; cold electrons without drift or temperature manipulation | Core tests |
| `density.param` | Transverse hard cuts, Gaussian ramps, fixed 3584×1792 transverse dimensions, longitudinal vacuum until `6720*dy`, then 100 µm linear ramp | 1352 analytic sample comparisons; rounding classification below |
| `memory.param` | 8³ supercells, one guard supercell/direction, 4 GiB reserve, electron exchanges 32/320/32 MiB and 2 MiB edges/corners, zero scaling/reference-domain vectors | Core and memory tests |
| `memory.param`, inherited field output | One temporary field slot with gather communication enabled, also with all diagnostics active | Final integration check and openPMD source inspection |
| `fieldAbsorber.param` | 110 PML cells on every face, sigma/kappa grading 2.5, sigma multiplier 1.25, kappa `(4,1,4)`, alpha grading 1, alpha maxima zero | Core and PML regressions |
| `TwtsBackgroundLaser.param` | Two pulses: λ=0.8 µm, waist=1 µm, a0=1.25, duration `25 fs/2.354820045`, incidence ±5°, polarization ±45.10716816968629°, focal y=68.5 µm, offsets ±12 µm, beta0=1 | Core and TWTS regressions |
| `incidentField.param` | Both pulses on YMin; positive pulse on ZMin, negative on ZMax; other faces inactive; surfaces `[[116,-116],[116,100000000],[116,-116]]`; window `(0,10000000,10000)` | Core/TWTS tests |
| Injection selection | Recorded flags `PARAM_FIELD_BACKGROUND=0`, `PARAM_COMPONENTWISE=1`; no active field background | Workflow-input checks; source inspection |
| Setup-local `TWTSTight.hpp` (added in 9e) | Internal `float_T=float_64`, cutoff 8 through generated `precision.param`; core remains float32 | Full normalized header comparison and TWTS configuration tests; `TASK9E.md` |
| `particleFilters.param` | `all` is the only selected output filter; unused on-axis/pinhole filters omitted | Output inventory and final integration |
| Energy options | Field and electron energy every 812 steps; electron filter `all` | `test_outputs.py` |
| PhaseSpace options | Seven reference coordinate pairs, SI-converted momentum limits, `all`, every 812 steps, `.bp` | Output tests; task 9b table |
| Field dump options and inherited `fileOutput.param` | E, B, electron charge/energy densities; full dump every 8120 steps, three 24-cell slices every 812; reference names/ranges, `_%T`, `.bp`, mapped memory and Blosc/Zstd settings | Four generated TOML files compared with active reference JSON/options |
| Checkpoint options | `0:1000:50000`, `_%T`, mapped memory; extension/backend/name/directory left at reference defaults; no restart requested | Output tests |
| `binningSetup.param` | `eSpec`: 1600 energy bins, 256 pointing bins, signed weighted charge, overflow bins, every 500 steps, `.bp`; ineffective ADIOS2 `chunks:auto` key omitted on rebase | `test_binning.py`; root `resolvedConflicts.md` |
| Inherited precision/defaults | Core float32 and standard special-operation precision; unchanged unoverridden C++ defaults | Generated precision and shared source |

The runtime comparison expands quoted TBG assignments as data and compares
effective arguments. It never sources a configuration file. The differently
represented diagnostic CLI/TOML settings are compared separately by the existing
output tests. Task 9e expanded the inventory from 17 parameter files to all 18
files under the reference `include/`, including the TWTS implementation header;
changes to that inventory cause the audit test to fail.

## Resolved runtime gaps

Two optional, PIConGPU-specific `Simulation` arguments were added:

- `picongpu_auto_adjust_grid: bool | None`: explicit false emits
  `--autoAdjustGrid false`; omission retains the C++ default.
- `picongpu_progress_percent: int | None`: integers 1–100 emit `--percent N`;
  omission retains the C++ default. This controls stdout statistics, not diagnostics.

Both frontend and backend validate types; booleans are not accepted as a progress
integer, and strings/numeric substitutes are not accepted as grid booleans.
The renderer distinguishes explicit false from omission. No arbitrary CLI
injection or setup-specific template override was introduced.

These controls preserve the reference's policy and progress cadence. With this
particular grid, automatic adjustment would already be unnecessary: each rank
has `(448,224,224)` cells, all divisible by 8, at least three supercells wide,
and large enough for the applicable 110-cell absorber. The C++ `DomainAdjuster`
checks these conditions; disabling adjustment retains validation and makes an
invalid later domain fail rather than silently change its dimensions.

No temporary-storage extension is needed. The openPMD derived-field writer
reuses temporary slot 0 for each field and requires `1 + RequiredExtraSlots`.
Both selected density solvers require zero extra slots. Binning uses its own
histogram buffer. The final generated setup retains exactly one FieldTmp slot.

## Remaining differences, classified

| Difference | Classification / consequence |
| --- | --- |
| Binning backend option after rebase | The reference's `adios2.dataset.chunks="auto"` is ignored by openPMD and rejected by upstream's typed model. Omitted with effective ADIOS2 defaults preserved; source evidence is recorded in root `resolvedConflicts.md`. |
| Density arithmetic | Same analytic branches and constants, but SymPy rearranges operations and the reference rounds some intermediates to float32. Not a bitwise-equivalence claim for longer plasma runs. No effect on the agreed active 1000-step vacuum run; see below. |
| Electron mass and laser amplitude conversions | Python particle data and the C++ electron-mass constant differ by about `3.4e-12` relatively. The mass ratio rounds to the same float32 value; laser amplitude comparison passes at `1e-12` relative tolerance. No different physical parameter was selected. |
| Phase-space `-50` momentum bound | SI round-trip prints `-49.999999999999993`; the float32 value is identical. |
| Binning dump period | Reference C++ default 0 and PICMI explicit 1 both dump/reset every notification in this source version. Raw charge per bin, dimensions, overflow and storage order match. |
| Output representation | Explicit field sources instead of `fields_all`, and TOML tables instead of inline CLI backend JSON. Effective selected sources and backend configuration match. |
| Available J output and radiation placeholder | J is available in the generated field typelist but never selected. The radiation placeholder has no species/period and enables no output. |
| Particle exchange configuration | Reference species-specific electron setting becomes the global setting. Electrons are the only active species, so effective buffers agree. |
| Inactive reference definitions | PNG/histogram/pinhole/resource-log definitions, ions, thermal/drift manipulators, exponential-absorber coefficients and alternate laser/binning files are not active and need not be reproduced. |
| Metadata / generated helpers | Generated names, UUID-bearing metadata and workflow helper scripts need not be byte-identical to the handwritten reference. Helpers are written but never executed in acceptance. |

The active profile has no particles until y reaches **207.5590272 µm**. A
conservative bound covering the initial box, motion through step 1001 and one
additional initializing GPU is **133.0756143 µm**. Thus the active run never
reaches nonzero density; both profiles return zero before any transverse
mixed-precision arithmetic matters. The profile remains defined for subsequent
moving-window regions, and tests compare its analytic values there. A longer
`--steps` override is supported but does not establish bitwise particle loading
or physical equivalence in plasma. The original density-rounding caveat remains
visible rather than being treated as a passed device-level check.

## Generation and acceptance

Use [README_PICMI.md](../README_PICMI.md) for the exact isolated `uv` preparation
and generation commands. The existing environment uses **Python 3.13.5** and
an editable install of this modified checkout. Although package metadata permits
3.11, the generation path uses newer `tempfile` options; the documented workflow
uses the tested 3.13 interpreter. No environment installation or network access
was needed for this task.

The CLI generated **`/tmp/tweac-picmi-final`**, containing:

- `include/picongpu/param/`: generated core, laser and diagnostic definitions;
- `etc/picongpu/N.cfg`: effective runtime settings, periods and plugin configuration references;
- four `etc/openPMD_config_*.toml` files for full fields and slices;
- `workflow/input.yaml`: recorded incident-field/componentwise CMake flags;
- metadata, ordinary workflow helpers and an empty workflow cache directory.

No binary, simulation output, checkpoint or diagnostic data file was produced.
The audit test forbids subprocess execution and simulation entry points during
generation and guards against reads/scans of the reference `.build` and `bin`
directories. Import/construction tests forbid writes and directory creation.
The existing core, output and Binning checks run with everything enabled.

The final regression command combines the four migration test files with
runtime controls, simulation conversion, diagnostics, TWTS/Gaussian lasers,
PML, minimum weighting, deposition/interpolation, composite pushers,
analytic distributions and memory tests. **471 tests and 3497 subtests passed**,
including 64 migration acceptance tests and 27 runtime-control regressions.
The CLI also generated the final artifact independently. No new test required
compilation or simulation.

Reproduce the final regression selection with:

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  tweac5DegreeSetup/validation/test_core.py \
  tweac5DegreeSetup/validation/test_outputs.py \
  tweac5DegreeSetup/validation/test_binning.py \
  tweac5DegreeSetup/validation/test_audit.py \
  lib/python/test/picongpu/quick/picmi/test_runtime_options.py \
  lib/python/test/picongpu/quick/picmi/test_simulation.py \
  lib/python/test/picongpu/quick/picmi/diagnostics \
  lib/python/test/picongpu/quick/picmi/test_twts_laser.py \
  lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py \
  lib/python/test/picongpu/quick/picmi/test_pml.py \
  lib/python/test/picongpu/quick/picmi/test_min_weighting.py \
  lib/python/test/picongpu/quick/picmi/test_current_deposition.py \
  lib/python/test/picongpu/quick/picmi/test_species_interpolation.py \
  lib/python/test/picongpu/quick/picmi/test_composite_pusher.py \
  lib/python/test/picongpu/quick/picmi/distribution/test_analytic_distribution.py \
  lib/python/test/picongpu/quick/pypicongpu/test_memory_config.py
```

**Unrun by design:** C++ compilation, MPI/GPU execution, cluster submission,
physical validation, actual openPMD files and backend compression availability
in a future installation. These are not prerequisites for the requested
Python-only migration acceptance.
