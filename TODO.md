# Migrate tweac5DegreeSetup to PICMI

## Objective and working rules

Produce a Python definition that reproduces the active settings in
[tweac5DegreeSetup](tweac5DegreeSetup), including numerical methods and diagnostics.
Do not silently substitute different physics or numerical defaults to make the
script generate successfully.

Work on one numbered task at a time. Each task should have a small, reviewable
implementation, focused Python tests, and documentation of its public parameters.
Prefer local overrides for TWTS behavior; do not change Gaussian or dispersive
laser behavior to accommodate TWTS. Shared changes are appropriate where the
feature itself is shared, such as species deposition or PML configuration.

Do not compile PIConGPU or run a simulation, locally or on a cluster, as part of
this work. Acceptance consists only of Python runs that generate the setup and
comparison of the generated parameters and runtime configuration with the
reference. Python construction, conversion and template-rendering checks can
support this comparison; they do not establish physical equivalence of a run.
Use the available `uv` for an isolated Python environment as needed, without
installing the full PIConGPU toolchain. Record unavailable checks as unrun.

This inventory describes the checkout at `4ee5ee0d7`, including the existing
TWTS-local centroid fix. API names suggested below are design work, not claims
that the corresponding parameters already exist.

## Task order and status

| ID | Task | Status / dependency |
| --- | --- | --- |
| 0 | Pin the reference run configuration | Complete; choices confirmed and active outputs recorded below |
| 1 | Validate the existing TWTS centroid fix | Complete; 35 Python tests passed, including five setup-generation cases |
| 2 | Correct TWTS polarization geometry | Complete; 62 Python tests passed, generation uses the actual setup angles |
| 3 | Expose the composite particle pusher | Complete; 139 Python tests passed, generated type matches the reference |
| 4 | Keep interpolation consistent with species shape | Complete; 151 Python tests passed, PQS interpolation matches the reference |
| 5 | Expose EmZ current deposition with the selected shape | Complete; 181 Python tests passed, EmZ/PQS matches the reference |
| 6 | Expose custom PML thickness and coefficients | Complete; 274 Python tests passed, generated PML matches the reference |
| 7 | Expose minimum macroparticle weighting | Complete; 337 Python tests passed, reference cutoff and normalization verified |
| 8 | Expose the phase-space output extension | Complete; 405 Python tests passed, all seven reference phase spaces match |
| 9 | Generate and audit the complete Python setup | Complete for Python-generation scope; task 9e closes the missed TWTS override |
| 9a | Assemble the core setup and exact density profile | Complete; core script generated, 268 Python tests passed |
| 9b | Add energy, phase-space, field and checkpoint outputs | Complete; active outputs matched, 131 selected tests passed |
| 9c | Add the Binning electron spectrometer | Complete; 59 migration tests passed, including 41 Binning checks |
| 9d | Audit and deliver the complete migration | Complete after the expanded task 9e audit |
| 9e | Preserve TWTS arithmetic precision and envelope cutoff | Complete; existing precision.param, 510 tests and 3497 subtests passed |

Tasks 2, 3, 6, 7 and 8 can be tackled independently. Task 1 is validation of work
already present, not a request to redesign the shared laser classes.

## 0. Pin the reference run configuration

The directory contains several `.cfg` files and commented alternatives. Use
[config.cfg](tweac5DegreeSetup/etc/picongpu/config.cfg) and the recorded
[cmakeFlagsSetup](tweac5DegreeSetup/cmakeFlagsSetup) as the confirmed reference.

- [x] Default the final script to the active **1000 steps**. Do not use the
  commented 230000-step setting.
- [x] Preserve the active grid `3584 × 3584 × 1792` and device decomposition
  `8 × 16 × 8`.
- [x] Target incident-field injection (`PARAM_FIELD_BACKGROUND=0`) with
  componentwise field evaluation (`PARAM_COMPONENTWISE=1`) only. The alternative
  background-field and non-componentwise presets are outside this migration.
- [x] Record the active diagnostics from `TBG_plugins` and `binningSetup.param`.
  Here, "inventory" means identifying enabled outputs, their schedules, species,
  filters, axes/ranges, filenames and backend settings so they can be reproduced.

| Active output | Reference schedule in timesteps |
| --- | --- |
| Total field energy and electron energy (`all` filter) | Every 812 |
| Seven electron phase spaces (`all` filter) | Every 812 |
| Full field dump, `simOutput` | `0:1000:8120` |
| Field slices, `simOutputX`, `simOutputY`, `simOutputZ` | `0:1000:812` |
| Checkpoints | `0:1000:50000` |
| Energy-versus-pointing electron spectrometer, `eSpec` | Every 500 |

The user confirmed that the migration must reproduce all active outputs listed
above, including their existing schedules and settings. This is the agreed
diagnostic scope, not a provisional selection.

Retain these schedules even where the interval exceeds the 1000-step run;
do not shorten an interval to force additional output. The full-field and
checkpoint schedules include step zero. Task 9 records the detailed output
settings. `eSpec` is registered in `binningSetup.param`, separately from TBG.
PNG outputs, energy histograms, macroparticle counts, the standalone field-slice
plugins and the resource log are defined but not enabled by `TBG_plugins`.
The active binomial-current option is a numerical setting, not a diagnostic.

**Acceptance:** The final script states which reference configuration it
reproduces and defaults to the confirmed 1000-step configuration. Acceptance
generates files for this grid; it does not allocate or evolve a simulation grid.

## 1. Validate the existing TWTS centroid fix

**Required behavior:** A focal y coordinate of `68.5e-6 m` and an equal centroid
y coordinate yield `TDELAY = 0`, with `AUTO_TDELAY = false`.

The implementation is already local to
[TWTSLaser](lib/python/picongpu/picmi/lasers/twts_laser.py): it overrides validation
to allow positive centroid y, retains direction/polarization checks, and supplies
the unused compatibility value `pulse_init = 0.0`. The shared validator and
backend laser classes retain their original implementations.

- [x] Run [test_twts_laser.py](lib/python/test/picongpu/quick/picmi/test_twts_laser.py)
  and the existing Gaussian laser regression tests in a suitable Python environment.
- [x] Verify positive, zero and negative manual delays survive conversion and
  template rendering, including both incidence-angle signs.
- [x] Verify the window remains `windowStart=0`, `windowEnd=10000000`,
  `windowLength=10000`, in timestep units.
- [x] Keep the limitation explicit: this configures boundary injection; it does
  not populate the domain with an analytic field at time zero.

**Acceptance result:** All 35 TWTS/Gaussian Python regression tests passed in an
isolated environment created with `uv`. Five complete setup-generation cases
cover both incidence signs and positive, zero and negative delays on the active
grid with 1000 steps. The zero-delay case's timing, window, geometry, grid,
timestep and recorded CMake settings were compared directly with the reference.
See [task 1 validation](tweac5DegreeSetup/validation/TASK1.md) for commands and scope.
No PIConGPU compilation, job submission or simulation execution was performed.
The original task-1 validation used polarization angles of ±45° to isolate
timing. Task 2 subsequently replaced those test inputs with the actual reference
angles and reran the generation cases; they are still not the final migrated setup.

## 2. Correct TWTS polarization geometry

**Original problem:** The `polarization_direction` calculation in
[twts_laser.py](lib/python/picongpu/picmi/lasers/twts_laser.py) used
`cos(polarizationAngle)` for both x and the z prefactor. For incidence `±5°`
and polarization `±45.10716816968629°`, the resulting vector has norm about
`0.998142049` and failed validation.

- [x] Derive the polarization vector from the incidence and polarization-angle
  conventions in the [C++ TWTS implementation](include/picongpu/fields/background/templates/twtstight).
  Check the transverse basis and signs; do not merely normalize the incorrect vector.
- [x] Implement the correction within `TWTSLaser` while preserving the scalar
  `Polarization` angle sent to C++.
- [x] Test unit norm and orthogonality to propagation for both incidence signs,
  zero and right-angle polarization, and the actual setup angles.
- [x] Combine the actual setup angles with task 1's zero-delay configuration and
  verify both laser objects convert and render successfully.

**Acceptance result:** Corrected the z prefactor to `sin(polarizationAngle)`.
All 62 TWTS/Gaussian Python tests passed, including five complete setup-generation
cases with the actual ±45.10716816968629° angles and unchanged `a0=1.25`.
Both pulses convert and render with zero delay, and their nominal polarization
axes are unit length and transverse. No normalization workaround or disabled
validation is used. See [task 2 validation](tweac5DegreeSetup/validation/TASK2.md)
for the derivation, reference comparison and reproduction command. No PIConGPU
compilation or simulation was performed.

## 3. Expose the composite particle pusher

**Required C++ configuration:**

```cpp
particles::pusher::Composite<
    particles::pusher::Free,
    particles::pusher::HigueraCary,
    particles::pusher::CompositeBinarySwitchActivationFunctor<6720>>
```

See [setup species.param](tweac5DegreeSetup/include/picongpu/param/species.param)
and [particlePusherComposite.hpp](include/picongpu/particles/pusher/particlePusherComposite.hpp).
The first pusher applies for `currentStep < 6720`; the second applies from 6720.

- [x] Design a typed PIConGPU-specific species option describing two supported
  pushers and an integer switching step. Define its interaction with `method`;
  reject conflicting specifications. Arbitrary C++ strings are unnecessary.
- [x] Extend the [PICMI species mapping](lib/python/picongpu/picmi/species.py),
  [backend species model](lib/python/picongpu/pypicongpu/species/species.py), and
  [species template](lib/python/picongpu/templates/include/picongpu/param/speciesDefinition.param.mustache).
- [x] Verify public pusher names render valid C++ names. Corrected the backend
  value from `Higuera-Cary` to `HigueraCary`, keeping the PICMI spelling unchanged.
  Removed the unsupported `other:Axel` mapping: this checkout has no such C++ type.
- [x] Test exact generated types, step zero and step 6720, invalid switching
  steps, and unchanged rendering for existing single-pusher inputs.

**Acceptance:** The generated species selects precisely the composite above.
Retain this setting even if the chosen short run ends before the switch.

**Acceptance result:** Added `picmi.CompositePusher(first="free-streaming",
second="Higuera-Cary", switch_step=6720)` through the species option
`picongpu_composite_pusher`. Omit `method` or use `None`; an explicit method
conflicts. Switching steps are strict uint32-range integers. Duplicate pusher
types (invalid duplicate C++ bases), unknown methods and nested composites are
rejected. The backend carries the structured selection and derives its C++ type
for the species template.

All 139 selected Python tests passed, including 52 new composite-pusher tests,
existing species tests and the 62 TWTS/Gaussian regressions. Generated cases at
steps 0 and 6720 preserve the active grid and 1000-step limit. The latter's
pusher type matches the active reference alias exactly after whitespace removal.
See [task 3 validation](tweac5DegreeSetup/validation/TASK3.md) for reproduction
and limitations. No C++ compilation or simulation was performed. The complete
setup still requires tasks 4–9.

## 4. Keep field interpolation consistent with species shape

**Original problem:** `particle_shape="cubic"` selected the PQS shape flag, but the
[species template](lib/python/picongpu/templates/include/picongpu/param/speciesDefinition.param.mustache)
still used `interpolation<UsedField2Particle>`. The
[default species.param](include/picongpu/param/species.param) defines that alias
using TSC. Your setup uses PQS for both particle shape and interpolation.

- [x] Generate the interpolation type from each species' selected shape,
  retaining `AssignedTrilinearInterpolation` for this setup.
- [x] Avoid changing a global alias as a shortcut: different species may select
  different shapes.
- [x] Add a rendering test with two different species shapes to catch accidental
  reuse of a global/default shape, plus a default-shape regression test.

**Acceptance:** The electron species renders
`FieldToParticleInterpolation<particles::shapes::PQS, AssignedTrilinearInterpolation>`
or an equivalent species-specific alias.

**Acceptance result:** Each species now defines `ParticleShape_species_<name>`
from its existing backend shape. Both its shape flag and
`FieldToParticleInterpolation<..., AssignedTrilinearInterpolation>` use that
alias. No new PICMI option is needed; omitted/`None` shapes still select TSC.
The diagnostic-only `other:counter` shape is rejected during species conversion
because it lacks the cached assignment interface required by interpolation.

All 151 selected Python tests and 12 subtests passed, including 12 new shape and
interpolation tests and the previous species, composite-pusher and laser tests.
A mixed-species setup generated PQS, CIC and default TSC interpolation on the
active grid with 1000 steps. Resolving the electron shape alias produced exactly
the reference interpolation type, with the reference composite pusher retained.
See [task 4 validation](tweac5DegreeSetup/validation/TASK4.md). No C++ compilation
or simulation was performed. Current deposition was unchanged in task 4;
task 5 below now also uses the per-species shape alias.

## 5. Expose EmZ current deposition with the selected shape

**Original problem:** Your setup uses `currentSolver::EmZ<PQS>`. The generated species
used the global `UsedParticleCurrentSolver`, which defaults to
`Esirkepov<TSC>`. Current smoothing is a separate feature and does not select EmZ.

- [x] Add a validated PIConGPU-specific current-deposition choice per species,
  initially supporting the existing default Esirkepov and the required EmZ.
- [x] Render deposition using that species' shape, using the same shape source
  established by task 4. Preserve existing default behavior for default inputs.
- [x] Propagate the selection through PICMI conversion, backend representation
  and rendering. Document accepted names and unsupported combinations.
- [x] Test generated `EmZ<PQS>`, default deposition, and mixed-species choices.

**Acceptance:** The electron species uses PQS consistently for shape,
interpolation and EmZ deposition. Binomial smoothing remains independently enabled.

**Acceptance result:** Added `Species.picongpu_current_deposition`, accepting
`"Esirkepov"` (default), `"EmZ"` or its equivalent spelling `"EZ"`. The backend
normalizes `"EZ"` to `"EmZ"`, carries a validated enum and the
template uses `currentSolver::<algorithm><ParticleShape_species_<name>>`.
Default inputs still produce `Esirkepov<TSC>`; selecting another shape now also
changes the deposition shape. The C++ default strategy is retained.

EmZ accepts CIC/TSC/PQS/PCS and rejects NGP because its support-one deposition
loops are empty. Esirkepov retains NGP support; Counter remains invalid as a
species shape. Unknown algorithms, `None` and arbitrary C++
expressions are rejected. The public extension and its limits are documented.

All 181 selected Python tests and 12 subtests passed, including 30 new deposition
tests. Two mixed-species setups generated on the active grid with 1000 steps,
with binomial smoothing enabled and disabled. Direct reference comparison
confirmed the electron's PQS shape, PQS interpolation, EmZ/PQS deposition and
step-6720 composite pusher in both cases. See
[task 5 validation](tweac5DegreeSetup/validation/TASK5.md). No PIConGPU compilation
or simulation was performed. Tasks 6–9 remain pending.

The subsequent EZ-alias addition passed all 40 deposition tests, including
four Python setup generations and checks that EZ and EmZ generate identical
types and enforce the same restrictions.

## 6. Expose custom PML thickness and coefficients

**Original problem:** [grid.py](lib/python/picongpu/picmi/grid.py) rejects `pml_cells`, and
the Python templates do not expose the custom
[fieldAbsorber.param](tweac5DegreeSetup/include/picongpu/param/fieldAbsorber.param).
Selecting open boundaries alone does not reproduce the absorber.

| Setting | Required value |
| --- | --- |
| Absorber | PML |
| Thickness | 110 cells on every face |
| Sigma/kappa grading order | 2.5 |
| Conductivity | `1.25 * SIGMA_OPT_SI` on each axis |
| `SIGMA_OPT_SI[d]` | `0.8 * (2.5 + 1) / (Z0 * cell_size[d])` |
| `KAPPA_MAX` | `(4, 1, 4)` |
| Alpha grading order | 1.0 |
| `ALPHA_MAX_SI` | `(0, 0, 0)` |

- [x] Define a typed absorber configuration and its ownership by the grid or
  solver. Map standard `pml_cells` where appropriate; use explicit PIConGPU
  extensions for the additional coefficients.
- [x] Add backend representation and `fieldAbsorber.param` generation, retaining
  current defaults when no custom absorber configuration is supplied.
- [x] Ensure the generated runtime configuration selects PML, rather than relying
  silently on a runtime default.
- [x] Validate thickness and coefficient ranges, periodic-boundary behavior and
  compatibility with the laser injection surfaces.
- [x] Render the exact values above and test anisotropic cell sizes. Preserve
  Huygens positions `[[116,-116], [116,100000000], [116,-116]]`; the far YMax
  position is deliberate in this moving-window setup and must not be clamped.

**Acceptance:** Both the generated absorber parameters and runtime selection
match the reference, including the conductivity's dependence on cell size.

**Acceptance result:** Both Cartesian grids now support standard `pml_cells`
(one thickness per axis, applied to both faces) and the PIConGPU extension
`picongpu_pml=picmi.PML(...)` for coefficients. A typed backend and new absorber
template preserve C++ defaults when omitted. The runtime configuration now
explicitly selects `--fieldAbsorber pml`. Python validates finite coefficient
ranges, integer thicknesses, periodic axes, local-domain fit and absorber
clearance of active incident-field faces. It preserves inactive Huygens positions.

All 274 selected tests and 12 subtests passed, including 40 PML tests. After a
final template-footer correction, all 40 PML tests passed again with added
namespace-closure checks. Default and custom setups generated successfully;
direct comparison confirmed all active reference PML constants, both conductivity
formulas and Huygens positions. The custom case retains the active grid,
1000 steps, two TWTS pulses and PQS/EmZ electrons with the step-6720 pusher switch.
Inactive exponential-absorber coefficients retain C++ defaults; that absorber
is outside this PML migration. See [task 6 validation](tweac5DegreeSetup/validation/TASK6.md)
for scope, commands and the remaining runtime checks. No PIConGPU compilation
or simulation was performed. Tasks 7–9 remain pending.

## 7. Expose minimum macroparticle weighting

**Original problem:** The reference [particle.param](tweac5DegreeSetup/include/picongpu/param/particle.param)
sets `MIN_WEIGHTING = 0.1`; the
[generated template](lib/python/picongpu/templates/include/picongpu/param/particle.param.mustache)
hardcodes `1.0`. This changes particle creation in dilute parts of the ramps.

- [x] Add a validated simulation-level PIConGPU option, since the C++ constant
  is global. Preserve the existing default of 1.0.
- [x] Pass the value through the backend and template without changing the
  one-macroparticle-per-cell layout or reference-density normalization.
- [x] Test default/custom rendering and the creation threshold for representative
  low-density cells against the C++ start-position algorithm. Validate the allowed
  range from that algorithm rather than guessing whether zero is legal.

**Acceptance:** The generated setup contains `MIN_WEIGHTING = 0.1` and samples
the density ramps with the intended cutoff.

**Original acceptance result:** Added `Simulation.picongpu_min_weighting` (default `1.0`),
passed through the backend to `particles::MIN_WEIGHTING`. The reference uses
`0.1`. Both models require finite, strictly positive numeric values; zero is
explicitly forbidden by the C++ `WeightMacroParticles` assertion and division.
Backend validation also rejects values that become zero or infinite in the
selected core precision.

**Rebase update:** Current `dev` already provides this option. The rebased version
uses upstream's omitted/`None` default of `10.0`, retains strict numeric and
selected-precision validation, and keeps the reference's explicit `0.1`.
The original acceptance counts below describe the earlier implementation.

All 337 selected Python tests and 12 subtests passed, including 30 new weighting
tests. Default/custom setup generations preserve the active grid, 1000 steps,
one-macroparticle layout and `1e25 m^-3` normalization. Only the weighting
constant differs among their generated parameter headers. Direct comparison
matches the reference value, normalization and typical particle count.
Python checks derived from the C++ algorithm confirm the cutoff at and around
the threshold and in representative low-density ramp cells in both precisions.
See [task 7 validation](tweac5DegreeSetup/validation/TASK7.md) for the source
derivation, reproduction and limits. No PIConGPU compilation or simulation was
performed. Tasks 8–9 remain pending.

## 8. Expose the phase-space output extension

**Original problem:** The active run configuration requests `--e_phaseSpace.ext bp`.
The [PICMI model](lib/python/picongpu/picmi/diagnostics/phase_space.py),
[backend model](lib/python/picongpu/pypicongpu/output/phase_space.py), and
[runtime template](lib/python/picongpu/templates/etc/picongpu/N.cfg.mustache)
currently have no corresponding option. This is an output-format gap, not a
missing phase-space diagnostic.

- [x] Add an optional extension setting and render the matching CLI option.
  Preserve the existing behavior when it is unspecified.
- [x] Check whether the C++ option is shared per species or per diagnostic;
  handle multiple phase-space instances and conflicting values accordingly.
- [x] Test the seven reference phase spaces with `.bp` selected. Their PICMI
  momentum limits must be supplied in SI: multiply the reference `m_e*c` limits
  by `m_e*c` before passing them to the Python frontend.

**Acceptance:** All seven outputs have the requested format, axes, limits and
812-step period, without relying on an incidental plugin default.

**Acceptance result:** Added `PhaseSpace.picongpu_extension`, converted to the
backend's optional `extension` and rendered as `--<species>_phaseSpace.ext`.
Omission/`None` preserves the C++ default. Extensions are per diagnostic, indexed
per species across all filters; different explicit formats are supported.
The CLI cannot express omitted entries before later explicit entries for the
same species, so simulation validation rejects that ordering. Trailing defaults
remain valid. Shared token validation rejects empty strings, leading dots,
whitespace, paths and shell syntax without assuming installed backend support.

All 405 selected Python tests and 3501 subtests passed, including 41 new
extension tests and four existing phase-space tests. Generation preserves the
active grid and 1000 steps. Direct comparison confirms all seven reference
axes, momentum bounds (after SI conversion), `all` filters, explicit `bp`
extensions and the open-ended 812-step schedule. See
[task 8 validation](tweac5DegreeSetup/validation/TASK8.md) for implementation,
ordering examples and reproduction. No PIConGPU compilation or simulation was
performed. Full setup integration remains task 9.

## 9. Generate and audit the complete Python setup

This is integration work using the extensions above, not another general API
redesign. Use the active files rather than `*Old.param` or `binningSetupNew.param`.

Work on **one subtask per request**, in the recommended order 9a, 9b, 9c, 9d.
Each subtask must leave a usable Python definition or increment to it, focused
Python validation, and a short report in `tweac5DegreeSetup/validation/` named
`TASK9A.md`, `TASK9B.md`, `TASK9C.md` or `TASK9D.md`. Update this checklist and
`changes.txt` at each checkpoint. Reports should record the reproduction command,
what was compared, remaining differences and the next subtask, so work can resume
in another session without repeating completed work. Do not mark task 9 complete
until 9d passes; intermediate definitions are explicitly partial migrations.

The following features already have Python support and should first be mapped
and verified, not reimplemented:

| Setup feature | Existing interface / migration detail |
| --- | --- |
| Grid and timestep | `Cartesian3DGrid`; cells `(6.177353e-8, 3.088676e-8, 6.177353e-8) m`, `dt=5.15136e-17 s` |
| Field solver | `ElectromagneticSolver(method="CKC")` |
| Current smoothing | `BinomialSmoother` through `source_smoother` |
| Moving window | `picongpu_moving_window_move_point=0.0` |
| Species | Cold electrons named `e`; ions and temperature initialization are inactive |
| Density | `AnalyticDistribution` with SymPy piecewise expressions; peak `1e25 m^-3` |
| Layout | `PseudoRandomLayout(n_macroparticles_per_cell=1)` |
| Two lasers | `TWTSLaser`, wavelength `0.8 µm`, waist `1 µm`, `a0=1.25`, duration `25 fs / 2.354820045` |
| Laser geometry | Incidence `±5°`, polarization `±45.10716816968629°`, focal y `68.5 µm`, lateral offsets `±12 µm`, `beta0=1` |
| Laser window and surfaces | Existing TWTS window fields and `picongpu_huygens_surface_positions` |
| Memory | `MemoryConfig` and grid supercell/guard settings; preserve `8×8×8` supercells and 4 GiB reserve |
| Field/particle energy | Existing energy diagnostics |
| Field dumps and slices | Native field dumps with openPMD options and ranges |
| Checkpoints | Existing checkpoint settings, including nested openPMD options |
| Electron spectrometer | Binning functors for kinetic energy, `atan2(px,py)`, and deposited charge |

### 9a. Assemble the core setup and exact density profile

**Scope:** A generating Python definition of the simulation and its physics,
using the completed extensions. Reference diagnostics are added in 9b and 9c.

- [x] Create the Python definition with an explicit generation entry point that
  calls `write_input_file(...)`, not `run()` or `step()`. Keep imports free of
  build, submission and simulation side effects. Default to 1000 steps and
  preserve the active grid and device decomposition. Generate into a separate
  output directory without overwriting the reference.
- [x] Assemble CKC, binomial smoothing, moving window, cold electrons, the
  one-macroparticle layout, PQS shape/interpolation, EmZ deposition, the step-6720
  composite pusher, reference PML and `MIN_WEIGHTING=0.1` using tasks 0–7.
- [x] Translate the density function exactly, including the transverse Gaussian
  ramps, hard vacuum cuts, and 100 µm longitudinal linear upramp after
  `6720 * dy = 207.5590272 µm`. Do not activate the commented density spike,
  drift or temperature. Keep the reference transverse dimensions rather than
  silently rescaling the profile; acceptance uses the confirmed reference grid.
- [x] Configure both TWTS pulses with the reference parameters, time offsets,
  polarization, switching windows and Huygens positions. Verify both lasers
  occur on YMin, the positive-angle laser on ZMin, and the negative-angle laser
  on ZMax, with no additional incident faces enabled. Select incident-field
  injection and component-wise evaluation explicitly.
- [x] Map the electron exchange buffers to `MemoryConfig`: 32/320/32 MiB for
  x/y/z and 2 MiB for edges/corners; preserve 4 GiB GPU reserve and 8×8×8
  supercells. Since electrons are the only active species, global exchange
  settings can reproduce the existing electron-specific values; a general
  per-species memory API is not needed for this setup.
- [x] Generate input with Python and compare the core parameters with the
  reference. Check representative density values inside/outside each hard cut,
  at ramp/plateau transitions, in combined transverse ramps and along the
  longitudinal upramp, including coordinates encountered after window slides.
  Carry forward task 7's low-density creation-cutoff checks.

**Acceptance:** The core script generates successfully on the active grid and
matches the reference physics and memory settings. Density and laser comparisons
are recorded in `TASK9A.md`. The report clearly lists diagnostic work still
pending; default or placeholder outputs do not count as migrated diagnostics.

**Acceptance result:** Added [picmi_setup.py](tweac5DegreeSetup/picmi_setup.py)
and [generation instructions](tweac5DegreeSetup/README_PICMI.md). The script
defaults to 1000 steps and generates outside the reference directory. Core
parameter comparisons and 1352 density samples pass, including exact hard-cut
boundaries and later moving-window regions. Both TWTS pulses, incident faces,
cold-electron initialization, PML and memory settings are preserved. One existing
renderer defect was fixed: large memory expressions now promote the first factor
to `ULL`, avoiding signed-int overflow for the 4 GiB reserve.

All 268 selected Python tests passed, and the CLI generated input separately.
No PIConGPU compilation or simulation occurred. See
[TASK9A.md](tweac5DegreeSetup/validation/TASK9A.md) for comparisons and remaining
differences, including the density expression's intermediate floating-point
rounding and runtime options reserved for 9d. Analytic density agreement is
validated, not bitwise device evaluation. At completion of 9a, the definition
had no active reference diagnostics; 9b adds them below. Final acceptance is recorded in 9d.

### 9b. Add energy, phase-space, field and checkpoint outputs

**Scope:** All active outputs from `TBG_plugins` and the checkpoint configuration.
The separate Binning `eSpec` diagnostic belongs to 9c.

- [x] Add field/electron energy and the seven PhaseSpace-plugin outputs every
  812 steps. Reuse task 8's `picongpu_extension="bp"`; supply momentum limits
  in SI and retain all coordinate pairs, ranges and `all` filters.
- [x] Add full field dumps with schedule `0:1000:8120`, x/y/z field slices with
  schedule `0:1000:812`, and checkpoints with schedule `0:1000:50000` at the
  default run length. If a step-count override is provided, use that count for
  the schedule endpoints, preserving the intervals and step-zero behavior.
- [x] Preserve slice ranges `1780:1804,:,:`, `:,1780:1804,:`, and `:,:,884:908`,
  output names, `.bp` extensions, `_%T` infixes, mapped-memory preparation and
  the active compression configuration. If a backend configuration file is
  necessary, have the Python definition generate it reproducibly.
- [x] Do not enable unused PNG, histogram or pinhole-filter outputs just because
  definitions for them exist in `config.cfg`. Compare the effective enabled
  output list, including any implicit Python defaults, against the reference.
- [x] Generate the updated setup and compare effective schedules, filters,
  ranges, filenames and backend configuration with the active reference.
  Record any interface gap explicitly rather than changing output semantics.

**Acceptance:** All outputs assigned to 9b match the reference configuration,
including schedules whose intervals exceed 1000 steps. Record the comparison
in `TASK9B.md`; writing actual diagnostic files in a simulation is not required
or permitted by this acceptance workflow.

**Completed:** The Python definition now includes all outputs assigned to 9b.
`fields_all` was resolved against the inherited default: E, B and electron
charge/energy density. New built-in selectors reuse the C++ density implementations;
inline backend dictionaries preserve the active compression. Two existing range
bugs were fixed: accepting `RangeSpec` instances and retaining unrestricted axes
as `:` instead of empty strings (which the C++ parser skips).

All 131 selected Python tests and 3489 subtests passed, including 18 migration
acceptance tests. Outputs were compared at 1000, 230000 and zero steps; the CLI
also generated input. No PIConGPU compilation or simulation occurred. See
[TASK9B.md](tweac5DegreeSetup/validation/TASK9B.md). Task 9c adds the Binning
spectrometer below; the completed final audit is recorded in 9d.

### 9c. Add the Binning electron spectrometer

**Scope:** The active `eSpec` definition in `binningSetup.param`, using the
existing PICMI Binning interface. This is distinct from the PhaseSpace plugin
covered by task 8 and subtask 9b.

- [x] Reproduce `eSpec`: 1600 energy bins from 0 to 1000 MeV, 256 pointing bins
  from -0.1 to +0.1 rad, deposited charge, overflow-bin behavior, and its own
  openPMD backend settings. Retain its name and 500-step period.
- [x] Map the reference kinetic-energy, `atan2(px,py)` pointing and deposited
  charge functors. Check units, charge sign, axis ordering, bin-edge conventions,
  normalization and macroparticle weighting explicitly; avoid counting a
  particle's weight twice or silently treating momentum as velocity.
- [x] Evaluate representative Python cases against the reference formulas:
  zero and nonzero momentum, pointing signs/quadrants, different weights, and
  values inside, at and outside the bin limits. Inspect generated functor and
  bin configuration code without compiling it.
- [x] Generate the setup with `eSpec` enabled and compare its registration,
  schedule, axes, accumulation settings and openPMD configuration with the
  reference. Generate any necessary configuration files reproducibly.

**Acceptance:** The generated Binning definition and source-based Python checks
match the active reference `eSpec`, independently of PhaseSpace. Record results
and the limits of Python-only validation in `TASK9C.md`.

**Completed:** Added `electron_spectrometer()` using the existing Binning and
ParticleFunctor interfaces, with no library or template changes. The definition
preserves energy/pointing axes, units, weighted signed charge, overflow behavior,
500-step notifications and separate bp output. After rebasing, the ineffective
`adios2.dataset.chunks="auto"` key is omitted: openPMD ignores it for ADIOS2,
and upstream's typed backend model rejects it. Effective ADIOS2 defaults remain
unchanged; see `resolvedConflicts.md` for source evidence.
Source checks establish that dumpPeriod 0 (reference default) and 1 (PICMI) both
write and reset at every notification, without bin-width or time normalization.

All **59 migration acceptance tests passed**, including 41 new Binning checks.
The CLI also generated the default 1000-step setup with all diagnostics enabled.
No network access, installation, PIConGPU compilation or simulation occurred.
See [TASK9C.md](tweac5DegreeSetup/validation/TASK9C.md) for source mappings and the
limits of Python-only validation. The completed final audit is recorded in **9d** below.

### 9d. Audit and deliver the complete migration

**Scope:** Integrate and audit the already implemented parts, resolve material
differences, and document the final generation workflow.

- [x] Generate the complete setup from the final Python definition, with all
  9a–9c settings enabled together. Verify that imports are side-effect free and
  generation does not depend on the old setup's `.build` directory or binaries.
- [x] Maintain a comparison table covering the active reference parameter files
  and effective runtime options. Include core physics, density, lasers, memory,
  every active diagnostic, checkpoint settings and generated backend files.
  Resolve C++ aliases and normalize harmless formatting for comparison rather
  than requiring byte-for-byte identity.
- [x] Audit remaining runtime differences, including `--autoAdjustGrid false`,
  `--percent 1`, and field temporary storage. Classify each as already expressible,
  an operational difference with no effect for this configuration, or an actual
  remaining gap. Split any confirmed gap into a new focused task; do not claim
  full migration while a material gap remains unresolved or unagreed.
- [x] Run the relevant Python regression checks and the combined generation/
  reference comparison. Recheck earlier results where integration changes
  affect them; reuse otherwise valid acceptance evidence from 9a–9c.
- [x] Document the final script, its 1000-step default, generated artifacts and
  exact Python-only generation requirements/commands using `uv`. Record all
  remaining differences and unrun checks in `TASK9D.md`. Do not invoke PIConGPU
  compilation, job submission or simulation execution.

**Acceptance for 9d and task 9:** A reviewable PICMI Python definition generates the intended
setup without handwritten C++ template overrides, with all material differences
resolved or explicitly agreed. Acceptance establishes agreement of generated
configuration with the reference; compilation and physical simulation validation
are outside the scope of this work.

**Superseded acceptance result (see correction below):** Core parameter settings and outputs are represented
in [picmi_setup.py](tweac5DegreeSetup/picmi_setup.py). Added optional typed runtime
controls for automatic grid adjustment and progress percentage; the definition
emits the reference's `--autoAdjustGrid false` and `--percent 1`. The original grid
and single temporary-field slot remain valid with all diagnostics enabled.

All **471 selected Python tests and 3497 subtests passed**, including 64 migration
acceptance checks and 27 runtime-control regressions. The CLI independently
created `/tmp/tweac-picmi-final`. Isolation checks forbid subprocess execution
and simulation entry points during generation and guard against reads/scans of
the reference `.build` and `bin` directories. No compilation or simulation ran.

[TASK9D.md](tweac5DegreeSetup/validation/TASK9D.md) contains the complete comparison
table, reproducible commands, and remaining numerical/representation differences.
The density's mixed-precision rounding caveat does not affect the active 1000-step
vacuum run; longer plasma runs have analytic agreement without a bitwise device
claim. The subsequent discovery below invalidated the earlier completion claim.
Tasks **9d and 9 were reopened**, and are now completed by the task 9e acceptance below.

### 9e. Preserve TWTS internal precision and envelope cutoff

The reference's setup-local `fields/background/templates/twtstight/TWTSTight.hpp`
changes `float_T` from `float_X` to `float_64` and `numSigmas` from 6 to 8.
The original generated setup did not carry this override. Core/special-operation
precision settings alone do not reproduce it. The cutoff changes the region where
the laser is evaluated, so this gap also applied to the active vacuum run.

- [x] Expand the reference inventory to every file under `include/`, not only
  `param/*.param`; compare implementation-header overrides against the repository.
- [x] Expose TWTS-only compile-time numerical configuration through the existing
  `precision.param`, consumed by the existing implementation. Keep legacy
  defaults (`float_X`, 6), with explicit 64-bit/8 selection for this setup.
  Prefer one shared TWTS configuration while its scalar aliases are global;
  per-laser precision would require parameterizing EField/BField and helpers.
- [x] Add typed Python configuration and render it into `precision.param`,
  without copying or rewriting the full TWTSTight implementation header and
  without changing Gaussian/dispersive lasers or the global simulation precision.
- [x] Validate generation, defaults, both pulses, componentwise injection and
  effective include/configuration resolution with Python/source-based tests.
  Re-run the integration audit and correct completion status only afterwards.

Whether newer numerical algorithms permit dropping 64-bit arithmetic or reducing
the cutoff is a separate accuracy decision. Preserve 64-bit/8 unless a change is
explicitly agreed; passing generation tests cannot establish numerical redundancy.
No C++ compilation or simulation is authorized by this follow-up task inventory.

**Acceptance result:** `picmi.PrecisionConfig(twts=64, twts_num_sigmas=8)` now
preserves the reference TWTS settings through the existing `precision.param`.
Core precision remains float32. Guarded defaults keep old custom precision files
compatible with core arithmetic/cutoff 6. `TWTSPulse.def` uses the shared settings
without modification; no additional parameter file or copied implementation is
needed. Both incident and background TWTS consumers share this configuration.

The expanded inventory accounts for all 18 reference include files. The normalized
full TWTS header matches after resolving the two settings. All **510 selected
Python tests and 3497 subtests passed**, including 66 migration checks. The CLI
generated `/tmp/tweac-picmi-final` at 1000 steps. No C++ compilation or simulation
ran. [TASK9E.md](tweac5DegreeSetup/validation/TASK9E.md) records configuration,
compatibility, generation evidence and limits. Tasks **9e, 9d and 9 are complete**
for the agreed Python-generation scope.
