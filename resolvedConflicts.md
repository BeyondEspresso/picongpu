# Rebase conflict resolutions

Rebased `topic-create-TWEAC-picmi-script-rebased` onto the local `dev` branch
at `b8f83fcfa`. Original tip: `05fc94955`; original common ancestor: `5b41865b0`.
The original branch `topic-create-TWEAC-picmi-script` is retained unchanged.
No fetch, push, squash, PIConGPU compilation, simulation or job submission is
part of this operation. The 11 original task commits are replayed individually,
with compatibility fixes made in the commit that introduces the affected feature.

The untracked `hintsPMACCComplexBessel.md`, `tweac5DegreeSetup/.build/` and
`tweac5DegreeSetup/bin/` are outside this work and are not staged or used.
Historical task reports retain their original test counts; the results below
record validation against the updated development branch.

## Per-commit decisions

### 1. `163451506` — First step at making TWEAC-setup PICMI compliant

Conflict: `lib/python/picongpu/picmi/lasers/twts_laser.py` imports.
Kept upstream's `PICMI_Laser` base class and `resolve_once` validator decorator,
and added NumPy for the TWTS-local direction checks. Retained positive-centroid
support, TWTS-specific delay/pulse-init semantics and transverse polarization.
Gaussian and dispersive laser implementations are unchanged by this commit.
Validation: TWTS and Gaussian tests, **62 passed**.

### 2. `a2f93efde` — Implement composite pushers in PICMI

Conflict: `lib/python/picongpu/picmi/species.py`, documentation and defaults.
Kept upstream's `particle_shape=None`, `method=None`, `DEFAULT_PARTICLE_SHAPE`
and simulation-level shape inheritance. Added the composite selection and its
explicit-method conflict checks. Removed `_composite_method_default`: upstream
now already leaves the method unset, so suppressing an implicit Boris constructor
default is unnecessary. Ordinary species still resolve to Boris at conversion.
Retained the HigueraCary C++ spelling fix and rejection of the nonexistent Axel
pusher. No upstream species-default behavior is replaced.
Updated the mutation regression for the new PICMI base's atomic assignment
rollback; an invalid `model_copy(update=...)` still fails conversion.
Validation: composite pushers, species and simulation, **161 passed and 24
subtests passed**.

### 3. `cde04f4c3` — Keep field interpolation consistent with species shape

Conflicts: `picmi/species.py` and `species_distributions_layouts.rst`.
Combined the interpolation/counter-shape rules with upstream's simulation-level
shape inheritance and retained the new MultiSpecies documentation. Updated the
new interpolation documentation so omission inherits the Simulation shape
instead of unconditionally selecting TSC. Extended the mixed-species generation
regression to cover both the default TSC and inherited PQS, while retaining an
explicit CIC override and the composite pusher.
Updated upstream's inherited-Counter case to expect rejection at conversion,
consistent with the existing C++ interpolation restriction; ordinary simulation
shape inheritance remains supported.
Validation: interpolation, species and composite tests, **95 passed and 23
subtests passed**.

### 4. `78d3b448e` — Expose EmZ current deposition with the selected shape

Conflicts: `picmi/species.py` (imports/defaults) and species documentation.
Kept both `ClassVar` and `Literal`, retained upstream's unset species defaults
and shape resolution, and added the deposition choice once. Interpolation and
deposition use the same resolved shape. Updated the documentation accordingly.
Extended generation checks to compare inherited PQS as well as the default TSC,
with explicit per-species overrides, EmZ/EZ and smoothing choices retained.
Validation: deposition, interpolation and species tests, **87 passed and 23
subtests passed**.

### 5. `4af4af7e1` — Expose custom PML thickness and coefficients

Conflicts: `picmi/__init__.py` exports and `pypicongpu/simulation.py` imports.
Kept upstream's MultiSpecies export, background-field and minimum-weighting
support, and added PML/model_validator alongside them. The PML feature remains
needed; no equivalent implementation is present in this dev snapshot.
Validation: PML, grid and simulation tests, **144 passed**.

### 6. `08de23047` — Expose minimum macroparticle weighting

Conflicts: `picmi/simulation.py`, `pypicongpu/simulation.py`, and
`simulation_settings.rst`. Upstream commit `0842668fa` already implements the
public option, backend conversion and particle template. Reused that implementation
and its `None -> 10.0` fallback, which matches the native C++ default. Removed the
second field declaration and duplicated conversion keyword introduced by the
otherwise automatic merge. No duplicate minimum-weighting feature remains.

Kept strict numeric input checks (reject booleans/strings) and the additional
float32 underflow/overflow checks. Retained upstream validation messages and
explicit-None behavior. Updated public docs and tests for the upstream default,
including explicit None and cutoff-boundary/ramp comparisons at 10.0. The migrated
setup still explicitly selects 0.1, so its effective setting does not change.
Validation: minimum weighting and simulation tests, **111 passed**.

### 7. `79b1cf5ff` — Expose the phase-space output extension

Applied without textual conflicts. Kept the extension support and per-species
consistency checks alongside upstream's typed diagnostic base and shape-aware
conversion. This feature is not duplicated by the newer openPMD backend model:
PhaseSpace still uses its own plugin CLI options.
Validation: PhaseSpace and extension tests, **45 passed**.

### 8. `115e5a490` — Assemble the core setup and exact density profile

Applied without textual conflicts. The explicit grid, shape, composite pusher,
EmZ, PML and minimum weighting continue to select the reference settings. Checked
core generation and the updated upstream analytic-distribution implementation
together; no replacement density implementation is introduced.
Validation: core generation and analytic-distribution tests, **45 passed and
20 subtests passed**.

### 9. `2fd25fb6c` — Add energy, phase-space, field and checkpoint outputs

Conflicts: openPMD documentation, `picmi/simulation.py` plugin conversion, and
`pypicongpu/output/openpmd_plugin.py` backend-config type.

Reused upstream `OpenPMDBackendConfig` from `6e1f1927d`, including nested typed
validation, dict input, omission of empty configuration and the full schema docs.
Dropped our broader untyped dict/string/path annotation. The setup's compression
dictionary is accepted by the upstream model and still emits the intended TOML.
Updated the round-trip test to cover typed-model and dict inputs, rather than
retaining an obsolete raw-string interface.

Retained the independent RangeSpec fixes (`:` preserves unrestricted axes, and
preconstructed RangeSpec instances are accepted), built-in charge/energy density
selectors and deduplication. Combined their plugin conversion with upstream's
`default_particle_shape` forwarding so particle dumps keep inherited shapes.
Python generation exposed an additional semantic conflict: upstream now registers
attributes requested by diagnostic functors, but our built-in selectors are strings.
Skip only those selectors when asking for functor requirements, while still
registering their enclosing particle filters. Strengthened the built-in-field
regression with a filter that requires `momentumPrev1`, ensuring its extra species
attribute is generated rather than accidentally skipped.
Validation: core/output acceptance, all diagnostic tests (including upstream
backend schema tests) and species, **164 passed and 3512 subtests passed**.

### 10. `a352391e4` — Add the Binning electron spectrometer

Applied without textual conflicts, but construction failed against the new typed
backend model: the reference contains `adios2.dataset.chunks="auto"`, which is not
an ADIOS2 option. Kept upstream's schema validation and omitted this ineffective
key from the Python setup. No replacement HDF5 configuration or engine selection
was introduced; `.bp`, periods, bin axes, weighting, charge and effective ADIOS2
defaults are preserved.

Evidence: [openPMD-api 0.17.1 backend documentation](https://openpmd-api.readthedocs.io/en/0.17.1/details/backendconfig.html)
documents `chunks` under `hdf5.dataset` and says unknown keys are ignored with a
warning. The matching [ADIOS2 source](https://github.com/openPMD/openPMD-api/blob/0.17.1/src/IO/ADIOS/ADIOS2IOHandler.cpp#L321)
returns no dataset operators when `operators` is absent; `getDatasetOperators`
retains the defaults and warns about unused configuration. There is no parser for
a `chunks` configuration key in that handler. This is an effective-configuration
comparison, not a byte-for-byte preservation of the old inert JSON.

Updated the Binning acceptance comparison and README/TODO to explicitly record
this difference. The source reference itself is unchanged.
Validation: all three migration suites available at this step plus upstream
openPMD backend configuration tests, **95 passed**.

### 11. `05fc94955` — Audit and deliver the complete migration

Conflicts: `simulation_settings.rst` and `picmi/simulation.py`.
Retained both the upstream particle-shape documentation and the TWTS numerical
controls. Added the runtime grid-adjustment/progress controls without restoring
`picongpu_distributions` or `_DensityImpl`: upstream has replaced those with its
current declarative/imperative density-registration mechanism, including
MultiSpecies support. Preserved the `precision.param` implementation and the
reference's explicit float32 core / float64 TWTS / cutoff 8 choices.

Updated the historical Binning/audit reports to identify the ineffective backend
key omitted on rebase. The expanded reference inventory and full TWTS header
comparison continue to pass.

## Final validation

The combined suite passed **889 tests and 3549 subtests** in 116.64 seconds. It
includes all migration acceptance tests, PICMI tests for the affected and newer
upstream interfaces, and backend memory/precision tests. Workflow and RO-Crate
validation tests were not included; those subsystems were not changed. No test
compiled PIConGPU, launched a simulation, submitted a job or used the reference
build/binaries. The migration's isolation checks still prohibit subprocess and
simulation execution during generation.

The temporary Python environment was recreated using `uv`, with an editable
install of this checkout's `lib/python[test]`. The upstream pin is
`picmistandard` commit `952708894e31b35075fd876a970827c45e96747f`
(`0.35.0.dev0`); tests used Python 3.13.5, Pydantic 2.14.1 and pytest 9.1.1.
Only Python dependencies were installed, not a PIConGPU compiler toolchain.

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  tweac5DegreeSetup/validation \
  lib/python/test/picongpu/quick/picmi \
  lib/python/test/picongpu/quick/pypicongpu/test_memory_config.py \
  --ignore=lib/python/test/picongpu/quick/picmi/test_workflow.py \
  --ignore=lib/python/test/picongpu/quick/picmi/test_rocrate.py \
  --tb=short
```

The standalone CLI also generated `/tmp/tweac-picmi-rebased-20261014` with the
1000-step default, original grid/devices, componentwise incident-field flags,
all active diagnostics and TWTS precision/cutoff settings. Generation emits the
existing Mustache warning about `grid.sim_dim`; the generated grid is verified
by the acceptance comparisons.

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python tweac5DegreeSetup/picmi_setup.py \
  /tmp/tweac-picmi-rebased-20261014
```

Use a new output directory when repeating the CLI command. Numerical/device
validation remains outside the agreed Python-generation scope; the longer-run
density-rounding qualifications in TASK9D.md still apply.

## Preserved commit mapping

| Original | Rebased | Task |
| --- | --- | --- |
| `163451506` | `3f403dea8` | First step at making TWEAC-setup PICMI compliant |
| `a2f93efde` | `fd4361c3b` | Implement composite pushers in PICMI |
| `cde04f4c3` | `daed4a2f8` | Keep field interpolation consistent with species shape |
| `78d3b448e` | `b4c702ccb` | Expose EmZ current deposition with the selected shape |
| `4af4af7e1` | `098bcc75e` | Expose custom PML thickness and coefficients |
| `08de23047` | `83ac52118` | Expose minimum macroparticle weighting |
| `79b1cf5ff` | `1e4dfe665` | Expose the phase-space output extension |
| `115e5a490` | `f2b059c3e` | Assemble the core setup and exact density profile |
| `2fd25fb6c` | `6e4447c96` | Add energy, phase-space, field and checkpoint outputs |
| `a352391e4` | `7e4791668` | Add the Binning electron spectrometer |
| `05fc94955` | `c312e02c8` | Audit and deliver the complete migration |

Verified that `dev` is an ancestor, all 11 task subjects remain in their
original order, no task commits were combined or dropped, and the original
branch still points to `05fc94955`. Reviewed `git range-diff` against the original
series. The setup reference include tree is byte-for-byte unchanged; its existing
whitespace issues are deliberately preserved. This report is recorded in a
separate documentation commit after the task series. The same documentation
commit places the task-7 rebase note before its historical acceptance results.
