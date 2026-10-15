# Specification: rebundle the TWEAC PICMI work for upstream PRs

## Assignment and deliverables

Implement this specification using the existing implementation as the source.
This is a history and patch-organization task, with the adaptations needed to
make each topic independently reviewable and testable. Do not redesign the
migration or change its physics settings.

Deliver:

1. A new production-prototype branch, `topic/tweac-picmi-production`, containing
   the same functionality as the source branch. Put 12 ordered topical commits
   above the pinned local `dev`, then the setup-specific commits above them.
2. Twelve new upstream-candidate branches, named below. **Eleven have exactly
   one commit, whose sole parent is the pinned local `dev` commit. PR-04 has
   exactly two commits: the exact PR-03 commit, then an EmZ-only topic commit.**
   Each topic commit includes its implementation, tests, user documentation and
   a root `PR.md`. Only PR-04 is stacked on another PR branch.
3. A root `PR.md` on the production branch describing the complete prototype,
   plus an archive of the 12 individual descriptions and a rebundling report.

Create and validate local branches only. Do not push, create remote PRs, compile
PIConGPU, execute a simulation, or submit jobs. No full toolchain installation.
Python dependency installation with `uv`, Python tests and setup generation are
within scope. Do not delegate work to additional agents.

## Fixed inputs and preservation

Specification prepared on 2026-10-15 in `/home/debus/src/picongpu`.

| Role | Ref | Pinned commit |
| --- | --- | --- |
| Source implementation | `topic-create-TWEAC-picmi-script-rebased` | `74b6a2089848c948dda9142b08246d82f453af34` |
| Local upstream base | `dev` | `b8f83fcfa07aa544ce7f3e1206adc797fa4a6b0b` |
| Earlier implementation, historical only | `topic-create-TWEAC-picmi-script` | `05fc94955645cf5a9ce9ac86e00fea28fb688e29` |

Verify these inputs before starting. If refs have moved, inspect the difference
and report it before deciding whether this specification still applies; do not
silently substitute a new base or discard newer work. Do not fetch to update
`dev`. Keep all three existing branches unchanged.

Read applicable repository instructions. Use separate worktrees, preferably
under `/tmp`, so the original checkout and untracked files remain undisturbed.
Never overwrite an existing destination branch without examining whether it
already contains progress. Resume completed work rather than recreating it.

Preserve and do not use these existing untracked artifacts:

- `hintsPMACCComplexBessel.md`
- `tweac5DegreeSetup/.build/`
- `tweac5DegreeSetup/bin/`

Do not copy them into any branch. Do not run reference binaries or build helpers.
Use explicit staging paths; avoid staging the entire original working folder.

## Source commits and extraction rules

The source series is a map of where to find changes, not a ready-made PR series:

| Source commit | Contents relevant to extraction |
| --- | --- |
| `3f403dea8` | TWTS fixes and the large original reference setup |
| `fd4361c3b` | Composite pusher and pusher-name corrections |
| `daed4a2f8` | Species-shape interpolation consistency |
| `b4c702ccb` | Shape-dependent deposition and EmZ/EZ selection |
| `098bcc75e` | PML configuration |
| `83ac52118` | Remaining minimum-weighting validation improvements |
| `1e4dfe665` | PhaseSpace filename extensions |
| `f2b059c3e` | Core Python setup and independent memory-literal fix |
| `6e4447c96` | Setup outputs, openPMD ranges and built-in density outputs |
| `7e4791668` | Setup-specific Binning electron spectrometer |
| `c312e02c8` | Runtime controls, shared TWTS precision/cutoff and final audit |
| `74b6a2089` | Rebase report and historical documentation placement |

Extract changes from the final source tree and the individual diffs. Do not
replace shared files wholesale with their final source versions on a standalone
PR branch: `simulation.py`, `species.py`, templates and tests contain several
independent features. Move matching tests and documentation with each feature.

Keep upstream behavior introduced by the rebase: inherited particle shapes,
MultiSpecies, current density registration, atomic validated assignments and
typed openPMD backend configuration. Do not reintroduce superseded code.

Each PR branch except PR-04 has one commit, even where its subject contains
multiple related fixes. In particular, the pusher-name fixes belong in that
single PR-02 commit;
the earlier suggestion of multiple commits inside that PR is superseded here.

## The 12 upstream branches

The numbering also defines the topical order on the production branch.
Every item includes focused Python conversion/rendering tests and relevant
public documentation. The examples of tests below are required behaviors,
not a demand for tests that merely duplicate implementation details.

### PR-01 — Fix TWTS validation and polarization metadata

- Branch: `pr/picmi-twts-validation`
- Source: library, tests and laser documentation in `3f403dea8`.
- Include the TWTS-local common-validator override, positive centroid support,
  unused `pulse_init` compatibility value, time-delay semantics and corrected
  transverse polarization metadata.
- Preserve the incoming-direction restrictions and normalization checks.
  Do not modify Gaussian/dispersive validation or add the numerical controls
  from PR-12.
- Verify positive/negative centroid cases, polarization/incidence angles,
  generated TWTS delay/window behavior, and unaffected Gaussian behavior.
- Start from `test_twts_laser.py` and the existing Gaussian tests. Ensure
  extracted tests do not depend on later precision or setup features.

### PR-02 — Support composite particle pushers

- Branch: `pr/picmi-composite-pushers`
- Source: `fd4361c3b`.
- Include the public and backend composite models, validated absolute switching
  timestep, C++ rendering and `method` conflict checks. Keep ordinary omitted
  methods resolving to Boris and upstream shape inheritance unchanged.
- Include the HigueraCary C++ spelling correction and removal of nonexistent
  Axel support. Explain both in the PR description.
- Verify two distinct supported pushers, timestep boundaries, conflicting
  selections, assignment rollback/invalid copied models and ordinary pushers.
- Use `test_composite_pusher.py` and affected species tests. Do not include
  shape-consistency or deposition-selection changes merely because they share
  `speciesDefinition.param.mustache`.

### PR-03 — Keep species shape consistent in interpolation and deposition

- Branch: `pr/picmi-species-shape-consistency`
- Source: `daed4a2f8` and the shape-consistency portion of `b4c702ccb`.
- Introduce the per-species shape alias and use the resolved shape for the
  particle, field interpolation and **existing Esirkepov deposition**. Reject
  the diagnostic-only Counter shape where species interpolation is required.
- Preserve simulation-level inheritance and explicit species overrides.
- Do not add an EmZ/EZ selection API or composite pushers in this branch.
  Hard-code Esirkepov in the appropriate generated type at this stage.
- Verify default TSC, inherited PQS, explicit CIC, mixed species, valid NGP
  with Esirkepov, and Counter rejection. Cover generated interpolation and
  deposition types together.
- Adapt `test_species_interpolation.py` to ordinary pushers and extract suitable
  Esirkepov tests from `test_current_deposition.py`.

### PR-04 — Expose EmZ/EZ current deposition

- Branch: `pr/picmi-emz-current-deposition`
- Source: `b4c702ccb`, with the PR-03 prerequisite described below.
- Include the public choice, backend enum, EZ alias, per-species rendering,
  default Esirkepov behavior and rejection of EmZ/EZ with NGP.
- **Create this branch from the exact PR-03 branch tip**, preserving its commit
  identity, then add one commit containing only the deposition-selection topic
  and its tests/documentation. Do not squash, duplicate or recreate the PR-03
  prerequisite. The final history is `dev -> PR-03 commit -> PR-04 commit`.
- The second commit replaces the inherited root `PR.md` with PR-04's description.
  Identify the dependency and explain that the EmZ-only review diff is against
  PR-03. The complete two-commit branch must pass its tests independently of
  any other checkout.
- During later review, if PR-03 is amended, rebase only PR-04's topic commit onto
  the updated PR-03 tip. Once PR-03 is merged, replay only that topic commit onto
  updated `dev`, accounting for any squash/rebase merge that changed the
  prerequisite's commit identity. Do not perform future review rebases during
  this assignment.
- On the production branch, PR-04 adds only the deposition-selection delta;
  PR-03's implementation already exists there and must not be duplicated.
- Adapt `test_current_deposition.py` to ordinary pushers. Retain inherited
  shapes, independent per-species choices, alias equivalence, smoothing cases
  and generated-code checks. No dependency on PR-02 or the TWEAC setup.

### PR-05 — Expose PML thickness and coefficients

- Branch: `pr/picmi-pml-configuration`
- Source: `098bcc75e`.
- Include standard `pml_cells` support, the PIConGPU coefficient model, backend
  model, absorber template, explicit runtime absorber selection, 2D/3D
  conversion, local-domain checks and incident-field clearance validation.
- Preserve existing defaults and periodic-axis behavior. Keep the current
  intentional handling of inactive Huygens faces.
- Verify conversion, coefficients, periodic/absent axes, thickness limits,
  solver margins, injection and generated parameters with `test_pml.py` and
  existing grid/simulation tests.
- If a fixture uses TWTS, choose parameters accepted by base `dev`; do not
  introduce a hidden dependency on PR-01's positive-centroid fix.

### PR-06 — Strengthen minimum-weighting validation

- Branch: `pr/picmi-min-weighting-validation`
- Source: the remaining diff in `83ac52118`.
- Reuse upstream's option and its omitted/None default of 10.0. Add strict
  numeric validation and the selected-float32 underflow/overflow checks.
- Do not present minimum-weighting configuration as a new feature, duplicate
  declarations/templates, or restore the older default of 1.0.
- Extract `test_min_weighting.py`, adapting any fixtures as needed. Verify
  None/default behavior, explicit valid values, bool/string rejection, finite
  positive values and 32/64-bit representability boundaries.

### PR-07 — Expose PhaseSpace filename extensions

- Branch: `pr/picmi-phase-space-extension`
- Source: `1e4dfe665`.
- Include the PICMI/backend option, validated bare suffix, CLI rendering and
  per-species ordering check that prevents omitted options shifting indices.
- Preserve default omission and filtered-species behavior. This extends the
  existing PhaseSpace plugin, not Binning or general openPMD backend schemas.
- Use `test_phase_space_extensions.py` and existing PhaseSpace tests.

### PR-08 — Fix openPMD spatial-range serialization

- Branch: `pr/picmi-openpmd-range`
- Source: the RangeSpec portion of `6e4447c96`.
- Render an unrestricted axis as `:` rather than an empty entry, and accept
  preconstructed `RangeSpec` instances without wrapping them incorrectly.
- Extract range tests from `diagnostics/test_field_dump.py`. Demonstrate that
  restrictions on later axes retain their original axis positions and that
  list/model forms agree.
- Do not include built-in density selectors, setup output choices or changes
  to upstream's typed backend configuration.

### PR-09 — Expose built-in charge- and energy-density outputs

- Branch: `pr/picmi-builtin-density-output`
- Source: the built-in-field portion of `6e4447c96`.
- Include the charge/energy selectors, backend representation, template
  definitions, output naming, deduplication and conversion changes.
- Preserve upstream shape forwarding and functor requirements. Strings do not
  request functor attributes, but enclosing particle filters still do; retain
  the regression involving `momentumPrev1`.
- Extract relevant tests from `diagnostics/test_field_dump.py`; cover multiple
  outputs and existing ordinary functors. Avoid restricted ranges needing
  PR-08 in this independent branch.
- Treat selectors as PIConGPU extensions; do not claim new PICMI-standard names.

### PR-10 — Prevent overflow in generated memory-size expressions

- Branch: `pr/picmi-memory-literal-overflow`
- Source: `pypicongpu/memory.py` and matching tests from `f2b059c3e`.
- Promote large integer literals before multiplication so a size_t target does
  not receive an already-overflowed int expression. Preserve small expressions.
- Include byte/KiB/MiB cases around the relevant boundary and the existing
  4-GiB case from `test_memory_config.py`.
- Do not include the core setup or TWTS precision tests sharing that test file.

### PR-11 — Expose runtime grid-adjustment and progress controls

- Branch: `pr/picmi-runtime-controls`
- Source: runtime-control portions of `c312e02c8`.
- Include `picongpu_auto_adjust_grid` and `picongpu_progress_percent`, backend
  conversion, validators, N.cfg rendering and public documentation.
- Preserve None as omission, explicit False as an emitted option, and strict
  progress bounds. Use `test_runtime_options.py`.
- Do not accidentally include PML's absorber option from the shared template,
  TWTS precision, or the setup's explicit runtime choices.

### PR-12 — Configure TWTS arithmetic precision and envelope cutoff

- Branch: `pr/twts-precision-controls`
- Source: numerical-control portions of `c312e02c8`.
- Include the shared controls in existing `precision.param`, compatibility
  defaults and checks in the library TWTSTight header, PICMI/backend precision
  models, rendering and documentation. Do not introduce another param file.
- Preserve defaults: core arithmetic and cutoff 6. Allow independent 32/64/core
  arithmetic and positive uint32 cutoff, shared by incident/background TWTS
  pulses. Field storage and switching arithmetic remain at core precision.
- Extract precision tests from `test_memory_config.py`; add or extract focused
  generated-header checks from the audit without depending on the setup.
  Base-compatible TWTS fixtures must work without PR-01.
- State clearly that this assignment provides Python/rendering validation.
  C++ compilation and numerical/device validation remain future review work;
  do not imply that the selected controls prove stability for arbitrary values.

## Independent branches versus production integration

PR-04 is the explicit exception to the one-commit-on-dev requirement: its first
commit must be identical to PR-03's commit, and its second commit is the EmZ-only
topic. All other topics should be independent without copying unrelated
prerequisites. Adapt test fixtures to base-supported features instead of
importing the whole migration. Complete PR-03 before constructing PR-04.

Use the same implementation in standalone and integrated variants wherever
possible. Resolve shared-file overlaps during integration. A standalone commit
and its production counterpart need not have the same SHA or identical patch:
their parents differ, and PR-04's prerequisite is a separate shared commit on
its upstream branch. Integrate only PR-04's second commit after PR-03 on the
production branch. Document substantive differences beyond integration context.

Do not blindly cherry-pick all branches expecting their root `PR.md` files or
shared template hunks to merge. On the production branch:

- Keep exactly 12 ordered topical commits for PR-01 through PR-12. Integrate
  each feature once, including its tests and public documentation.
- Store each standalone PR description at `pr-descriptions/NN-<topic>.md`,
  introduced with its corresponding topical commit. Its text should match
  the standalone branch's `PR.md` at delivery.
- Introduce the production root `PR.md` in the setup/documentation portion;
  it describes the full prototype and links the topic archive.
- Follow with five setup-specific commits: (1) reference setup files,
  (2) core Python setup/density and core validation, (3) energy/PhaseSpace/field/
  checkpoint outputs and their validation, (4) Binning electron spectrometer
  and its validation, (5) final setup settings, audit, README and migration/
  rebundling records. Assign historical task reports to the matching stage
  when practical. Do not make another upstream-example branch in this task.
- Keep reusable library changes out of those final five commits. In particular,
  the memory fix belongs to PR-10 and precision/runtime controls to PR-12/11.
- Each setup stage should have coherent documentation and pass the tests
  introduced by that stage. The final production tree must preserve all source
  functionality and acceptance coverage.

This gives the production branch **17 commits above the pinned dev**. Produce
that final structure without leaving temporary fixup or merge commits.

## What remains setup-specific

Preserve the tracked `tweac5DegreeSetup` reference files, Python script, validation
suite and README on the production branch. Preserve the migration records
`TODO.md`, `changes.txt`, `resolvedConflicts.md` and this specification there.
Historical reports retain their historical context; new results belong in the
new report. None of these setup/history files belongs in an upstream PR branch.

The Binning electron spectrometer uses existing infrastructure; it is not a new
plugin implementation. Likewise, do not contribute a replacement for upstream
minimum-weighting support or typed openPMD backend configuration.

The final prototype must retain:

- Default 1000 steps; grid 3584 x 3584 x 1792; devices 8 x 16 x 8.
- Exact existing density specification, species/numerical choices, moving and
  switching windows, memory/PML settings, and two TWTS pulses.
- Incident-field injection with componentwise evaluation only.
- Core float32, TWTS internal float64, envelope cutoff 8 through precision.param.
- Explicit minimum weighting 0.1, disabled automatic grid adjustment and 1%
  progress reporting.
- All 27 active diagnostic objects, including the separate PhaseSpace and
  Binning outputs, field slices, energies, compression choices and checkpoints.
- The documented omission of ineffective `adios2.dataset.chunks="auto"` and
  the longer-run density-rounding qualifications. Do not restore that rejected
  backend key or claim bitwise equivalence for long-run density evaluation.
- Import/construction without file creation and generation without subprocess,
  compilation, job submission or simulation execution.

## PR.md requirements

Every upstream branch's root `PR.md` must be included in its topic commit.
For PR-04, this is the second commit, replacing the description inherited from
PR-03; PR-03's original commit and description remain unchanged in its history.
Write it for a reviewer who has not seen this conversation. It is a local draft
artifact that the user can remove before final submission, not permanent API
documentation. Include:

1. A proposed PR title, then the concrete problem and resulting behavior.
2. Scope, public API/example when useful, defaults and compatibility behavior.
3. Whether names/options are standard PICMI or PIConGPU extensions; do not
   claim standardization merely because an option is exposed through PICMI.
4. Exact validation commands and actual results for that branch. Distinguish
   Python generation checks from C++/device validation not performed.
5. Dependencies/overlap where relevant, especially PR-04's included PR-03 fix.

Keep descriptions proportional to the change. Include conventional user docs
under the existing documentation tree as well; PR.md is not their replacement.
Do not reference absent TWEAC files or temporary absolute paths as prerequisites.

The production `PR.md` instead describes the complete prototype, its preserved
settings, validation/limitations and links to all archived topic descriptions
and the rebundling report. It must not present the whole branch as one proposed
upstream feature PR.

## Work and validation procedure

1. Inventory source-vs-dev changes and create a change-to-topic checklist.
   Account for every reusable change, test and documentation hunk, including
   incidental fixes hidden in setup commits.
2. Create/test the upstream branches against the pinned dev, with PR-04 based
   on PR-03 as specified. Work one topic at a time; record its results before
   proceeding. Tests must import the code from the branch being checked, not
   the original checkout.
3. Assemble the production branch in the order above, resolving overlaps and
   moving setup-only material to the final five commits.
4. Compare the final production tree and generated setup with the source.
5. Finish the report and descriptions, verify all branch histories and leave
   the original checkout on its original branch with user files untouched.

Use an isolated Python 3.13 environment with this checkout's declared test
dependencies. `uv` is available. The earlier `/tmp/picmi-task1-venv` may still
exist, but its editable install can point at the original checkout. For every
worktree, verify `picongpu.__file__` and its template location before testing.
Use a separate environment per worktree or deliberately rebind the editable
install sequentially; do not let branches share an incorrectly bound package.
Avoid accidentally importing modules from the original working directory.

Set `PIC_RC` to a verified nonexistent path and `MPLCONFIGDIR` to a temporary
directory. Use `PYTHONDONTWRITEBYTECODE=1` and pytest `-p no:cacheprovider` where
practical. Do not execute generated workflow helper scripts.

On each upstream branch, run its focused tests plus relevant existing
regressions. These must cover the feature without relying on another PR or on
`tweac5DegreeSetup`, except for PR-04's explicitly included PR-03 prerequisite.
Run both shape-consistency and deposition tests on PR-04. On the final
production branch, run the combined suite:

```bash
PIC_RC=/tmp/picmi-rebundle-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-rebundle-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
<production-venv>/bin/python -m pytest -q -p no:cacheprovider \
  tweac5DegreeSetup/validation \
  lib/python/test/picongpu/quick/picmi \
  lib/python/test/picongpu/quick/pypicongpu/test_memory_config.py \
  --ignore=lib/python/test/picongpu/quick/picmi/test_workflow.py \
  --ignore=lib/python/test/picongpu/quick/picmi/test_rocrate.py \
  --tb=short
```

The source previously passed 889 tests and 3549 subtests. Those are historical
reference counts, not a substitute for running the new branches or a requirement
to keep exactly the same test count after reorganizing tests. Explain changes in
coverage; do not drop meaningful tests to make independent branches pass.

Run the production script with its default arguments into a fresh temporary
output path. Generate from the pinned source in a separate temporary worktree
as a baseline, also using the correct package. Compare parameter headers,
N.cfg, relevant workflow settings and rendering context. Account explicitly for
path/provenance/timestamp metadata; do not hide differences using broad filters.

Aim for byte-identical final production implementation/templates and reference
setup relative to the source. Adapted tests, redistributed public documentation
and new handoff/report files may differ. Inventory every remaining implementation
or setup difference and justify it; this task does not authorize new numerical
behavior. `tweac5DegreeSetup/include` must be byte-for-byte unchanged. No source
implementation fix may be lost merely because it appeared incidental.

## Reporting and final acceptance

Create `REBUNDLING_REPORT.md` on the production branch in the final setup/docs
commit. Include:

- Pinned base/source commits and destination branch names.
- A mapping of source hunks/topics to upstream topic commit hashes and the 12
  production topical commit hashes. Do not embed a final commit's own hash in
  its contents; report the final production tip in the completion message.
- The shared PR-03 prerequisite hash on PR-04, and any differences between
  upstream and integrated variants.
- Actual per-branch commands/results, final combined results and generated
  baseline comparisons, with any unresolved validation limitation stated.
- An accounting of setup-only files, omitted-from-upstream records, and final
  source-tree differences. Preserve historical counts in older reports.

Verify mechanically for every upstream branch except PR-04:

```text
branch^ == pinned dev
rev-list --count dev..branch == 1
no merge commits
PR.md exists in branch's committed tree
no TWEAC setup or migration-history files added
```

Verify separately for PR-04:

```text
PR-04 branch^ == PR-03 branch tip (same commit hash)
PR-04 branch^^ == pinned dev
rev-list --count dev..PR-04 == 2
rev-list --count PR-03..PR-04 == 1
second commit contains only the EmZ topic and its tests/docs/PR.md update
no merge commits; no duplicated or squashed prerequisite
root PR.md describes PR-04 and its dependency
no TWEAC setup or migration-history files added
```

Verify for production:

```text
pinned dev is the base
rev-list --count dev..production == 17
12 topical commits followed by 5 setup-specific commits; no merge/fixup commits
root PR.md, 12 archived descriptions and REBUNDLING_REPORT.md exist
same final functionality and reference include tree as pinned source
all required Python tests and generation comparisons pass
```

Check whitespace/conflict markers in changed material without reformatting
preexisting reference files. Confirm original branch tips, local dev and user
untracked artifacts are unchanged. The final response should list the production
branch, the 12 upstream branches, validation results and the report location.
Do not claim remote PRs were opened. Stop only when all local deliverables are
complete, or report a concrete blocker without silently weakening requirements.
