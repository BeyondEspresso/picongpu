# PICMI migration rebundling report

## Inputs and branch layout

The pinned base is local `dev` at `b8f83fcfa07aa544ce7f3e1206adc797fa4a6b0b`. The source is `topic-create-TWEAC-picmi-script-rebased` at `74b6a2089848c948dda9142b08246d82f453af34`. The earlier implementation `topic-create-TWEAC-picmi-script` at `05fc94955645cf5a9ce9ac86e00fea28fb688e29` was historical context only. The integration branch is `topic/tweac-picmi-production`. Each `pr/*` branch below is a local upstream candidate and contains its own `PR.md`. No remote PR was opened.

| Topic and source commit | Candidate branch tip | Production topic commit | Focused Python result |
| --- | --- | --- | --- |
| 01 TWTS validation/polarization, `3f403dea8` | `pr/picmi-twts-validation` `90a5d6e1f` | `975c4cfc5` | 62 passed |
| 02 composite pushers/names, `fd4361c3b` | `pr/picmi-composite-pushers` `88a4f1844` | `b8435f153` | 161 passed, 24 subtests |
| 03 species shape, `daed4a2f8` and `b4c702ccb` | `pr/picmi-species-shape-consistency` `9eca73fac` | `4a2dc2efc` | 123 passed, 22 subtests |
| 04 EmZ/EZ deposition, `b4c702ccb` | `pr/picmi-emz-current-deposition` `81d369d2a` | `80666ca35` | 87 passed, 22 subtests |
| 05 PML, `098bcc75e` | `pr/picmi-pml-configuration` `5d5c7f27a` | `2fad1ca81` | 144 passed |
| 06 minimum-weighting validation, `83ac52118` | `pr/picmi-min-weighting-validation` `1dc1a21e6` | `cbb0074c9` | 111 passed |
| 07 PhaseSpace suffix, `1e4dfe665` | `pr/picmi-phase-space-extension` `c98fcdc4c` | `99db6e05b` | 45 passed |
| 08 openPMD range, `6e4447c96` | `pr/picmi-openpmd-range` `3ee9eb9ac` | `d74f826fd` | 11 passed |
| 09 built-in densities, `6e4447c96` | `pr/picmi-builtin-density-output` `5cf33b082` | `ef72d693c` | 41 passed |
| 10 memory literal, `f2b059c3e` | `pr/picmi-memory-literal-overflow` `accc4876e` | `ea5f8d9c5` | 26 passed |
| 11 runtime controls, `c312e02c8` | `pr/picmi-runtime-controls` `94cd52977` | `47c476637` | 107 passed |
| 12 TWTS precision/cutoff, `c312e02c8` | `pr/twts-precision-controls` `f29775085` | `3eb4d733c` | 59 passed |

PR-04 begins with the *exact* PR-03 commit `9eca73fac`, followed by its EmZ-only commit `81d369d2a`; its EmZ review diff is against PR-03. The other eleven candidate branches each have one commit directly on the pinned `dev`. Production applies PR-03 once, then only PR-04's EmZ delta. The standalone commits and production counterparts differ in SHA and, where files overlap, patch context and independent test fixtures. No setup files occur in candidate branches. Each archived `pr-descriptions/NN-*.md` matches the corresponding candidate's committed root `PR.md`.

The five production setup commits are `9416d33a4` (tracked reference files from `3f403dea8`), `b505ce3e0` (core Python setup from `f2b059c3e`), `d2c7300b7` (energy/PhaseSpace/field/checkpoint outputs from `6e4447c96`), `a25311a29` (Binning from `7e4791668`), and the final setup/audit/documentation commit (this report deliberately omits its own hash). `c312e02c8` supplies the final script settings and audit; `74b6a2089` supplies the historical rebase record. `TODO.md`, `changes.txt`, `resolvedConflicts.md`, `PR_REBUNDLING_SPEC.md`, and the task reports are retained only on production. The user-owned, untracked hint file, `.build`, and `bin` were neither read for generation nor copied.

## Validation

Each candidate was tested in its own worktree against the pinned base, with the worktree's Python package selected. Exact focused test commands and coverage are in its archived PR description. The setup stages passed their newly introduced tests: core 6, core plus outputs 18, and core plus outputs plus Binning 59. The final combined command was:

```bash
PIC_RC=/tmp/picmi-rebundle-no-config.toml MPLCONFIGDIR=/tmp/picmi-rebundle-matplotlib PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/picmi-rebundle-production/lib/python /tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  tweac5DegreeSetup/validation lib/python/test/picongpu/quick/picmi \
  lib/python/test/picongpu/quick/pypicongpu/test_memory_config.py \
  --ignore=lib/python/test/picongpu/quick/picmi/test_workflow.py \
  --ignore=lib/python/test/picongpu/quick/picmi/test_rocrate.py --tb=short
```

Result: **888 passed, 3549 subtests passed in 131.61 seconds**. The previous source result of 889 passed and 3549 subtests is historical. The one-test net difference reflects redistributed/adapted shared tests and the added PR-12 generated-header regression; meaningful behavior remains covered.

The default script was run with Python 3.13 and generated 1000-step input independently from production and the pinned source. All generated `include/` headers, `etc/picongpu/N.cfg`, and other files except provenance/metadata and path-bearing workflow files are byte-identical. Differences in `workflow/input.yaml` are only absolute output paths. `workflow/scripts/*` differences are only checkout paths. `metadata/rc_params.json` records the different source checkout, and `metadata/pypicongpu_runner.json` records checkout/setup/run paths plus random generated diagnostic type identifiers. `metadata/pypicongpu_rendering_context.json` differs only in those random identifiers. `ro-crate-metadata.json` differs only in generation timestamp. No generated parameter or active output differs. Python generation only was performed; no workflow script was run.

The final `tweac5DegreeSetup/include` tree and script match the pinned source byte-for-byte. Reusable implementation/templates also match except for a three-line wording change in `lib/python/picongpu/picmi/species.py`'s docstring ("particle_shape" versus "resolved particle shape"); this has no runtime effect. Adapted tests and redistributed public documentation differ intentionally. In particular, the source's generic runtime-control prose in `foundations/defining_simulation.rst` is represented on production in `selected_topics/simulation_settings.rst`; the source's detailed shape/deposition prose is condensed in `selected_topics/species_distributions_layouts.rst`. Precision, openPMD and memory explanations were grouped with their standalone topics. No reusable numerical behavior was changed by rebundling.

The historical audit and setup README explain the remaining acceptance limits: active 1000-step vacuum run, longer-run density rounding, omitted ineffective ADIOS2 chunk hint, and no C++ compilation/device/numerical validation. The TWTS precision controls were validated through Python conversion and generated-header checks only.

History checks confirm 17 linear production commits and the required one-commit candidate histories (two for PR-04), with no merge commits. `git diff --check dev..HEAD` reports only seven existing whitespace/EOF issues in the copied reference `tweac5DegreeSetup/include` tree. That tree is byte-identical to the pinned source and was intentionally not reformatted; excluding it, the full diff passes `git diff --check`. No conflict markers were found in the new setup and handoff files. The original checkout remains on its original branch with its untracked artifacts unchanged.
