# Task 9c — Binning electron spectrometer

**Rebase update:** The new typed backend model rejects the reference's ineffective
`adios2.dataset.chunks="auto"` key. The Python setup now omits that key, retaining
the same effective ADIOS2 defaults and `.bp` output. The reference source is
unchanged. See root `resolvedConflicts.md` for openPMD source evidence and current
test results; the original report below describes the earlier configuration.

The active `eSpec` from `include/picongpu/param/binningSetup.param` is now
implemented by `electron_spectrometer()` in [picmi_setup.py](../picmi_setup.py)
and added alongside the task 9b diagnostics. It uses the existing PIConGPU
PICMI Binning and ParticleFunctor interfaces; no new interface, library change,
handwritten C++ template override, or external backend file was necessary.
This is the Binning plugin, separate from the seven PhaseSpace instances.

## Reference mapping

| Setting | Generated definition |
| --- | --- |
| Binner/species | `eSpec`, unfiltered electrons `e` |
| First axis | `Energy`: 1600 linear bins, 0 to 1000 MeV |
| Energy limits in SI | 0 to `1.602176634e-10` J, converted to PIC units by the plugin |
| Energy value | `KinEnergy(weighted momentum, weighted mass) / weighting` |
| Second axis | `pointingXY`: 256 linear bins, -0.1 to +0.1 rad |
| Pointing value | `atan2(px, py)` using stored momentum components |
| Deposited quantity | `Charge`: signed macroparticle charge from `getCharge(weighting, particle)` |
| Types | `float_X` for both axes and deposition; the setup selects float32 |
| Overflow | Enabled for each axis, matching the C++ default |
| Notification | Every 500 steps, including zero; default run schedules 0, 500, 1000 |
| Particle region | Bounded enabled, Leaving disabled, matching the reference default |
| Backend | `.bp`, `{"adios2":{"dataset":{"chunks":"auto"}}}` |
| Filename | Default `simOutput/binningOpenPMD/eSpec_%06T.bp` |

The generated notification string `0:-1:500` is equivalent to reference `500`.
The interval and axes do not change with `--steps`. The reference leaves its
filename infix at the C++ default `_%06T.`; the writer strips the trailing dot
before adding `.bp`. The Python definition also leaves this setting unspecified.
Its backend JSON is embedded directly in the generated `binningSetup.param`,
independently of the four field-output TOML files and their Blosc compression.

## Weighting, units and bin semantics

Particle accessors expose weighted quantities in PIC units. The energy functor
divides the macroparticle energy by weight exactly once, giving energy per
electron. It reuses PIConGPU's `KinEnergy` implementation, including its
nonrelativistic branch below `GAMMA_THRESH=1.005`; no substitute energy formula
is emitted. The pointing functor keeps the reference argument order and all
quadrants. It does not substitute velocity or a one-argument arctangent.

The charge accessor already includes macroparticle weight. Deposition returns
that negative electron charge directly, without dividing by weight, applying
an absolute value, or multiplying by weight again. The Binning kernel adds the
functor result directly to the selected bin. Unit vectors in L,M,T,I,Θ,N,J order
are (2,1,-2,0,0,0,0) for energy, all zeros for pointing, and (0,0,1,1,0,0,0)
for charge. The plugin converts SI axis limits into PIC units and records the
deposited charge's conversion factor and dimensions in openPMD.

`LinearAxis` uses intervals closed on the lower edge and open on the upper
edge. Index 0 is underflow; regular bins are 1 through N; values at or above
the upper limit enter overflow index N+1. Each axis therefore has two extra
bins: 1602 energy bins and 258 pointing bins in storage. Tuple order is energy,
pointing; the writer reverses dimensions and axis labels for C-order storage,
giving shape **[258, 1602]** and labels **[pointingXY, Energy]**. Both definitions
use that same implementation, including its float32 rounding near bin edges.

The current C++ `BinningData` initializes `dumpPeriod` to 0, while the PICMI
interface defaults to 1. The definition explicitly uses 1. In `Binner::notify`,
the reduction counter increments before `counter >= dumpPeriod` is checked;
both settings consequently dump on every notification and reset the histogram
and counter afterwards. At each dump the counter is 1. The current writer
stores summed charge per bin and the counter as metadata; it does not divide
by energy width, angular width, or notification count. There is no accumulation
across successive 500-step notifications in either configuration.

## Acceptance

All work used local source and the existing isolated Python environment.
No network access, installation, PIConGPU compilation, job submission or
simulation execution was needed.

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  tweac5DegreeSetup/validation/test_core.py \
  tweac5DegreeSetup/validation/test_outputs.py \
  tweac5DegreeSetup/validation/test_binning.py
```

**59 tests passed**, including **41 new Binning checks**:

- Actual Python functors evaluated for zero, nonrelativistic and relativistic
  momenta; four different positive weights; pointing signs and all quadrants.
- Reference source comparisons for axis names/order, limits/counts, dimensions,
  formulas, species, notification period, output format and backend JSON.
- Float32 axis-limit rendering, in-range bins, interior/lower/upper boundaries,
  underflow/overflow, and SI-to-PIC limit conversion semantics.
- Raw negative charge accumulation, per-notification reset, equivalent dump
  periods, storage dimensions and run-length overrides.
- Generated registration and accessor code, alongside the preserved active
  grid, seven PhaseSpace instances and four field-output groups.

The CLI separately generated `/tmp/tweac-task9c-final` at the default 1000 steps.
The previous core and runtime-output comparisons pass with `eSpec` enabled,
including the existing 1352 density samples and 1000/230000/zero-step cases.

These checks establish Python generation and agreement with the checked-out
reference and C++ implementations. Numerical functor checks use Python math;
bin-index checks translate the source's float32 arithmetic. They do not execute
the generated C++, validate GPU math at signed-zero/rounding corner cases, or
create/read actual openPMD histogram files. The physical direction of exactly
zero momentum is undefined; the Python check follows `atan2(0,0)=0`, while the
generated definition preserves the reference's C++ call unchanged. Actual
compilation and simulation remain deliberately unrun.

Task **9c is complete**. The final migration audit, remaining runtime settings,
and previously documented density-rounding caveat remain assigned to **9d**.
