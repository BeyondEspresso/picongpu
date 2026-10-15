# Task 9a: core simulation and density profile

Task 9a is complete. [picmi_setup.py](../picmi_setup.py) constructs the core
simulation with the active 1000-step default and generates it through an explicit
entry point. It has no handwritten C++ template overrides and does not read
the old setup's `.build` directory or binaries. Import and construction have
no generation, build or execution side effects.

The generation and isolated `uv` environment commands are in
[README_PICMI.md](../README_PICMI.md). No new Python environment or full PIConGPU
toolchain was installed for this task; the existing isolated environment was reused.

## Implemented configuration

| Setting | Core definition / reference |
| --- | --- |
| Cells / devices | `(3584,3584,1792)` / `(8,16,8)` |
| Cell sizes | `(6.177353e-8,3.088676e-8,6.177353e-8)` m |
| Timestep / default steps | `5.15136e-17` s / 1000 |
| Field solver / smoothing | CKC / binomial |
| Moving window | Move point 0 |
| Species | Cold electrons `e`; no ions, temperature or drift initialization |
| Shape / interpolation / deposition | PQS / PQS AssignedTrilinearInterpolation / EmZ |
| Pusher | Free then HigueraCary, switch at step 6720 |
| Layout / normalization | One random macroparticle per cell; `1e25 m^-3`; typical PPC 1 |
| Minimum weighting | 0.1 |
| PML | 110 cells, grading 2.5, sigma multiplier 1.25, kappa `(4,1,4)`, alpha grading 1 and maxima 0 |
| Memory | 4 GiB reserve; exchanges 32/320/32 MiB; edges/corners 2 MiB |
| Supercells / guard | `(8,8,8)` cells / one supercell per direction |
| Exchange scaling | Reference size and scaling factors all zero |
| Temporary field storage | One slot; gather communication enabled |
| Precision / walltime | Default core 32-bit / 2 hours |

The global exchange configuration reproduces the reference's electron-specific
configuration because electrons are the only active species. The existing backend
already generates one temporary field slot without collisions, matching the
reference; no extra option was necessary.

The two TWTS pulses retain wavelength 0.8 µm, waist 1 µm, a0 1.25, duration
`25 fs / 2.354820045`, incidence ±5°, polarization ±45.10716816968629°, focus y
68.5 µm, lateral offsets ±12 µm and beta0 1. Equal focus and centroid y produce
zero manual delay; automatic delay remains disabled. Both windows retain
`(start,end,length) = (0,10000000,10000)`.

Generated faces contain both lasers on YMin, positive incidence on ZMin and
negative incidence on ZMax. XMin/XMax/YMax are inactive. Huygens positions remain
`[[116,-116],[116,100000000],[116,-116]]`. The recorded CMake options select
incident-field injection and componentwise evaluation.

## Density translation and checks

The SymPy expression translates the active [density.param](../include/picongpu/param/density.param):
hard transverse vacuum cuts at 200 cells, 23.6 µm Gaussian ramps with sigma
5.9 µm, the fixed reference transverse box dimensions, and a 100 µm longitudinal
linear ramp starting at `6720*dy = 207.5590272 µm`. The commented density spike
is absent. Transverse boundary points retain the Gaussian tail; cuts apply
strictly outside those bounds. Coordinates are absolute and therefore apply to
new regions introduced by window slides.

The definition supplies SI density; the existing generated density template
divides by `BASE_DENSITY_SI`. This preserves the reference normalization.
The initial box ends at y = 110.69814784 µm, before the longitudinal vacuum
boundary, so it initially contains no plasma. The profile must still be defined
beyond that box for subsequent moving-window initialization.

[test_core.py](test_core.py) extracts the active reference density constants and
uses a separate imperative Python translation of its branches. It compares
**1352 points** covering both sides and exact values of hard cuts, Gaussian
transitions, combined x/z ramps, vacuum, longitudinal ramp and later plateau.
The comparison uses normalized density with relative tolerance `2e-11` and
absolute tolerance `2e-15`. SymPy constants retain 17 digits so Python evaluation
does not round a hard boundary onto the wrong side. Task 7's representative
low-density cells also retain the expected one-particle creation cutoff.

These are analytic/source-based checks. SymPy may rearrange arithmetic, while
the reference explicitly rounds some intermediate density values to `float_X`.
Bitwise equality of the generated double-precision expression with that mixed
precision evaluation is not established. This numerical-rounding difference
must remain visible in task 9d; the profile branches, parameters and sampled
analytic values agree, but no device evaluation or physical equivalence is claimed.

## Memory renderer correction

The existing [memory renderer](../../lib/python/picongpu/pypicongpu/memory.py)
rendered 4 GiB as `4096 * 1024 * 1024`. Those unsuffixed literals multiply as
C++ `int` before assignment to `size_t`, overflowing a 32-bit signed int.
The renderer now adds `ULL` to the first factor when the value exceeds
`2**31-1`, producing `4096ULL * 1024 * 1024`. Smaller/default expressions
are unchanged. Five regression cases cover the signed-int limit and larger
MiB-, KiB- and byte-valued reserves. This is the only library implementation
change needed by 9a.

## Acceptance and remaining work

**268 Python tests passed**, comprising the six new core acceptance tests,
memory/precision tests (including five new cases), and the existing minimum
weighting, PML, deposition, interpolation, composite-pusher and TWTS/Gaussian
tests. Generated core parameters are compared directly with the active source
files, resolving pusher/shape aliases and inherited laser values. Tests also
check cold initialization, no extra active species, absence of diagnostic CLI
options, the unchanged normalization and the workflow's componentwise flags.

CLI generation was also exercised separately. Its final artifact is
`/tmp/tweac-task9a-final`; pytest artifacts are temporary and reproducible.
No PIConGPU compilation, job submission or simulation was performed.

Remaining differences and next steps:

- **9b:** Add the energy, seven PhaseSpace, field/slice and checkpoint outputs.
  The current diagnostic list is empty. The backend's empty radiation
  configuration emits no active CLI output and is not a migrated diagnostic.
- **9c:** Add the Binning `eSpec` spectrometer.
- **9d:** Audit effective runtime options, including the absent explicit
  `--autoAdjustGrid false` and `--percent 1`, and retain the density-rounding
  caveat above. Inactive exponential-absorber coefficients still use C++ defaults
  as documented in task 6. PICMI obtains electron mass from its Python particle
  data; it differs from the C++ constant by about `3.4e-12` relatively, but the
  generated mass ratio rounds to the reference's 1 in the selected float32.
  The generated laser amplitude agrees with the reference conversion within
  `1e-12` relative tolerance.

Task 9 remains incomplete. These checks establish the core configuration and
analytic profile comparison only; full output integration, runtime validation
and physical simulation validation have not been performed.
