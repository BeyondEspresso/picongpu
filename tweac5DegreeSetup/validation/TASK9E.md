# Task 9e — preserve TWTS numerical settings through precision.param

The reference's setup-local `TWTSTight.hpp` changes internal arithmetic from
`float_X` to `float_64` and the envelope cutoff from 6 to 8. Both settings are now
represented by the Python definition through the existing `precision.param`:

```python
picongpu_precision=32,
picongpu_precision_config=picmi.PrecisionConfig(twts=64, twts_num_sigmas=8),
```

These are PIConGPU extensions to the simulation configuration. There is no new
parameter file, copied implementation header, or per-laser numerical policy.
The two reference pulses share the settings. The default run remains 1000 steps
on the original grid, with incident-field injection and componentwise evaluation.

## Configuration and compatibility

| Python setting | Definition in precision.param | Default |
| --- | --- | --- |
| `twts="core"`, `32`, or `64` | `PICONGPU_TWTS_PRECISION` = `0`, `32`, or `64` | Core precision (`float_X`) |
| `twts_num_sigmas` | `PICONGPU_TWTS_NUM_SIGMAS` | 6 |

The cutoff accepts positive integers representable by `uint32_t`. Frontend and
backend validation reject fractional values, booleans, strings, zero, negative
values and overflow. Precision and cutoff remain independent; selecting double
precision does not implicitly enlarge the envelope.

`TWTSTight.hpp` already includes `defines.hpp`, which loads `precision.param`
before the TWTS aliases are declared. The implementation selects `float_T` from
the precision definition and `numSigmas` from the cutoff definition. Its existing
vector and complex aliases then follow `float_T`; the field formulas are unchanged.
Invalid numeric precision selections and out-of-range cutoffs have preprocessor
errors when the header is used.

Guarded defaults are present both in the repository `precision.param` and at the
TWTS consumer. The latter preserve compatibility with old setups that supply a
custom `precision.param` without these controls. Explicit build definitions take
precedence over the guarded settings; acceptance uses no such override. Native
setups can configure the same macros in their existing `precision.param`.

`TWTSPulse.def` needs no modification: its EField/BField functors already consume
the shared implementation. Background fields use that same implementation.
Neither `incidentField.param` nor `fieldBackground.param` is included from the
TWTS implementation, avoiding circular dependencies and include-order precedence.
Returned field components, amplitude scaling and the switching window retain
`float_X`. The simulation core remains float32. Gaussian/dispersive laser code
and special-operation precision selections are unchanged.

## Expanded acceptance

The inventory in `test_audit.py` now covers every file under the reference
`include/`: 17 parameter files and the TWTS header. An unknown file fails the
inventory. A new comparison resolves the two configuration selections as source
text, removes comments/whitespace, and compares the entire repository TWTS header
with the reference header. This verifies there are no additional unported header
differences; it is explicitly not a C++ preprocessor or compilation test.

Additional checks cover:

- defaults, both core precisions and independent TWTS precision/cutoff choices;
- typed validation in the PICMI and pypicongpu layers;
- conversion, serialization and generated `precision.param` definitions;
- 64-bit/8 in the complete setup with both reference pulses and core float32;
- include order, guarded legacy defaults and existing float_X component returns;
- the absence of a generated TWTS implementation override.

The previous integration tests continue to check all active outputs and runtime
options, enforce the original grid and componentwise incident-field flags, and
prohibit subprocess/simulation execution or reads of the reference build/binaries
during generation. The density-rounding and representation qualifications in
[TASK9D.md](TASK9D.md) still apply.

## Reproduction and limits

Use the isolated `uv` setup and generation instructions in
[README_PICMI.md](../README_PICMI.md). The previous temporary environment was no
longer present, so it was recreated with Python 3.13.5 and an editable installation
of `./lib/python[test]`. Python dependencies were downloaded; no PIConGPU compiler
toolchain was installed. This run used Pydantic 2.14.0 and pytest 9.1.1.

Run the full regression command recorded in [TASK9D.md](TASK9D.md); its existing
file selection includes all new tests in `test_audit.py` and `test_memory_config.py`.
All **510 tests and 3497 subtests passed** (102.31 s), including 66 migration
acceptance checks. Tasks 9e, 9d and 9 are complete for the agreed generation scope.
The standalone CLI generated `/tmp/tweac-picmi-final`, including the precision
settings above. Generation emitted the existing Mustache warning about
`grid.sim_dim`; the generated grid is covered by acceptance comparisons.

No C++ preprocessing/compilation, device execution, job submission, simulation,
or actual diagnostic-file validation was performed. Python/source comparison
preserves the requested configuration; it does not prove numerical stability,
bitwise device equivalence, or that newer algorithms make 64-bit/8 unnecessary.
