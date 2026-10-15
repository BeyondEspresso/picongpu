# Task 2: TWTS polarization geometry

Task 2 is complete at the Python setup-generation level: **62 tests passed**.
Both beams now generate using the reference polarization angles
`±45.10716816968629°`, including with zero delay. No PIConGPU C++ compilation,
job submission or simulation was performed.

## Geometry and source convention

Let `phi = laserIncidenceAngle` and `alpha = polarizationAngle`, in radians.
The propagation direction is `k = (0, cos(phi), sin(phi))`. Choose orthonormal
transverse axes `ex = (1,0,0)` and `q = (0,-sin(phi),cos(phi))`. The nominal
linear-polarization axis is

```text
e = cos(alpha) ex + sin(alpha) q
  = (cos(alpha), -sin(alpha) sin(phi), sin(alpha) cos(phi)).
```

Its squared length is `cos(alpha)^2 + sin(alpha)^2 = 1`, and `e·k = 0`.
At zero polarization angle it selects x; at a right angle it selects the
transverse axis in the yz plane. These limits agree with the documented C++
angle convention in
[TWTSTight.hpp](../../include/picongpu/fields/background/templates/twtstight/TWTSTight.hpp).

[TWTSTight.tpp](../../include/picongpu/fields/background/templates/twtstight/TWTSTight.tpp)
evaluates the core expressions at `abs(phi)`. For negative incidence,
[EField.tpp](../../include/picongpu/fields/background/templates/twtstight/EField.tpp)
implements a 180° rotation about y, changing the signs of the x and z
components. Applying that rotation to the positive-incidence nominal axis
gives the negative of the representative `e` above at negative incidence.
Both describe the same linear-polarization axis.

The Python vector is nominal transverse metadata, not the full spatially varying
electric field of the tightly focused pulse. Its choice of overall sign does
not change the field calculation: the TWTS template passes the scalar `PHI`
and `Polarization` values unchanged to the C++ field functors, which handle the
beam rotation and signs themselves.

## Correction

In [TWTSLaser.polarization_direction](../../lib/python/picongpu/picmi/lasers/twts_laser.py),
the z prefactor changed from `cos(alpha)` to `sin(alpha)`. The incorrect expression
was generally neither normalized nor transverse. Dividing it by its norm would
not repair its direction.

The correction is local to TWTSLaser. No shared validator, Gaussian/dispersive
implementation, backend model or C++ field formula was changed in task 2.

## Validation

- The new basis test was run before the correction and failed as expected:
  zero polarization angle at -30° incidence produced
  `(1, 0, 0.8660254...)`, with norm `1.32287565...`.
- Tests now cover incidence angles ±5° and ±60° with polarization angles
  0°, ±90°, 180°, and ±45.10716816968629°, asserting unit norm and
  transversality with absolute tolerance `1e-14`.
- Independent basis anchors at ±30° incidence check the signs, and a rotation
  test checks equivalence with the negative-incidence C++ polarization axis.
- The five full generation cases from task 1 now use the actual reference
  polarization angles. They preserve the grid, device decomposition, 1000 steps,
  switching window and incident-field/componentwise options.
- Generated metadata and `incidentField.param` preserve both scalar polarization
  angles, the corrected vectors, and the electric-field amplitudes. Positive,
  zero and negative manual delays still generate successfully.
- Existing Gaussian/dispersive regression tests and the shared restrictions
  tested by the TWTS suite still pass.

Direct comparison of all five generated setups with
[TwtsBackgroundLaser.param](../include/picongpu/param/TwtsBackgroundLaser.param)
confirmed the original angles and `a0=1.25`. For the zero-delay setup:

| Beam incidence | Polarization angle (rad) | Nominal polarization axis |
| --- | --- | --- |
| +5° | +0.78726860081181 | `(0.7057829460593135, -0.06174358094082977, 0.7057323595134627)` |
| -5° | -0.78726860081181 | `(0.7057829460593135, -0.06174358094082977, -0.7057323595134627)` |

Both generated amplitudes are `5.016720467141577e12 V/m`, using the Python
physical constants to convert the unchanged reference `a0` and wavelength.

## Reproduction and artifacts

Reuse the isolated `uv` environment described in [TASK1.md](TASK1.md):

```bash
PIC_RC=/tmp/picmi-task1-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-task1-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/picmi-task1-venv/bin/python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_twts_laser.py \
  lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py
```

Result: `62 passed in 19.46s`. No additional dependencies were installed for
task 2. The zero-delay generated setup for this run is:

```text
/tmp/pytest-of-debus/pytest-1/test_twts_setup_generation_6_80/setup
```

Temporary artifact paths may change on later test runs. These are still focused
laser-validation setups: particle species, custom PML and diagnostics await
their respective migration tasks. Acceptance verifies generated configuration
and Python geometry, not compiled field evaluation or simulation physics.
