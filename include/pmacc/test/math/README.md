# Complex Bessel tests

`besselUT.cpp` tests `pmacc::math::bessel::{j0,j1,j0e,j1e}` for
`alpaka::Complex<float>` and `alpaka::Complex<double>`, on the host and the
configured accelerator. `j0e` and `j1e` return `exp(-abs(Im(z))) * J_n(z)`;
they compute the scaled value directly, including when the ordinary function
overflows.

The PMacc test CMake project discovers `*UT.cpp` automatically. Reconfigure an
existing build to discover the new file, then build these targets:

```sh
cmake --build <pmacc-test-build> --target PMaccTest-bessel-2D PMaccTest-bessel-3D
```

From the build directory, run `ctest -R '^bessel-[23]D$' --output-on-failure`.
The CTest launcher uses `mpiexec`; on a cluster requiring `srun`, launch the two
executables with the site's usual single-rank allocation instead. For CUDA-only
builds, use `alpaka_ACC_GPU_CUDA_ONLY_MODE=ON` and disable CPU backends.
Use `alpaka_FAST_MATH=OFF` for these accuracy checks: reassociation can remove
the compensated summation, and approximate trigonometry changes the error budget.

The checked-in reference data has no runtime Python dependency. Regenerate it
with mpmath (generated with version 1.3.0):

```sh
python3 include/pmacc/test/math/bessel_reference.py
clang-format -i include/pmacc/test/math/BesselReference.hpp
```

Coordinates are exact binary values. The generator starts with at least 100
decimal digits, adds precision for extremely large/small inputs, and verifies
that the binary64 reference constants agree at a further 80 digits. Each point
is tested in all four quadrants using the exact parity/conjugation identities.

Coverage includes zero, tiny normal inputs whose squares underflow, real and
imaginary axes, both sides of the Taylor/Miller/asymptotic boundaries, the former
35/50 cutoffs, neighbors of the first positive zero of each order, large real
phases, and scaled values at imaginary arguments far beyond the overflow limit.
At `91i` (float) and `713i` (double), the ordinary Bessel values are still finite
although forming `exp(abs(Im(z)))` directly would overflow. Separate checks cover
legitimate unscaled overflow, exact zero axis components, and NaN/infinite inputs.

Errors are checked componentwise, without squaring tiny differences. The budget
is 32 float epsilons or 64 double epsilons times the largest reference component
of the J0/J1 pair (scaled or unscaled as appropriate). This controls absolute error
near zeros and relative error away from zeros. For inputs with both components
smaller than one, each order uses its own amplitude, so an incorrect zero for a
tiny J1 value cannot pass. These are test tolerances, not correctly-rounded or
global relative-error guarantees. Axis components required to vanish are checked
exactly. Both host and accelerator results are independently compared with the
references; host/device agreement alone is not sufficient.

The Bessel implementation intentionally avoids general complex `abs`, `sqrt`,
trigonometric functions and unrestricted complex division in its large-argument
path. Its local scaling addresses their intermediate-range limitations without
changing alpaka's general complex implementation.
