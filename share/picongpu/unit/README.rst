PIConGPU unit test
==================

Test components in an as best as possible isolated environment.

Example how to compile and execute the tests.compile

.. code-block:: bash
    # compile for NVIDIA GPUs
    cmake <path_to_picongpu_source>/share/picongpu/unit/ -Dalpaka_ACC_GPU_CUDA_ENABLE=ON
    make -j
    ctest

TWTSTight tests
--------------

The targets ``UnitTest-TWTSTight.cpp-2D`` and ``UnitTest-TWTSTight.cpp-3D`` each
contain all four TWTSTight test cases. Field evaluations, float Bessel ratios,
double Bessel ratios, and envelope guard checks compile in separate translation
units to avoid a CUDA frontend stall observed when compiling them together.
The ratio test case invokes both precision variants through host functions;
its GPU kernels remain in their respective translation units. A single PMacc
fixture in ``TWTSTight.cpp`` initializes the combined executable.

After updating the source, rerun CMake in the existing configured build directory
so it discovers the additional sources, then build and run both executables:

.. code-block:: bash

    cmake -S <path_to_picongpu_source>/share/picongpu/unit -B .
    cmake --build . --target UnitTest-TWTSTight.cpp-2D UnitTest-TWTSTight.cpp-3D --parallel 16
    ./dimensional_tests/UnitTest-TWTSTight.cpp-2D
    ./dimensional_tests/UnitTest-TWTSTight.cpp-3D

Run CUDA tests in a GPU allocation. A new build directory also needs the backend
and architecture options; the command above assumes they are already cached.
The temporary ``PIC_TWTSTIGHT_COMPILE_PROBES`` option and probe targets have been
removed. The reference generator continues to read field constants from
``TWTSTight.cpp``; ratio constants are in ``twtstight/BesselRatios.hpp``.
