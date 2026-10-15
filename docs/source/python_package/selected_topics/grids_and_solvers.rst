Grids and Solvers
=================

Every simulation needs a spatial domain and a field solver.
In PICMI the grid lives *inside* the solver
(the solver defines how the fields are advanced on the grid),
so the two are configured together.

.. literalinclude:: ../snippets/selected_topics/grids_and_solvers.py
   :language: python
   :start-at: grid = picmi.Cartesian3DGrid
   :end-before: simulation.write_input_file

Grids
-----

PIConGPU supports two grids:

:class:`~picongpu.picmi.grid.Cartesian3DGrid`
   A 3D Cartesian grid (3D3V).

:class:`~picongpu.picmi.grid.Cartesian2DGrid`
   A 2D Cartesian grid (2D3V);
   the position along the third axis is a fixed thickness
   (``picongpu_cell_depth_si``).
   A 3D grid can be reduced to a 2D one with ``Cartesian3DGrid.to_2d()``.

The domain of a Cartesian grid is given by

* ``number_of_cells``: the number of cells per dimension,
* ``lower_bound`` / ``upper_bound``: the extent in metres
  (**lower bound must be ``[0, 0, 0]``**),
* ``lower_boundary_conditions`` / ``upper_boundary_conditions``:
  ``"open"`` (absorbing) or ``"periodic"`` per axis.
  PIConGPU chooses boundary conditions **per axis**, so the lower and the
  upper condition of a dimension must be equal.

The cell size per dimension is derived as
``(upper_bound - lower_bound) / number_of_cells``.

Distributing over GPUs
^^^^^^^^^^^^^^^^^^^^^^

By default a simulation runs on a single GPU.
To distribute it, give the grid

* ``picongpu_n_gpus``: a list of GPU counts per dimension.
  A single number ``[N]`` distributes over ``N`` GPUs in ``y`` direction
  (the direction PIConGPU is optimized for); ``[Nx, Ny, Nz]``
  distributes over all three.
* ``picongpu_grid_dist`` (optional):
  the explicit number of cells assigned to each GPU per dimension.
  If given, each chunk must be a multiple of the super-cell size
  and the chunks must sum to ``number_of_cells``.
* ``picongpu_super_cell_size``:
  the super-cell size (default ``(8, 8, 4)``).
  The grid must be divisible by both the GPU count and the super-cell size;
  the frontend checks this and reports which dimension fails.

.. _grids_guard_cells:

Guard cells
^^^^^^^^^^^

``guard_cells`` (per axis) sets the number of guard cells.
It must be a non-negative multiple of the super-cell size;
if unset, PIConGPU's default is used.

PML absorber
^^^^^^^^^^^^

The standard PICMI grid argument ``pml_cells`` sets the thickness in cells:
one integer per spatial axis, applied to both faces of that axis.
For example, ``pml_cells=[110, 110, 110]`` sets all six faces in 3D to 110 cells.
Values must be integers from zero to ``2**32 - 1``; zero disables absorption
on that axis. Periodic axes require zero when thickness is explicitly supplied.
Thickness need not be a multiple of the super-cell size.

Additional coefficients use the PIConGPU extension ``picongpu_pml``.
Pass a ``picmi.PML`` object alongside the grid's usual domain arguments:

.. code-block:: python

   grid = picmi.Cartesian3DGrid(
       number_of_cells=[3584, 3584, 1792],
       lower_bound=[0, 0, 0],
       upper_bound=[3584 * 6.177353e-8, 3584 * 3.088676e-8, 1792 * 6.177353e-8],
       lower_boundary_conditions=["open"] * 3,
       upper_boundary_conditions=["open"] * 3,
       picongpu_n_gpus=[8, 16, 8],
       picongpu_super_cell_size=[8, 8, 8],
       pml_cells=[110, 110, 110],
       picongpu_pml=picmi.PML(
           sigma_kappa_grading_order=2.5,
           sigma_opt_multiplier=1.25,
           kappa_max=[4, 1, 4],
           alpha_grading_order=1.0,
           alpha_max_si=[0, 0, 0],
       ),
   )

``picmi.PML`` and its coefficient names are PIConGPU extensions, not additions
adopted by the PICMI standard. Supported coefficients and defaults are:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Allowed values
   * - ``sigma_kappa_grading_order``
     - ``4.0``
     - Finite, nonnegative
   * - ``sigma_opt_multiplier``
     - ``1.0``
     - Finite, nonnegative
   * - ``kappa_max``
     - ``(1.0, 1.0, 1.0)``
     - Three finite components, each at least one
   * - ``alpha_grading_order``
     - ``1.0``
     - Finite, nonnegative
   * - ``alpha_max_si``
     - ``(0.2, 0.2, 0.2)``
     - Three finite, nonnegative components, in siemens/metre

The generated conductivity is calculated separately for each axis ``d``:

.. code-block:: text

   SIGMA_OPT_SI[d] = 0.8 * (sigma_kappa_grading_order + 1) / (Z0 * cell_size[d])
   SIGMA_MAX_SI[d] = sigma_opt_multiplier * SIGMA_OPT_SI[d]

Here ``Z0`` is PIConGPU's vacuum impedance. The expression retains the actual
cell size per axis, including anisotropic grids. Coefficient vectors always
have three components, including for 2D. ``to_2d()`` preserves coefficients
and the x/y thicknesses; an explicitly configured 2D absorber has zero z thickness.

Omitting both options preserves the C++ default coefficients and 12-cell
thickness (periodic faces are disabled by PIConGPU at runtime). Supplying only
``picongpu_pml`` chooses 12 cells on open axes and zero on periodic axes;
supplying only ``pml_cells`` retains default coefficients.
Generated runtime configuration explicitly selects ``--fieldAbsorber pml``.
PIConGPU disables the absorber when every axis is periodic.
An exponential-absorber selector and arbitrary conductivity formulas are not exposed.

For explicit PML configuration, Python checks that layers fit every local
domain, accounting for nonuniform GPU partitions and opposing layers on a
single rank. Active incident-field injection faces must also clear the PML
by the field solver's required margin. No Huygens positions are moved or
clamped; inactive faces, such as the distant YMax used by the TWTS moving
window, are preserved. These checks cover absorber clearance, not every
incident-field stencil constraint at GPU interfaces; C++ runtime checks remain.

Solvers
-------

:class:`~picongpu.picmi.solver.ElectromagneticSolver` advances the electromagnetic fields.
The ``method`` selects the solver:

``"Yee"``
   The standard second-order Yee solver (fixed order).

``"Lehe"``
   The Cherenkov-free Lehe solver (fixed order).

``"CKC"``
   The extended Cole-Karkkainen-Cowan solver (fixed order).

``"other:ArbitraryOrderFDTD"``
   The arbitrary-order FDTD solver;
   ``stencil_order`` selects the order
   (it must be the same for all axes and even).

``"other:None"``
   Disables the vacuum update of E and B (no CFL limit).

Time step: ``cfl`` or ``time_step_size``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The time step is fixed by one of two equivalent quantities:

* ``solver.cfl``: the Courant-Friedrichs-Lewy number, or
* ``simulation.time_step_size`` in seconds.

Give exactly one and the other is derived from the cell size;
if you give both, they must be consistent or the frontend raises an error.
``other:None`` has no CFL limit, so any time step is legal.
Note that a 2D simulation uses the 2D CFL factor (``sqrt(2)`` on a square
grid, not ``sqrt(3)``).

Field smoothing
^^^^^^^^^^^^^^^

The current can be smoothed with :class:`~picongpu.picmi.solver.BinomialSmoother`
(passed via ``source_smoother``).
PIConGPU's binomial current deposition is a fixed, single-pass filter,
so ``n_pass`` may only be ``None``, ``1`` or an all-ones vector.
Any other field-smoother option is rejected.
