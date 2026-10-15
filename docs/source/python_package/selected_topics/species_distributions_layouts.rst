.. _species:

Species, Distributions and Layouts
==================================

Adding particles to a simulation takes three components:
a **species** (what the particles are),
a **distribution** (where they are placed and how they move), and a
**layout** (their positions *within* a cell).
They are passed to the simulation together:

.. code-block:: python

   sim = picmi.Simulation(
       ...,
       species=[ion, electron],
       layouts=[ion_layout, electron_layout],
   )

.. literalinclude:: ../snippets/selected_topics/species_distributions_layouts.py
   :language: python
   :start-at: plasma = picmi.UniformDistribution
   :end-before: simulation.write_input_file

Species
-------

A :class:`~picongpu.picmi.species.Species` describes one type of particle.
Its most important parameters are:

* ``name``:
  the name of the species (also used in the output files).
  If not given, the name is derived from the particle type.
* ``particle_type``:
  the physical identity of the particle.
  This is either an element symbol (``"H"``, ``"He"``, ``"C"``, ...),
  one of the predefined particle types
  (``"electron"``, ``"positron"``, ``"proton"``, ``"anti-proton"``, ``"photon"``, ...),
  or a custom particle of the form ``"other:<name>"``
  (which then requires an explicit ``name``).
  The mass and charge of known particle types are filled in automatically.
* ``charge_state``:
  the initial charge state of an ion
  (0 for neutral, 1 for singly ionized, ...).
  Only meaningful together with an element ``particle_type``.
  Ions that can be ionized further during the simulation
  (see :ref:`Interactions <python_package/selected_topics/interactions:Interactions>`)
  must specify their initial charge state explicitly.
* ``picongpu_fixed_charge``:
  for ion species that are *not* subject to ionization,
  this fixes the charge of all their particles for the entire simulation.
* ``mass`` / ``charge``:
  override the (element-)derived mass and charge in SI units.
  This is how you define custom particles.
* ``particle_shape``:
  the species' shape flag, field-to-particle interpolation shape and
  current-deposition shape.
  If left unset, it is inherited from
  :attr:`~picongpu.picmi.simulation.Simulation.particle_shape`
  and, if that too is unset, falls back to the PIConGPU default
  ``"quadratic"`` (i.e. TSC).
* ``picongpu_current_deposition``:
  the current-deposition algorithm, ``"Esirkepov"`` (default), ``"EmZ"`` or
  its alias ``"EZ"``, using the species' resolved ``particle_shape``.
* ``method``:
  the particle pusher (default ``"Boris"``;
  ``"Vay"`` and ``"Higuera-Cary"`` are relativistic variants,
  ``"LLRK4"`` adds radiation reaction).
* ``picongpu_composite_pusher``:
  a :class:`~picongpu.picmi.species.CompositePusher` selecting two pushers
  and an absolute timestep at which to switch between them (see below).
* ``density_scale``:
  rescales the species' density relative to a shared profile
  (see below).

The per-species shape and pusher method default to the Simulation:
:attr:`~picongpu.picmi.simulation.Simulation.particle_shape`
is inherited by every species that does not set its own
(an explicit species ``particle_shape`` overrides it), while an unset
``method`` falls back to ``"Boris"``.
If neither the species nor the Simulation sets a shape,
PIConGPU uses its native default ``"quadratic"`` (TSC):

.. literalinclude:: ../snippets/selected_topics/species_shape_and_method.py
   :language: python
   :start-after: BEGIN-SPECIES_SHAPE
   :end-before: END-SPECIES_SHAPE

By default, every species is initialised **independently**: even when several
species happen to share the same distribution and layout, each one draws its
own in-cell positions.
To place several species *collectively* -- i.e. on exactly the same in-cell
positions, the standard way to build charge-neutral plasmas -- group them in a
:class:`~picongpu.picmi.multi_species.MultiSpecies`
(see :ref:`multi_species`).

.. _multi_species:

MultiSpecies: collective initialisation
---------------------------------------

A :class:`~picongpu.picmi.multi_species.MultiSpecies` is the explicit way to
request **collective (coordinated) initialisation**: all its members share one
``initial_distribution`` and one layout, and the whole group is placed with a
single density operation, so the members occupy exactly the same in-cell
positions -- and are therefore charge-neutral by construction, irrespective of
per-member momentum or temperature.

.. literalinclude:: ../snippets/selected_topics/multi_species.py
   :language: python
   :start-after: BEGIN-MULTI-SPECIES
   :end-before: END-MULTI-SPECIES

The whole :class:`~picongpu.picmi.multi_species.MultiSpecies` is one entry of
the simulation's ``species`` list, paired with a single layout in ``layouts``
(passed declaratively as shown above or via
:meth:`~picongpu.picmi.simulation.Simulation.add_species`). Its individual
members can be addressed by index or, if named, by name (e.g.
``multispecies["electrons"]``) and used elsewhere as needed.
The value at each position of ``proportions`` becomes the corresponding member's
``density_scale`` (its ``DensityRatio`` on the C++ level), so a
``proportions=[1.0, 1.0]`` ion/electron pair yields a neutral plasma.

.. note::

   Grouping is **structural**: the :class:`~picongpu.picmi.Simulation` stores
   each ``species`` entry as given. A whole ``MultiSpecies`` becomes one density
   operation covering all of its members; every standalone
   :class:`~picongpu.picmi.species.Species` entry becomes its own operation.
   There is therefore no implicit merging of look-alike species: to initialise
   species collectively, you must group them in a ``MultiSpecies``.

Composite particle pusher
^^^^^^^^^^^^^^^^^^^^^^^^^

To stream freely initially and use Higuera--Cary from step 6720 onwards:

.. code-block:: python

   electrons = picmi.Species(
       name="e",
       particle_type="electron",
       picongpu_composite_pusher=picmi.CompositePusher(
           first="free-streaming",
           second="Higuera-Cary",
           switch_step=6720,
       ),
   )

``first`` is active for ``currentStep < switch_step``; ``second`` is active
from ``switch_step`` onwards. ``switch_step`` is an absolute simulation step,
not a duration. It must be an integer from 0 through ``2**32 - 1`` (the C++
``uint32_t`` range); floats, strings and booleans are rejected. A switch at
zero selects the second pusher immediately. A switch beyond ``max_steps``
is retained, so a 1000-step run can still specify the switch at 6720.

The two pushers must be distinct because the C++ composite inherits from
both types. Each name uses the same spelling as ``Species.method``:
``"Boris"``, ``"Vay"``, ``"Higuera-Cary"``, ``"free-streaming"``,
``"LLRK4"``, ``"other:Acceleration"``, ``"other:Photon"`` or
``"other:Probe"``. Nested composites and arbitrary C++ expressions are not
supported. ``"Li"`` and ``"other:Axel"`` have no implementation in this
checkout and are rejected during conversion for a single pusher, or during
construction of a composite. These choices do not validate the physical
suitability of a pusher for a particular species.

When using ``picongpu_composite_pusher``, omit ``method`` or set it to
``None``. An explicit non-``None`` method, including ``"Boris"``, conflicts
with the composite and raises an error. Without a composite, omitting
``method`` or setting it to ``None`` still selects Boris.

The PICMI name ``"Higuera-Cary"`` maps to the C++ type
``particles::pusher::HigueraCary`` for both single and composite pushers.

Particle shape, interpolation and current deposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Species.particle_shape`` determines the shape used to gather grid fields at
particle positions and to deposit current. The generated definitions use a
separate shape alias for each species. For example, ``particle_shape="cubic"``
selects PQS for the species' shape flag, field interpolation and default
Esirkepov current deposition, even when another species uses ``"linear"``
(CIC). Omitting the species shape inherits the Simulation shape, falling back
to TSC if that too is unset.

The diagnostic-only ``"other:counter"`` shape is rejected during conversion:
it cannot provide the cached assignment functions used by field interpolation.
NGP remains valid with Esirkepov.

``picongpu_current_deposition`` is a PIConGPU-specific extension. ``"EZ"``
is an alias of ``"EmZ"``; both generate the canonical C++ name ``EmZ``.
NGP is rejected with EmZ because its support is only one point, leaving the
in-plane EmZ current-deposition loops empty. Current smoothing remains an
independent solver choice. For example:

.. code-block:: python

   electrons = picmi.Species(
       name="e", particle_type="electron", particle_shape="cubic",
       picongpu_current_deposition="EZ",
   )

.. _distributions:

Particle Distributions
----------------------

A distribution describes *where* the particles of a species are placed
(their density profile) and *how they move* initially.
All distributions take

* ``rms_velocity``:
  a 3D vector of thermal velocity spreads in m/s
  (they are converted to a temperature internally), and
* ``directed_velocity``:
  a 3D vector of a collective drift velocity in m/s.

The available distributions are:

:class:`~picongpu.picmi.distribution.UniformDistribution`
   A constant density throughout the box (``density`` in m⁻³).

   .. note::

      The ``lower_bound``/``upper_bound`` and ``fill_in`` parameters
      are not supported: setting them to non-default values raises an
      ``UnsupportedFeatureError`` at input-file generation, while the
      density fills the entire simulation box. For sub-volume densities
      use ``AnalyticDistribution``, ``GaussianDistribution`` or
      ``FoilDistribution`` instead.

:class:`~picongpu.picmi.distribution.GaussianDistribution`
   A constant-density region with Gaussian ramps at the front and the rear
   of the box (in ``y`` direction):
   ``center_front``/``center_rear`` and ``sigma_front``/``sigma_rear``
   give the position and width of the ramps,
   ``power`` the exponent (2 is Gaussian, 4 and up super-Gaussian),
   ``factor`` the (negative) scaling of the ramps,
   and ``vacuum_front`` the vacuum in front of the profile.

:class:`~picongpu.picmi.distribution.FoilDistribution`
   A thin foil of constant ``thickness`` at position ``front``
   (perpendicular to ``y``),
   with optional exponential pre- and post-plasma ramps
   (``exponential_pre_plasma_length``/``_cutoff`` and
   ``exponential_post_plasma_length``/``_cutoff``).

:class:`~picongpu.picmi.distribution.CylindricalDistribution`
   A cylinder of ``radius`` around the axis ``cylinder_axis``
   through the point ``center_position``,
   with an optional exponential pre-plasma ramp
   (``exponential_pre_plasma_length``/``_cutoff``).

:class:`~picongpu.picmi.distribution.AnalyticDistribution`
   A density given by an analytic expression
   (see :ref:`the functors page <functors>`).

:class:`~picongpu.picmi.distribution.GaussianBunchDistribution`
   A finite 3D Gaussian particle bunch (the PICMI-standard
   ``GaussianBunchDistribution``).
   It is described by the number of physical particles
   ``n_physical_particles``, the per-axis RMS size ``rms_bunch_size``
   and the ``centroid_position`` of the bunch,
   plus an optional rigid ``centroid_velocity`` (given as ``gamma * v``)
   and a thermal ``rms_velocity``.
   The correlated ``velocity_divergence`` is not supported
   (a non-zero value raises an ``UnsupportedFeatureError``), and
   because the bunch is inherently three-dimensional it is rejected on a
   2D grid (an ``UnsupportedFeatureError`` at input-file generation).

   .. literalinclude:: ../snippets/selected_topics/gaussian_bunch.py
      :language: python
      :start-at: bunch = picmi.GaussianBunchDistribution
      :end-before: electrons = picmi.Species

The reference density used to normalize the code units is
``simulation.picongpu_base_density`` (default ``1.0e25`` m⁻³).

Layouts
-------

The layout determines the positions of the particles *within* a cell.
It is given per species via the ``layouts`` list:

:class:`~picongpu.picmi.layout.PseudoRandomLayout`
  ``n_macroparticles_per_cell`` particles per cell at pseudo-random positions.
  This is the default choice for most simulations.

:class:`~picongpu.picmi.layout.GriddedLayout`
  A regular sub-grid of ``n_macroparticle_per_cell = [nx, ny, nz]``
  positions per cell (``nx * ny * nz`` particles per cell).
  Useful for well-resolved, low-noise configurations.

:class:`~picongpu.picmi.layout.OnePositionLayout`
  A single position per cell
  (``n_macroparticles_per_cell`` particles per cell, all at the same point,
  shifted by ``in_cell_offset`` in units of the cell size).

A species that starts out empty (``initial_distribution=None``,
e.g. electrons filled by ionization) gets the layout ``None``.
Giving a layout to a species without a distribution raises an error.
