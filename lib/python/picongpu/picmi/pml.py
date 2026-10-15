"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

from ..pypicongpu.pml import PMLParameters


class PML(PMLParameters):
    """PIConGPU PML coefficients, passed as grid.picongpu_pml.

    Set thickness through the grid's standard pml_cells parameter. Coefficient
    vectors always have three components (x, y, z), including for 2D grids.
    """
