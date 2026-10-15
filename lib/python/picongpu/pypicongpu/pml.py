"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer


CellCount = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Kappa = Annotated[float, Field(ge=1, allow_inf_nan=False)]


def _vector_context(value) -> dict[str, float]:
    return dict(zip("xyz", value))


class PMLParameters(BaseModel):
    """PML coefficients; defaults match fieldAbsorber.param. SI units for alpha."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    sigma_kappa_grading_order: Nonnegative = 4.0
    sigma_opt_multiplier: Nonnegative = 1.0
    kappa_max: Annotated[tuple[Kappa, Kappa, Kappa], PlainSerializer(_vector_context)] = (1.0, 1.0, 1.0)
    alpha_grading_order: Nonnegative = 1.0
    alpha_max_si: Annotated[tuple[Nonnegative, Nonnegative, Nonnegative], PlainSerializer(_vector_context)] = (
        0.2,
        0.2,
        0.2,
    )


def _thickness_context(value) -> dict[str, dict[str, int]]:
    return {axis: dict(zip(("negative", "positive"), pair)) for axis, pair in zip("xyz", value)}


class PML(PMLParameters):
    """Backend PML configuration with thickness on each of the six faces."""

    num_cells: Annotated[
        tuple[tuple[CellCount, CellCount], tuple[CellCount, CellCount], tuple[CellCount, CellCount]],
        PlainSerializer(_thickness_context),
    ] = ((12, 12), (12, 12), (12, 12))


def validate_grid_pml(grid):
    """Check explicitly configured PML against periodicity and slab allocation."""
    if grid.pml is None:
        return
    for axis in range(grid.sim_dim):
        negative, positive = grid.pml.num_cells[axis]
        if grid.boundary_condition[axis].get_cfg_str() == "1":
            if negative or positive:
                raise ValueError("PML thickness must be zero on periodic axes.")
            continue
        local_sizes = (
            grid.grid_dist[axis] if grid.grid_dist is not None else [grid.cell_cnt[axis] // grid.gpu_cnt[axis]]
        )
        # C++ allocates slabs for global thickness on every rank, even interior ranks.
        if max(negative, positive) > min(local_sizes):
            raise ValueError("PML thickness exceeds the smallest local domain on axis " + "xyz"[axis])
        if grid.gpu_cnt[axis] == 1 and negative + positive > grid.cell_cnt[axis]:
            raise ValueError("Opposing PML layers exceed the local domain on axis " + "xyz"[axis])
    if grid.sim_dim == 2 and grid.pml.num_cells[2] != (0, 0):
        raise ValueError("PML thickness must be zero on the absent z axis in 2D.")


def validate_pml_injection(simulation):
    """Match the absorber-clearance check on faces used by incidentField.param.

    This applies to explicitly configured PML. It does not clamp inactive
    surfaces, notably the distant YMax used by moving-window setups.
    """
    from .field_solver import ArbitraryOrderFDTDSolver, LeheSolver, NoneSolver

    grid = simulation.grid
    if grid.pml is None or not simulation.laser:
        return
    if isinstance(simulation.solver, NoneSolver):
        raise ValueError("Incident-field injection with configured PML requires an FDTD solver.")
    # The template always injects on YMin; TWTS additionally uses one Z face.
    faces = {(1, 0)}
    for laser in simulation.laser:
        if getattr(laser, "type_twts", False) and grid.sim_dim == 3:
            faces.add((2, 0 if laser.laserIncidenceAnglePositive else 1))
    positions = simulation.laser[0].huygens_surface_positions
    for axis, side in faces:
        margin = 1
        if isinstance(simulation.solver, ArbitraryOrderFDTDSolver):
            margin = simulation.solver.neighbors
        elif isinstance(simulation.solver, LeheSolver) and axis == 1:
            margin = 2
        position = positions[axis][side]
        offset = position if side == 0 else (grid.cell_cnt[axis] - position if position > 0 else -position)
        required = grid.pml.num_cells[axis][side] + margin - 1
        if offset < required:
            raise ValueError(
                f"Huygens surface on {'xyz'[axis]}{'Min' if side == 0 else 'Max'} is too close to the PML: "
                f"offset {offset} cells, required at least {required}."
            )
