"""PICMI migration of the active tweac5DegreeSetup (tasks 9a/9b).

Generate input only: python picmi_setup.py /path/to/new-output-directory
The Binning eSpec diagnostic remains pending task 9c. No C++ template overrides.
"""

import argparse
import math
from datetime import timedelta
from pathlib import Path

from sympy import Float, Or, Piecewise, exp

from picongpu import picmi
from picongpu.picmi.constants import GiB, MiB
from picongpu.picmi import diagnostics as diag
from picongpu.picmi.diagnostics.backend_config import RangeSpec


NUMBER_OF_CELLS = (3584, 3584, 1792)
CELL_SIZE_SI = (6.177353e-8, 3.088676e-8, 6.177353e-8)
TIMESTEP_SI = 5.15136e-17
BASE_DENSITY_SI = 1e25
DEFAULT_STEPS = 1000
CMAKE_FLAGS = "-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'"

# Fixed reference dimensions: changing the run length must not rescale the target.
VACUUM_X_SI = 200 * CELL_SIZE_SI[0]
VACUUM_Y_SI = 6720 * CELL_SIZE_SI[1]
VACUUM_Z_SI = 200 * CELL_SIZE_SI[2]
RAMP_LENGTH_SI = 47.2e-6 / 2
RAMP_SIGMA_SI = RAMP_LENGTH_SI / 4
X_PLATEAU_START = VACUUM_X_SI + RAMP_LENGTH_SI
X_PLATEAU_END = NUMBER_OF_CELLS[0] * CELL_SIZE_SI[0] - X_PLATEAU_START
Z_PLATEAU_START = VACUUM_Z_SI + RAMP_LENGTH_SI
Z_PLATEAU_END = NUMBER_OF_CELLS[2] * CELL_SIZE_SI[2] - Z_PLATEAU_START
LONGITUDINAL_RAMP_SI = 100e-6


def add_diagnostics(simulation, electrons, max_steps):
    """Active config.cfg outputs; the separate Binning eSpec follows in 9c."""
    period = diag.TS[::812]
    simulation.add_diagnostic(diag.FieldEnergyMonitor(period=period))
    simulation.add_diagnostic(diag.ParticleEnergy(species=electrons, period=period))
    mc = electrons.picongpu_get_mass_si() * picmi.constants.c
    for position, momentum, lower, upper in [
        ("x", "px", -20, 20),
        ("x", "pz", -20, 20),
        ("y", "px", -20, 20),
        ("y", "pz", -20, 20),
        ("y", "py", -50, 2000),
        ("x", "py", -50, 2000),
        ("z", "py", -50, 2000),
    ]:
        simulation.add_diagnostic(
            diag.PhaseSpace(
                species=electrons,
                period=period,
                spatial_coordinate=position,
                momentum_coordinate=momentum,
                min_momentum=lower * mc,
                max_momentum=upper * mc,
                picongpu_extension="bp",
            )
        )
    # Only the compression object is active in config.cfg, not its unused
    # BP5 engine/buffer configuration. Embed it reproducibly in each TOML file.
    compression = {
        "adios2": {
            "dataset": {
                "operators": [
                    {
                        "type": "blosc",
                        "parameters": {"clevel": "1", "compressor": "zstd", "doshuffle": "BLOSC_BITSHUFFLE"},
                    }
                ]
            }
        }
    }
    for filename, interval, bounds in [
        ("simOutput", 8120, (None, None, None)),
        ("simOutputX", 812, ((1780, 1804), None, None)),
        ("simOutputY", 812, (None, (1780, 1804), None)),
        ("simOutputZ", 812, (None, None, (884, 908))),
    ]:
        options = diag.OpenPMDConfig(
            file=filename,
            infix="_%T",
            ext="bp",
            range=RangeSpec(*bounds),
            data_preparation_strategy="mappedMemory",
            backend_config=compression,
        )
        schedule = diag.TS[0:max_steps:interval]
        for field in ("E", "B"):
            simulation.add_diagnostic(diag.NativeFieldDump(fieldname=field, period=schedule, options=options))
        # The reference inherits fileOutput.param: fields_all means E, B and
        # these two built-in electron fields. Explicit sources exclude J.
        for attribute in ("ChargeDensity", "EnergyDensity"):
            simulation.add_diagnostic(
                diag.DerivedFieldDump(species=electrons, functor=attribute, period=schedule, options=options)
            )
    simulation.add_diagnostic(
        diag.Checkpoint(
            period=diag.TS[0:max_steps:50000],
            openPMD={"infix": "_%T", "dataPreparationStrategy": "mappedMemory"},
        )
    )


def density_si(x, y, z):
    """Active density.param translated to a SymPy expression, in particles/m³.

    Coordinates are absolute SI coordinates, including moving-window slides.
    At a transverse vacuum boundary the Gaussian tail is retained; only points
    strictly outside are cut. The longitudinal ramp is zero at its start.
    """

    def transverse(coordinate, vacuum, start, end):
        # Keep enough digits for binary64 boundaries in both lambdify and C++.
        # In particular, rounding a hard cut must not remove its boundary point.
        upper_cut = Float(end + RAMP_LENGTH_SI, 17)
        vacuum, start, end, sigma = [Float(v, 17) for v in (vacuum, start, end, RAMP_SIGMA_SI)]
        return Piecewise(
            (0, Or(coordinate < vacuum, coordinate > upper_cut)),
            (exp(-0.5 * ((coordinate - start) / sigma) ** 2), coordinate < start),
            (exp(-0.5 * ((coordinate - end) / sigma) ** 2), coordinate > end),
            (1, True),
        )

    longitudinal = Piecewise(
        (0, y < Float(VACUUM_Y_SI, 17)),
        (
            (y - Float(VACUUM_Y_SI, 17)) / Float(LONGITUDINAL_RAMP_SI, 17),
            y < Float(VACUUM_Y_SI + LONGITUDINAL_RAMP_SI, 17),
        ),
        (1, True),
    )
    return (
        BASE_DENSITY_SI
        * transverse(x, VACUUM_X_SI, X_PLATEAU_START, X_PLATEAU_END)
        * transverse(z, VACUUM_Z_SI, Z_PLATEAU_START, Z_PLATEAU_END)
        * longitudinal
    )


def build_simulation(max_steps=DEFAULT_STEPS):
    """Construct the setup without generating, compiling or running it."""
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    grid = picmi.Cartesian3DGrid(
        number_of_cells=list(NUMBER_OF_CELLS),
        lower_bound=[0.0, 0.0, 0.0],
        upper_bound=[n * dx for n, dx in zip(NUMBER_OF_CELLS, CELL_SIZE_SI)],
        lower_boundary_conditions=["open"] * 3,
        upper_boundary_conditions=["open"] * 3,
        picongpu_n_gpus=[8, 16, 8],
        picongpu_super_cell_size=[8, 8, 8],
        guard_cells=[8, 8, 8],
        pml_cells=[110, 110, 110],
        picongpu_pml=picmi.PML(
            sigma_kappa_grading_order=2.5,
            sigma_opt_multiplier=1.25,
            kappa_max=[4, 1, 4],
            alpha_grading_order=1,
            alpha_max_si=[0, 0, 0],
        ),
    )
    lasers = [
        picmi.TWTSLaser(
            wavelength=0.8e-6,
            waist=1e-6,
            duration=25e-15 / 2.354820045,
            a0=1.25,
            laserIncidenceAngle=sign * math.radians(5),
            polarizationAngle=sign * math.radians(45.10716816968629),
            focal_position=[0, 68.5e-6, 0],
            centroid_position=[0, 68.5e-6, 0],
            focus_lateral_offset_si=sign * 12e-6,
            beta0=1,
            windowStart=0,
            windowEnd=10000000,
            windowLength=10000,
            picongpu_huygens_surface_positions=[[116, -116], [116, 100000000], [116, -116]],
        )
        for sign in (1, -1)
    ]
    simulation = picmi.Simulation(
        solver=picmi.ElectromagneticSolver(grid=grid, method="CKC", source_smoother=picmi.BinomialSmoother()),
        time_step_size=TIMESTEP_SI,
        max_steps=max_steps,
        lasers=lasers,
        picongpu_base_density=BASE_DENSITY_SI,
        picongpu_typical_ppc=1,
        picongpu_min_weighting=0.1,
        picongpu_moving_window_move_point=0.0,
        picongpu_precision=32,
        picongpu_walltime=timedelta(hours=2),
        picongpu_memory_config=picmi.MemoryConfig(
            reserved_gpu_memory_size=4 * GiB,
            bytes_exchange_x=32 * MiB,
            bytes_exchange_y=320 * MiB,
            bytes_exchange_z=32 * MiB,
            bytes_edges=2 * MiB,
            bytes_corner=2 * MiB,
            ref_local_dom_size=(0, 0, 0),
            dir_scaling_factor=(0, 0, 0),
            field_tmp_support_gather_communication=True,
        ),
    )
    simulation.add_species(
        picmi.Species(
            name="e",
            particle_type="electron",
            particle_shape="cubic",
            picongpu_current_deposition="EmZ",
            picongpu_composite_pusher=picmi.CompositePusher(
                first="free-streaming", second="Higuera-Cary", switch_step=6720
            ),
            initial_distribution=picmi.AnalyticDistribution(density_function=density_si),
        ),
        layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=1),
    )
    add_diagnostics(simulation, simulation.species[0], max_steps)
    return simulation


def generate(output_directory, max_steps=DEFAULT_STEPS):
    """Write a new input directory outside the reference; never build or run."""
    output = Path(output_directory).resolve()
    if output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Generate outside tweac5DegreeSetup to keep the reference directory intact.")
    build_simulation(max_steps).write_input_file(output, cmake=CMAKE_FLAGS)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path, help="new directory outside the reference setup")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="default: active 1000-step run")
    args = parser.parse_args()
    output = generate(args.output_directory, args.steps)
    print(f"Generated input at {output}. Binning eSpec remains pending task 9c; final audit is task 9d.")


if __name__ == "__main__":
    main()
