"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import json
import math
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from picongpu import picmi
from picongpu.pypicongpu.pml import PML as BackendPML
from picongpu.pypicongpu.rendering import Renderer


PACKAGE = Path(picmi.__file__).resolve().parents[1]
TEMPLATE = PACKAGE / "templates/include/picongpu/param/fieldAbsorber.param.mustache"


def make_grid(dim=3, **overrides):
    parameters = dict(
        number_of_cells=[128] * dim,
        lower_bound=[0.0] * dim,
        upper_bound=[128e-7] * dim,
        lower_boundary_conditions=["open"] * dim,
        upper_boundary_conditions=["open"] * dim,
    )
    cls = picmi.Cartesian3DGrid if dim == 3 else picmi.Cartesian2DGrid
    return cls(**(parameters | overrides))


def make_simulation(grid, lasers=(), method="CKC", **solver_options):
    # Derive dt from CFL for non-reference solvers; avoid an unrelated exact
    # floating-point dt/CFL round-trip check in the existing simulation model.
    time_options = {"time_step_size": 5.15136e-17} if method == "CKC" else {}
    if method != "CKC":
        solver_options["cfl"] = 0.5
    return picmi.Simulation(
        solver=picmi.ElectromagneticSolver(grid=grid, method=method, **solver_options),
        **time_options,
        max_steps=1000,
        lasers=list(lasers),
        picongpu_moving_window_move_point=0.0,
    )


def gaussian(surface):
    return picmi.GaussianLaser(
        wavelength=0.8e-6,
        waist=1e-6,
        duration=10e-15,
        focal_position=[0.0, 1e-6, 0.0],
        centroid_position=[0.0, -1e-6, 0.0],
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        a0=1.0,
        picongpu_huygens_surface_positions=surface,
    )


def render_absorber(grid):
    context = {"grid": grid.get_as_pypicongpu().get_rendering_context()}
    return Renderer.get_rendered_template(Renderer.get_context_preprocessed(context), TEMPLATE.read_text())


def numbers(text, name):
    expression = re.search(rf"\b{name}(?:\[\d+\])*\s*=\s*([^;]+);", text)[1]
    return [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?", expression)]


def test_default_coefficients_match_cpp_defaults():
    rendered = render_absorber(make_grid())
    source = (PACKAGE.parents[2] / "include/picongpu/param/fieldAbsorber.param").read_text()
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    assert numbers(rendered, "NUM_CELLS") == [12] * 6
    assert numbers(source, "THICKNESS") == [12]
    for name in [
        "SIGMA_KAPPA_GRADING_ORDER",
        "SIGMA_OPT_MULTIPLIER",
        "KAPPA_MAX",
        "ALPHA_GRADING_ORDER",
        "ALPHA_MAX_SI",
    ]:
        assert numbers(rendered, name) == numbers(source, name)
    assert "{1.0e-3, 1.0e-3}" in rendered  # Preserve the exponential parameters as well.


@pytest.mark.parametrize(
    "field,value",
    [
        ("sigma_kappa_grading_order", -1),
        ("sigma_kappa_grading_order", float("nan")),
        ("sigma_opt_multiplier", -0.1),
        ("sigma_opt_multiplier", float("inf")),
        ("kappa_max", [0.9, 1, 1]),
        ("kappa_max", [1, 1]),
        ("kappa_max", [1, 1, float("inf")]),
        ("alpha_grading_order", -1),
        ("alpha_grading_order", float("nan")),
        ("alpha_max_si", [0, -1, 0]),
        ("alpha_max_si", [0, 0, float("nan")]),
        ("alpha_max_si", [0, 0]),
        ("thickness", 10),
    ],
)
def test_invalid_pml_coefficients(field, value):
    with pytest.raises(ValidationError):
        picmi.PML(**{field: value})


def test_zero_coefficients_and_unit_kappa_allowed():
    pml = picmi.PML(
        sigma_kappa_grading_order=0,
        sigma_opt_multiplier=0,
        kappa_max=[1, 1, 1],
        alpha_grading_order=0,
        alpha_max_si=[0, 0, 0],
    )
    assert make_grid(picongpu_pml=pml).get_as_pypicongpu().pml.sigma_opt_multiplier == 0


@pytest.mark.parametrize(
    "cells", [[-1, 1, 1], [1.5, 1, 1], [True, 1, 1], ["12", 1, 1], [2**32, 1, 1], [12, 12], [12, 12, 12, 12], []]
)
def test_invalid_pml_cells(cells):
    with pytest.raises(ValueError):
        make_grid(pml_cells=cells).get_as_pypicongpu()


@pytest.mark.parametrize("dim", [2, 3])
def test_pml_cells_map_both_faces_and_2d_has_no_z(dim):
    backend = make_grid(dim, pml_cells=[7, 11, 13][:dim]).get_as_pypicongpu()
    assert backend.pml.num_cells == ((7, 7), (11, 11), (13, 13) if dim == 3 else (0, 0))
    assert numbers(render_absorber(make_grid(dim, pml_cells=[7, 11, 13][:dim])), "NUM_CELLS") == (
        [7, 7, 11, 11, 13, 13] if dim == 3 else [7, 7, 11, 11, 0, 0]
    )


def test_periodic_thickness_handling():
    boundaries = dict(
        lower_boundary_conditions=["periodic", "open", "periodic"],
        upper_boundary_conditions=["periodic", "open", "periodic"],
    )
    with pytest.raises(ValueError, match="zero on periodic"):
        make_grid(**boundaries, pml_cells=[12, 12, 12]).get_as_pypicongpu()
    assert make_grid(**boundaries, pml_cells=[0, 12, 0]).get_as_pypicongpu().pml.num_cells == ((0, 0), (12, 12), (0, 0))
    assert make_grid(**boundaries, picongpu_pml=picmi.PML()).get_as_pypicongpu().pml.num_cells == (
        (0, 0),
        (12, 12),
        (0, 0),
    )


def test_all_periodic_or_zero_thickness():
    grid = make_grid(
        lower_boundary_conditions=["periodic"] * 3, upper_boundary_conditions=["periodic"] * 3, picongpu_pml=picmi.PML()
    )
    assert grid.get_as_pypicongpu().pml.num_cells == ((0, 0),) * 3
    assert make_grid(pml_cells=[0, 0, 0]).get_as_pypicongpu().pml.num_cells == ((0, 0),) * 3


@pytest.mark.parametrize(
    "options",
    [
        dict(pml_cells=[65, 0, 0]),  # Opposing layers on a single rank overlap.
        dict(pml_cells=[65, 0, 0], picongpu_n_gpus=[2, 1, 1]),  # Exceeds a 64-cell rank.
        dict(pml_cells=[33, 0, 0], picongpu_n_gpus=[2, 1, 1], picongpu_grid_dist=[[32, 96], [128], [128]]),
    ],
)
def test_pml_must_fit_local_domains(options):
    with pytest.raises(ValueError, match="local domain"):
        make_grid(**options).get_as_pypicongpu()


def test_backend_rejects_invalid_geometry():
    backend = make_grid().get_as_pypicongpu()
    parameters = dict(backend)
    parameters["cell_size_si"] = parameters.pop("cell_size")
    with pytest.raises(ValueError, match="local domain"):
        type(backend)(**(parameters | {"pml": BackendPML(num_cells=((129, 0), (0, 0), (0, 0)))}))


def test_to_2d_preserves_pml_coefficients_and_xy_thickness():
    original = make_grid(pml_cells=[7, 11, 13], picongpu_pml=picmi.PML(kappa_max=[4, 1, 4]))
    reduced = original.to_2d()
    assert reduced.pml_cells == [7, 11]
    assert reduced.picongpu_pml.kappa_max == (4, 1, 4)
    assert reduced.get_as_pypicongpu().pml.num_cells == ((7, 7), (11, 11), (0, 0))
    assert original.pml_cells == [7, 11, 13]


@pytest.mark.parametrize(
    "method,options,margin",
    [
        ("Yee", {}, 1),
        ("CKC", {}, 1),
        ("Lehe", {}, 2),
        ("other:ArbitraryOrderFDTD", {"stencil_order": [4, 4, 4]}, 2),
    ],
)
def test_huygens_clearance_includes_solver_margin(method, options, margin):
    grid = make_grid(pml_cells=[10, 10, 10])
    surface = [[16, -16], [10 + margin - 1, 100000000], [16, -16]]
    make_simulation(grid, [gaussian(surface)], method, **options).get_as_pypicongpu()
    surface[1][0] -= 1
    with pytest.raises(ValueError, match="too close to the PML"):
        make_simulation(grid, [gaussian(surface)], method, **options).get_as_pypicongpu()


def reference_simulation():
    cells = [3584, 3584, 1792]
    cell_size = [6.177353e-8, 3.088676e-8, 6.177353e-8]
    surface = [[116, -116], [116, 100000000], [116, -116]]
    grid = make_grid(
        number_of_cells=cells,
        upper_bound=[n * dx for n, dx in zip(cells, cell_size)],
        picongpu_n_gpus=[8, 16, 8],
        picongpu_super_cell_size=[8, 8, 8],
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
            focal_position=[0, 68.5e-6, 0],
            centroid_position=[0, 0, 0],
            laserIncidenceAngle=sign * math.radians(5),
            # Base dev's TWTS vector metadata is not yet corrected by PR-01.
            # Choose an angle that makes that legacy vector unit length; this
            # fixture tests PML clearance, not polarization physics.
            polarizationAngle=math.pi / 4,
            focus_lateral_offset_si=sign * 12e-6,
            windowStart=0,
            windowEnd=10000000,
            windowLength=10000,
            picongpu_huygens_surface_positions=surface,
        )
        for sign in [1, -1]
    ]
    simulation = make_simulation(grid, lasers, source_smoother=picmi.BinomialSmoother())
    return simulation


@pytest.mark.parametrize("side", [0, 1])
def test_twts_z_injection_must_clear_pml(side):
    sim = reference_simulation()
    for laser in sim.lasers:
        laser.picongpu_huygens_surface_positions[2][side] = 109 if side == 0 else -109
    with pytest.raises(ValueError, match="zMin|zMax"):
        sim.get_as_pypicongpu()


@pytest.mark.parametrize("custom", [False, True])
def test_pml_setup_generation(custom, tmp_path):
    """Generate Python input only; reference plasma profile and outputs are still pending."""
    simulation = reference_simulation() if custom else make_simulation(make_grid())
    output = tmp_path / "setup"
    simulation.write_input_file(
        output, cmake="-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'"
    )
    cfg = (output / "etc/picongpu/N.cfg").read_text()
    assert 'TBG_fieldAbsorber="--fieldAbsorber pml"' in cfg
    assert "!TBG_fieldAbsorber" in cfg.split('TBG_programParams="', 1)[1]
    rendered = (output / "include/picongpu/param/fieldAbsorber.param").read_text()
    assert rendered.rstrip().endswith("} // namespace picongpu")
    assert rendered.count("namespace picongpu") == 2  # Declaration and closing comment.
    assert "{{" not in rendered
    assert numbers(rendered, "NUM_CELLS") == [110 if custom else 12] * 6
    assert numbers(rendered, "SIGMA_KAPPA_GRADING_ORDER") == [2.5 if custom else 4.0]
    assert numbers(rendered, "SIGMA_OPT_MULTIPLIER") == [1.25 if custom else 1.0]
    assert numbers(rendered, "KAPPA_MAX") == ([4, 1, 4] if custom else [1, 1, 1])
    assert numbers(rendered, "ALPHA_GRADING_ORDER") == [1]
    assert numbers(rendered, "ALPHA_MAX_SI") == ([0, 0, 0] if custom else [0.2, 0.2, 0.2])
    for axis in "xyz":
        assert f"0.8 * (SIGMA_KAPPA_GRADING_ORDER + 1.0) / (sim.si.getZ0() * sim.si.getCellSize().{axis}())" in rendered
    for i in range(3):
        assert f"SIGMA_OPT_SI[{i}] * SIGMA_OPT_MULTIPLIER" in rendered
    context = json.loads((output / "metadata/pypicongpu_rendering_context.json").read_text())
    assert context["time_steps"] == 1000
    if custom:
        assert context["grid"]["cell_cnt"] == dict(x=3584, y=3584, z=1792)
        assert context["grid"]["gpu_cnt"] == dict(x=8, y=16, z=8)
        assert context["grid"]["cell_size"] == pytest.approx(dict(x=6.177353e-8, y=3.088676e-8, z=6.177353e-8))
        incident = (output / "include/picongpu/param/incidentField.param").read_text()
        assert re.search(r"\{\s*116,\s*100000000\s*\}", incident)
        assert len(context["laser"]) == 2
        assert all(laser["time_offset_si"] > 0 for laser in context["laser"])
        assert "--currentInterpolation binomial" in cfg
    assert not (output / ".build").exists()
