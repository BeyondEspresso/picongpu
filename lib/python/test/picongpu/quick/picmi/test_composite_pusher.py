"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from picongpu import picmi
from picongpu.pypicongpu.rendering import Renderer
from picongpu.pypicongpu.species import species as backend


PACKAGE = Path(picmi.__file__).resolve().parents[1]
SINGLE_PUSHERS = {
    "Boris": "Boris",
    "Vay": "Vay",
    "Higuera-Cary": "HigueraCary",
    "free-streaming": "Free",
    "LLRK4": "ReducedLandauLifshitz",
    "other:Acceleration": "Acceleration",
    "other:Photon": "Photon",
    "other:Probe": "Probe",
}


def composite(**overrides):
    parameters = dict(first="free-streaming", second="Higuera-Cary", switch_step=6720)
    return picmi.CompositePusher(**(parameters | overrides))


def render_species(species):
    context = species.get_as_pypicongpu().get_rendering_context()
    template = (PACKAGE / "templates/include/picongpu/param/speciesDefinition.param.mustache").read_text()
    return Renderer.get_rendered_template(Renderer.get_context_preprocessed({"species": [context]}), template)


@pytest.mark.parametrize("step", [0, 6720, 2**32 - 1])
def test_composite_conversion_and_rendering(step):
    species = picmi.Species(particle_type="electron", picongpu_composite_pusher=composite(switch_step=step))
    converted = species.get_as_pypicongpu()
    assert species.method is None
    assert converted.pusher == backend.CompositePusher(
        first=backend.Pusher.Free, second=backend.Pusher.Higuera, switch_step=step
    )
    assert (
        "particlePusher<particles::pusher::Composite<particles::pusher::Free, "
        "particles::pusher::HigueraCary, "
        f"particles::pusher::CompositeBinarySwitchActivationFunctor<{step}>>>,"
    ) in render_species(species)
    # The structured selection survives JSON serialization and schema validation.
    context = converted.get_rendering_context()
    assert context["pusher"] == {"first": "Free", "second": "HigueraCary", "switch_step": step}
    assert backend.Species.model_validate(context).pusher == converted.pusher


@pytest.mark.parametrize("step", [-1, 2**32, 1.5, 6720.0, "6720", True, False, None])
@pytest.mark.parametrize("layer", ["picmi", "backend"])
def test_invalid_switch_step(step, layer):
    with pytest.raises(ValidationError):
        if layer == "picmi":
            composite(switch_step=step)
        else:
            backend.CompositePusher(first=backend.Pusher.Free, second=backend.Pusher.Higuera, switch_step=step)


@pytest.mark.parametrize("field", ["first", "second"])
@pytest.mark.parametrize("method", ["Li", "other:Axel", "other:Missing", "particles::pusher::Free", "Composite"])
def test_unsupported_composite_method(field, method):
    with pytest.raises(ValidationError, match="does not support pusher method"):
        composite(**{field: method})


@pytest.mark.parametrize("layer", ["picmi", "backend"])
def test_duplicate_pusher_types_rejected(layer):
    with pytest.raises(ValidationError, match="distinct pushers"):
        if layer == "picmi":
            composite(second="free-streaming")
        else:
            backend.CompositePusher(first=backend.Pusher.Free, second=backend.Pusher.Free, switch_step=0)


@pytest.mark.parametrize("layer", ["picmi", "backend"])
def test_nested_composites_and_extra_options_rejected(layer):
    cls = picmi.CompositePusher if layer == "picmi" else backend.CompositePusher
    first, second = ("free-streaming", "Higuera-Cary") if layer == "picmi" else ("Free", "HigueraCary")
    inner = cls(first=first, second=second, switch_step=1)
    with pytest.raises(ValidationError):
        cls(first=inner, second=second, switch_step=2)
    with pytest.raises(ValidationError, match="Extra inputs"):
        cls(first=first, second=second, switch_step=2, switch_time=1)


@pytest.mark.parametrize("method", ["Boris", "Vay", "free-streaming", "Higuera-Cary"])
def test_method_conflict(method):
    with pytest.raises(ValidationError, match="cannot both be specified"):
        picmi.Species(particle_type="electron", method=method, picongpu_composite_pusher=composite())


def test_method_none_and_later_conflict():
    species = picmi.Species(particle_type="electron", method=None, picongpu_composite_pusher=composite())
    assert isinstance(species.get_as_pypicongpu().pusher, backend.CompositePusher)
    with pytest.raises(ValidationError, match="cannot both be specified"):
        species.method = "Boris"
    # The updated PICMI base rolls back failed assignments atomically.
    assert species.method is None
    assert isinstance(species.get_as_pypicongpu().pusher, backend.CompositePusher)
    # model_copy(update=...) deliberately bypasses validation; conversion must
    # still refuse a conflicting state introduced through that interface.
    conflicting = species.model_copy(update={"method": "Boris"})
    with pytest.raises(ValueError, match="cannot both be specified"):
        conflicting.get_as_pypicongpu()


@pytest.mark.parametrize("method,cpp_name", SINGLE_PUSHERS.items())
def test_single_pusher_rendering(method, cpp_name):
    species = picmi.Species(particle_type="electron", method=method)
    assert f"particlePusher<particles::pusher::{cpp_name}>," in render_species(species)
    assert species.get_as_pypicongpu().get_rendering_context()["pusher"] == cpp_name


@pytest.mark.parametrize("parameters", [{}, {"method": None}, {"picongpu_composite_pusher": None}])
def test_default_pusher_is_boris(parameters):
    species = picmi.Species(particle_type="electron", **parameters)
    assert "particlePusher<particles::pusher::Boris>," in render_species(species)


@pytest.mark.parametrize("step", [0, 6720])
def test_composite_setup_generation(step, tmp_path):
    """Generate on the active reference grid without compiling or running it.

    Only task 3 is compared here. Interpolation, deposition, PML, density profile,
    lasers and diagnostics are outside this isolated pusher acceptance case.
    """
    cells = [3584, 3584, 1792]
    cell_size = [6.177353e-8, 3.088676e-8, 6.177353e-8]
    grid = picmi.Cartesian3DGrid(
        number_of_cells=cells,
        lower_bound=[0.0] * 3,
        upper_bound=[n * dx for n, dx in zip(cells, cell_size)],
        lower_boundary_conditions=["open"] * 3,
        upper_boundary_conditions=["open"] * 3,
        picongpu_n_gpus=[8, 16, 8],
        picongpu_super_cell_size=[8, 8, 8],
    )
    electrons = picmi.Species(
        name="e",
        particle_type="electron",
        particle_shape="cubic",
        initial_distribution=picmi.UniformDistribution(density=1e25),
        picongpu_composite_pusher=composite(switch_step=step),
    )
    simulation = picmi.Simulation(
        solver=picmi.ElectromagneticSolver(grid=grid, method="CKC", source_smoother=picmi.BinomialSmoother()),
        time_step_size=5.15136e-17,
        max_steps=1000,
        picongpu_moving_window_move_point=0.0,
        species=[electrons],
        layouts=[picmi.PseudoRandomLayout(n_macroparticles_per_cell=1)],
    )
    output = tmp_path / "setup"
    simulation.write_input_file(
        output, cmake="-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'"
    )
    context = json.loads((output / "metadata/pypicongpu_rendering_context.json").read_text())
    assert context["time_steps"] == 1000
    assert context["grid"]["cell_cnt"] == dict(zip("xyz", cells))
    assert context["grid"]["gpu_cnt"] == dict(zip("xyz", [8, 16, 8]))
    rendered = (output / "include/picongpu/param/speciesDefinition.param").read_text()
    cpp_type = (
        "particles::pusher::Composite<particles::pusher::Free,particles::pusher::HigueraCary,"
        f"particles::pusher::CompositeBinarySwitchActivationFunctor<{step}>>"
    )
    assert f"particlePusher<{cpp_type}>" in re.sub(r"\s+", "", rendered)
    assert not (output / ".build").exists()


def test_all_backend_pushers_have_cpp_definitions():
    # Audit names against this checkout's actual C++ declarations (no compilation).
    root = PACKAGE.parents[2]
    definitions = (root / "include/picongpu/unitless/pusher.unitless").read_text()
    assert {p.value for p in backend.Pusher} == set(SINGLE_PUSHERS.values())
    for pusher in backend.Pusher:
        assert re.search(rf"\bstruct {pusher.value}\b", definitions)
