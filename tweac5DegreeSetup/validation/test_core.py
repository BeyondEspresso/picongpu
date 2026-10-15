"""Task 9a acceptance: Python generation and source comparisons only."""

import ast
import importlib.util
import json
import math
import operator
import re
from pathlib import Path

import numpy as np
import pytest
from sympy import lambdify, symbols

from picongpu import picmi


REFERENCE = Path(__file__).resolve().parents[1]
REPO = REFERENCE.parent


def load_setup():
    spec = importlib.util.spec_from_file_location("tweac_core", REFERENCE / "picmi_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_setup()


def source(name, root=REFERENCE):
    text = (root / "include/picongpu/param" / name).read_text()
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def expression(text, name):
    return re.search(rf"\b{name}(?:\[\d+\])*\s*=\s*([^;]+);", text)[1].strip()


def arithmetic(text, names=None):
    """Evaluate only numeric C++ arithmetic, never arbitrary source code."""
    names = names or {}
    text = re.sub(r"(?<=\d)(?:ULL|u)\b", "", text)
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        raise ValueError(f"Unsupported arithmetic: {text}")

    return evaluate(ast.parse(text, mode="eval").body)


def reference_density_constants():
    simulation = source("simulation.param")
    constants = {
        axis: arithmetic(expression(simulation, name))
        for axis, name in zip("xyz", ["CELL_WIDTH_SI", "CELL_HEIGHT_SI", "CELL_DEPTH_SI"])
    }
    for name, expr in re.findall(r"constexpr double (\w+)\s*=\s*([^;]+);", source("density.param")):
        expr = re.sub(r"sim.si.getCellSize\(\).([xyz])\(\)", r"\1", expr)
        constants[name] = arithmetic(expr, constants)
    return constants


def reference_density_fraction(x, y, z, c):
    """Independent imperative translation of active density.param, float64.

    Tests analytic agreement; the C++ float_X intermediate rounding is not
    reproduced, and no C++ or GPU evaluation is performed.
    """
    if y < c["VACUUM_Y_SI"]:
        return 0.0
    value = 1.0
    for coordinate, axis in [(x, "X"), (z, "Z")]:
        if coordinate < c[f"VACUUM_{axis}_SI"] or coordinate > c[f"{axis}_PLAT_END"] + c[f"{axis}_RAMP_LENGTH"]:
            return 0.0
        if coordinate < c[f"{axis}_PLAT_START"]:
            exponent = abs((coordinate - c[f"{axis}_PLAT_START"]) / c[f"{axis}_SIGMA"])
            value *= math.exp(-0.5 * exponent * exponent)
        elif coordinate > c[f"{axis}_PLAT_END"]:
            exponent = abs((coordinate - c[f"{axis}_PLAT_END"]) / c[f"{axis}_SIGMA"])
            value *= math.exp(-0.5 * exponent * exponent)
    if y < c["VACUUM_Y_SI"] + c["INITIAL_UPRAMP_SI"]:
        return value * ((y - c["VACUUM_Y_SI"]) / c["INITIAL_UPRAMP_SI"])
    return value


def test_density_cuts_transitions_and_combined_ramps():
    c = reference_density_constants()
    coordinates = []
    for axis in "XZ":
        boundaries = [
            c[f"VACUUM_{axis}_SI"],
            c[f"{axis}_PLAT_START"],
            c[f"{axis}_PLAT_END"],
            c[f"{axis}_PLAT_END"] + c[f"{axis}_RAMP_LENGTH"],
        ]
        epsilon = c[axis.lower()] * 1e-6
        coordinates.append(
            [v + offset for v in boundaries for offset in [-epsilon, 0, epsilon]]
            + [(boundaries[1] + boundaries[2]) / 2]
        )
    ys = [
        0,
        c["VACUUM_Y_SI"] - c["y"],
        c["VACUUM_Y_SI"],
        c["VACUUM_Y_SI"] + 0.5 * c["y"],
        c["VACUUM_Y_SI"] + 0.5 * c["INITIAL_UPRAMP_SI"],
        c["VACUUM_Y_SI"] + c["INITIAL_UPRAMP_SI"] - c["y"],
        c["VACUUM_Y_SI"] + c["INITIAL_UPRAMP_SI"],
        1e-3,
    ]
    points = np.array([(x, y, z) for x in coordinates[0] for y in ys for z in coordinates[1]])
    evaluate = lambdify(symbols("x y z"), setup.density_si(*symbols("x y z")) / setup.BASE_DENSITY_SI, "numpy")
    actual = evaluate(*points.T)
    expected = [reference_density_fraction(*p, c) for p in points]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-15)
    assert len(points) == 1352


def test_initial_box_vacuum_and_later_window_cutoff():
    evaluate = lambdify(symbols("x y z"), setup.density_si(*symbols("x y z")), "numpy")
    x = setup.NUMBER_OF_CELLS[0] * setup.CELL_SIZE_SI[0] / 2
    z = setup.NUMBER_OF_CELLS[2] * setup.CELL_SIZE_SI[2] / 2
    assert evaluate(x, setup.NUMBER_OF_CELLS[1] * setup.CELL_SIZE_SI[1], z) == 0
    counts = []
    for i in range(4):
        y = (6720 + i + 0.5) * setup.CELL_SIZE_SI[1]
        real_count = float(evaluate(x, y, z)) * math.prod(setup.CELL_SIZE_SI)
        counts.append((math.floor(min(1, real_count / 0.1)), math.floor(min(1, real_count / 1.0))))
    assert counts == [(1, 0), (1, 0), (1, 0), (1, 1)]


def test_import_and_construction_do_not_generate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Import or construction attempted generation/build/run")

    for name in ["write_input_file", "run", "step", "picongpu_run"]:
        monkeypatch.setattr(picmi.Simulation, name, forbidden)
    imported = load_setup()
    sim = imported.build_simulation()
    assert sim.max_steps == 1000
    assert sim.diagnostics
    assert imported.build_simulation(230000).max_steps == 230000
    with pytest.raises(ValueError, match="outside"):
        imported.generate(REFERENCE / "generated")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("tweac-core") / "setup"
    setup.generate(output)
    return output, json.loads((output / "metadata/pypicongpu_rendering_context.json").read_text())


def test_generated_core_and_cold_species(generated):
    output, context = generated
    for name in [
        "CELL_WIDTH_SI",
        "CELL_HEIGHT_SI",
        "CELL_DEPTH_SI",
        "DELTA_T_SI",
        "BASE_DENSITY_SI",
        "TYPICAL_PARTICLES_PER_CELL",
    ]:
        assert arithmetic(expression(source("simulation.param", output), name)) == arithmetic(
            expression(source("simulation.param"), name)
        )
    assert context["time_steps"] == 1000
    assert context["grid"]["cell_cnt"] == dict(x=3584, y=3584, z=1792)
    assert context["grid"]["gpu_cnt"] == dict(x=8, y=16, z=8)
    assert context["grid"]["guard_size"] == dict(x=1, y=1, z=1)
    assert "maxwellSolver::CKC" in source("fieldSolver.param", output)
    particle = source("particle.param", output)
    assert arithmetic(expression(particle, "MIN_WEIGHTING")) == arithmetic(
        expression(source("particle.param"), "MIN_WEIGHTING")
    )
    assert "numParticlesPerCell = 1;" in particle
    assert "RandomImpl<random_ppc_species_e>" in particle
    species = re.sub(r"\s+", "", source("speciesDefinition.param", output))
    assert "ParticleShape_species_e=particles::shapes::PQS" in species
    assert "FieldToParticleInterpolation<ParticleShape_species_e,AssignedTrilinearInterpolation>" in species
    assert "currentSolver::EmZ<ParticleShape_species_e>" in species
    reference_pusher = re.search(r"using UsedParticlePusher\s*=\s*([^;]+);", source("species.param"))[1]
    assert re.sub(r"\s+", "", reference_pusher) in species
    assert "VectorAllSpecies=MakeSeq_t<species_e>" in species
    initialization = source("speciesInitialization.param", output)
    assert initialization.count("CreateDensity<") == 1
    assert "Manipulate<" not in initialization
    assert "SI::BASE_DENSITY_SI" in source("density.param", output)
    cfg = (output / "etc/picongpu/N.cfg").read_text()
    for option in [
        "--fieldAbsorber pml",
        "--currentInterpolation binomial",
        "--windowMovePoint 0.0",
        'TBG_steps="1000"',
    ]:
        assert option in cfg
    workflow = json.loads((output / "workflow/input.yaml").read_text())
    assert workflow["build_cmake"] == setup.CMAKE_FLAGS
    assert not (output / ".build").exists()


def test_generated_memory_and_pml_match_reference(generated):
    output, _ = generated
    memory = source("memory.param", output)
    original = source("memory.param")
    assert arithmetic(expression(memory, "reservedGpuMemorySize")) == arithmetic(
        expression(original, "reservedGpuMemorySize")
    )
    assert "4096ULL * 1024 * 1024" in memory
    original_electron = original.split("struct ElectronExchangeMemCfg", 1)[1]
    for name in ["BYTES_EXCHANGE_X", "BYTES_EXCHANGE_Y", "BYTES_EXCHANGE_Z", "BYTES_EDGES", "BYTES_CORNER"]:
        assert arithmetic(expression(memory, name)) == arithmetic(expression(original_electron, name))
    assert "mCT::Int<8, 8, 8>" in memory
    assert "mCT::Int<1, 1, 1>" in memory
    assert "REF_LOCAL_DOM_SIZE = mCT::Int<0, 0, 0>" in memory
    assert arithmetic(expression(memory, "fieldTmpNumSlots")) == 1
    assert arithmetic(expression(original, "fieldTmpNumSlots")) == 1
    assert expression(memory, "fieldTmpSupportGatherCommunication") == "true"
    absorber = source("fieldAbsorber.param", output)
    original_absorber = source("fieldAbsorber.param")
    assert re.findall(r"\d+", expression(absorber, "NUM_CELLS")) == ["110"] * 6
    for name in ["SIGMA_KAPPA_GRADING_ORDER", "SIGMA_OPT_MULTIPLIER", "ALPHA_GRADING_ORDER"]:
        assert arithmetic(expression(absorber, name)) == arithmetic(expression(original_absorber, name))
    for name in ["KAPPA_MAX", "ALPHA_MAX_SI"]:
        values = lambda text: [float(x) for x in re.findall(r"[\d.]+", expression(text, name))]
        assert values(absorber) == values(original_absorber)
    for name in ["SIGMA_OPT_SI", "SIGMA_MAX_SI"]:
        assert re.sub(r"\s+", "", expression(absorber, name)) == re.sub(r"\s+", "", expression(original_absorber, name))


def test_generated_laser_parameters_and_faces(generated):
    output, context = generated
    incident = source("incidentField.param", output)
    original = source("TwtsBackgroundLaser.param")
    assert re.sub(r"\s+", "", expression(incident, "POSITION")) == re.sub(
        r"\s+", "", expression(source("incidentField.param"), "POSITION")
    )
    for face, indices in [("YMin", [0, 1]), ("ZMin", [0]), ("ZMax", [1])]:
        profile = re.search(rf"using {face}\s*=\s*([^;]+);", incident)[1]
        assert list(map(int, re.findall(r"LaserProfile_(\d+)", profile))) == indices
    for face in ["XMin", "XMax", "YMax"]:
        assert re.search(rf"using {face}\s*=\s*profiles::None;", incident)
    physical = source("physicalConstants.param", REPO)
    mass = arithmetic(expression(physical, "ELECTRON_MASS_SI"))
    charge = arithmetic(expression(physical, "ELECTRON_CHARGE_SI"))
    amplitude = -2 * math.pi / 0.8e-6 * mass * 299792458**2 / charge * 1.25
    assert len(context["laser"]) == 2
    for index, sign in enumerate([1, -1]):
        block = re.search(rf"struct PyPIConGPUTWTSParam_{index}[^{{]*\{{(.*?)\n    \}};", incident, re.S)[1]
        block += re.search(rf"struct PyPIConGPULaserBaseParam_{index}[^{{]*\{{(.*?)\n    \}};", incident, re.S)[1]
        expected = {
            "WAVE_LENGTH_SI": arithmetic(expression(original, "WAVE_LENGTH_SI")),
            "PULSE_DURATION_SI": arithmetic(expression(original, "PULSE_DURATION_SI")),
            "W0_SI": 1e-6,
            "FOCUS_Z_OFFSET_SI": sign * 12e-6,
            "PHI": sign * math.radians(5),
            "BETA_0": 1,
            "TDELAY": 0,
            "Polarization": sign * math.radians(45.10716816968629),
            "windowStart": 0,
            "windowEnd": 10000000,
            "windowLength": 10000,
        }
        for name, value in expected.items():
            assert arithmetic(expression(block, name)) == pytest.approx(value, rel=1e-12, abs=0)
        assert expression(block, "AUTO_TDELAY") == "false"
        assert expression(block, "FOCUS_Y_SI") == "focus_position[1]"
        assert context["laser"][index]["focus_pos_si"][1]["component"] == 68.5e-6
        assert context["laser"][index]["E0_si"] == pytest.approx(amplitude, rel=1e-12)
