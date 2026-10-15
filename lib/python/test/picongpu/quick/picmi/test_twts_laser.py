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
from pydantic import ValidationError, computed_field

from picongpu import picmi
from picongpu.pypicongpu import laser as backend
from picongpu.pypicongpu.rendering import Renderer


REFERENCE_POLARIZATION = math.radians(45.10716816968629)


def make_twts(centroid_y, **overrides):
    parameters = dict(
        wavelength=0.8e-6,
        waist=1.0e-6,
        duration=25.0e-15 / 2.354820045,
        laserIncidenceAngle=math.radians(5.0),
        polarizationAngle=REFERENCE_POLARIZATION,
        focal_position=[0.0, 68.5e-6, 0.0],
        centroid_position=[0.0, centroid_y, 0.0],
        a0=1.25,
        windowStart=0.0,
        windowEnd=10000000.0,
        windowLength=10000.0,
    )
    return picmi.TWTSLaser(**(parameters | overrides))


@pytest.mark.parametrize("incidence_angle", [-60.0, -5.0, 5.0, 60.0])
@pytest.mark.parametrize("polarization_angle", [0.0, 90.0, -90.0, 180.0, 45.10716816968629, -45.10716816968629])
def test_twts_polarization_is_transverse_unit_vector(incidence_angle, polarization_angle):
    pulse = make_twts(
        68.5e-6,
        laserIncidenceAngle=math.radians(incidence_angle),
        polarizationAngle=math.radians(polarization_angle),
    )
    direction = pulse.polarization_direction
    propagation = pulse.propagation_direction
    assert math.hypot(*direction) == pytest.approx(1.0, rel=0, abs=1e-14)
    assert sum(e * k for e, k in zip(direction, propagation)) == pytest.approx(0.0, abs=1e-14)
    assert pulse.get_as_pypicongpu().polarizationAngle == math.radians(polarization_angle)


@pytest.mark.parametrize("incidence_sign", [-1, 1])
def test_twts_polarization_basis(incidence_sign):
    # At |phi|=30 degrees, alpha=0 selects x and alpha=90 selects the
    # transverse axis in the yz plane. These anchors also fix the signs.
    for angle, expected in [
        (0.0, [1.0, 0.0, 0.0]),
        (math.pi / 2, [0.0, -incidence_sign / 2, math.sqrt(3) / 2]),
        (-math.pi / 2, [0.0, incidence_sign / 2, -math.sqrt(3) / 2]),
    ]:
        pulse = make_twts(
            68.5e-6,
            laserIncidenceAngle=incidence_sign * math.pi / 6,
            polarizationAngle=angle,
        )
        assert pulse.polarization_direction == pytest.approx(expected, rel=0, abs=1e-14)


def test_twts_negative_incidence_matches_cpp_rotation_axis():
    # TWTSTight computes at |phi| and rotates negative-incidence beams by pi
    # about y: (Ex,Ey,Ez) -> (-Ex,Ey,-Ez). PICMI chooses the equivalent axis
    # with x=cos(alpha), so its representative vector has the opposite sign.
    for angle in (0.0, math.pi / 2, REFERENCE_POLARIZATION, -REFERENCE_POLARIZATION):
        positive = make_twts(68.5e-6, polarizationAngle=angle)
        negative = make_twts(68.5e-6, laserIncidenceAngle=-math.radians(5.0), polarizationAngle=angle)
        ex, ey, ez = positive.polarization_direction
        rotated = [-ex, ey, -ez]
        assert negative.polarization_direction == pytest.approx([-v for v in rotated], rel=0, abs=1e-14)


@pytest.mark.parametrize("centroid_y", [-10e-6, 0.0, 30e-6, 68.5e-6, 100e-6])
@pytest.mark.parametrize("incidence_angle", [-5.0, 5.0])
def test_twts_centroid_sets_manual_delay(centroid_y, incidence_angle):
    pulse = make_twts(centroid_y, laserIncidenceAngle=math.radians(incidence_angle))
    translated = pulse.get_as_pypicongpu()
    expected_delay = (68.5e-6 - centroid_y) / picmi.constants.c
    assert translated.time_offset_si == pytest.approx(expected_delay, rel=1e-12, abs=1e-25)
    assert translated.focus_pos_si[1] == 68.5e-6
    # The shared backend requires pulse_init, but TWTS does not use it.
    assert pulse.pulse_init == 0.0
    assert translated.pulse_init == 0.0
    context = translated.model_dump(mode="json")
    assert context["pulse_init"] == 0.0


def test_twts_zero_delay_renders_with_switching_window():
    pulse = make_twts(68.5e-6).get_as_pypicongpu()
    context = Renderer.get_context_preprocessed({"laser": [pulse.model_dump(mode="json")]})
    template = (
        Path(backend.__file__).resolve().parents[1]
        / "templates/include/picongpu/param/incidentField.param.mustache"
    ).read_text()
    rendered = Renderer.get_rendered_template(context, template)
    assert "TDELAY = 0.0;" in rendered
    assert "AUTO_TDELAY = false;" in rendered
    assert "windowStart = 0.0;" in rendered
    assert "windowEnd = 10000000.0;" in rendered
    assert "windowLength = 10000.0;" in rendered
    assert "PULSE_INIT" not in rendered
    assert "RAMP_INIT" not in rendered


@pytest.mark.parametrize("centroid_y", [-10e-6, 0.0, 30e-6, 68.5e-6, 100e-6])
def test_twts_setup_generation(centroid_y, tmp_path):
    """Generate both beams on the reference grid; never build or run PIConGPU.

    This uses the reference polarization angles, but excludes the pending
    particle, PML and diagnostic migrations. It is not the complete migrated setup.
    """
    number_of_cells = [3584, 3584, 1792]
    cell_size = [6.177353e-8, 3.088676e-8, 6.177353e-8]
    surface = [[116, -116], [116, 100000000], [116, -116]]
    pulses = [
        make_twts(
            centroid_y,
            laserIncidenceAngle=sign * math.radians(5.0),
            polarizationAngle=sign * REFERENCE_POLARIZATION,
            focus_lateral_offset_si=sign * 12e-6,
            picongpu_huygens_surface_positions=surface,
        )
        for sign in (1, -1)
    ]
    grid = picmi.Cartesian3DGrid(
        number_of_cells=number_of_cells,
        lower_bound=[0.0, 0.0, 0.0],
        upper_bound=[n * dx for n, dx in zip(number_of_cells, cell_size)],
        lower_boundary_conditions=["open"] * 3,
        upper_boundary_conditions=["open"] * 3,
        picongpu_n_gpus=[8, 16, 8],
        picongpu_super_cell_size=[8, 8, 8],
    )
    simulation = picmi.Simulation(
        solver=picmi.ElectromagneticSolver(
            grid=grid, method="CKC", source_smoother=picmi.BinomialSmoother()
        ),
        time_step_size=5.15136e-17,
        max_steps=1000,
        picongpu_moving_window_move_point=0.0,
        lasers=pulses,
    )
    output = tmp_path / "setup"
    cmake = "-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'"
    simulation.write_input_file(output, cmake=cmake)

    context = json.loads((output / "metadata/pypicongpu_rendering_context.json").read_text())
    assert context["time_steps"] == 1000
    assert context["grid"]["cell_cnt"] == dict(zip("xyz", number_of_cells))
    assert context["grid"]["gpu_cnt"] == dict(zip("xyz", [8, 16, 8]))
    assert context["grid"]["cell_size"] == pytest.approx(dict(zip("xyz", cell_size)))

    parameters = output / "include/picongpu/param"
    rendered = (parameters / "incidentField.param").read_text()
    blocks = re.findall(r"struct PyPIConGPUTWTSParam_\d+[^\{]*\{(.*?)\n    \};", rendered, re.S)
    assert len(blocks) == 2
    for block, sign in zip(blocks, (1, -1)):
        for name, expected in {
            "TDELAY": (68.5e-6 - centroid_y) / picmi.constants.c,
            "W0_SI": 1e-6,
            "PHI": sign * math.radians(5.0),
            "Polarization": sign * REFERENCE_POLARIZATION,
            "FOCUS_Z_OFFSET_SI": sign * 12e-6,
            "BETA_0": 1.0,
            "windowStart": 0.0,
            "windowEnd": 10000000.0,
            "windowLength": 10000.0,
        }.items():
            value = re.search(rf"\b{name} = ([^;]+);", block)
            assert value is not None, name
            assert float(value[1]) == pytest.approx(expected, rel=1e-12, abs=1e-25), name
        assert "AUTO_TDELAY = false;" in block
        assert "FOCUS_Y_SI = focus_position[1];" in block
    for pulse_context, pulse in zip(context["laser"], pulses):
        assert pulse_context["focus_pos_si"][1]["component"] == 68.5e-6
        assert pulse_context["pulse_init"] == 0.0
        assert pulse_context["polarizationAngle"] == pulse.polarizationAngle
        assert pulse_context["E0_si"] == pytest.approx(
            1.25 * picmi.constants.m_e * picmi.constants.c**2 * (2 * math.pi / 0.8e-6) / picmi.constants.q_e
        )
        assert [v["component"] for v in pulse_context["polarization_direction"]] == pulse.polarization_direction
    base_blocks = re.findall(r"struct PyPIConGPULaserBaseParam_\d+[^\{]*\{(.*?)\n    \};", rendered, re.S)
    assert len(base_blocks) == 2
    for block, pulse in zip(base_blocks, pulses):
        vector = re.search(r"polarisation_direction\[3u\] = \{(.*?)\};", block, re.S)
        assert vector is not None
        assert [float(v) for v in vector[1].split(",")] == pytest.approx(
            pulse.polarization_direction, rel=0, abs=1e-14
        )
        amplitude = re.search(r"AMPLITUDE_SI = ([^;]+);", block)
        assert amplitude is not None
        assert float(amplitude[1]) == pytest.approx(pulse.E0, rel=1e-12)
    for face, expected_profiles in {
        "YMin": ["LaserProfile_0", "LaserProfile_1"],
        "ZMin": ["LaserProfile_0"],
        "ZMax": ["LaserProfile_1"],
    }.items():
        definition = re.search(rf"using {face} = MakeSeq_t<(.*?)>;", rendered, re.S)
        assert definition is not None, face
        assert re.findall(r"LaserProfile_\d+", definition[1]) == expected_profiles
    assert "PULSE_INIT" not in rendered
    assert "RAMP_INIT" not in rendered
    assert not (output / ".build").exists()


@pytest.mark.parametrize("kind", ["GaussianLaser", "DispersivePulseLaser", "PlaneWaveLaser"])
def test_other_lasers_still_reject_positive_centroid(kind):
    parameters = dict(
        wavelength=0.8e-6,
        duration=10e-15,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        centroid_position=[0.0, 68.5e-6, 0.0],
        a0=1.0,
    )
    if kind != "PlaneWaveLaser":
        parameters.update(waist=1e-6, focal_position=[0.0, 68.5e-6, 0.0])
    with pytest.raises(ValidationError, match="centroid_y <= 0"):
        getattr(picmi, kind)(**parameters)

    parameters["centroid_position"] = [0.0, -10e-6, 0.0]
    translated = getattr(picmi, kind)(**parameters).get_as_pypicongpu()
    assert translated.pulse_init > 0


def test_twts_still_checks_propagation_direction():
    with pytest.raises(ValidationError, match="Laser propagation"):
        make_twts(68.5e-6, laserIncidenceAngle=math.pi)


def test_twts_still_checks_polarization_normalization():
    class InvalidPolarizationTWTS(picmi.TWTSLaser):
        @computed_field
        def polarization_direction(self) -> list[float]:
            return [2.0, 0.0, 0.0]

    valid = make_twts(68.5e-6)
    parameters = {name: getattr(valid, name) for name in type(valid).model_fields}
    parameters["E0"] = None  # a0 and E0 are both populated after construction.
    with pytest.raises(ValidationError, match="polarization direction vector must be normalized"):
        InvalidPolarizationTWTS(**parameters)
