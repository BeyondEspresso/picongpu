"""Task 9c: source-based eSpec checks and Python input generation only."""

import json
import math
import re

import numpy as np
import pytest
from sympy import lambdify, symbols

from picongpu import picmi
from picongpu.picmi.particle_functor.particle_functor import AbstractParticle

from test_core import REPO, arithmetic, expression, setup, source
from test_outputs import events


REFERENCE = source("binningSetup.param")
BINNING_SOURCE = REPO / "include/picongpu/plugins/binning"


def spectrometer():
    return setup.electron_spectrometer(picmi.Species(name="e", particle_type="electron"))


def numeric_constant(name):
    return arithmetic(expression(REFERENCE, name).replace("_X", ""))


def reference_unit_vector(name):
    indices = {
        key: i
        for i, key in enumerate(
            ["length", "mass", "time", "electricCurrent", "temperature", "amountOfSubstance", "luminousIntensity"]
        )
    }
    vector = [0.0] * 7
    for key, value in re.findall(rf"{name}\[SIBaseUnits::(\w+)\]\s*=\s*([\d.-]+)", REFERENCE):
        vector[indices[key]] = float(value)
    return vector


def functor_evaluators():
    """Evaluate the actual Python functors, with C++ accessor values supplied separately."""
    diagnostic = spectrometer()
    arguments = symbols("Ekin weighting px py pz charge")
    return [
        lambdify(arguments, f(AbstractParticle()), modules="math")
        for f in [diagnostic.axes[0].functor, diagnostic.axes[1].functor, diagnostic.deposition_functor]
    ]


def kinetic_energy(momentum, mass):
    """Source translation of KinEnergy.hpp in units where c=1, binary64 arithmetic."""
    threshold = arithmetic(expression(source("speciesConstants.param", REPO), "GAMMA_THRESH").replace("_X", ""))
    p_squared = sum(p * p for p in momentum)
    gamma = math.sqrt(1 + p_squared / mass**2)
    return p_squared / (2 * mass) if gamma < threshold else (gamma - 1) * mass


@pytest.mark.parametrize("weight", [0.1, 1.0, 1e3, 1e8])
@pytest.mark.parametrize(
    "single_momentum",
    [
        (0, 0, 0),
        (0, 0.01, 0),
        (0, 0.2, 0),
        (0.03, 2, 0.4),
        (-2, -3, 1),
        (5, 2000, 0),
    ],
)
def test_weighting_energy_pointing_and_charge(weight, single_momentum):
    energy, pointing, charge = functor_evaluators()
    # Macroparticle momenta, mass and charge carry weight. Rest mass and c
    # are one in this unit system; use the same low-energy branch as KinEnergy.
    macro_momentum = tuple(weight * p for p in single_momentum)
    macro_energy = kinetic_energy(macro_momentum, weight)
    args = (macro_energy, weight, *macro_momentum, -weight)
    assert energy(*args) == pytest.approx(kinetic_energy(single_momentum, 1), rel=2e-13, abs=1e-15)
    assert pointing(*args) == pytest.approx(math.atan2(single_momentum[0], single_momentum[1]), abs=1e-15)
    assert charge(*args) == -weight


@pytest.mark.parametrize(
    "px, py, angle",
    [
        (0, 1, 0),
        (1, 0, math.pi / 2),
        (-1, 0, -math.pi / 2),
        (1, 1, math.pi / 4),
        (-1, 1, -math.pi / 4),
        (1, -1, 3 * math.pi / 4),
        (-1, -1, -3 * math.pi / 4),
        (0, 0, 0),
    ],
)
def test_pointing_quadrants_and_zero_momentum(px, py, angle):
    _, pointing, _ = functor_evaluators()
    assert pointing(0, 1, px, py, 0, -1) == pytest.approx(angle)


def test_axes_units_limits_and_reference_formulas():
    diagnostic = spectrometer()
    energy, pointing = diagnostic.axes
    charge_si = arithmetic(expression(source("physicalConstants.param", REPO), "ELECTRON_CHARGE_SI"))
    assert (
        [a.name for a in diagnostic.axes]
        == re.findall(r'createFunctorDescription<float_X>\(\w+, "([^"]+)", \w+\)', REFERENCE)[:2]
        == ["Energy", "pointingXY"]
    )
    assert (
        [a.bin_spec.nsteps for a in diagnostic.axes]
        == [int(n) for n in re.findall(r"axis::Range\{[^}]+\}, (\d+)\)", REFERENCE)]
        == [1600, 256]
    )
    assert energy.bin_spec.start == numeric_constant("minEnergy") == 0
    assert energy.bin_spec.stop == numeric_constant("maxEnergy_MeV") * 1e6 * -charge_si
    assert (pointing.bin_spec.start, pointing.bin_spec.stop) == (
        -numeric_constant("pointing_range"),
        numeric_constant("pointing_range"),
    )
    for functor, unit_name in [
        (energy.functor, "energyDimension"),
        (pointing.functor, "pointingXYDimension"),
        (diagnostic.deposition_functor, "depositedUnits"),
    ]:
        assert list(functor.unit_dimension) == reference_unit_vector(unit_name)
        assert functor.return_type == "float_X"
    assert "KinEnergy<>()(mom, mass) / weighting" in REFERENCE
    assert "math::atan2(particle[momentum_][0], particle[momentum_][1])" in REFERENCE
    assert "getCharge(particle[weighting_], particle)" in REFERENCE


def source_linear_index(value, lower, upper, n_bins):
    """LinearAxisKernel's floating-point path with overflow bins, in float32."""
    value, lower, upper = map(np.float32, (value, lower, upper))
    if value < lower:
        return 0
    if value >= upper:
        return n_bins + 1
    scale = np.float32(n_bins) / np.float32(upper - lower)
    return int(np.float32(value - lower) * scale) + 1


@pytest.mark.parametrize("axis_index", [0, 1])
def test_bin_boundaries_overflow_and_unit_conversion(axis_index):
    axis = spectrometer().axes[axis_index]
    assert axis.use_overflow_bins
    n = axis.bin_spec.nsteps
    # C++ divides SI axis limits by their dimension's unit conversion factor.
    # Energy here uses a 1 MeV unit; pointing is dimensionless (radians).
    unit = 1e6 * picmi.constants.q_e if axis_index == 0 else 1.0
    lower, upper = np.float32(axis.bin_spec.start / unit), np.float32(axis.bin_spec.stop / unit)
    width = (upper - lower) / n
    for value, expected in [
        (lower - width, 0),
        (lower, 1),
        (lower + width / 2, 1),
        (lower + (n // 2 + 0.5) * width, n // 2 + 1),
        (upper - width / 2, n),
        (upper, n + 1),
        (upper + width, n + 1),
    ]:
        assert source_linear_index(value, lower, upper, n) == expected
    # An exactly representable interior energy edge enters the next bin.
    if axis_index == 0:
        assert source_linear_index(lower + width, lower, upper, n) == 2
    linear_source = (BINNING_SOURCE / "axis/LinearAxis.hpp").read_text()
    assert "if(val >= picRange.min)" in linear_source
    assert "if(val < picRange.max)" in linear_source
    assert "binIdx = (val - picRange.min) * scaling;" in linear_source
    assert "bool enableOverflow = true" in (BINNING_SOURCE / "axis/Axis.hpp").read_text()


def test_snapshot_accumulation_and_storage_order():
    diagnostic = spectrometer()
    data = (BINNING_SOURCE / "BinningData.hpp").read_text()
    default_dump_period = int(re.search(r"uint32_t dumpPeriod = (\d+)u", data)[1])
    assert default_dump_period == 0
    assert diagnostic.dumpPeriod == 1
    charge = functor_evaluators()[2]
    snapshots = [[0.1, 2, 3], [5], [1, 10]]

    def dumps(period):
        counter, accumulated = 0, 0
        result = []
        for weights in snapshots:
            accumulated += sum(charge(0, w, 0, 1, 0, -w) for w in weights)
            counter += 1
            if counter >= period:
                result.append((accumulated, counter))
                accumulated, counter = 0, 0
        return result

    assert dumps(default_dump_period) == dumps(diagnostic.dumpPeriod) == [(-5.1, 1), (-5, 1), (-11, 1)]
    assert "if(reduceCounter >= binningData.dumpPeriod)" in (BINNING_SOURCE / "binners/Binner.hpp").read_text()
    # The current writer stores raw charge per bin, not charge divided by bin
    # width, angular width or the reduce counter. Both paths use this writer.
    writer = (BINNING_SOURCE / "WriteHist.hpp").read_text()
    assert "mesh.storeChunk<Type>(std::move(data), histOffset, histExtent);" in writer
    assert 'iteration.setAttribute("reduceCounter", reduceCounter);' in writer
    assert "std::reverse(labelVector.begin(), labelVector.end());" in writer
    assert [a.bin_spec.nsteps + 2 for a in reversed(diagnostic.axes)] == [258, 1602]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tweac-binning") / "setup"
    setup.generate(directory)
    context = json.loads((directory / "metadata/pypicongpu_rendering_context.json").read_text())
    return directory, context


def test_generated_registration_functors_and_backend(generated):
    directory, context = generated
    binners = [o for o in context["output"] if o.get("type_binning")]
    assert len(binners) == 1
    binner = binners[0]
    text = source("binningSetup.param", directory)
    assert binner["binner_name"] == "eSpec"
    assert [s["name"] for s in binner["species"]] == ["e"]
    assert text.count("eSpec(binningCreator);") == REFERENCE.count("eSpec(binningCreator);") == 1
    assert "createTuple(axis_Energy,axis_pointingXY)" in text
    ranges = re.findall(r"axis::Range<float_X>\{\s*([^,]+),\s*([^}]+)\},\s*(\d+),\s*(true|false)", text)
    assert len(ranges) == 2
    for (lower, upper, count, overflow), axis in zip(ranges, spectrometer().axes):
        # Both reference and generated axes use float_X (32 bits here).
        assert np.float32(float(lower)) == np.float32(axis.bin_spec.start)
        assert np.float32(float(upper)) == np.float32(axis.bin_spec.stop)
        assert int(count) == axis.bin_spec.nsteps
        assert overflow == "true"
    assert "auto const mass = picongpu::traits::attribute::getMass(particle[weighting_], particle);" in text
    assert "picongpu::KinEnergy()(momentum::type{px, py, pz}, mass)" in text
    assert "return Ekin/weighting;" in text
    assert "return pmacc::math::atan2(px, py);" in text
    assert "auto const charge = picongpu::traits::attribute::getCharge(particle[weighting_], particle);" in text
    assert "return charge;" in text
    for axis in binner["axes"]:
        assert axis["axis_functor"]["return_type"] == "float_X"
        assert axis["use_overflow_bins"] is True
    assert binner["deposition_functor"]["name"] == "Charge"
    assert binner["deposition_functor"]["return_type"] == "float_X"
    for unit in ["2.0,1.0,-2.0,0.0,0.0,0.0,0.0", "0.0,0.0,0.0,0.0,0.0,0.0,0.0", "0.0,0.0,1.0,1.0,0.0,0.0,0.0"]:
        assert f"std::array<double, 7u>{{{unit}}}" in text
    backend = re.search(r'setOpenPMDBackendConfig\(R"\((.*?)\)"\)', REFERENCE)[1]
    assert json.loads(backend) == {"adios2": {"dataset": {"chunks": "auto"}}}
    # openPMD's ADIOS2 handler ignores this HDF5-only key. The typed backend
    # rejects it, so the migrated setup leaves the equivalent defaults unset.
    # Source evidence and the configuration difference are in resolvedConflicts.md.
    assert binner["openPMDBackendConfig"] is None
    assert ".setOpenPMDBackendConfig(" not in text
    assert binner["openPMDExtension"] == re.search(r'setOpenPMDExtension\("(.*?)"\)', REFERENCE)[1] == "bp"
    assert binner["openPMDInfix"] is None
    assert ".setOpenPMDInfix(" not in text
    defaults = (BINNING_SOURCE / "BinningData.hpp").read_text()
    assert 'std::string openPMDInfix = "_%06T.";' in defaults
    assert "uint32_t particleRegion{ParticleRegion::Bounded};" in defaults
    assert ".enableRegion(ParticleRegion::Bounded)" in text
    assert ".disableRegion(ParticleRegion::Leaving)" in text
    reference_period = re.search(r'setNotifyPeriod\("(.*?)"\)', REFERENCE)[1]
    assert binner["period"]["specs"] == [{"start": 0, "stop": -1, "step": int(reference_period)}]
    assert events("0:-1:500", context["time_steps"]) == events(reference_period, 1000) == [0, 500, 1000]
    assert context["grid"]["cell_cnt"] == dict(x=3584, y=3584, z=1792)
    assert len([o for o in context["output"] if o.get("type_phasespace")]) == 7
    assert len(list(directory.glob("etc/openPMD_config_*.toml"))) == 4
    assert not (directory / ".build").exists()
    assert not (directory / "simOutput").exists()


@pytest.mark.parametrize("steps, expected", [(0, [0]), (499, [0]), (500, [0, 500]), (1250, [0, 500, 1000])])
def test_espec_period_uses_run_length_without_rescaling(steps, expected):
    sim = setup.build_simulation(steps)
    binner = next(d for d in sim.diagnostics if isinstance(d, picmi.diagnostics.Binning))
    period = binner.period.get_as_pypicongpu(sim.time_step_size, steps).get_rendering_context()["specs"]
    assert period == [{"start": 0, "stop": -1, "step": 500}]
    assert events("0:-1:500", steps) == expected
    assert [a.bin_spec.nsteps for a in binner.axes] == [1600, 256]
