"""Final integration audit: source/runtime comparison and generation isolation."""

import builtins
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

import pytest
from sympy import lambdify, symbols

from picongpu import picmi
from picongpu.pypicongpu.runner import Runner

from test_core import REFERENCE, REPO, arithmetic, expression, load_setup, setup, source


def program_options(text, generated=False):
    """Expand TBG strings as data, including the generated diagnostic aggregate."""
    assignments = dict(re.findall(r'^[ \t]*(TBG_\w+)="((?:\\.|[^"\\])*)"', text, re.M | re.S))
    assignments.update(re.findall(r"^[ \t]*(TBG_\w+)=(\d+)[ \t]*$", text, re.M))
    assignments = {k: v.replace("\\\n", "").replace('\\"', '"') for k, v in assignments.items()}
    if generated:
        assignments["TBG_pypicongpu_output"] = re.search(r'pypicongpu_output_with_newlines="(.*?)"', text, re.S)[1]

    def expand(value):
        return re.sub(r"!(TBG_\w+)", lambda m: expand(assignments[m[1]]), value)

    tokens = shlex.split(expand(assignments["TBG_programParams"]))
    result = {}
    key = None
    for token in tokens:
        if token.startswith("--") or token in ("-d", "-g", "-s", "-m"):
            key = token
            result.setdefault(key, []).append([])
        else:
            assert key is not None
            result[key][-1].append(token)
    return result


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tweac-final-audit") / "setup"
    # Enforce the generation-only contract, including no subprocess execution.
    with pytest.MonkeyPatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("Generation attempted subprocess execution or simulation")

        patch.setattr(subprocess, "Popen", forbidden)
        patch.setattr(os, "system", forbidden)
        patch.setattr(Runner, "run", forbidden)
        for name in ("run", "step", "picongpu_run"):
            patch.setattr(picmi.Simulation, name, forbidden)

        def guarded(original):
            def call(path, *args, **kwargs):
                if isinstance(path, (str, bytes, os.PathLike)):
                    candidate = Path(os.fsdecode(path)).absolute()
                    assert not any(candidate.is_relative_to(REFERENCE / name) for name in (".build", "bin"))
                return original(path, *args, **kwargs)

            return call

        patch.setattr(builtins, "open", guarded(builtins.open))
        patch.setattr(io, "open", guarded(io.open))
        patch.setattr(os, "scandir", guarded(os.scandir))
        setup.generate(directory)
    return directory, json.loads((directory / "metadata/pypicongpu_rendering_context.json").read_text())


def test_import_and_construction_do_not_write_files(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Import/construction attempted filesystem creation")

    def readonly(original):
        def opened(file, mode="r", *args, **kwargs):
            assert not any(flag in mode for flag in "wax+")
            return original(file, mode, *args, **kwargs)

        return opened

    monkeypatch.setattr(builtins, "open", readonly(builtins.open))
    monkeypatch.setattr(io, "open", readonly(io.open))
    monkeypatch.setattr(os, "mkdir", forbidden)
    sim = load_setup().build_simulation()
    assert sim.max_steps == 1000
    assert len(sim.diagnostics) == 27
    assert not sim.picongpu_template_dir


def test_effective_runtime_options(generated):
    directory, context = generated
    reference = program_options((REFERENCE / "etc/picongpu/config.cfg").read_text())
    actual = program_options((directory / "etc/picongpu/N.cfg").read_text(), generated=True)
    diagnostic_prefixes = ("--fields_energy.", "--e_energy.", "--e_phaseSpace.", "--checkpoint.", "--openPMD.")
    # The diagnostics' different CLI/TOML representations are compared by
    # test_outputs and test_binning. All other effective options agree exactly.
    runtime = lambda opts: {k: v for k, v in opts.items() if not k.startswith(diagnostic_prefixes)}
    expected = runtime(reference)
    expected["--periodic"] = [["0", "0", "0"]]  # C++ default explicitly emitted by PICMI
    assert runtime(actual) == expected
    assert actual["--autoAdjustGrid"] == [["false"]]
    assert actual["--percent"] == [["1"]]
    assert actual["-g"] == [["3584", "3584", "1792"]]
    assert actual["-d"] == [["8", "16", "8"]]
    assert context["auto_adjust_grid"] == "false"
    assert context["progress_percent"] == 1


def test_combined_fields_memory_and_grid_require_no_adjustment(generated):
    directory, context = generated
    local = [n // d for n, d in zip(setup.NUMBER_OF_CELLS, [8, 16, 8])]
    assert local == [448, 224, 224]
    assert all(n % 8 == 0 and n >= 3 * 8 and n >= 110 for n in local)
    memory = source("memory.param", directory)
    assert (
        arithmetic(expression(memory, "fieldTmpNumSlots"))
        == arithmetic(expression(source("memory.param"), "fieldTmpNumSlots"))
        == 1
    )
    assert expression(memory, "fieldTmpSupportGatherCommunication") == "true"
    assert len(context["builtin_derived_fields"]) == 2
    assert len([o for o in context["output"] if o.get("type_openPMD")]) == 4
    assert len([o for o in context["output"] if o.get("type_binning")]) == 1
    assert len([o for o in context["output"] if o.get("type_phasespace")]) == 7
    # Dormant filters/species/physics from the reference are not made active.
    assert context["particle_filters"] == []
    assert [s["name"] for s in context["species"]] == ["e"]
    assert not (directory / ".build").exists()
    assert not (directory / "simOutput").exists()


def test_active_run_never_reaches_nonzero_density():
    # Conservative moving-window bound: motion through step N+1 plus a full
    # local domain accounts for the advance of the initializing GPU on a slide.
    # See MovingWindow::ComputedConstants and virtualParticlePassesGPUBorder.
    dy = setup.CELL_SIZE_SI[1]
    local_y = setup.NUMBER_OF_CELLS[1] // 16
    upper_bound = (setup.NUMBER_OF_CELLS[1] + local_y) * dy + (setup.DEFAULT_STEPS + 1) * setup.TIMESTEP_SI * 299792458
    assert upper_bound < setup.VACUUM_Y_SI
    density = lambdify(symbols("x y z"), setup.density_si(*symbols("x y z")), "math")
    for x in [0, setup.X_PLATEAU_START, setup.X_PLATEAU_END]:
        for z in [0, setup.Z_PLATEAU_START, setup.Z_PLATEAU_END]:
            assert density(x, upper_bound, z) == 0
    # The only active profile returns zero before y reaches its vacuum boundary.
    assert re.search(r"if\s*\(y < VACUUM_Y_SI\)\s*return float_X\(0.0\)", source("density.param"))


def test_all_reference_include_files_are_accounted_for():
    include = REFERENCE / "include"
    files = {p.relative_to(include).as_posix() for p in include.rglob("*") if p.is_file()}
    active = {
        "simulation.param",
        "memory.param",
        "species.param",
        "speciesDefinition.param",
        "particle.param",
        "speciesInitialization.param",
        "density.param",
        "fieldSolver.param",
        "fieldAbsorber.param",
        "incidentField.param",
        "TwtsBackgroundLaser.param",
        "binningSetup.param",
        "particleFilters.param",
    }
    inactive = {
        "fieldBackground.param",
        "incidentFieldOld.param",
        "TwtsBackgroundLaserOld.param",
        "binningSetupNew.param",
    }
    assert files == {f"picongpu/param/{name}" for name in active | inactive} | {
        "picongpu/fields/background/templates/twtstight/TWTSTight.hpp"
    }
    assert setup.CMAKE_FLAGS == "-DPARAM_OVERWRITES:LIST='-DPARAM_FIELD_BACKGROUND=0;-DPARAM_COMPONENTWISE=1'"


def test_generated_twts_numerics_preserve_reference_header(generated):
    directory, context = generated
    header = Path("include/picongpu/fields/background/templates/twtstight/TWTSTight.hpp")
    reference = re.sub(r"/\*.*?\*/|//[^\n]*", "", (REFERENCE / header).read_text(), flags=re.S)
    implementation = (REPO / header).read_text()
    precision = source("precision.param", directory)
    bits = int(re.search(r"#\s*define PICONGPU_TWTS_PRECISION (\d+)", precision)[1])
    cutoff = int(re.search(r"#\s*define PICONGPU_TWTS_NUM_SIGMAS (\d+)", precision)[1])
    assert bits == 64
    assert cutoff == 8
    assert expression(reference, "float_T") == f"float_{bits}"
    assert arithmetic(expression(reference, "numSigmas")) == cutoff
    assert "namespace precisionPIConGPU = precision32Bit;" in precision
    assert len(context["laser"]) == 2
    assert context["precision_overrides"]["twts"] == bits
    assert context["precision_overrides"]["twts_num_sigmas"] == cutoff
    assert not (directory / header).exists()  # no copied implementation override

    # Resolve only the new configuration blocks as source text. This is not a
    # C++ preprocessing/compilation test. Compare the entire remaining header so
    # another setup-local implementation difference cannot silently be missed.
    implementation, count = re.subn(
        r"#ifndef PICONGPU_TWTS_(?:PRECISION|NUM_SIGMAS)\n.*?#endif", "", implementation, flags=re.S
    )
    assert count == 2
    implementation, count = re.subn(r"#if PICONGPU_TWTS_NUM_SIGMAS <.*?#endif", "", implementation, flags=re.S)
    assert count == 1
    selection = re.search(r"#if PICONGPU_TWTS_PRECISION == 0\n.*?#endif", implementation, re.S)[0]
    assert re.search(rf"#elif PICONGPU_TWTS_PRECISION == {bits}\s+using float_T = float_{bits};", selection)
    implementation = implementation.replace(selection, f"using float_T = float_{bits};")
    implementation = implementation.replace("numSigmas = PICONGPU_TWTS_NUM_SIGMAS;", f"numSigmas = {cutoff};")

    def normalized(text):
        return re.sub(r"\s+", "", re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S))

    assert normalized(implementation) == normalized(reference)


def test_twts_configuration_include_order_and_legacy_defaults():
    root = REPO / "include/picongpu"
    header = (root / "fields/background/templates/twtstight/TWTSTight.hpp").read_text()
    defaults = (root / "param/precision.param").read_text()
    # Old custom precision.param files need not define the new macros. In either
    # case the guarded defaults follow float_X/6; prior definitions win.
    for name, value in [("PRECISION", 0), ("NUM_SIGMAS", 6)]:
        guard = rf"#ifndef PICONGPU_TWTS_{name}\s+#\s*define PICONGPU_TWTS_{name} {value}\s+#endif"
        assert re.search(guard, defaults)
        assert re.search(guard, header)
    assert '#include "picongpu/param/precision.param"' in (root / "defines.hpp").read_text()
    assert header.index('#include "picongpu/defines.hpp"') < header.index("#ifndef PICONGPU_TWTS_PRECISION")
    assert re.search(r"#if PICONGPU_TWTS_PRECISION == 0\s+using float_T = float_X;", header)
    assert re.search(r"#elif PICONGPU_TWTS_PRECISION == 32\s+using float_T = float_32;", header)
    assert re.search(r"#elif PICONGPU_TWTS_PRECISION == 64\s+using float_T = float_64;", header)
    assert "constexpr uint32_t numSigmas = PICONGPU_TWTS_NUM_SIGMAS;" in header
    includes = re.findall(r"^#include [<\"]([^>\"]+)", header, re.M)
    assert "picongpu/param/incidentField.param" not in includes
    assert "picongpu/param/fieldBackground.param" not in includes

    pulse = (root / "fields/incidentField/profiles/TWTSPulse.def").read_text()
    assert '#include "picongpu/fields/background/templates/twtstight/TWTSTight.hpp"' in pulse
    assert "templates::twtstight::EField const twtsFieldE;" in pulse
    assert "templates::twtstight::BField twtsFieldB;" in pulse
    implementation = (root / "fields/background/templates/twtstight/TWTSTight.tpp").read_text()
    for component in "XYZ":
        assert f"static_cast<float_X>(calcTWTSField{component}(pos, time_SI))" in implementation
    assert "(numSigmas * tauG * cspeed)" in implementation
