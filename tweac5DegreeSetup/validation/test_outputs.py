"""Task 9b: compare generated input against active reference outputs, without C++."""

import json
import re
import shlex
import tomllib
from collections import defaultdict
from hashlib import sha256

import pytest

from test_core import REFERENCE, REPO, setup, source


def reference_plugins(steps):
    """Expand only quoted TBG assignments as data; never source config.cfg."""
    text = (REFERENCE / "etc/picongpu/config.cfg").read_text()
    assignments = dict(re.findall(r'^(TBG_\w+)="((?:\\.|[^"\\])*)"', text, re.M | re.S))
    assignments = {k: v.replace("\\\n", "").replace('\\"', '"') for k, v in assignments.items()}
    assignments["TBG_steps"] = str(steps)

    def expand(text):
        return re.sub(r"!(TBG_\w+)", lambda m: expand(assignments[m[1]]), text)

    return options(expand(assignments["TBG_plugins"]))


def options(text):
    tokens = shlex.split(text)
    result = defaultdict(list)
    for i, token in enumerate(tokens):
        if token.startswith("--"):
            result[token[2:]].append(tokens[i + 1])
    return dict(result)


def events(spec, steps):
    if ":" not in spec:
        start, end, interval = 0, steps, int(spec)
    else:
        start, end, interval = map(int, spec.split(":"))
        end = steps if end == -1 else min(end, steps)
    return list(range(start, end + 1, interval))


@pytest.fixture(scope="module", params=[1000, 230000, 0])
def output(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("tweac-outputs") / "setup"
    setup.generate(directory, max_steps=request.param)
    cfg = (directory / "etc/picongpu/N.cfg").read_text()
    body = re.search(r'pypicongpu_output_with_newlines="(.*?)"', cfg, re.S)[1]
    context = json.loads((directory / "metadata/pypicongpu_rendering_context.json").read_text())
    return directory, request.param, options(body), reference_plugins(request.param), context


def test_energy_phase_space_and_checkpoint(output):
    _, steps, actual, reference, _ = output
    prefixes = ("fields_energy.", "e_energy.", "e_phaseSpace.", "checkpoint.")
    assert {k for k in actual if k.startswith(prefixes)} == {k for k in reference if k.startswith(prefixes)}
    for key, values in reference.items():
        if not key.startswith(prefixes):
            continue
        if key.endswith(".period"):
            assert [events(v, steps) for v in actual[key]] == [events(v, steps) for v in values]
        elif key in ("e_phaseSpace.min", "e_phaseSpace.max"):
            assert list(map(float, actual[key])) == pytest.approx(list(map(float, values)), rel=1e-15)
        elif key == "e_phaseSpace.ext":
            # Reference has an unused eighth extension, but only seven instances.
            assert len(values) == 8
            assert actual[key] == values[:7] == ["bp"] * 7
        else:
            assert actual[key] == values
    assert actual["checkpoint.period"] == [f"0:{steps}:50000"]
    assert set(k for k in actual if k.startswith("checkpoint.")) == {
        "checkpoint.period",
        "checkpoint.openPMD.infix",
        "checkpoint.openPMD.dataPreparationStrategy",
    }


def test_field_files_ranges_sources_and_compression(output):
    directory, steps, actual, reference, _ = output
    files = sorted(directory.glob("etc/openPMD_config_*.toml"))
    assert len(files) == len(actual["openPMD.pluginConfig"]) == 4
    assert {"../input/etc/" + p.name for p in files} == set(actual["openPMD.pluginConfig"])
    configs = {data["file"]: data for p in files if (data := tomllib.loads(p.read_text()))}
    for i, filename in enumerate(reference["openPMD.file"]):
        config = configs[filename]
        for name in ("file", "infix", "ext", "range"):
            assert config[name] == reference[f"openPMD.{name}"][i]
        assert config["data_preparation_strategy"] == reference["openPMD.dataPreparationStrategy"][i]
        assert config["backend_config"] == json.loads(reference["openPMD.backendConfig"][i])
        assert reference["openPMD.source"][i] == "fields_all"
        periods = config["sink"]["dummy_application_name"]["period"]
        assert periods == {reference["openPMD.period"][i]: ["E", "B", "e_all_chargeDensity", "e_all_energyDensity"]}
        expected_events = [0] if steps == 0 or (steps == 1000 and filename == "simOutput") else None
        if expected_events is not None:
            assert events(next(iter(periods)), steps) == expected_events
    # Keep generated configuration portable and deterministic, without temporary paths.
    assert all(str(directory) not in p.read_text() for p in files)
    assert all(p.name == f"openPMD_config_{sha256(p.read_bytes()).hexdigest()}.toml" for p in files)


def test_inherited_fields_use_the_same_cpp_implementations_once(output):
    directory, _, _, _, context = output
    assert not (REFERENCE / "include/picongpu/param/fileOutput.param").exists()
    reference = source("fileOutput.param", REPO)
    assert "MakeSeq_t<FieldE, FieldB>" in reference
    generated = source("fileOutput.param", directory)
    assert len(context["builtin_derived_fields"]) == 2
    for attribute in ("ChargeDensity", "EnergyDensity"):
        assert f"deriveField::derivedAttributes::{attribute}>" in reference
        assert generated.count(f"deriveField::derivedAttributes::{attribute}") == 1
        assert f'derivedAttributes/{attribute}.def"' in generated
        assert re.search(rf"CreateEligible_t<\s*species_e, deriveField::derivedAttributes::{attribute}\s*>", generated)
    assert not re.search(r"struct (?:ChargeDensity|EnergyDensity)", generated)
    assert "species_e" in source("speciesDefinition.param", directory)


def test_effective_output_inventory(output):
    directory, _, actual, _, context = output
    assert set(actual) == {
        "fields_energy.period",
        "e_energy.period",
        "e_energy.filter",
        "e_phaseSpace.period",
        "e_phaseSpace.filter",
        "e_phaseSpace.space",
        "e_phaseSpace.momentum",
        "e_phaseSpace.min",
        "e_phaseSpace.max",
        "e_phaseSpace.ext",
        "checkpoint.period",
        "checkpoint.openPMD.infix",
        "checkpoint.openPMD.dataPreparationStrategy",
        "openPMD.pluginConfig",
    }
    # The mandatory radiation template placeholder has no species or active period.
    radiation = [o for o in context["output"] if o.get("type_radiation")]
    assert len(radiation) == 1
    assert radiation[0]["species"] == []
    assert radiation[0]["period"]["specs"] == []
    assert not (directory / ".build").exists()
    assert not (directory / "simOutput").exists()
