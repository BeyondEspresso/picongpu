"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import re
import json
import tomllib
from unittest import TestCase

import pytest
from pydantic import ValidationError

from picongpu import picmi
from picongpu import templates
from picongpu.picmi import diagnostics as diag
from picongpu.picmi.diagnostics.backend_config import RangeSpec
from picongpu.pypicongpu.output.openpmd_plugin import FieldDump
from picongpu.pypicongpu.particle_functor import ParticleFunctor
from picongpu.pypicongpu.rendering.renderer import Renderer

TEMPLATE_PATH = templates.path() / "include" / "picongpu" / "param" / "fileOutput.param.mustache"


def render_file_output(derived_fields):
    """Render the real fileOutput template with the given derived-field contexts."""
    context = {"output": [{"type_openPMD": True, "derived_fields": derived_fields}], "builtin_derived_fields": []}
    return Renderer.get_rendered_template(Renderer.get_context_preprocessed(context), TEMPLATE_PATH.read_text())


def make_derived_field_context(species_name, functor_name="energy"):
    """Build the rendering context for one derived field dump on ``species_name``.

    Returns ``(field_context, typename)`` where ``typename`` is the emitted
    derived-attribute struct name (carries a per-functor UUID).
    """
    functor = ParticleFunctor(
        name=functor_name,
        functor_expression="Ekin",
        functor_preamble=[],
        return_type="float_64",
    )
    field = FieldDump(
        name=f"{species_name}_all_{functor_name}",
        functor=functor,
        filtername=None,
        species_name=species_name,
    )
    field_context = field.get_rendering_context()
    return field_context, field_context["functor"]["typename"]


class TestFileOutputFieldDump(TestCase):
    def test_derived_field_eligibility_specialization_is_qualified(self):
        """The emitted SpeciesEligibleForSolver specialization must fully qualify the
        derived attribute (``particleToGrid::derivedAttributes::``), not the bare
        ``derivedAttributes::`` which does not resolve from ``namespace particles::traits``.
        (Regression: the unqualified form is a hard C++ compile error.)
        """
        field_context, typename = make_derived_field_context("electron")
        rendered = render_file_output([field_context])

        qualified = f"particleToGrid::derivedAttributes::{typename}"
        self.assertIn(qualified, rendered, "emitted trait must fully qualify the derived attribute")

        # the unqualified specialization form must not appear anywhere
        unqualified = re.search(r"SpeciesEligibleForSolver<T_Species,\s*derivedAttributes::", rendered)
        self.assertIsNone(unqualified, "emitted trait must not reference a bare 'derivedAttributes::'")

    def test_derived_field_eligibility_keys_on_species_name(self):
        """The emitted specialization narrows on the species' compile-time name via
        ``GetCTName`` / ``PMACC_CSTRING``."""
        field_context, _ = make_derived_field_context("electron")
        rendered = render_file_output([field_context])

        self.assertIn('std::is_same_v<pmacc::traits::GetCTName_t<T_Species>, PMACC_CSTRING("electron")>', rendered)

    def test_native_field_dump_emits_no_eligibility_specialization(self):
        """A native E/B/J dump (functor=None) must not emit a derived attribute struct
        or a species-eligibility narrowing block."""
        native = FieldDump(name="E", functor=None, filtername=None, species_name=None).get_rendering_context()
        rendered = render_file_output([native])

        self.assertNotIn("namespace particles::traits", rendered)
        self.assertNotIn("struct SpeciesEligibleForSolver", rendered)
        self.assertNotIn("CreateEligible_t", rendered)


@pytest.mark.parametrize(
    "bounds, expected",
    [
        ((None, None, None), ":,:,:"),
        (((1780, 1804), None, None), "1780:1804,:,:"),
        ((None, (1780, 1804), None), ":,1780:1804,:"),
        ((None, None, (884, 908)), ":,:,884:908"),
    ],
)
@pytest.mark.parametrize("as_object", [False, True])
def test_range_preserves_axis_positions(bounds, expected, as_object):
    # C++ toRangeSlice skips empty components, so commas alone are insufficient.
    config = diag.OpenPMDConfig(file="fields", range=RangeSpec(*bounds) if as_object else bounds)
    assert config.model_dump(mode="json")["range"] == expected


def test_builtin_selectors_reject_unknown_attributes():
    species = picmi.Species(name="e", particle_type="electron")
    with pytest.raises(ValidationError):
        diag.DerivedFieldDump(species=species, functor="UnknownDensity")


def test_builtin_fields_species_filters_and_shared_files(tmp_path):
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[32] * 3,
        lower_bound=[0] * 3,
        upper_bound=[1e-5] * 3,
        lower_boundary_conditions=["periodic"] * 3,
        upper_boundary_conditions=["periodic"] * 3,
    )
    sim = picmi.Simulation(
        solver=picmi.ElectromagneticSolver(grid=grid, method="Yee"),
        time_step_size=1e-16,
        max_steps=10,
    )
    electron = picmi.Species(name="e", particle_type="electron")
    proton = picmi.Species(name="p", particle_type="proton")
    for species in (electron, proton):
        sim.add_species(species, layout=None)
    filtered = picmi.FilteredSpecies(
        species=electron, functor=picmi.ParticleFilter(lambda p: p.get("momentumPrev1")[0] > 0, name="forward")
    )
    for file in ("fields", "slice"):
        for species, attribute in [(electron, "ChargeDensity"), (proton, "EnergyDensity"), (filtered, "ChargeDensity")]:
            sim.add_diagnostic(
                diag.DerivedFieldDump(
                    species=species,
                    functor=attribute,
                    options=diag.OpenPMDConfig(file=file),
                )
            )
    output = tmp_path / "setup"
    sim.write_input_file(output)
    context = json.loads((output / "metadata/pypicongpu_rendering_context.json").read_text())
    assert "momentumPrev1" in (output / "include/picongpu/param/speciesDefinition.param").read_text()
    # Two output files must not register duplicate solvers; species and filter differ.
    assert len(context["builtin_derived_fields"]) == 3
    header = (output / "include/picongpu/param/fileOutput.param").read_text()
    assert len(re.findall(r"CreateEligible_t<\s*species_e,", header)) == 2
    assert len(re.findall(r"CreateEligible_t<\s*species_p,", header)) == 1
    assert "picongpu::particles::filter::forward" in header
    filters = (output / "include/picongpu/param/particleFilters.param").read_text()
    assert "using forward =" in filters
    configs = list(output.glob("etc/openPMD_config_*.toml"))
    assert len(configs) == 2
    for file in configs:
        periods = tomllib.loads(file.read_text())["sink"]["dummy_application_name"]["period"]
        assert list(periods.values()) == [["e_all_chargeDensity", "p_all_energyDensity", "e_forward_chargeDensity"]]
