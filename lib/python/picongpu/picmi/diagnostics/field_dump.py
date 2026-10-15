"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from os import PathLike
from pathlib import Path
from typing import Literal

from picmistandard import PICMI_Diagnostic
from pydantic import ConfigDict, computed_field

from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.openpmd_plugin import NATIVE_FIELDS, BuiltinDerivedAttribute
from .backend_config import BackendConfig, OpenPMDConfig
from .timestepspec import TimeStepSpec
from picongpu.picmi.particle_functor import ParticleFunctor


class _FieldDump(PICMI_Diagnostic):
    period: TimeStepSpec = TimeStepSpec[:]("steps")
    options: BackendConfig = OpenPMDConfig(file="simData")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def result_path(self, prefix_path: PathLike):
        return self.options.result_path(prefix_path=Path(prefix_path) / "simOutput" / "openPMD")


class NativeFieldDump(_FieldDump):
    fieldname: Literal[*NATIVE_FIELDS]
    filtername: None = None


class DerivedFieldDump(_FieldDump):
    """Deposit a particle functor or a PIConGPU built-in density attribute.

    ``functor="ChargeDensity"`` and ``functor="EnergyDensity"`` select the
    existing C++ implementations, including their units and weighting rules.
    These string selectors are PIConGPU extensions, not PICMI-standard names.
    """

    species: Species | FilteredSpecies
    functor: ParticleFunctor | BuiltinDerivedAttribute

    @computed_field
    def filtername(self) -> None | str:
        return None if isinstance(self.species, Species) else self.species.functor.name

    @computed_field
    def species_name(self) -> str:
        """Compile-time name of the source species, for species-eligibility narrowing."""
        return self.species.name if isinstance(self.species, Species) else self.species.species.name

    @computed_field
    def fieldname(self) -> str:
        name = self.functor[0].lower() + self.functor[1:] if isinstance(self.functor, str) else self.functor.name
        return f"{self.species_name}_{self.filtername or 'all'}_{name}"
