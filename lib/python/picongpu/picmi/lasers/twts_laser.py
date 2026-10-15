"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Alexander Debus
License: GPLv3+
"""

import math
from collections.abc import Sequence

import numpy as np
from picmistandard import PICMI_Laser, resolve_once
from pydantic import Field, computed_field, model_validator

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser, PositiveFloat
from .polarization_type import PolarizationType

from .. import constants


@default_converts_to(laser.TWTSLaser)
class TWTSLaser(PICMI_Laser, BaseLaser):
    """
    Specifies a Traveling-Wave Thomson Scattering (TWTS) laser

    Parameters
    ----------
    wavelength: float
        Central wavelength of the laser [m], must be > 0
    waist: float
        Spot size (1/e^2 radius) of the laser at focus [m], must be > 0
    duration: float
        Duration of the TWTS pulse [s], must be > 0
    laserIncidenceAngle: float
        Laser incidence angle [rad]
    polarizationAngle: float
        Linear laser polarization direction as rotation angle [rad]
    focal_position: list[float]
        3D coordinates of the laser focus [m]
    centroid_position: list[float]
        3D coordinates of the initial laser centroid [m]
        Its y coordinate sets the manual time delay relative to the focus and
        may be positive. This does not initialize fields in the domain at t=0;
        the user must choose a suitable incident-field switching window and
        allow time for the injected field to propagate into the domain.
    focus_lateral_offset_si: float
        Offset from the middle of the simulation domain to the laser focus [m]
    a0: float, optional
        Normalized vector potential at focus. Specify either a0 or E0.
    E0: float, optional
        Peak electric field amplitude [V/m]. Specify either a0 or E0.
    beta0: float, default 1.0
        Laser centroid speed normalized to speed of light, must be > 0
    windowStart: float, default 0.0
    windowEnd: float, default 0.0
    windowLength: float, default 0.0
    """

    wavelength: PositiveFloat
    waist: PositiveFloat
    duration: PositiveFloat
    laserIncidenceAngle: float
    polarizationAngle: float
    focal_position: Sequence[float]
    centroid_position: Sequence[float]
    focus_lateral_offset_si: float = 0.0
    a0: float | None = None
    E0: float | None = None
    beta0: PositiveFloat = 1.0
    windowStart: float = 0.0
    windowEnd: float = 0.0
    windowLength: float = 0.0

    picongpu_huygens_surface_positions: list[list[int]] = Field(
        default_factory=lambda: [[16, -16], [16, -16], [16, -16]]
    )
    picongpu_polarization_type: PolarizationType = PolarizationType.LINEAR

    @computed_field
    def pulse_init(self) -> float:
        """Unused compatibility value required by the common backend model.

        The TWTS template uses TDELAY and the switching window, not PULSE_INIT
        or RAMP_INIT. Do not derive this value from the centroid position.
        """
        return 0.0

    @computed_field
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @computed_field
    def phi0(self) -> float:
        # TWTS has no carrier phase input; the phase is always zero.
        return 0.0

    @computed_field
    def propagation_direction(self) -> list[float]:
        return [0.0, math.cos(self.laserIncidenceAngle), math.sin(self.laserIncidenceAngle)]

    @computed_field
    def polarization_direction(self) -> list[float]:
        # For k=(0, cos(phi), sin(phi)), the transverse basis is
        # ex=(1, 0, 0), e_perp=(0, -sin(phi), cos(phi)). The nominal linear
        # polarization axis is cos(alpha)*ex + sin(alpha)*e_perp.
        # For negative phi, TWTSTight rotates the positive-|phi| beam by pi
        # around y (EField.tpp); its vector is the negative of this equivalent
        # axis representative. The field functor uses the unchanged scalar
        # polarizationAngle, so this metadata does not change the field phase.
        return [
            math.cos(self.polarizationAngle),
            -math.sin(self.polarizationAngle) * math.sin(self.laserIncidenceAngle),
            math.sin(self.polarizationAngle) * math.cos(self.laserIncidenceAngle),
        ]

    @computed_field
    def laserIncidenceAnglePositive(self) -> bool:
        return self.laserIncidenceAngle > 0

    @computed_field
    def time_offset_si(self) -> float:
        return (self.focal_position[1] - self.centroid_position[1]) / (self.beta0 * constants.c)

    def _validate_common_properties(self):
        """Validate TWTS directions independently of the incoming-pulse model.

        TWTS uses a manual TDELAY and a separate incident-field time window.
        Its centroid sets the analytic pulse's time origin, not an initial
        field distribution, so the base validator's y <= 0 restriction does
        not apply. Keep the direction checks without calling that validator.
        """
        if not np.allclose(n := np.linalg.norm(self.polarization_direction), 1):
            raise ValueError(
                "The polarization direction vector must be normalized. "
                f"You gave {self.polarization_direction=} with norm {n}."
            )

        if not np.allclose(n := np.linalg.norm(self.propagation_direction), 1):
            raise ValueError(
                "The propagation direction vector must be normalized. "
                f"You gave {self.propagation_direction=} with norm {n}."
            )

        if self.propagation_direction[1] <= 0.0:
            raise ValueError(
                "Laser propagation parallel to the y-plane or pointing outside "
                "from the inside of the simulation box is not supported by this "
                f"laser in PICMI. You gave {self.propagation_direction=}."
            )

    @model_validator(mode="after")
    @resolve_once
    def _validate(self):
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, self.E0, self.a0)
        self._validate_common_properties()
        return self
