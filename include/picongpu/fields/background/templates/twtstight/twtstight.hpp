/* Copyright 2014-2024 Alexander Debus
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file
 *
 * This background field implements a obliquely incident, cylindrically-focused, pulse-front tilted laser for some
 * incidence angle phi as used for [1].
 *
 * The TWTS implementation is based on the definition of eq. (7) in [1]. Additionally, techniques from [2] and [3]
 * are used to allow for strictly Maxwell-conform solutions for tight foci wx or small incident angles phi.
 *
 * Specifically, this TWTStight implementation assumes a special case, where the transverse extent (but not its height
 * wx or its pulse duration) of the TWTS-laser wy is assumed to be infinite. While this special case of the TWTS laser
 * applies to a large range of use cases, the resulting form allows to use different spatial and time coordinates
 * (timeMod, yMod and zMod), which allow long term numerical stability beyond 100000 timesteps at single precision,
 * as well as for mitigating errors of the approximations far from the coordinate origin.
 *
 * We exploit the wavelength-periodicity and the known propagation direction for realizing the laser pulse
 * using relative coordinates (i.e. from a finite coordinate range) only. All these quantities have to be calculated
 * in double precision.
 *
 * float_64 const tanAlpha = (1.0 - beta_0 * math::cos(phi)) / (beta_0 * math::sin(phi));
 * float_64 const tanFocalLine = math::tan(PI / 2.0 - phi);
 * float_64 const deltaT = wavelength_SI / sim.si.getSpeedOfLight() * (1.0 + tanAlpha / tanFocalLine);
 * float_64 const deltaY = wavelength_SI * math::cos(phi) + wavelength_SI * math::sin(phi) * math::sin(phi) /
 * math::sin(phi); float_64 const numberOfPeriods = math::floor(time / deltaT); auto const timeMod = float_T(time -
 * numberOfPeriods * deltaT); auto const yMod = float_T(pos.y() - numberOfPeriods * deltaY);
 *
 * Literature:
 * [1] Steiniger et al., "Optical free-electron lasers with Traveling-Wave Thomson-Scattering",
 *     Journal of Physics B: Atomic, Molecular and Optical Physics, Volume 47, Number 23 (2014),
 *     https://doi.org/10.1088/0953-4075/47/23/234011
 * [2] Mitri, F. G., "Cylindrical quasi-Gaussian beams", Opt. Lett., 38(22), pp. 4727-4730 (2013),
 *     https://doi.org/10.1364/OL.38.004727
 * [3] Hua, J. F., "High-order corrected fields of ultrashort, tightly focused laser pulses",
 *     Appl. Phys. Lett. 85, 3705-3707 (2004),
 *     https://doi.org/10.1063/1.1811384
 *
 */

#pragma once
#include "picongpu/defines.hpp"
// include "picongpu/fields/background/templates/twtstight/numComponents.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
// include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    namespace templates
    {
        namespace twtstight
        {
            /** To avoid underflows in computation, numsigmas controls where a zero cutoff is made.
             *  The fields thus are set to zero at a position (numSigmas * tauG * cspeed) ahead
             *  and behind the respective TWTS pulse envelope.
             *  Developer note: In case the float_T-type is set to float_64 instead of float_X,
             *  numSigma can be increased to numSigmas = 10 without running into numerical issues.
             */
            constexpr uint32_t numSigmas = 6;

            class TWTSTight
            {
            public:
                using float_T = float_X;

                /** Center of simulation volume in number of cells */
                PMACC_ALIGN(halfSimSize, DataSpace<simDim>);
                /** y-position of TWTS coordinate origin inside the simulation coordinates [meter]
                    The other origin coordinates (x and z) default to globally centered values
                    with respect to the simulation volume. */
                PMACC_ALIGN(focus_y_SI, float_64 const);
                /** Laser wavelength [meter] */
                PMACC_ALIGN(wavelength_SI, float_64 const);
                /** TWTS laser pulse duration [second] */
                PMACC_ALIGN(pulselength_SI, float_64 const);
                /** line focus height of TWTS pulse [meter] */
                PMACC_ALIGN(w_x_SI, float_64 const);
                /** TWTS interaction angle
                 *  Enclosed by the laser propagation direction and the y-axis.
                 *  For a positive value of the interaction angle, the laser propagation direction
                 *  points along the y-axis and against the z-axis.
                 *  That is, for phi = 90 degree the laser propagates in the -z direction.
                 * [rad]
                 */
                PMACC_ALIGN(phi, float_X const);
                /** Takes value 1.0 for phi > 0 and -1.0 for phi < 0. */
                PMACC_ALIGN(phiPositive, float_X);
                /** propagation speed of TWTS laser overlap
                normalized to the speed of light. [Default: beta0=1.0] */
                PMACC_ALIGN(beta_0, float_X const);
                /** If auto_tdelay=FALSE, then a user defined delay is used. [second] */
                PMACC_ALIGN(tdelay_user_SI, float_64 const);
                /** Make time step constant accessible to device. */
                PMACC_ALIGN(dt, float_64 const);
                /** Make length normalization constant accessible to device. */
                PMACC_ALIGN(unit_length, float_64 const);
                /** TWTS laser time delay */
                PMACC_ALIGN(tdelay, float_64);
                /** Should the TWTS laser delay be chosen automatically, such that
                 *  the laser gradually enters the simulation volume? [Default: TRUE]
                 */
                PMACC_ALIGN(auto_tdelay, bool const);
                /** Polarization of TWTS laser with respect to x-axis around propagation direction [rad, default = 0. *
                 * (PI/180.)] */
                PMACC_ALIGN(polAngle, float_X const);

                /** Electric or magnetic field of the TWTS laser
                 *
                 * @param focus_y_SI the distance to the laser focus in y-direction [m]
                 * @param wavelength_SI central wavelength [m]
                 * @param pulselength_SI sigma of std. gauss for intensity (E^2),
                 *  pulselength_SI = FWHM_of_Intensity / 2.35482 [seconds (sigma)]
                 * @param w_x beam waist: distance from the axis where the pulse electric field
                 *  decreases to its 1/e^2-th part at the focus position of the laser [m]
                 * @param phi interaction angle between TWTS laser propagation vector and
                 *  the y-axis [rad, default = 90. * (PI/180.)]
                 * @param beta_0 propagation speed of overlap normalized to
                 *  the speed of light [c, default = 1.0]
                 * @param tdelay_user manual time delay if auto_tdelay is false
                 * @param auto_tdelay calculate the time delay such that the TWTS pulse is not
                 *  inside the simulation volume at simulation start timestep = 0 [default = true]
                 * @param polAngle determines the TWTS laser polarization angle with respect to x-axis around
                 * propagation direction [rad, default = 0. * (PI/180.)] Normal to laser pulse front tilt plane:
                 * polAngle = 0.0 * (PI/180.) (linear polarization parallel to x-axis) Parallel to laser pulse front
                 * tilt plane: polAngle = 90.0 * (PI/180.) (linear polarization parallel to yz-plane)
                 */
                HINLINE
                TWTSTight(
                    float_64 const focus_y_SI,
                    float_64 const wavelength_SI,
                    float_64 const pulselength_SI,
                    float_64 const w_x_SI,
                    float_X const phi = 90. * (PI / 180.),
                    float_X const beta_0 = 1.0,
                    float_64 const tdelay_user_SI = 0.0,
                    bool const auto_tdelay = true,
                    float_X const polAngle = 0. * (PI / 180.));

                HDINLINE
                std::tuple<
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T>
                defineBasicHelperVariables() const;

                HDINLINE
                std::tuple<float_T, float_T, float_T, float_T> defineMinimalCoordinates(
                    float3_64 const& pos,
                    float_64 const& time) const;

                HDINLINE
                std::tuple<float_T, float_T, float_T, float_T, float_T> defineTrigonometryShortcuts(
                    float_T const& absPhi,
                    float_T const& sinPhi) const;

                HDINLINE
                std::tuple<
                    alpaka::Complex<float_T>,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    float_T,
                    alpaka::Complex<float_T>,
                    alpaka::Complex<float_T>,
                    float_T,
                    alpaka::Complex<float_T>,
                    alpaka::Complex<float_T>,
                    alpaka::Complex<float_T>>
                defineCommonHelperVariables(
                    float_T const& absPhi,
                    float_T const& sinPhi,
                    float_T const& cosPhi,
                    float_T const& beta0,
                    float_T const& tanAlpha,
                    float_T const& cspeed,
                    float_T const& lambda0,
                    float_T const& omega0,
                    float_T const& tauG,
                    float_T const& w0,
                    float_T const& k,
                    float_T const& x,
                    float_T const& y,
                    float_T const& z,
                    float_T const& t,
                    float_T const& cotPhi,
                    float_T const& sinPhi_2) const;
            };

        } // namespace twtstight
    } // namespace templates
} // namespace picongpu

#include "picongpu/fields/background/templates/twtstight/BField.hpp"
#include "picongpu/fields/background/templates/twtstight/EField.hpp"
#include "picongpu/fields/background/templates/twtstight/twtstight.tpp"
