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


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/background/templates/twtstight/BField.tpp"
#include "picongpu/fields/background/templates/twtstight/EField.tpp"
// include "picongpu/fields/YeeCell.hpp"
// include "picongpu/fields/background/templates/twtstight/EField.hpp"
#include "picongpu/fields/background/templates/twtstight/GetInitialTimeDelay_SI.tpp"
// include "picongpu/fields/background/templates/twtstight/getFieldPositions_SI.tpp"
#include "picongpu/fields/background/templates/twtstight/twtstight.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Complex.hpp>
// include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    /* Load pre-defined background field */
    namespace templates
    {
        /* Traveling-wave Thomson scattering laser pulse */
        namespace twtstight
        {
            HINLINE
            TWTSTight::TWTSTight(
                float_64 const focus_y_SI,
                float_64 const wavelength_SI,
                float_64 const pulselength_SI,
                float_64 const w_x_SI,
                float_X const phi,
                float_X const beta_0,
                float_64 const tdelay_user_SI,
                bool const auto_tdelay,
                float_X const polAngle)
                : focus_y_SI(focus_y_SI)
                , wavelength_SI(wavelength_SI)
                , pulselength_SI(pulselength_SI)
                , w_x_SI(w_x_SI)
                , phi(phi)
                , phiPositive(float_X(1.0))
                , beta_0(beta_0)
                , tdelay_user_SI(tdelay_user_SI)
                , dt(sim.si.getDt())
                , unit_length(sim.unit.length())
                , auto_tdelay(auto_tdelay)
                , polAngle(polAngle)
            {
                /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
                         on host (see fieldBackground.param), this is no problem.
                 */
                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                halfSimSize = subGrid.getGlobalDomain().size / 2;
                tdelay = detail::getInitialTimeDelay_SI(
                    auto_tdelay,
                    tdelay_user_SI,
                    halfSimSize,
                    pulselength_SI,
                    focus_y_SI,
                    phi,
                    beta_0);
                if(phi < 0.0_X)
                    phiPositive = float_X(-1.0);
            }

            HDINLINE
            std::tuple<
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T>
            TWTSTight::defineBasicHelperVariables() const
            {
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * y-axis of the coordinate system in this function.
                 */
                auto const absPhi = float_T(math::abs(phi));

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                auto const beta0 = float_T(beta_0);

                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(absPhi, sinPhi, cosPhi);
                float_T const tanAlpha = (float_T(1.0) - beta0 * cosPhi) / (beta0 * sinPhi);

                auto const cspeed = float_T(sim.si.getSpeedOfLight() / sim.unit.speed());
                auto const lambda0 = float_T(wavelength_SI / sim.unit.length());
                float_T const omega0 = float_T(2.0 * PI) * cspeed / lambda0;
                /* factor 2  in tauG arises from definition convention in laser formula */
                auto const tauG = float_T(pulselength_SI * 2.0 / sim.unit.time());
                /* w0 is wx here --> w0 could be replaced by wx */
                auto const w0 = float_T(w_x_SI / sim.unit.length());
                auto const k = float_T(2.0 * PI / lambda0);

                return std::make_tuple(absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k);
            }

            HDINLINE
            std::tuple<TWTSTight::float_T, TWTSTight::float_T, TWTSTight::float_T, TWTSTight::float_T> TWTSTight::
                defineMinimalCoordinates(float3_64 const& pos, float_64 const& time) const
            {
                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 const deltaT = wavelength_SI / sim.si.getSpeedOfLight()
                    / (1.0 - beta_0 * pmacc::math::cos(precisionCast<float_64>(phi)));
                float_64 const deltaY = beta_0 * sim.si.getSpeedOfLight() * deltaT;
                float_64 const numberOfPeriods = math::floor(time / deltaT);
                auto const timeMod = float_T(time - numberOfPeriods * deltaT);
                auto const yMod = float_T(pos.y() - numberOfPeriods * deltaY);

                auto const x = float_T(phiPositive * pos.x() / sim.unit.length());
                auto const y = float_T(yMod / sim.unit.length());
                auto const z = float_T(phiPositive * pos.z() / sim.unit.length());
                auto const t = float_T(timeMod / sim.unit.time());

                return std::make_tuple(x, y, z, t);
            }

            HDINLINE
            std::tuple<
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T>
            TWTSTight::defineTrigonometryShortcuts(TWTSTight::float_T const& absPhi, TWTSTight::float_T const& sinPhi)
                const
            {
                /* Calculating shortcuts for speeding up field calculation */
                float_T const tanPhi = math::tan(absPhi);
                float_T const cotPhi = float_T(1.0) / tanPhi;
                float_T const sinPhi_2 = sinPhi * sinPhi;
                float_T const sinPolAngle = math::sin(polAngle);
                float_T const cosPolAngle = math::cos(polAngle);

                return std::make_tuple(tanPhi, cotPhi, sinPhi_2, sinPolAngle, cosPolAngle);
            }

            HDINLINE
            std::tuple<
                alpaka::Complex<TWTSTight::float_T>,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                TWTSTight::float_T,
                alpaka::Complex<TWTSTight::float_T>,
                alpaka::Complex<TWTSTight::float_T>,
                TWTSTight::float_T,
                alpaka::Complex<TWTSTight::float_T>,
                alpaka::Complex<TWTSTight::float_T>,
                alpaka::Complex<TWTSTight::float_T>>
            TWTSTight::defineCommonHelperVariables(
                TWTSTight::float_T const& absPhi,
                TWTSTight::float_T const& sinPhi,
                TWTSTight::float_T const& cosPhi,
                TWTSTight::float_T const& beta0,
                TWTSTight::float_T const& tanAlpha,
                TWTSTight::float_T const& cspeed,
                TWTSTight::float_T const& lambda0,
                TWTSTight::float_T const& omega0,
                TWTSTight::float_T const& tauG,
                TWTSTight::float_T const& w0,
                TWTSTight::float_T const& k,
                TWTSTight::float_T const& x,
                TWTSTight::float_T const& y,
                TWTSTight::float_T const& z,
                TWTSTight::float_T const& t,
                TWTSTight::float_T const& cotPhi,
                TWTSTight::float_T const& sinPhi_2) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                complex_T I = complex_T(0, 1);
                float_T const x2 = x * x;
                float_T const tauG2 = tauG * tauG;
                float_T const psi0 = float_T(2.0) / k;
                float_T const w02 = w0 * w0;
                float_T const beta02 = beta0 * beta0;

                float_T const nu = (y * cosPhi - z * sinPhi) / cspeed;
                float_T const xi = (-z * cosPhi - y * sinPhi) * tanAlpha / cspeed;
                complex_T const Xm = -z - complex_T(0, 0.5) * (k * w02);
                complex_T const rhom = math::sqrt(x2 + pmacc::math::cPow(Xm, static_cast<uint32_t>(2u)));
                float_T const besselI0const = pmacc::math::bessel::i0(k * k * sinPhi * w02 / float_T(2.0));
                complex_T const besselJ0const = pmacc::math::bessel::j0(k * sinPhi * rhom);
                complex_T const besselJ1const = pmacc::math::bessel::j1(k * sinPhi * rhom);

                complex_T const zeroOrder = (beta0 * tauG)
                    / (math::sqrt(float_T(2.0))
                       * math::exp(
                           beta02 * omega0 * pmacc::math::cPow(t - nu + xi, static_cast<uint32_t>(2u))
                           / (beta02 * omega0 * tauG2 - complex_T(0, 2) * (beta02 * (nu - xi) * cotPhi * cotPhi)
                              + complex_T(0, 2) * (beta0 * (float_T(2.0) * nu - xi) * cotPhi / sinPhi)
                              - complex_T(0, 2) * (nu / sinPhi_2)))
                       * math::sqrt(
                           ((beta02 * omega0 * tauG2) / float_T(2.0) - I * (beta02 * (nu - xi) * cotPhi * cotPhi)
                            + I * (beta0 * (float_T(2.0) * nu - xi) * cotPhi / sinPhi) - I * (nu / sinPhi_2))
                           / omega0));

                return std::make_tuple(
                    I,
                    x2,
                    tauG2,
                    psi0,
                    w02,
                    beta02,
                    nu,
                    xi,
                    rhom,
                    Xm,
                    besselI0const,
                    besselJ0const,
                    besselJ1const,
                    zeroOrder);
            }

        } /* namespace twtstight */
    } /* namespace templates */
} /* namespace picongpu */
