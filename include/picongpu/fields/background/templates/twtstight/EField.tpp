/* Copyright 2014-2024 Alexander Debus, Axel Huebl, Sergei Bastrakov
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
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/fields/background/templates/twtstight/EField.hpp"
#include "picongpu/fields/background/templates/twtstight/GetInitialTimeDelay_SI.tpp"
#include "picongpu/fields/background/templates/twtstight/getFieldPositions_SI.tpp"
#include "picongpu/fields/background/templates/twtstight/twtstight.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    /* Load pre-defined background field */
    namespace templates
    {
        /* Traveling-wave Thomson scattering laser pulse */
        namespace twtstight
        {
            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized<DIM3>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& eFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    for(uint32_t i = 0; i < simDim; ++i)
                        pos[k][i] = eFieldPositions_SI[k][i];
                }

                /* E-field normalized to the peak amplitude. */
                return float3_X(
                    float_X(calcTWTSEx(pos[0], time)),
                    float_X(calcTWTSEy(pos[1], time)),
                    float_X(calcTWTSEz(pos[2], time)));
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized<DIM2>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& eFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
                    for(uint32_t i = 0; i < DIM2; ++i)
                    {
                        pos[k][i + 1] = eFieldPositions_SI[k][i];
                    }
                }

                /* E-field normalized to the peak amplitude. */
                return float3_X(calcTWTSEx(pos[0], time), calcTWTSEy(pos[1], time), calcTWTSEz(pos[2], time));
            }

            HDINLINE float3_X EField::operator()(DataSpace<simDim> const& cellIdx, uint32_t const currentStep) const
            {
                traits::FieldPosition<fields::YeeCell, FieldE> const fieldPosE;
                return getValue(precisionCast<float_X>(cellIdx), fieldPosE(), static_cast<float_X>(currentStep));
            }

            HDINLINE
            float3_X EField::operator()(floatD_X const& cellIdx, float_X const currentStep) const
            {
                pmacc::math::Vector<floatD_X, detail::numComponents> zeroShifts;
                for(uint32_t component = 0; component < detail::numComponents; ++component)
                    zeroShifts[component] = floatD_X::create(0.0);
                return getValue(cellIdx, zeroShifts, currentStep);
            }

            HDINLINE
            float3_X EField::getValue(
                floatD_X const& cellIdx,
                pmacc::math::Vector<floatD_X, detail::numComponents> const& extraShifts,
                float_X const currentStep) const
            {
                float_64 const time_SI = float_64(currentStep) * dt - tdelay;

                pmacc::math::Vector<floatD_64, detail::numComponents> const eFieldPositions_SI
                    = detail::getFieldPositions_SI(cellIdx, halfSimSize, extraShifts, unit_length, focus_y_SI, phi);

                /* Single TWTS-Pulse */
                return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI);
            }

            template<uint32_t T_component>
            HDINLINE float_X EField::getComponent(floatD_X const& cellIdx, float_X const currentStep) const
            {
                // The optimized way is only implemented for 3D, fall back to full field calculation in 2d
                if constexpr(simDim == DIM3)
                {
                    float_64 const time_SI = float_64(currentStep) * dt - tdelay;
                    pmacc::math::Vector<floatD_X, detail::numComponents> zeroShifts;
                    for(uint32_t component = 0; component < detail::numComponents; ++component)
                        zeroShifts[component] = floatD_X::create(0.0);
                    pmacc::math::Vector<floatD_64, detail::numComponents> const eFieldPositions_SI
                        = detail::getFieldPositions_SI(cellIdx, halfSimSize, zeroShifts, unit_length, focus_y_SI, phi);
                    // Explicitly use a 3D vector so that this function compiles for 2D as well
                    auto const pos = float3_64{
                        eFieldPositions_SI[T_component][0],
                        eFieldPositions_SI[T_component][1],
                        eFieldPositions_SI[T_component][2]};
                    if constexpr(T_component == 0)
                        return static_cast<float_X>(calcTWTSEx(pos, time_SI));
                    else
                    {
                        if constexpr(T_component == 1)
                        {
                            return static_cast<float_X>(calcTWTSEy(pos, time_SI));
                        }
                        if constexpr(T_component == 2)
                        {
                            return static_cast<float_X>(calcTWTSEz(pos, time_SI));
                        }
                    }

                    // we should never be here
                    return NAN;
                }
                if constexpr(simDim != DIM3)
                    return (*this)(cellIdx, currentStep)[T_component];
            }

            /** Calculate the Ex(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEx(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                auto const& [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
                    = defineBasicHelperVariables();
                auto const& [x, y, z, t] = defineMinimalCoordinates(pos, time);

                /* To avoid underflows in computation, fields are set to zero
                 * before and after the respective TWTS pulse envelope.
                 */
                if(math::abs(y - z * tanAlpha - (beta0 * cspeed * t)) > (numSigmas * tauG * cspeed))
                    return float_T(0.0);

                auto const& [tanPhi, cotPhi, sinPhi_2, sinPolAngle, cosPolAngle]
                    = defineTrigonometryShortcuts(absPhi, sinPhi);
                auto const& [I, x2, tauG2, psi0, w02, beta02, nu, xi, rhom, Xm, besselI0const, besselJ0const, besselJ1const, zeroOrder]
                    = defineCommonHelperVariables(
                        absPhi,
                        sinPhi,
                        cosPhi,
                        beta0,
                        tanAlpha,
                        cspeed,
                        lambda0,
                        omega0,
                        tauG,
                        w0,
                        k,
                        x,
                        y,
                        z,
                        t,
                        cotPhi,
                        sinPhi_2);

                /* Calculating shortcuts for speeding up field calculation */
                float_T const cosPhi_2 = cosPhi * cosPhi;
                complex_T const rhom2 = rhom * rhom;

                complex_T const result
                    = (complex_T(0, 0.25) * math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder
                       * (k * rhom * besselJ0const
                              * ((rhom2 - x2 + x * Xm * cosPhi) * sinPolAngle * sinPhi_2
                                 + cosPolAngle
                                     * (rhom2 + rhom2 * cosPhi_2 - x2 * sinPhi_2 - x * Xm * cosPhi * sinPhi_2))
                          + besselJ1const * sinPhi
                              * (+sinPolAngle
                                     * (-rhom2 + float_T(2.0) * x2 - I * k * rhom2 * Xm * sinPhi
                                        + x * cosPhi * (float_T(-2.0) * Xm - I * k * rhom2 * sinPhi))
                                 + cosPolAngle
                                     * (-rhom2 + float_T(2.0) * x2 + I * k * rhom2 * Xm * sinPhi
                                        + x * cosPhi * (float_T(+2.0) * Xm + I * k * rhom2 * sinPhi))))
                       * psi0)
                    / (rhom * rhom2 * besselI0const);

                /* A 180deg-rotation of the field vector around the y-axis
                 * leads to a sign flip in the x- and z- components, respectively.
                 * This is implemented by multiplying the result by "phiPositive".
                 */
                return phiPositive * result.real();
            }

            /** Calculate the Ey(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEy(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                auto const& [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
                    = defineBasicHelperVariables();
                auto const& [x, y, z, t] = defineMinimalCoordinates(pos, time);

                /* To avoid underflows in computation, fields are set to zero
                 * before and after the respective TWTS pulse envelope.
                 */
                if(math::abs(y - z * tanAlpha - (beta0 * cspeed * t)) > (numSigmas * tauG * cspeed))
                    return float_T(0.0);

                auto const& [tanPhi, cotPhi, sinPhi_2, sinPolAngle, cosPolAngle]
                    = defineTrigonometryShortcuts(absPhi, sinPhi);
                auto const& [I, x2, tauG2, psi0, w02, beta02, nu, xi, rhom, Xm, besselI0const, besselJ0const, besselJ1const, zeroOrder]
                    = defineCommonHelperVariables(
                        absPhi,
                        sinPhi,
                        cosPhi,
                        beta0,
                        tanAlpha,
                        cspeed,
                        lambda0,
                        omega0,
                        tauG,
                        w0,
                        k,
                        x,
                        y,
                        z,
                        t,
                        cotPhi,
                        sinPhi_2);

                /* Calculating shortcuts for speeding up field calculation */
                float_T const cosPhi_2 = cosPhi * cosPhi;

                complex_T const result = (math::exp(I * (omega0 * t - k * y * cosPhi)) * k * zeroOrder * sinPhi
                                          * (besselJ1const
                                                 * (cosPolAngle * (Xm - float_T(2.0) * x * cosPhi - Xm * cosPhi_2)
                                                    + Xm * (float_T(1.0) + cosPhi_2) * sinPolAngle)
                                             + I * rhom * besselJ0const * (cosPolAngle - sinPolAngle) * sinPhi_2)
                                          * psi0)
                    / (float_T(4.0) * rhom * besselI0const);

                return result.real();
            }

            /** Calculate the Ez(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEz(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                auto const& [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
                    = defineBasicHelperVariables();
                auto const& [x, y, z, t] = defineMinimalCoordinates(pos, time);

                /* To avoid underflows in computation, fields are set to zero
                 * before and after the respective TWTS pulse envelope.
                 */
                if(math::abs(y - z * tanAlpha - (beta0 * cspeed * t)) > (numSigmas * tauG * cspeed))
                    return float_T(0.0);

                auto const& [tanPhi, cotPhi, sinPhi_2, sinPolAngle, cosPolAngle]
                    = defineTrigonometryShortcuts(absPhi, sinPhi);
                auto const& [I, x2, tauG2, psi0, w02, beta02, nu, xi, rhom, Xm, besselI0const, besselJ0const, besselJ1const, zeroOrder]
                    = defineCommonHelperVariables(
                        absPhi,
                        sinPhi,
                        cosPhi,
                        beta0,
                        tanAlpha,
                        cspeed,
                        lambda0,
                        omega0,
                        tauG,
                        w0,
                        k,
                        x,
                        y,
                        z,
                        t,
                        cotPhi,
                        sinPhi_2);

                /* Calculating shortcuts for speeding up field calculation */
                float_T const sin2Phi = math::sin(float_T(2.0) * absPhi);
                complex_T const rhom2 = rhom * rhom;
                complex_T const Xm2 = Xm * Xm;

                complex_T const result
                    = (complex_T(0, 0.125) * math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder
                       * (float_T(2.0) * k * rhom * besselJ0const
                              * (x * Xm * (cosPolAngle + sinPolAngle) * sinPhi_2
                                 + cosPhi
                                     * (Xm2 * cosPolAngle * sinPhi_2
                                        + sinPolAngle * (float_T(2.0) * rhom2 - Xm2 * sinPhi_2)))
                          + besselJ1const * sinPhi
                              * (cosPolAngle
                                     * (float_T(-4.0) * x * Xm + float_T(2.0) * (rhom2 - float_T(2.0) * Xm2) * cosPhi
                                        + complex_T(0, 2) * k * rhom2 * (x - Xm * cosPhi) * sinPhi)
                                 + sinPolAngle
                                     * (float_T(-4.0) * x * Xm - float_T(2.0) * (rhom2 - float_T(2.0) * Xm2) * cosPhi
                                        - complex_T(0, 2) * k * rhom2 * x * sinPhi + I * k * rhom2 * Xm * sin2Phi)))
                       * psi0)
                    / (rhom * rhom2 * besselI0const);

                /* A 180deg-rotation of the field vector around the y-axis
                 * leads to a sign flip in the x- and z- components, respectively.
                 * This is implemented by multiplying the result by "phiPositive".
                 */
                return phiPositive * result.real();
            }

        } /* namespace twtstight */
    } /* namespace templates */
} /* namespace picongpu */
