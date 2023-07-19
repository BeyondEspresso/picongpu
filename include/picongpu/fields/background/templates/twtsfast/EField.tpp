/* Copyright 2014-2022 Alexander Debus, Axel Huebl, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/CellType.hpp"
#include "picongpu/fields/background/templates/twtsfast/EField.hpp"
#include "picongpu/fields/background/templates/twtsfast/GetInitialTimeDelay_SI.tpp"
#include "picongpu/fields/background/templates/twtsfast/RotateField.tpp"
#include "picongpu/fields/background/templates/twtsfast/getFieldPositions_SI.tpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>
#include <stdio.h>

namespace picongpu
{
    /* Load pre-defined background field */
    namespace templates
    {
        /* Traveling-wave Thomson scattering laser pulse */
        namespace twtsfast
        {
            HINLINE
            EField::EField(
                float_64 const focus_y_SI,
                float_64 const wavelength_SI,
                float_64 const pulselength_SI,
                float_64 const w_x_SI,
                float_X const phi,
                float_X const beta_0,
                float_64 const tdelay_user_SI,
                bool const auto_tdelay,
                PolarizationType const pol)
                : focus_y_SI(focus_y_SI)
                , wavelength_SI(wavelength_SI)
                , pulselength_SI(pulselength_SI)
                , w_x_SI(w_x_SI)
                , phi(phi)
                , phiPositive(float_X(1.0))
                , beta_0(beta_0)
                , tdelay_user_SI(tdelay_user_SI)
                , dt(SI::DELTA_T_SI)
                , unit_length(UNIT_LENGTH)
                , auto_tdelay(auto_tdelay)
                , pol(pol)
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

                /* Calculate Ez-component with the intra-cell offset of a Ey-field */
                float_64 const Ez_Ey = calcTWTSEz_Ex(pos[1], time);
                /* Calculate Ez-component with the intra-cell offset of a Ez-field */
                float_64 const Ez_Ez = calcTWTSEz_Ex(pos[2], time);

                /* Since we rotated all position vectors before calling calcTWTSEz_Ex,
                 * we need to back-rotate the resulting E-field vector.
                 *
                 * RotationMatrix[-(PI/2+phi)].(Ey,Ez) for rotating back the field-vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const Ey_rot = +cosPhi * float_X(Ez_Ey);
                float_X const Ez_rot = -sinPhi * float_X(Ez_Ez);

                /* Finally, the E-field normalized to the peak amplitude. */
                return float3_X(float_X(calcTWTSEx(pos[0], time)), Ey_rot, Ez_rot);
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized_Ey<DIM3>(
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

                /* Calculate Ey-component with the intra-cell offset of a Ey-field */
                float_64 const Ey_Ey = calcTWTSEy(pos[1], time);
                /* Calculate Ey-component with the intra-cell offset of a Ez-field */
                float_64 const Ey_Ez = calcTWTSEy(pos[2], time);
                /* Calculate Ez-component with the intra-cell offset of a Ey-field */
                float_64 const Ez_Ey = calcTWTSEz_Ey(pos[1], time);
                /* Calculate Ez-component with the intra-cell offset of a Ez-field */
                float_64 const Ez_Ez = calcTWTSEz_Ey(pos[2], time);

                /* Since we rotated all position vectors before calling calcTWTSEy,
                 * we need to back-rotate the resulting E-field vector.
                 *
                 * RotationMatrix[-(PI/2+phi)].(Ey,Ez) for rotating back the field-vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const Ey_rot = -sinPhi * float_X(Ey_Ey) + cosPhi * float_X(Ez_Ey);
                float_X const Ez_rot = -cosPhi * float_X(Ey_Ez) - sinPhi * float_X(Ez_Ez);

                /* Finally, the E-field normalized to the peak amplitude. */
                return float3_X(0.0_X, Ey_rot, Ez_rot);
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

                /* General background comment for the rest of this function:
                 *
                 * Corresponding position vector for the field components in 2D simulations.
                 *  3D     3D vectors in 2D space (x, y)
                 *  x -->  z (Meaning: In 2D-sim, insert cell-coordinate x
                 *            into TWTS field function coordinate z.)
                 *  y -->  y
                 *  z --> -x (Since z=0 for 2D, we use the existing
                 *            3D TWTS-field-function and set x = -0)
                 *  The transformed 3D coordinates are used to calculate the field components.
                 *  Ex --> Ez (Meaning: Calculate Ex-component of existing 3D TWTS-field (calcTWTSEx) using
                 *             transformed position vectors to obtain the corresponding Ez-component in 2D.
                 *             Note: Swapping field component coordinates also alters the
                 *                   intra-cell position offset.)
                 *  Ez --> -Ex (Yes, the sign is necessary.)
                 *  By --> By
                 *  Bz --> -Bx
                 *
                 * An example of intra-cell position offsets is the staggered Yee-grid.
                 *
                 * This procedure is analogous to 3D case, but replace Ex --> Bz and Ez --> -Ex. Hence the
                 * grid cell offset for Ex has to be used instead of Ez. Mind the "-"-sign.
                 */

                /* Calculate Ex-component with the intra-cell offset of a Ey-field */
                float_64 const Ex_Ey = -calcTWTSEz_Ex(pos[1], time);
                /* Calculate Ex-component with the intra-cell offset of a Ex-field */
                float_64 const Ex_Ex = -calcTWTSEz_Ex(pos[0], time);
                /* Since we rotated all position vectors before calling calcTWTSEz_Ex, we
                 * need to back-rotate the resulting E-field vector. Now the rotation is done
                 * analogously in the (y,x)-plane. (Reverse of the position vector transformation.)
                 *
                 * RotationMatrix[-(PI / 2+phi)].(Ey,Ex) for rotating back the field vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const Ey_rot = + cosPhi * float_X(Ex_Ey);
                float_X const Ex_rot = - sinPhi * float_X(Ex_Ex);

                /* Finally, the E-field normalized to the peak amplitude. */
                return float3_X(Ex_rot, Ey_rot, calcTWTSEx(pos[2], time));
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized_Ey<DIM2>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& eFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    /* The 2D output of getFieldPositions_SI only returns
                     * the y- and z-component of a 3D vector.
                     */
                    for(uint32_t i = 0; i < DIM2; ++i)
                    {
                        pos[k][i + 1] = eFieldPositions_SI[k][i];
                    }
                }

                /* General background comment for the rest of this function:
                 *
                 * Corresponding position vector for the field components in 2D simulations.
                 *  3D     3D vectors in 2D space (x, y)
                 *  x -->  z (Meaning: In 2D-sim, insert cell-coordinate x
                 *            into TWTS field function coordinate z.)
                 *  y -->  y
                 *  z --> -x (Since z=0 for 2D, we use the existing
                 *            3D TWTS-field-function and set x = -0)
                 *  The transformed 3D coordinates are used to calculate the field components.
                 *  Ex --> Ez (Meaning: Calculate Ex-component of existing 3D TWTS-field (calcTWTSEx) using
                 *             transformed position vectors to obtain the corresponding Ez-component in 2D.
                 *             Note: Swapping field component coordinates also alters the
                 *                   intra-cell position offset.)
                 *  Ez --> -Ex (Yes, the sign is necessary.)
                 *  By --> By
                 *  Bz --> -Bx
                 *
                 * An example of intra-cell position offsets is the staggered Yee-grid.
                 *
                 * This procedure is analogous to 3D case, but replace Ex --> Bz and Ez --> -Ex. Hence the
                 * grid cell offset for Ex has to be used instead of Ez. Mind the "-"-sign.
                 */


                /* Calculate Ey-component with the intra-cell offset of a Ey-field */
                float_64 const Ey_Ey = calcTWTSEy(pos[1], time);
                /* Calculate Ex-component with the intra-cell offset of a Ey-field */
                float_64 const Ex_Ey = -calcTWTSEz_Ey(pos[1], time);
                /* Calculate Ey-component with the intra-cell offset of a Ex-field */
                float_64 const Ey_Ex = calcTWTSEy(pos[0], time);
                /* Calculate Ex-component with the intra-cell offset of a Ex-field */
                float_64 const Ex_Ex = -calcTWTSEz_Ey(pos[0], time);

                /* Since we rotated all position vectors before calling calcTWTSEz_Ey, we
                 * need to back-rotate the resulting E-field vector. Now the rotation is done
                 * analogously in the (y,x)-plane. (Reverse of the position vector transformation.)
                 *
                 * RotationMatrix[-(PI / 2+phi)].(Ey,Ex)
                 * for rotating back the field-vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const Ey_rot = -sinPhi * float_X(Ey_Ey) + cosPhi * float_X(Ex_Ey);
                float_X const Ex_rot = -cosPhi * float_X(Ey_Ex) - sinPhi * float_X(Ex_Ex);

                /* Finally, the E-field normalized to the peak amplitude. */
                return float3_X(Ex_rot, Ey_rot, 0.0_X);
            }

            HDINLINE float3_X EField::operator()(DataSpace<simDim> const& cellIdx, uint32_t const currentStep) const
            {
                traits::FieldPosition<fields::CellType, FieldE> const fieldPosE;
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
                switch(pol)
                {
                case LINEAR_X:
                    return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI);

                case LINEAR_YZ:
                    return getTWTSEfield_Normalized_Ey<simDim>(eFieldPositions_SI, time_SI);
                }
                return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI); // defensive default
            }

            template<uint32_t T_component>
            HDINLINE float_X EField::getComponent(floatD_X const& cellIdx, float_X const currentStep) const
            {
                // The optimized way is only implemented for 3d, fall back to full field calculation in 2d
                if constexpr(simDim == DIM3)
                {
                    float_64 const time_SI = float_64(currentStep) * dt - tdelay;
                    pmacc::math::Vector<floatD_X, detail::numComponents> zeroShifts;
                    for(uint32_t component = 0; component < detail::numComponents; ++component)
                        zeroShifts[component] = floatD_X::create(0.0);
                    pmacc::math::Vector<floatD_64, detail::numComponents> const eFieldPositions_SI
                        = detail::getFieldPositions_SI(cellIdx, halfSimSize, zeroShifts, unit_length, focus_y_SI, phi);
                    // Explicitly use a 3d vector so that this function compiles for 2d as well
                    auto const pos = float3_64{
                        eFieldPositions_SI[T_component][0],
                        eFieldPositions_SI[T_component][1],
                        eFieldPositions_SI[T_component][2]};
                    switch(pol)
                    {
                    case LINEAR_X:
                        if constexpr(T_component == 0)
                            return static_cast<float_X>(calcTWTSEx(pos, time_SI));
                        else
                            return 0.0_X;

                    case LINEAR_YZ:
                        if constexpr(T_component == 0)
                            return 0.0_X;
                        else
                        {
                            auto const field = calcTWTSEy(pos, time_SI);
                            float_X sinPhi;
                            float_X cosPhi;
                            pmacc::math::sincos(phi, sinPhi, cosPhi);
                            if constexpr(T_component == 1)
                                return -sinPhi * field;
                            if constexpr(T_component == 2)
                                return -cosPhi * field;
                        }
                    }
                    // we should never be here
                    return 0.0_X;
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

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                auto const beta0 = float_T(beta_0);
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                auto const phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                float_T const alphaTilt = math::atan2(float_T(1.0) - beta0 * cosPhiReal, beta0 * sinPhiReal);
                /* Definition of the laser pulse front tilt angle for the laser field below.
                 *
                 * For beta0 = 1.0, this is equivalent to our standard definition. Question: Why is the
                 * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
                 * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
                 * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
                 * the dispersion will (although physically correct) be slightly off the ideal TWTS
                 * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
                 * scenarios close to beta0 = 1.
                 */
                float_T const phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
                 * documentation purposes.
                 * float_T const eta = (PI / 2) - (phiReal - alphaTilt);
                 */

                auto const cspeed = float_T(SI::SPEED_OF_LIGHT_SI / UNIT_SPEED);
                auto const lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
                float_T const om0 = float_T(2.0 * PI) * cspeed / lambda0;
                /* factor 2  in tauG arises from definition convention in laser formula */
                auto const tauG = float_T(pulselength_SI * 2.0 / UNIT_TIME);
                /* w0 is wx here --> w0 could be replaced by wx */
                auto const w0 = float_T(w_x_SI / UNIT_LENGTH);
                auto const rho0 = float_T(PI * w0 * w0 / lambda0);
                auto const k = float_T(2.0 * PI / lambda0);

                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 sinPhiVal;
                float_64 cosPhiVal;
                pmacc::math::sincos(precisionCast<float_64>(phi), sinPhiVal, cosPhiVal);
                float_64 const tanAlpha = (1.0 - beta_0 * cosPhiVal) / (beta_0 * sinPhiVal);
                float_64 const tanFocalLine = math::tan(PI / 2.0 - phi);
                float_64 const deltaT = wavelength_SI / SI::SPEED_OF_LIGHT_SI * (1.0 + tanAlpha / tanFocalLine);
                float_64 const deltaY = wavelength_SI / tanFocalLine;
                float_64 const deltaZ = -wavelength_SI;
                float_64 const numberOfPeriods = math::floor(time / deltaT);
                auto const timeMod = float_T(time - numberOfPeriods * deltaT);
                auto const yMod = float_T(pos.y() + numberOfPeriods * deltaY);
                auto const zMod = float_T(pos.z() + numberOfPeriods * deltaZ);

                auto const x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
                auto const y = float_T(phiPositive * yMod / UNIT_LENGTH);
                auto const z = float_T(zMod / UNIT_LENGTH);
                auto const t = float_T(timeMod / UNIT_TIME);

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                float_T const cotPhi = float_T(1.0) / math::tan(phiT);
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));

                float_T const tanPhi2_2 = tanPhi2 * tanPhi2;

                float_T const tauG2 = tauG * tauG;

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                const complex_T helpVar1 = float_T(2.0)*cspeed*t - complex_T(0,1)*cspeed*om0*tauG2
                                         - float_T(2.0)*z + float_T(2.0)*y*tanPhi2 - float_T(2.0)*(z + y*cotPhi)*tanPhi2_2;
                const complex_T helpVar2 = (
                            - (om0*om0*tauG2) - (complex_T(0,2)*k*x*x)/(complex_T(0,1)*rho0 - y*cosPhi - z*sinPhi)
                            - (complex_T(0,4)*om0*y*tanPhi2)/cspeed
                            + (complex_T(0,2)*om0*(z + y*cotPhi)*tanPhi2_2)/cspeed
                            - (om0*helpVar1*helpVar1)/(cspeed*(cspeed*om0*tauG2 - complex_T(0,2)*(z + y*cotPhi)*tanPhi2_2))
                           )/float_T(4.0);
                complex_T const result =
                    (
                        math::exp(helpVar2)*tauG
                        *math::sqrt
                        (
                            rho0/(complex_T(0,1)*rho0 - y*cosPhi - z*sinPhi)
                        )
                    )/
                    math::sqrt(tauG2 - (complex_T(0,2)*(z + y*cotPhi)*tanPhi2_2)/(cspeed*om0));

                /* A 180°-rotation of the field vector around the z-axis
                 * leads to a sign flip in the x- and y- components, respectively.
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
                /* The field function of Ey (polarization in pulse-front-tilt plane)
                 * is by definition identical to Ex (polarization normal to pulse-front-tilt plane)
                 */
                return calcTWTSEx(pos, time);
            }

            /** Calculate the Ez(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEz_Ex(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                auto const beta0 = float_T(beta_0);
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                auto const phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                float_T const alphaTilt = math::atan2(float_T(1.0) - beta0 * cosPhiReal, beta0 * sinPhiReal);
                /* Definition of the laser pulse front tilt angle for the laser field below.
                 *
                 * For beta0 = 1.0, this is equivalent to our standard definition. Question: Why is the
                 * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
                 * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
                 * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
                 * the dispersion will (although physically correct) be slightly off the ideal TWTS
                 * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
                 * scenarios close to beta0 = 1.
                 */
                float_T const phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
                 * documentation purposes.
                 * float_T const eta = (PI / 2) - (phiReal - alphaTilt);
                 */

                auto const cspeed = float_T(SI::SPEED_OF_LIGHT_SI / UNIT_SPEED);
                auto const lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
                float_T const om0 = float_T(2.0 * PI) * cspeed / lambda0;
                /* factor 2  in tauG arises from definition convention in laser formula */
                auto const tauG = float_T(pulselength_SI * 2.0 / UNIT_TIME);
                /* w0 is wx here --> w0 could be replaced by wx */
                auto const w0 = float_T(w_x_SI / UNIT_LENGTH);
                auto const rho0 = float_T(PI * w0 * w0 / lambda0);
                auto const k = float_T(2.0 * PI / lambda0);

                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 sinPhiVal;
                float_64 cosPhiVal;
                pmacc::math::sincos(precisionCast<float_64>(phi), sinPhiVal, cosPhiVal);
                float_64 const tanAlpha = (1.0 - beta_0 * cosPhiVal) / (beta_0 * sinPhiVal);
                float_64 const tanFocalLine = math::tan(PI / 2.0 - phi);
                float_64 const deltaT = wavelength_SI / SI::SPEED_OF_LIGHT_SI * (1.0 + tanAlpha / tanFocalLine);
                float_64 const deltaY = wavelength_SI / tanFocalLine;
                float_64 const deltaZ = -wavelength_SI;
                float_64 const numberOfPeriods = math::floor(time / deltaT);
                auto const timeMod = float_T(time - numberOfPeriods * deltaT);
                auto const yMod = float_T(pos.y() + numberOfPeriods * deltaY);
                auto const zMod = float_T(pos.z() + numberOfPeriods * deltaZ);

                auto const x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
                auto const y = float_T(phiPositive * yMod / UNIT_LENGTH);
                auto const z = float_T(zMod / UNIT_LENGTH);
                auto const t = float_T(timeMod / UNIT_TIME);

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                float_T const cotPhi = float_T(1.0) / math::tan(phiT);
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));
                float_T const sinPhi2 = math::sin(phiT / float_T(2.0));

                float_T const tanPhi2_2 = tanPhi2 * tanPhi2;
                float_T const sinPhi2_2 = sinPhi2 * sinPhi2;
                float_T const sinPhi_2 = sinPhi * sinPhi;
                float_T const cosPhi_2 = cosPhi * cosPhi;

                float_T const tauG2 = tauG * tauG;
                float_T const cspeed2 = cspeed * cspeed;
                float_T const rho02 = rho0 * rho0;
                float_T const om02 = om0 * om0;
                float_T const x2 = x * x;
                float_T const y2 = y * y;
                float_T const z2 = z * z;

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                const complex_T helpVar1 =
                                  float_T(2.0) * cspeed * t - complex_T(0,1) * cspeed * om0 * tauG2
                                - float_T(2.0) * z + float_T(2.0) * y * tanPhi2 - float_T(2.0) * (z + y * cotPhi) * tanPhi2_2;

                const complex_T helpVar2 = float_T(0.25) * (
                                - (om02 * tauG2) - (complex_T(0,2) * k * x2) / (complex_T(0,1) * rho0 - y * cosPhi - z * sinPhi)
                                - (complex_T(0,4) * om0 * y * tanPhi2) / cspeed + (complex_T(0,2) * om0 * (z + y * cotPhi) * tanPhi2_2) / cspeed
                                - (om0 * helpVar1 * helpVar1) / (cspeed * (cspeed * om0 * tauG2 - complex_T(0,2) * (z + y * cotPhi) * tanPhi2_2))
                          );

                const complex_T helpVar3 = rho0 + complex_T(0,1) * y * cosPhi + complex_T(0,1) * z * sinPhi;

                const complex_T result = (
                              cspeed * math::exp(helpVar2) * k * tauG * x * rho0 *
                               (
                                - float_T(4.0) * cspeed * om0 * t * rho02 + complex_T(0,2) * cspeed * om02 * tauG2 * rho02 + float_T(4.0) * om0 * z * rho02
                                + float_T(8.0) * z2 * (complex_T(0,3) * cspeed - float_T(2.0) * om0 * z) * sinPhi2_2 * sinPhi2_2
                                - complex_T(0,1) * cspeed2 * k * om0 * tauG2 * x2 * sinPhi
                                + complex_T(0,3) * cspeed2 * om0 * tauG2 * rho0 * sinPhi
                                - complex_T(0,8) * cspeed * om0 * t * z * rho0 * sinPhi
                                - float_T(4.0) * cspeed * om02 * tauG2 * z * rho0 * sinPhi + complex_T(0,8) * om0 * z2 * rho0 * sinPhi
                                - float_T(3.0) * cspeed2 * om0 * tauG2 * z * sinPhi_2 + float_T(4.0) * cspeed * om0 * t * z2 * sinPhi_2
                                - complex_T(0,2) * cspeed * om02 * tauG2 * z2 * sinPhi_2 - float_T(4.0) * om0 * z2 * z * sinPhi_2
                                - float_T(4.0) * om0 * y * rho02 * tanPhi2 - complex_T(0,8) * om0 * y * z * rho0 * sinPhi * tanPhi2
                                + float_T(4.0) * om0 * y * z2 * sinPhi_2 * tanPhi2 + float_T(4.0) * om0 * z * rho02 * tanPhi2_2
                                + float_T(4.0) * om0 * y * rho02 * cotPhi * tanPhi2_2 - float_T(2.0) * cspeed * k * x2 * z * sinPhi * tanPhi2_2
                                + float_T(6.0) * cspeed * z * rho0 * sinPhi * tanPhi2_2 + complex_T(0,8) * om0 * z2 * rho0 * sinPhi * tanPhi2_2
                                + float_T(2.0) * y2 * cosPhi_2 * (
                                  om0 * (float_T(2.0) * cspeed * t - complex_T(0,1) * cspeed * om0 * tauG2 - float_T(2.0) * z)
                                  + float_T(2.0) * om0 * y * tanPhi2 + (complex_T(0,3) * cspeed - float_T(6.0) * om0 * z - float_T(2.0) * om0 * y * cotPhi) * tanPhi2_2
                                )
                                - y * cosPhi * (
                                  + float_T(4.0) * om0 * (complex_T(0,2) * cspeed * t + cspeed * om0 * tauG2 - complex_T(0,2) * z) * rho0
                                  + complex_T(0,8) * om0 * y * rho0 * tanPhi2 + float_T(2.0) * (
                                    + cspeed * k * x2 - float_T(3.0) * cspeed * rho0 - complex_T(0,8) * om0 * z * rho0
                                    - complex_T(0,4) * om0 * y * rho0 * cotPhi
                                  ) * tanPhi2_2
                                  + sinPhi * (
                                    + om0 * (float_T(3.0) * cspeed2 * tauG2 - float_T(8.0) * cspeed * t * z + complex_T(0,4) * cspeed * om0 * tauG2 * z + float_T(8.0) * z2)
                                    - float_T(8.0) * om0 * y * z * tanPhi2 - float_T(12.0) * z * (complex_T(0,1) * cspeed - om0 * z) * tanPhi2_2
                                  )
                                )
               )
             )
             /* The "round-trip" conversion in the lines below fixes a gross accuracy bug
              * in floating-point arithmetics leading to nans, when float_T is set to float_X.
              */
             * complex_T(
               complex_64(1,0) /
               complex_64(
                float_T(2.0) * om02 * math::sqrt( rho0 / ( complex_T(0,1) * rho0 - y * cosPhi - z * sinPhi ) )
                * pmacc::math::cPow(helpVar3, static_cast<uint32_t>(4u)) * ( complex_T(0,-1) * cspeed * om0 * tauG2 - float_T(2.0) * ( z + y * cotPhi) * tanPhi2_2)
                * math::sqrt(tauG2 - (complex_T(0,2) * (z + y * cotPhi) * tanPhi2_2) / (cspeed * om0))
               )
             );
             /* Explanation for the change of the sign below: The original solution propagates in (-z)-direction.
              * For this reason we fix this by inverting both the propagation direction and the pulse front tilt by two transforms.
              * 1) Rotate 180° around x: coordinates (y,z)->(-y,-z) and vector components ((E,B)_(y,z)-> -(E,B)_(y,z))
              * 2) Reverse propagation ( B = (1/c^2) *  n x E ): (E,B)->(-E,B) or (E,B)->(E,-B)
              * 3) Phase-shift 180°: (E,B)-> -(E,B) (This does not change relevant physics, but is a clean-up to give the main E-component a positive sign.)
              *
              * x-pol:
              * Ex,Ez_Ex,By,Bz_Ex (coordinates in expressions already transformed)
              * Rotate vectors 180° around x: Ex,-Ez_Ex,-By,-Bz_Ex
              * Flip propagation & Phase-shift: -Ex,Ez_Ex,-By,-Bz_Ex -> Ex,-Ez_Ex,By,Bz_Ex
              * --> Thus the Ez_Ex component needs to be multiplied by -1 .
              */

                return -result.real();
            }

            /** Calculate the Ez(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEz_Ey(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                auto const beta0 = float_T(beta_0);
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                auto const phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                float_T const alphaTilt = math::atan2(float_T(1.0) - beta0 * cosPhiReal, beta0 * sinPhiReal);
                /* Definition of the laser pulse front tilt angle for the laser field below.
                 *
                 * For beta0 = 1.0, this is equivalent to our standard definition. Question: Why is the
                 * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
                 * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
                 * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
                 * the dispersion will (although physically correct) be slightly off the ideal TWTS
                 * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
                 * scenarios close to beta0 = 1.
                 */
                float_T const phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
                 * documentation purposes.
                 * float_T const eta = (PI / 2) - (phiReal - alphaTilt);
                 */

                auto const cspeed = float_T(SI::SPEED_OF_LIGHT_SI / UNIT_SPEED);
                auto const lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
                float_T const om0 = float_T(2.0 * PI) * cspeed / lambda0;
                /* factor 2  in tauG arises from definition convention in laser formula */
                auto const tauG = float_T(pulselength_SI * 2.0 / UNIT_TIME);
                /* w0 is wx here --> w0 could be replaced by wx */
                auto const w0 = float_T(w_x_SI / UNIT_LENGTH);
                auto const rho0 = float_T(PI * w0 * w0 / lambda0);
                auto const k = float_T(2.0 * PI / lambda0);

                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 sinPhiVal;
                float_64 cosPhiVal;
                pmacc::math::sincos(precisionCast<float_64>(phi), sinPhiVal, cosPhiVal);
                float_64 const tanAlpha = (1.0 - beta_0 * cosPhiVal) / (beta_0 * sinPhiVal);
                float_64 const tanFocalLine = math::tan(PI / 2.0 - phi);
                float_64 const deltaT = wavelength_SI / SI::SPEED_OF_LIGHT_SI * (1.0 + tanAlpha / tanFocalLine);
                float_64 const deltaY = wavelength_SI / tanFocalLine;
                float_64 const deltaZ = -wavelength_SI;
                float_64 const numberOfPeriods = math::floor(time / deltaT);
                auto const timeMod = float_T(time - numberOfPeriods * deltaT);
                auto const yMod = float_T(pos.y() + numberOfPeriods * deltaY);
                auto const zMod = float_T(pos.z() + numberOfPeriods * deltaZ);

                auto const x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
                auto const y = float_T(phiPositive * yMod / UNIT_LENGTH);
                auto const z = float_T(zMod / UNIT_LENGTH);
                auto const t = float_T(timeMod / UNIT_TIME);

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                float_T const cotPhi = float_T(1.0) / math::tan(phiT);
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));
                float_T const sin2Phi = math::sin(float_T(2.0) * phiT);
                float_T const sinPhi2 = math::sin(phiT / float_T(2.0));

                float_T const sinPhi2_2 = sinPhi2 * sinPhi2;
                float_T const cosPhi_2 = cosPhi * cosPhi;
                float_T const tanPhi2_2 = tanPhi2 * tanPhi2;
                float_T const sinPhi_2 = sinPhi * sinPhi;
                float_T const sinPhi_3 = sinPhi_2 * sinPhi;

                float_T const tauG2 = tauG * tauG;
                float_T const k2 = k * k;
                float_T const cspeed2 = cspeed * cspeed;
                float_T const rho02 = rho0 * rho0;
                float_T const rho03 = rho02 * rho0;
                float_T const om02 = om0 * om0;
                float_T const x2 = x * x;
                float_T const y2 = y * y;
                float_T const z2 = z * z;
                float_T const z3 = z2 * z;
                float_T const x4 = x2 * x2;

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                const complex_T helpVar1 = float_T(2.0) * cspeed * t - complex_T(0,1) * cspeed * om0 * tauG2
                    - float_T(2.0) * z + float_T(2.0) * y * tanPhi2 - float_T(2.0) * (z + y * cotPhi) * tanPhi2_2;
                const complex_T helpVar2 = float_T(0.25) * (
                    - (om02 * tauG2 ) - (complex_T(0,2) * k * x2) / (complex_T(0,1) * rho0 - y * cosPhi - z * sinPhi)
                    - (complex_T(0,4) * om0 * y * tanPhi2) / cspeed + (complex_T(0,2) * om0 * (z + y * cotPhi) * tanPhi2_2) / cspeed
                    - (om0 * helpVar1 * helpVar1) / (cspeed * (cspeed * om0 * tauG2 - complex_T(0,2) * (z + y * cotPhi) * tanPhi2_2))
                    );
                const complex_T helpVar3 = rho0 + complex_T(0,1) * y * cosPhi + complex_T(0,1) * z * sinPhi;

                const complex_T result = (
                cspeed * math::exp(helpVar2) * tauG * rho0 * cosPhi * (
                complex_T(0,-4) * cspeed * k * om0 * t * x2 * rho02 - float_T(2.0) * cspeed * k * om02 * tauG2 * x2 * rho02
              + complex_T(0,4) * k * om0 * x2 * z * rho02 + complex_T(0,4) * cspeed * om0 * t * rho03
              + float_T(2.0) * cspeed * om02 * tauG2 * rho03 - complex_T(0,4) * om0 * z * rho03
              - float_T(24.0) * om0 * y * z * rho02 * sinPhi2_2 + cspeed2 * k2 * om0 * tauG2 * x4 * sinPhi
              - float_T(6.0) * cspeed2 * k * om0 * tauG2 * x2 *rho0 * sinPhi + float_T(8.0) * cspeed * k * om0 * t * x2 * z * rho0 * sinPhi
              - complex_T(0,4) * cspeed * k * om02 * tauG2 * x2 * z * rho0 * sinPhi - float_T(8.0) * k * om0 * x2 * z2 * rho0 * sinPhi
              + float_T(3.0) * cspeed2 * om0 * tauG2 * rho02 * sinPhi - float_T(12.0) * cspeed * om0 * t * z * rho02 * sinPhi
              + complex_T(0,6) * cspeed * om02 * tauG2 * z * rho02 * sinPhi + float_T(12.0) * om0 * z2 * rho02 * sinPhi
              - complex_T(0,6) * cspeed2 * k * om0 * tauG2 * x2 * z * sinPhi_2 + complex_T(0,4) * cspeed * k * om0 * t * x2 * z2 * sinPhi_2
              + float_T(2.0) * cspeed * k * om02 * tauG2 * x2 * z2 * sinPhi_2 - complex_T(0,4) * k * om0 * x2 * z3 * sinPhi_2
              + complex_T(0,6) * cspeed2 * om0 * tauG2 * z * rho0 * sinPhi_2 - complex_T(0,12) * cspeed * om0 * t * z2 * rho0 * sinPhi_2
              - float_T(6.0) * cspeed * om02 * tauG2 * z2 * rho0 * sinPhi_2 + complex_T(0,12) * om0 * z3 * rho0 * sinPhi_2
              - float_T(3.0) * cspeed2 * om0 * tauG2 * z2 * sinPhi_3 + float_T(4.0) * cspeed * om0 * t * z3 * sinPhi_3
              - complex_T(0,2) * cspeed * om02 * tauG2 * z3 * sinPhi_3 - float_T(4.0) * om0 * z2 * z2 * sinPhi_3
              - float_T(8.0) * z2 * sinPhi2_2 * sinPhi2_2 * (
                + complex_T(0,2) * om0 * z * (k * x2 - float_T(3.0) * rho0) + float_T(6.0) * cspeed * (k * x2 - rho0)
                - z * (complex_T(0,3) * cspeed - float_T(2.0) * om0 * z) * sinPhi
              )
              - complex_T(0,12) * cspeed * om0 * t * y * z * rho0 * sin2Phi - float_T(6.0) * cspeed * om02 * tauG2 * y * z * rho0 * sin2Phi
              + complex_T(0,12) * om0 * y * z2 * rho0 * sin2Phi + float_T(6.0) * cspeed * om0 * t * y * z2 * sinPhi * sin2Phi
              - float_T(6.0) * om0 * y * z3 * sinPhi * sin2Phi - complex_T(0,4) * k * om0 * x2 * y * rho02 * tanPhi2
              + complex_T(0,4) * om0 * y * rho03 * tanPhi2 + float_T(8.0) * k * om0 * x2 * y * z * rho0 * sinPhi * tanPhi2
              + complex_T(0,4) * k * om0 * x2 * y * z2 * sinPhi_2 * tanPhi2 - complex_T(0,12) * om0 * y * z2 * rho0 * sinPhi_2 * tanPhi2
              + float_T(4.0) * om0 * y * z3 * sinPhi_3 * tanPhi2 + complex_T(0,4) * k * om0 * x2 * z * rho02 * tanPhi2_2
              - complex_T(0,4) * om0 * z * rho03 * tanPhi2_2 + complex_T(0,4) * k * om0 * x2 * y * rho02 * cotPhi * tanPhi2_2
              - complex_T(0,4) * om0 * y * rho03 * cotPhi * tanPhi2_2
              - complex_T(0,2) * cspeed * k2 * x4 * z * sinPhi * tanPhi2_2
              + complex_T(0,12) * cspeed * k * x2 * z * rho0 * sinPhi *tanPhi2_2 - float_T(8.0) * k * om0 * x2 * z2 * rho0 * sinPhi * tanPhi2_2
              - complex_T(0,6) * cspeed * z * rho02 * sinPhi * tanPhi2_2 + float_T(12.0) * om0 * z2 * rho02 * sinPhi * tanPhi2_2
              + float_T(2.0) * y2 * y * cosPhi_2 * cosPhi * (
                + om0 * (float_T(2.0) * cspeed * t - complex_T(0,1) * cspeed * om0 * tauG2 - float_T(2.0) * z) + float_T(2.0) * om0 * y * tanPhi2
                + (complex_T(0,3) * cspeed - float_T(8.0) * om0*z - float_T(2.0) * om0 * y * cotPhi) * tanPhi2_2
              )
              - y2 * cosPhi_2 * (
                - float_T(2.0) * om0 * (complex_T(0,2) * cspeed * t + cspeed * om0 * tauG2 - complex_T(0,2) * z) * (k * x2 - float_T(3.0) * rho0)
                - float_T(24.0) * om0 * y * z * sinPhi2_2
                + float_T(3.0) * om0 * (cspeed2 * tauG2 - float_T(4.0) * cspeed * t * z + complex_T(0,2) * cspeed * om0 * tauG2 * z + float_T(4.0) * z2) * sinPhi
                - complex_T(0,4) * om0 * y * (k * x2 - float_T(3.0) * rho0) * tanPhi2 - float_T(2.0) * (complex_T(0,-6) * om0 * z * (k * x2 - float_T(3.0) * rho0)
                - cspeed * (float_T(6.0) * k * x2 - float_T(6.0) * rho0) - complex_T(0,2) * om0 * y * (k * x2 - float_T(3.0) * rho0) * cotPhi) * tanPhi2_2
                - float_T(6.0) * z * (complex_T(0,3) * cspeed - float_T(4.0) * om0 * z) * sinPhi * tanPhi2_2
              )
              - float_T(2.0) * y * cosPhi * (
                - float_T(4.0) * cspeed * k * om0 * t * x2 * rho0 + complex_T(0,2) * cspeed * k * om02 * tauG2 * x2 * rho0
                + float_T(4.0) * k * om0 * x2 * z * rho0 + float_T(6.0) * cspeed * om0 * t * rho02 - complex_T(0,3) * cspeed * om02 * tauG2 * rho02
                - float_T(6.0) * om0 * z * rho02 - complex_T(0,8) * om0 * y * z * (k * x2 - float_T(3.0) * rho0) * sinPhi2_2
                - float_T(4.0) * z2 * (complex_T(0,9) * cspeed - float_T(8.0) * om0 * z) * sinPhi2_2 * sinPhi2_2
                - float_T(4.0) * k * om0 * x2 * y * rho0 * tanPhi2
                + float_T(6.0) * om0 * y * rho02 * tanPhi2 + complex_T(0,1) * cspeed * k2 * x4 * tanPhi2_2
                - complex_T(0,6) * cspeed * k * x2 * rho0 * tanPhi2_2 + float_T(8.0) * k * om0 * x2 * z * rho0 * tanPhi2_2
                + complex_T(0,3) * cspeed * rho02 * tanPhi2_2 - float_T(12.0) * om0 * z * rho02 * tanPhi2_2
                + float_T(4.0) * k * om0 * x2 * y * rho0 * cotPhi * tanPhi2_2 - float_T(6.0) * om0 * y * rho02 * cotPhi * tanPhi2_2
                - float_T(3.0) * om0 * z * sinPhi_2 * (
                  - (cspeed * tauG2 * (cspeed + complex_T(0,1) * om0 * z)) + float_T(2.0) * y * z * tanPhi2
                )
                + complex_T(0,1) * sinPhi * (
                  om0 * (
                    - float_T(2.0) * cspeed * k * (float_T(2.0) * t - complex_T(0,1) * om0 * tauG2) * x2 * z + float_T(4.0) * k * x2 * z2
                    + float_T(3.0) * cspeed2 * tauG2 * (k * x2 - rho0)
                  )
                  - float_T(6.0) * z * (
                    - (om0 * z * (k * x2 - 3 * rho0)) + complex_T(0,2) * cspeed * (k * x2 - rho0)
                  ) * tanPhi2_2
                )
             )
               )
             )
                /* The "round-trip" conversion in the lines below fixes a gross accuracy bug
                 * in floating-point arithmetics leading to nans, when float_T is set to float_X.
                 */
             * complex_T(
               complex_64(1,0) /
               complex_64(
               + float_T(4.0) * om02 * math::sqrt(rho0 / (complex_T(0,1) * rho0 - y * cosPhi - z * sinPhi))
               * pmacc::math::cPow(helpVar3, static_cast<uint32_t>(5u)) * (complex_T(0,-1) * cspeed * om0 * tauG2 - float_T(2.0) * (z + y * cotPhi) * tanPhi2_2)
               * math::sqrt(tauG2 - (complex_T(0,2) * (z + y * cotPhi) * tanPhi2_2) / (cspeed * om0))
               )
             );
                return result.real();
            }

        } /* namespace twtsfast */
    } /* namespace templates */
} /* namespace picongpu */
