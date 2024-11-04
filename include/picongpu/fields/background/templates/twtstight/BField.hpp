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
#include "picongpu/fields/background/templates/twtstight/numComponents.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

#include <tuple>

namespace picongpu
{
    /* Load pre-defined background field */
    namespace templates
    {
        /* Traveling-wave Thomson scattering laser pulse */
        namespace twtstight
        {
            class BField : public TWTSTight
            {
            public:
                using TWTSTight::TWTSTight;

                /** Specify your background field B(r,t) here
                 *
                 * @param cellIdx The total cell id counted from the start at t=0, note it can be fractional
                 * @param currentStep The current time step for the field to be calculated at, note it can be
                 * fractional
                 * @return float3_X with field normalized to amplitude in range [-1.:1.]
                 *
                 * @{
                 */

                //! Integer index version, adds in-cell shifts according to the grid used; t = currentStep * dt
                //! This interface is used by the fieldBackground approach for implementing fields.
                HDINLINE float3_X operator()(DataSpace<simDim> const& cellIdx, uint32_t const currentStep) const;

                //! Floating-point index version, uses fractional cell index as provided; t = currentStep * dt
                //! This interface is used by the incidentField approach for implementing fields.
                HDINLINE float3_X operator()(floatD_X const& cellIdx, float_X const currentStep) const;

                /** @} */

                /** Calculate the given component of B(r, t)
                 *
                 * Result is same as for the fractional version of operator()(cellIdx, currentStep)[T_component].
                 * This version exists for optimizing usage in incident field where single components are needed.
                 *
                 * @tparam T_component field component, 0 = x, 1 = y, 2 = z
                 *
                 * @param cellIdx The total fractional cell id counted from the start at t=0
                 * @param currentStep The current time step for the field to be calculated at
                 * @return float_X with field component normalized to amplitude in range [-1.:1.]
                 */
                template<uint32_t T_component>
                HDINLINE float_X getComponent(floatD_X const& cellIdx, float_X const currentStep) const;

                /** Calculate B(r, t) for given position, time, and extra in-cell shifts
                 *
                 * @param cellIdx The total cell id counted from the start at t=0, note it is fractional
                 * @param extraShifts The extra in-cell shifts to be added to calculate the position
                 * @param currentStep The current time step for the field to be calculated at, note it is fractional
                 */
                HDINLINE float3_X getValue(
                    floatD_X const& cellIdx,
                    pmacc::math::Vector<floatD_X, detail::numComponents> const& extraShifts,
                    float_X const currentStep) const;

                /** Calculate the By(r,t) field
                 *
                 * @param pos Spatial position of the target field.
                 * @param time Absolute time (SI, including all offsets and transformations)
                 *  for calculating the field
                 * @return By-field component of the TWTS field in SI units */
                HDINLINE float_T calcTWTSBy(float3_64 const& pos, float_64 const time) const;

                /** Calculate the Bz(r,t) field
                 *
                 * @param pos Spatial position of the target field.
                 * @param time Absolute time (SI, including all offsets and transformations)
                 *  for calculating the field
                 * @return Bz-field component of the TWTS field in SI units */
                HDINLINE float_T calcTWTSBz(float3_64 const& pos, float_64 const time) const;

                /** Calculate the Bx(r,t) field
                 *
                 * @param pos Spatial position of the target field.
                 * @param time Absolute time (SI, including all offsets and transformations)
                 *  for calculating the field
                 * @return Bx-field component of the TWTS field in SI units */
                HDINLINE float_T calcTWTSBx(float3_64 const& pos, float_64 const time) const;

                /** Calculate the B-field vector of the TWTS laser in SI units.
                 * @tparam T_dim Specializes for the simulation dimension
                 * @param cellIdx The total cell id counted from the start at timestep 0
                 * @return B-field vector of the TWTS field in SI units */
                template<unsigned T_dim>
                HDINLINE float3_X getTWTSBfield_Normalized(
                    pmacc::math::Vector<floatD_64, detail::numComponents> const& bFieldPositions_SI,
                    float_64 const time) const;
            };

        } /* namespace twtstight */
    } /* namespace templates */
} /* namespace picongpu */
