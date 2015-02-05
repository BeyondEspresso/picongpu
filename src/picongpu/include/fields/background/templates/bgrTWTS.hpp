/**
 * Copyright 2014 Alexander Debus, Axel Huebl
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

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_classTypes.hpp"

#include "math/Vector.hpp"
#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/SubGrid.hpp"

/** \todo not great... if complex is that general, refactor it to libPMacc! */
#include "plugins/radiation/complex.hpp"


namespace picongpu
{
    /** Load external TWTS field
     *
     */
    namespace templates
    {
        using namespace PMacc;

        HDINLINE
        TWTSFieldE::TWTSFieldE( const float_64 focus_y_SI,
                                const float_64 wavelength_SI,
                                const float_64 pulselength_SI,
                                const float_64 w_x_SI,
                                const float_64 w_y_SI,
                                const float_X phi,
                                const float_X beta_0,
                                const float_64 tdelay_user_SI,
                                const bool auto_tdelay ) :
            focus_y_SI(focus_y_SI), wavelength_SI(wavelength_SI),
            pulselength_SI(pulselength_SI), w_x_SI(w_x_SI),
            w_y_SI(w_y_SI), phi(phi), beta_0(beta_0),
            tdelay_user_SI(tdelay_user_SI), auto_tdelay(auto_tdelay)
        {
            const uint32_t numComponents = DIM3 ;
#if !defined(__CUDA_ARCH__)
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);
#endif
        }

        HDINLINE PMacc::math::Vector<floatD_64,FieldE::numComponents>
        TWTSFieldE::getEfieldPositions_SI<DIM3>(const DataSpace<simDim>& cellIdx) const
        {
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center (usually maximum of intensity) in y.
            floatD_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI, halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<floatD_X, FieldE::numComponents> eFieldPositions = fieldSolver::NumericalCellType::getEFieldPosition();
            PMacc::math::Vector<floatD_64,FieldE::numComponents> eFieldPositions_SI;
            
            const float_64 unit_length = UNIT_LENGTH;
            const floatD_64 cellDimensions = precisionCast<floatD_64>(cellSize) * unit_length;
            for( uint32_t i = 0; i < FieldE::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                eFieldPositions[i]   += floatD_X(cellIdx) - laserOrigin;
                eFieldPositions_SI[i] = precisionCast<float_64>(eFieldPositions[i]) * cellDimensions;
            }

            return eFieldPositions_SI;
        }
        
        HDINLINE PMacc::math::Vector<floatD_64,FieldE::numComponents>
        TWTSFieldE::getEfieldPositions_SI<DIM3>(const DataSpace<simDim>& cellIdx) const
        {
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center (usually maximum of intensity) in y.
            floatD_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI, halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<floatD_X, FieldE::numComponents> eFieldPositions = fieldSolver::NumericalCellType::getEFieldPosition();
            PMacc::math::Vector<floatD_64,FieldE::numComponents> eFieldPositions_SI;
            
            const float_64 unit_length = UNIT_LENGTH;
            const floatD_64 cellDimensions = precisionCast<floatD_64>(cellSize) * unit_length;
            for( uint32_t i = 0; i < FieldE::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                eFieldPositions[i]   += floatD_X(cellIdx) - laserOrigin;
                eFieldPositions_SI[i] = precisionCast<float_64>(eFieldPositions[i]) * cellDimensions;
                // Rotate 90° around y-axis, so that TWTS laser propagates within the 2D (x,y)-plane.
                /** Corresponding position vector for the Ez-components in 2D simulations.
                 *  3D     2D
                 *  x -->  z
                 *  y -->  y
                 *  z --> -x (Since z=0 for 2D, we use the existing TWTS-field-function and set -x=0)
                 *  Ex --> Ez (--> Same function values can be used in 2D, but with Yee-Cell-Positions for Ez.)
                 *  By --> By
                 *  Bz --> -Bx
                 */
                eFieldPositions_SI[i] = ( -(eFieldPositions_SI[i]).z(), (eFieldPositions_SI[i]).y(), (eFieldPositions_SI[i]).x() );
            }
            
            return eFieldPositions_SI;
        }
        
        HDINLINE float_64 getTime_SI(const uint32_t currentStep) const
        {
            const float_64 time_SI=currentStep*::picongpu::SI::DELTA_T_SI;
                       
            if ( auto_tdelay ) {
                
                /* halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff
                 * of the TWTS-pulse. The abs()-function is for keeping the same offset for -phiReal and +phiReal. */
                const float_64 y1=(float_64)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/abs(tan(eta)); 
                const float_64 m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
                const float_64 y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
                const float_64 y3=::picongpu::bgrTWTS::SI::FOCUS_POS_SI; // Position of maximum intensity in simulation volume along y
                const float_64 tdelay= (y1+y2+y3)/(cspeed*beta0);
                
                return time-tdelay;
            }
            else
                return time-tdelay_user_SI;
        }
        
        HDINLINE float3_X getTWTSEfield_SI<DIM3>( const PMacc::math::Vector<floatD_64,FieldE::numComponents>& eFieldPositions_SI,
                                                  const float_64 time) const;
        {
            return float3_X( precisionCast<float_X>( calcTWTSEx(eFieldPositions_SI[0],time,halfSimSize, phi) ), floatX( 0. ), float_X( 0. ) );
        }
        
        HDINLINE float3_X getTWTSEfield_SI<DIM2>( const PMacc::math::Vector<floatD_64,FieldE::numComponents>& eFieldPositions_SI,
                                                  const float_64 time) const;
        {
            // Ex->Ez, so also the grid cell offset for Ez has to be used.
            return float3_X( float_X( 0. ), float_X( 0. ), calcTWTSEx(eFieldPositions_SI[2],time,halfSimSize, phi) );
        }
        
        HDINLINE float3_X
        TWTSFieldE::operator()( const DataSpace<simDim>& cellIdx,
                                const uint32_t currentStep ) const
        {
            /** \todo fixme, can be done without SI */
            const float_64 time=getTime_SI(currentStep);
            const PMacc::math::Vector<floatD_64,FieldE::numComponents> eFieldPositions_SI=getFieldPosition(cellIdx);
            // Single TWTS-Pulse
            return getTWTSEfield(eFieldPositions_SI, time, halfSimSize, phi);
        }

        /** Calculate the Ex(r,t) field here
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field
         * \param halfSimSize Center of simulation volume in number of cells
         * \param phiReal interaction angle between TWTS laser propagation vector and the y-axis */
        HDINLINE float_64
        TWTSFieldE::calcTWTSEx( const floatD_64& pos, const float_X time, const DataSpace<simDim> halfSimSize, const float_X phiReal ) const
        {
            
            return float_X(0.);
        }

        HDINLINE
        TWTSFieldB::TWTSFieldB()
        {
            const uint32_t numComponents = DIM3 ;
#if !defined(__CUDA_ARCH__)
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);
#endif
        }
        
        HDINLINE PMacc::math::Vector<floatD_64,FieldB::numComponents>
        TWTSFieldB::getBfieldPositions_SI<DIM3>(const DataSpace<simDim>& cellIdx) const
        {
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center (usually maximum of intensity) in y.
            floatD_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI, halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<floatD_X, FieldB::numComponents> bFieldPositions = fieldSolver::NumericalCellType::getBFieldPosition();
            PMacc::math::Vector<floatD_64,FieldB::numComponents> bFieldPositions_SI;
            
            const float_64 unit_length = UNIT_LENGTH;
            const floatD_64 cellDimensions = precisionCast<floatD_64>(cellSize) * unit_length;
            for( uint32_t i = 0; i < FieldB::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                bFieldPositions[i]   += floatD_X(cellIdx) - laserOrigin;
                bFieldPositions_SI[i] = precisionCast<float_64>(bFieldPositions[i]) * cellDimensions;
            }

            return bFieldPositions_SI;
        }
        
        HDINLINE PMacc::math::Vector<floatD_64,FieldB::numComponents>
        TWTSFieldE::getBfieldPositions_SI<DIM3>(const DataSpace<simDim>& cellIdx) const
        {
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center (usually maximum of intensity) in y.
            floatD_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI, halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<floatD_X, FieldB::numComponents> bFieldPositions = fieldSolver::NumericalCellType::getBFieldPosition();
            PMacc::math::Vector<floatD_64,FieldB::numComponents> bFieldPositions_SI;
            
            const float_64 unit_length = UNIT_LENGTH;
            const floatD_64 cellDimensions = precisionCast<floatD_64>(cellSize) * unit_length;
            for( uint32_t i = 0; i < FieldB::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                bFieldPositions[i]   += floatD_X(cellIdx) - laserOrigin;
                bFieldPositions_SI[i] = precisionCast<float_64>(bFieldPositions[i]) * cellDimensions;
                // Rotate 90° around y-axis, so that TWTS laser propagates within the 2D (x,y)-plane.
                /** Corresponding position vector for the Ez-components in 2D simulations.
                 *  3D     2D
                 *  x -->  z
                 *  y -->  y
                 *  z --> -x (Since z=0 for 2D, we use the existing TWTS-field-function and set -x=0)
                 *  Ex --> Ez (--> Same function values can be used in 2D, but with Yee-Cell-Positions for Ez.)
                 *  By --> By
                 *  Bz --> -Bx
                 */
                bFieldPositions_SI[i] = ( -(bFieldPositions_SI[i]).z(), (bFieldPositions_SI[i]).y(), (bFieldPositions_SI[i]).x() );
            }
            
            return bFieldPositions_SI;
        }
        
        HDINLINE float_64 getTime_SI(const uint32_t currentStep) const
        {
            const float_64 time_SI=currentStep*::picongpu::SI::DELTA_T_SI;
                       
            if ( auto_tdelay ) {
                
                /* halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff
                 * of the TWTS-pulse. The abs()-function is for keeping the same offset for -phiReal and +phiReal. */
                const float_64 y1=(float_64)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/abs(tan(eta)); 
                const float_64 m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
                const float_64 y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
                const float_64 y3=::picongpu::bgrTWTS::SI::FOCUS_POS_SI; // Position of maximum intensity in simulation volume along y
                const float_64 tdelay= (y1+y2+y3)/(cspeed*beta0);
                
                return time-tdelay;
            }
            else
                return time-tdelay_user_SI;
        }
        
        HDINLINE float3_X getTWTSBfield_SI<DIM3>( const PMacc::math::Vector<floatD_64,FieldB::numComponents>& bFieldPositions_SI,
                                                  const float_64 time) const;
        {
            return float3_X( precisionCast<float_X>( calcTWTSEx(bFieldPositions_SI[0],time,halfSimSize, phi) ), floatX( 0. ), float_X( 0. ) );
        }
        
        HDINLINE float3_X getTWTSBfield_SI<DIM2>( const PMacc::math::Vector<floatD_64,FieldB::numComponents>& bFieldPositions_SI,
                                                  const float_64 time) const;
        {
            // Ex->Ez, so also the grid cell offset for Ez has to be used.
            return float3_X( float_X( 0. ), float_X( 0. ), calcTWTSEx(bFieldPositions_SI[2],time,halfSimSize, phi) );
        }
        
        HDINLINE float3_X
        TWTSFieldB::operator()( const DataSpace<simDim>& cellIdx,
                                const uint32_t currentStep ) const
        {
            return float3_X(0.);
        }

        /** Calculate the By(r,t) field here
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field
         * \param halfSimSize Center of simulation volume in number of cells
         * \param phiReal interaction angle between TWTS laser propagation vector and the y-axis */
        HDINLINE float_X
        TWTSFieldB::calcTWTSBy( const float3_X& pos, const float_X time, const DataSpace<simDim> halfSimSize, const float_X phiReal ) const
        {

            return float_X(0.);
        }
        
        /** Calculate the Bz(r,t) field here
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field
         * \param halfSimSize Center of simulation volume in number of cells
         * \param phiReal interaction angle between TWTS laser propagation vector and the y-axis */
        HDINLINE float_X
        TWTSFieldB::calcTWTSBz( const float3_X& pos, const float_X time, const DataSpace<simDim> halfSimSize, const float_X phiReal ) const
        {

            return float_X(0.);
        }

    } /* namespace templates */
} /* namespace picongpu */
