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
                                const float3_64& unitField,
                                const float_X phi,
                                const float_X beta_0,
                                const float_64 tdelay_user_SI,
                                const bool auto_tdelay ) :
            focus_y_SI(focus_y_SI), wavelength_SI(wavelength_SI),
            pulselength_SI(pulselength_SI), w_x_SI(w_x_SI),
            w_y_SI(w_y_SI), unitField(unitField), phi(phi), beta_0(beta_0),
            tdelay_user_SI(tdelay_user_SI), auto_tdelay(auto_tdelay)
        {
#if !defined(__CUDA_ARCH__)
            // These objects cannot be instantiated on CUDA GPU device. Since this is done on host (see fieldBackground.param), this is no problem.
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);
#endif
        }

        template<>
        HDINLINE PMacc::math::Vector<floatD_64,FieldE::numComponents>
        TWTSFieldE::getEfieldPositions_SI<DIM3>(const DataSpace<simDim>& cellIdx) const
        {
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center (usually maximum of intensity) in y.
            floatD_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI, halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<floatD_X, FieldE::numComponents> eFieldPositions = fieldSolver::NumericalCellType::getEFieldPosition();
            PMacc::math::Vector<floatD_64,FieldE::numComponents> eFieldPositions_SI;
            
            const float_64 unit_length = picongpu::UNIT_LENGTH;
            const floatD_64 cellDimensions = precisionCast<floatD_64>(picongpu::cellSize) * unit_length;
            for( uint32_t i = 0; i < FieldE::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                eFieldPositions[i]   += floatD_X(cellIdx) - laserOrigin;
                eFieldPositions_SI[i] = precisionCast<float_64>(eFieldPositions[i]) * cellDimensions;
                
                /*  Since, the laser propagation direction encloses an angle of phi with the simulation y-axis (i.e. direction of sliding window),
                 *  the positions vectors are rotated around the simulation x-axis before calling the TWTS field functions. Note: The TWTS field
                 *  functions are in non-rotated frame and only use the angle phi to determine the required amount of pulse front tilt.
                 *  RotationMatrix[PI/2+phiReal].(y,z) (180Deg-flip at phiReal=90Deg since coordinate system in paper is oriented the other way round.) */
                eFieldPositions_SI[i] = ( (eFieldPositions_SI[i]).x(),
                                          -sin(phi)*(eFieldPositions_SI[i]).y()-cos(phi)*(eFieldPositions_SI[i]).z(),
                                          +cos(phi)*(eFieldPositions_SI[i]).y()-sin(phi)*(eFieldPositions_SI[i]).z()  );
            }

            return eFieldPositions_SI;
        }
        
        template<>
        HDINLINE PMacc::math::Vector<floatD_64,FieldE::numComponents>
        TWTSFieldE::getEfieldPositions_SI<DIM2>(const DataSpace<simDim>& cellIdx) const
        {
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center in y (usually maximum of intensity).
            floatD_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI, halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<floatD_X, FieldE::numComponents> eFieldPositions = fieldSolver::NumericalCellType::getEFieldPosition();
            PMacc::math::Vector<floatD_64,FieldE::numComponents> eFieldPositions_SI;
            
            const float_64 unit_length = picongpu::UNIT_LENGTH;
            const floatD_64 cellDimensions = precisionCast<floatD_64>(picongpu::cellSize) * unit_length;
            for( uint32_t i = 0; i < FieldE::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                eFieldPositions[i]   += floatD_X(cellIdx) - laserOrigin;
                eFieldPositions_SI[i] = precisionCast<float_64>(eFieldPositions[i]) * cellDimensions;
                // Rotate 90� around y-axis, so that TWTS laser propagates within the 2D (x,y)-plane.
                /** Corresponding position vector for the Ez-components in 2D simulations.
                 *  3D     eD vectors in 2D space (x,y)
                 *  x -->  z
                 *  y -->  y
                 *  z --> -x (Since z=0 for 2D, we use the existing TWTS-field-function and set -x=0)
                 *  Ex --> Ez (--> Same function values can be used in 2D, but with Yee-Cell-Positions for Ez.)
                 *  By --> By
                 *  Bz --> -Bx
                 */
                eFieldPositions_SI[i] = ( -(eFieldPositions_SI[i]).z(), (eFieldPositions_SI[i]).y(), (eFieldPositions_SI[i]).x() );
                
                /*  Since, the laser propagation direction encloses an angle of phi with the simulation y-axis (i.e. direction of sliding window),
                 *  the positions vectors are rotated around the simulation x-axis before calling the TWTS field functions. Note: The TWTS field
                 *  functions are in non-rotated frame and only use the angle phi to determine the required amount of pulse front tilt.
                 *  RotationMatrix[PI/2+phiReal].(y,z) (180Deg-flip at phiReal=90Deg since coordinate system in paper is oriented the other way round.) */
                
                /* Note: The x-axis of rotation is fine in 2D, because that component now contains the (non-existing) simulation z-coordinate. */
                eFieldPositions_SI[i] = ( (eFieldPositions_SI[i]).x(),
                                          -sin(phi)*(eFieldPositions_SI[i]).y()-cos(phi)*(eFieldPositions_SI[i]).z(),
                                          +cos(phi)*(eFieldPositions_SI[i]).y()-sin(phi)*(eFieldPositions_SI[i]).z()  );
            }
            
            return eFieldPositions_SI;
        }
        
        HDINLINE float_64
        TWTSFieldE::getTime_SI(const uint32_t currentStep) const
        {
            const float_64 time_SI=currentStep*picongpu::SI::DELTA_T_SI;
                       
            if ( auto_tdelay ) {
                
                // angle between the laser pulse front and the y-axis. (Good approximation for beta0 \simeq 1. For exact relation look in TWTS core routines for Ex, By or Bz.)
                const float_64 eta = PI/2 - (phi/2);
                /* halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff
                 * of the TWTS-pulse. The abs()-function is for correct offset for -phiReal<-90Deg and +phiReal>+90Deg. */
                const float_64 y1=precisionCast<float_64>(halfSimSize[2]*picongpu::SI::CELL_DEPTH_SI)*abs(cos(eta)); 
                const float_64 m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
                const float_64 y2=m*(pulselength_SI*picongpu::SI::SPEED_OF_LIGHT_SI)/cos(eta); // Approximate cross section of laser pulse through y-axis, scaled with "fudge factor" m.
                const float_64 y3=focus_y_SI; // y-position of laser coordinate system origin within simulation.
                const float_64 tdelay= (y1+y2+y3)/(picongpu::SI::SPEED_OF_LIGHT_SI*beta_0);
                
                return time_SI-tdelay;
            }
            else
                return time_SI-tdelay_user_SI;
        }
        
        template<>
        HDINLINE float3_X
        TWTSFieldE::getTWTSEfield_SI<DIM3>( const PMacc::math::Vector<floatD_64,FieldE::numComponents>& eFieldPositions_SI, const float_64 time) const
        {
            return float3_X( precisionCast<float_X>( calcTWTSEx(eFieldPositions_SI[0],time)/unitField[0] ), float_X(0.), float_X(0.) );
        }
        
        template<>
        HDINLINE float3_X
        TWTSFieldE::getTWTSEfield_SI<DIM2>( const PMacc::math::Vector<floatD_64,FieldE::numComponents>& eFieldPositions_SI, const float_64 time) const
        {
            // Ex->Ez, so also the grid cell offset for Ez has to be used.
            return float3_X( float_X(0.), float_X(0.), precisionCast<float_X>( calcTWTSEx(eFieldPositions_SI[2],time)/unitField[2] ) );
        }
        
        HDINLINE float3_X
        TWTSFieldE::operator()( const DataSpace<simDim>& cellIdx,
                                const uint32_t currentStep ) const
        {
            const float_64 time=getTime_SI(currentStep);
            const PMacc::math::Vector<floatD_64,FieldE::numComponents> eFieldPositions_SI=getEfieldPositions_SI<simDim>(cellIdx);
            // Single TWTS-Pulse
            return getTWTSEfield_SI<simDim>(eFieldPositions_SI, time);
        }

        /** Calculate the Ex(r,t) field here
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field
         * \param phi interaction angle between TWTS laser propagation vector and the y-axis */
        HDINLINE float_64
        TWTSFieldE::calcTWTSEx( const floatD_64& pos, const float_64 time) const
        {
            
            return float_64(0.);
        }

        /* Here comes the B-field part of the TWTS laser pulse. */
        
        HDINLINE
        TWTSFieldB::TWTSFieldB( const float_64 focus_y_SI,
                                const float_64 wavelength_SI,
                                const float_64 pulselength_SI,
                                const float_64 w_x_SI,
                                const float_64 w_y_SI,
                                const float3_64& unitField,
                                const float_X phi,
                                const float_X beta_0,
                                const float_64 tdelay_user_SI,
                                const bool auto_tdelay ) :
            focus_y_SI(focus_y_SI), wavelength_SI(wavelength_SI),
            pulselength_SI(pulselength_SI), w_x_SI(w_x_SI),
            w_y_SI(w_y_SI), unitField(unitField), phi(phi), beta_0(beta_0),
            tdelay_user_SI(tdelay_user_SI), auto_tdelay(auto_tdelay)
        {
#if !defined(__CUDA_ARCH__)
            // These objects cannot be instantiated on CUDA GPU device. Since this is done on host (see fieldBackground.param), this is no problem.
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);
#endif
        }
        
        template<>
        HDINLINE PMacc::math::Vector<floatD_64,FieldB::numComponents>
        TWTSFieldB::getBfieldPositions_SI<DIM3>(const DataSpace<simDim>& cellIdx) const
        {
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center in y (usually maximum of intensity).
            floatD_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI, halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<floatD_X, FieldB::numComponents> bFieldPositions = fieldSolver::NumericalCellType::getBFieldPosition();
            PMacc::math::Vector<floatD_64,FieldB::numComponents> bFieldPositions_SI;
            
            const float_64 unit_length = picongpu::UNIT_LENGTH;
            const floatD_64 cellDimensions = precisionCast<floatD_64>(picongpu::cellSize) * unit_length;
            for( uint32_t i = 0; i < FieldB::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                bFieldPositions[i]   += floatD_X(cellIdx) - laserOrigin;
                bFieldPositions_SI[i] = precisionCast<float_64>(bFieldPositions[i]) * cellDimensions;
                
                /*  Since, the laser propagation direction encloses an angle of phi with the simulation y-axis (i.e. direction of sliding window),
                 *  the positions vectors are rotated around the simulation x-axis before calling the TWTS field functions. Note: The TWTS field
                 *  functions are in non-rotated frame and only use the angle phi to determine the required amount of pulse front tilt.
                 *  RotationMatrix[PI/2+phiReal].(y,z) (180Deg-flip at phiReal=90Deg since coordinate system in paper is oriented the other way round.) */
                bFieldPositions_SI[i] = ( (bFieldPositions_SI[i]).x(),
                                          -sin(phi)*(bFieldPositions_SI[i]).y()-cos(phi)*(bFieldPositions_SI[i]).z(),
                                          +cos(phi)*(bFieldPositions_SI[i]).y()-sin(phi)*(bFieldPositions_SI[i]).z()  );
            }

            return bFieldPositions_SI;
        }
        
        template<>
        HDINLINE PMacc::math::Vector<floatD_64,FieldB::numComponents>
        TWTSFieldB::getBfieldPositions_SI<DIM2>(const DataSpace<simDim>& cellIdx) const
        {
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center (usually maximum of intensity) in y.
            floatD_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI, halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<floatD_X, FieldB::numComponents> bFieldPositions = fieldSolver::NumericalCellType::getBFieldPosition();
            PMacc::math::Vector<floatD_64,FieldB::numComponents> bFieldPositions_SI;
            
            const float_64 unit_length = picongpu::UNIT_LENGTH;
            const floatD_64 cellDimensions = precisionCast<floatD_64>(picongpu::cellSize) * unit_length;
            for( uint32_t i = 0; i < FieldB::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                bFieldPositions[i]   += floatD_X(cellIdx) - laserOrigin;
                bFieldPositions_SI[i] = precisionCast<float_64>(bFieldPositions[i]) * cellDimensions;
                /* Rotate position vector by 90Deg around y-axis, so that the TWTS laser ( k-vector is (0,0,-k) in laser coordinate system )
                 * propagates within the 2D (x,y)-plane of the simulation. */
                /** Corresponding position vector for the Field-components in 2D simulations.
                 *  3D     3D vectors in 2D spaces (x, y)
                 *  x -->  z (Meaning: In 2D-sim, insert cell-coordinate x into TWTS field function coordinate z.)
                 *  y -->  y
                 *  z --> -x (Since z=0 for 2D, we use the existing 3D TWTS-field function and set x=-0)
                 *  Ex --> Ez   (Meaning: Calculate Ex-component of existing 3D TWTS-field to obtain corresponding Ez-component in 2D.
                 *               Note: the position offset due to the Yee-Cell for Ez.)
                 *  By --> By
                 *  Bz --> -Bx (Yes, the sign is necessary.)
                 */
                bFieldPositions_SI[i] = ( -(bFieldPositions_SI[i]).z(), (bFieldPositions_SI[i]).y(), (bFieldPositions_SI[i]).x() );
                
                /*  Since, the laser propagation direction encloses an angle of phi with the simulation y-axis (i.e. direction of sliding window),
                 *  the positions vectors are rotated around the simulation x-axis before calling the TWTS field functions. Note: The TWTS field
                 *  functions are in non-rotated frame and only use the angle phi to determine the required amount of pulse front tilt.
                 *  RotationMatrix[PI/2+phiReal].(y,z) (180Deg-flip at phiReal=90Deg since coordinate system in paper is oriented the other way round.) */
                
                /* Note: The x-axis of rotation is fine in 2D, because that component now contains the (non-existing) simulation z-coordinate. */
                bFieldPositions_SI[i] = ( (bFieldPositions_SI[i]).x(),
                                          -sin(phi)*(bFieldPositions_SI[i]).y()-cos(phi)*(bFieldPositions_SI[i]).z(),
                                          +cos(phi)*(bFieldPositions_SI[i]).y()-sin(phi)*(bFieldPositions_SI[i]).z()  );
            }
            
            return bFieldPositions_SI;
        }
        
        HDINLINE float_64
        TWTSFieldB::getTime_SI(const uint32_t currentStep) const
        {
            const float_64 time_SI=currentStep*picongpu::SI::DELTA_T_SI;
                       
            if ( auto_tdelay ) {
                
                // angle between the laser pulse front and the y-axis. (Good approximation for beta0 \simeq 1. For exact relation look in TWTS core routines for Ex, By or Bz.)
                const float_64 eta = PI/2 - (phi/2);
                /* halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff
                 * of the TWTS-pulse. The abs()-function is for correct offset for -phiReal<-90Deg and +phiReal>+90Deg. */
                const float_64 y1=precisionCast<float_64>(halfSimSize[2]*picongpu::SI::CELL_DEPTH_SI)*abs(cos(eta)); 
                const float_64 m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
                const float_64 y2=m*(pulselength_SI*picongpu::SI::SPEED_OF_LIGHT_SI)/cos(eta); // Approximate cross section of laser pulse through y-axis, scaled with "fudge factor" m.
                const float_64 y3=focus_y_SI; // y-position of laser coordinate system origin within simulation.
                const float_64 tdelay= (y1+y2+y3)/(picongpu::SI::SPEED_OF_LIGHT_SI*beta_0);
                
                return time_SI-tdelay;
            }
            else
                return time_SI-tdelay_user_SI;
        }
        
        template<>
        HDINLINE float3_X
        TWTSFieldB::getTWTSBfield_SI<DIM3>( const PMacc::math::Vector<floatD_64,FieldB::numComponents>& bFieldPositions_SI, const float_64 time) const
        {
            const float_64 By_By=calcTWTSBy(bFieldPositions_SI[1], time); // Calculate By-component with the Yee-Cell offset of a By-field
            const float_64 Bz_By=calcTWTSBz(bFieldPositions_SI[1], time); // Calculate Bz-component the Yee-Cell offset of a Bz-field
            const float_64 By_Bz=calcTWTSBy(bFieldPositions_SI[2], time);
            const float_64 Bz_Bz=calcTWTSBz(bFieldPositions_SI[2], time);
            /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz, we need to back-rotate the resulting B-field vector. */
            const float_64 By_rot=-sin(+phi)*By_By+cos(+phi)*Bz_By;  // RotationMatrix[-(PI/2+phiReal)].(y,z)
            const float_64 Bz_rot=-cos(+phi)*By_Bz-sin(+phi)*Bz_Bz;  // for rotating back the Field-Vektors.
            
            // Finally, the B-field in PIConGPU units.
            return float3_X( float_X(0.0), precisionCast<float_X>(By_rot/unitField[1]), precisionCast<float_X>(Bz_rot/unitField[2]) );
        }
        
        template<>
        HDINLINE float3_X
        TWTSFieldB::getTWTSBfield_SI<DIM2>( const PMacc::math::Vector<floatD_64,FieldB::numComponents>& bFieldPositions_SI, const float_64 time) const
        {
            // By->By and Bz->-Bx, so the grid cell offset for Bx has to be used instead of Bz. Mind the -sign.
            const float_64 By_By= calcTWTSBy(bFieldPositions_SI[1], time); // Calculate By-component with the Yee-Cell offset of a By-field
            const float_64 Bx_By=-calcTWTSBz(bFieldPositions_SI[1], time); // Calculate Bx-component with the Yee-Cell offset of a By-field
            const float_64 By_Bx= calcTWTSBy(bFieldPositions_SI[0], time);
            const float_64 Bx_Bx=-calcTWTSBz(bFieldPositions_SI[0], time);
            /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz, we need to back-rotate the resulting B-field vector. */
            const float_64 By_rot=-sin(phi)*By_By+cos(phi)*Bx_By;  // RotationMatrix[-(PI/2+phiReal)].(y,x) PLEASE CHECK THIS!!!
            const float_64 Bx_rot=-cos(phi)*By_Bx-sin(phi)*Bx_Bx;  // for rotating back the Field-Vektors.
            
            // Finally, the B-field in PIConGPU units.
            return float3_X( float_X(0.0), precisionCast<float_X>(By_rot/unitField[1]), precisionCast<float_X>(Bx_rot/unitField[2]) );
        }
        
        HDINLINE float3_X
        TWTSFieldB::operator()( const DataSpace<simDim>& cellIdx,
                                const uint32_t currentStep ) const
        {
            const float_64 time=getTime_SI(currentStep);
            const PMacc::math::Vector<floatD_64,FieldB::numComponents> bFieldPositions_SI=getBfieldPositions_SI<simDim>(cellIdx);
            // Single TWTS-Pulse
            return getTWTSBfield_SI<simDim>(bFieldPositions_SI, time);
        }

        /** Calculate the By(r,t) field
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field */
        HDINLINE float_64
        TWTSFieldB::calcTWTSBy( const floatD_64& pos, const float_64 time ) const
        {

            return float_64(0.);
        }
        
        /** Calculate the Bz(r,t) field here
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field */
        HDINLINE float_64
        TWTSFieldB::calcTWTSBz( const floatD_64& pos, const float_64 time ) const
        {

            return float_64(0.);
        }

    } /* namespace templates */
} /* namespace picongpu */
