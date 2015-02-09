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

        HINLINE
        TWTSFieldE::TWTSFieldE( const float_64 focus_y_SI,
                                const float_64 wavelength_SI,
                                const float_64 pulselength_SI,
                                const float_64 w_x_SI,
                                const float_64 w_y_SI,
                                const floatD_64& unitField,
                                const float_X phi,
                                const float_X beta_0,
                                const float_64 tdelay_user_SI,
                                const bool auto_tdelay ) :
            focus_y_SI(focus_y_SI), wavelength_SI(wavelength_SI),
            pulselength_SI(pulselength_SI), w_x_SI(w_x_SI),
            w_y_SI(w_y_SI), unitField(unitField), phi(phi), beta_0(beta_0),
            tdelay_user_SI(tdelay_user_SI), auto_tdelay(auto_tdelay)
        {
            // These objects cannot be instantiated on CUDA GPU device. Since this is done on host (see fieldBackground.param), this is no problem.
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            halfSimSize=subGrid.getGlobalDomain().size / 2;
        }

        template<>
        HDINLINE PMacc::math::Vector<float3_64,FieldE::numComponents>
        TWTSFieldE::getEfieldPositions_SI<DIM3>(const DataSpace<simDim>& cellIdx) const
        {
            const float_64 unit_length = picongpu::UNIT_LENGTH;
            // Direct precisionCast on picongpu::cellSize does not work.
            const float3_64 cellDimensions = ( precisionCast<float_64>( picongpu::cellSize.x() ),
                                               precisionCast<float_64>( picongpu::cellSize.y() ),
                                               precisionCast<float_64>( picongpu::cellSize.z() ) ) * unit_length;
            
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center (usually maximum of intensity) in y.
            float3_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI/cellDimensions.y(), halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<float3_X, FieldE::numComponents> eFieldPositions = fieldSolver::NumericalCellType::getEFieldPosition();
            PMacc::math::Vector<float3_64,FieldE::numComponents> eFieldPositions_SI;
            
            for( uint32_t i = 0; i < FieldE::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                eFieldPositions[i]   += float3_X(cellIdx) - laserOrigin;
                eFieldPositions_SI[i] = precisionCast<float_64>(eFieldPositions[i]) * cellDimensions;
                
                /*  Since, the laser propagation direction encloses an angle of phi with the simulation y-axis (i.e. direction of sliding window),
                 *  the positions vectors are rotated around the simulation x-axis before calling the TWTS field functions. Note: The TWTS field
                 *  functions are in non-rotated frame and only use the angle phi to determine the required amount of pulse front tilt.
                 *  RotationMatrix[PI/2+phi].(y,z) (180Deg-flip at phi=90Deg since coordinate system in paper is oriented the other way round.) */
                eFieldPositions_SI[i] = ( (eFieldPositions_SI[i]).x(),
                                          -sin(phi)*(eFieldPositions_SI[i]).y()-cos(phi)*(eFieldPositions_SI[i]).z(),
                                          +cos(phi)*(eFieldPositions_SI[i]).y()-sin(phi)*(eFieldPositions_SI[i]).z()  );
            }

            return eFieldPositions_SI;
        }
        
        template<>
        HDINLINE PMacc::math::Vector<float3_64,FieldE::numComponents>
        TWTSFieldE::getEfieldPositions_SI<DIM2>(const DataSpace<simDim>& cellIdx) const
        {
            const float_64 unit_length = picongpu::UNIT_LENGTH;
            const float2_64 cellDimensions = ( precisionCast<float_64>( picongpu::cellSize.x() ),
                                               precisionCast<float_64>( picongpu::cellSize.y() ) ) * unit_length;
            
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center in y (usually maximum of intensity).
            float2_X laserOrigin = float2_X( halfSimSize.x(), focus_y_SI/cellDimensions.y() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<float3_X, FieldE::numComponents> eFieldPositions = fieldSolver::NumericalCellType::getEFieldPosition();
            PMacc::math::Vector<float3_64,FieldE::numComponents> eFieldPositions_SI;
            
            for( uint32_t i = 0; i < FieldE::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                eFieldPositions[i]   += ( precisionCast<float_X>(cellIdx.x() - laserOrigin.x()),
                                          precisionCast<float_X>(cellIdx.y() - laserOrigin.y()),
                                          float_X(0.0) );
                eFieldPositions_SI[i] = ( precisionCast<float_64>( (eFieldPositions[i]).x() ) * cellDimensions.x(),
                                          precisionCast<float_64>( (eFieldPositions[i]).y() ) * cellDimensions.y(),
                                          float_64(0.0) );
                
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
                 *  RotationMatrix[PI/2+phi].(y,z) (180Deg-flip at phi=90Deg since coordinate system in paper is oriented the other way round.) */
                
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
                 * of the TWTS-pulse. The abs()-function is for correct offset for -phi<-90Deg and +phi>+90Deg. */
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
        TWTSFieldE::getTWTSEfield_SI<DIM3>( const PMacc::math::Vector<float3_64,FieldE::numComponents>& eFieldPositions_SI, const float_64 time) const
        {
            return float3_X( precisionCast<float_X>( calcTWTSEx(eFieldPositions_SI[0],time)/unitField[0] ), float_X(0.), float_X(0.) );
        }
        
        template<>
        HDINLINE float3_X
        TWTSFieldE::getTWTSEfield_SI<DIM2>( const PMacc::math::Vector<float3_64,FieldE::numComponents>& eFieldPositions_SI, const float_64 time) const
        {
            // Ex->Ez, so also the grid cell offset for Ez has to be used.
            return float3_X( float_X(0.), float_X(0.), precisionCast<float_X>( calcTWTSEx(eFieldPositions_SI[2],time)/unitField[2] ) );
        }
        
        HDINLINE float3_X
        TWTSFieldE::operator()( const DataSpace<simDim>& cellIdx,
                                const uint32_t currentStep ) const
        {
            const float_64 time=getTime_SI(currentStep);
            const PMacc::math::Vector<float3_64,FieldE::numComponents> eFieldPositions_SI=getEfieldPositions_SI<simDim>(cellIdx);
            // Single TWTS-Pulse
            return getTWTSEfield_SI<simDim>(eFieldPositions_SI, time);
        }

        /** Calculate the Ex(r,t) field here
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field */
        HDINLINE float_64
        TWTSFieldE::calcTWTSEx( const float3_64& pos, const float_64 time) const
        {
            const float_64 beta0=precisionCast<float_64>(beta_0); // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
            const float_64 phiReal=precisionCast<float_64>(this->phi);
            const float_64 alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
            const float_64 phi=2*alphaTilt; /* Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent
            to our standard definition. Question: Why is the local "phi" not equal in value to the object member "phiReal" or "this->phi"? Because the
            standard TWTS pulse is defined for beta0=1.0 and in the coordinate-system of the TWTS model phi is responsible for pulse front tilt and
            dispersion only. Hence the dispersion will (although physically correct) be slightly off the ideal TWTS pulse for beta0!=1.0. This only shows
            that this TWTS pulse is primarily designed for scenarios close to beta0=1. */
            
            /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for documentation purposes.
             * const float_64 eta = PI/2 - (phiReal - alphaTilt); */
            
            const float_64 cspeed=picongpu::SI::SPEED_OF_LIGHT_SI;
            const float_64 lambda0=precisionCast<float_64>(wavelength_SI);
            const float_64 om0=2*PI*cspeed/lambda0;
            const float_64 tauG=precisionCast<float_64>(pulselength_SI)*2.0; // factor 2 arises from definition convention in laser formula
            const float_64 w0=precisionCast<float_64>(w_x_SI); // w0 is wx here --> w0 could be replaced by wx
            const float_64 rho0=PI*w0*w0/lambda0;
            const float_64 wy=precisionCast<float_64>(w_y_SI); // Width of TWTS pulse
            const float_64 k=2*PI/lambda0;
            const float_64 x=pos.x();
            const float_64 y=pos.y();
            const float_64 z=pos.z();
            const float_64 t=time;
            
            //Calculating shortcuts for speeding up field calculation
            const float_64 sinPhi = sin(phi);
            const float_64 cosPhi = cos(phi);
            const float_64 sinPhi2 = sin(phi/2.);
            const float_64 cosPhi2 = cos(phi/2.);
            const float_64 tanPhi2 = tan(phi/2.);
            
            //The "helpVar" variables decrease the nesting level of the evaluated expressions and thus help with formal code verification through manual code inspection.
            const Complex_64 helpVar1=Complex_64(0,1)*rho0 - y*cosPhi - z*sinPhi;
            const Complex_64 helpVar2=Complex_64(0,-1)*cspeed*om0*tauG*tauG - y*cosPhi/cosPhi2/cosPhi2*tanPhi2 - 2*z*tanPhi2*tanPhi2;
            const Complex_64 helpVar3=Complex_64(0,1)*rho0 - y*cosPhi - z*sinPhi;

            const Complex_64 helpVar4=(
            -(cspeed*cspeed*k*om0*tauG*tauG*wy*wy*x*x)
            - 2*cspeed*cspeed*om0*t*t*wy*wy*rho0 
            + Complex_64(0,2)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*rho0
            - 2*cspeed*cspeed*om0*tauG*tauG*y*y*rho0
            + 4*cspeed*om0*t*wy*wy*z*rho0
            - Complex_64(0,2)*cspeed*om0*om0*tauG*tauG*wy*wy*z*rho0
            - 2*om0*wy*wy*z*z*rho0
            - Complex_64(0,8)*om0*wy*wy*y*(cspeed*t - z)*z*sinPhi2*sinPhi2
            + Complex_64(0,8)/sinPhi*(
                    +2*z*z*(cspeed*om0*t*wy*wy + Complex_64(0,1)*cspeed*y*y - om0*wy*wy*z)
                    + y*(
                        + cspeed*k*wy*wy*x*x
                        - Complex_64(0,2)*cspeed*om0*t*wy*wy*rho0
                        + 2*cspeed*y*y*rho0
                        + Complex_64(0,2)*om0*wy*wy*z*rho0
                    )*tan(PI/2-phi)/sinPhi
                )*sinPhi2*sinPhi2*sinPhi2*sinPhi2
            - Complex_64(0,2)*cspeed*cspeed*om0*t*t*wy*wy*z*sinPhi
            - 2*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*z*sinPhi
            - Complex_64(0,2)*cspeed*cspeed*om0*tauG*tauG*y*y*z*sinPhi
            + Complex_64(0,4)*cspeed*om0*t*wy*wy*z*z*sinPhi
            + 2*cspeed*om0*om0*tauG*tauG*wy*wy*z*z*sinPhi
            - Complex_64(0,2)*om0*wy*wy*z*z*z*sinPhi
            - 4*cspeed*om0*t*wy*wy*y*rho0*tanPhi2
            + 4*om0*wy*wy*y*z*rho0*tanPhi2
            + Complex_64(0,2)*y*y*(cspeed*om0*t*wy*wy + Complex_64(0,1)*cspeed*y*y - om0*wy*wy*z)*cosPhi*cosPhi/cosPhi2/cosPhi2*tanPhi2
            + Complex_64(0,2)*cspeed*k*wy*wy*x*x*z*tanPhi2*tanPhi2
            - 2*om0*wy*wy*y*y*rho0*tanPhi2*tanPhi2
            + 4*cspeed*om0*t*wy*wy*z*rho0*tanPhi2*tanPhi2
            + Complex_64(0,4)*cspeed*y*y*z*rho0*tanPhi2*tanPhi2
            - 4*om0*wy*wy*z*z*rho0*tanPhi2*tanPhi2
            - Complex_64(0,2)*om0*wy*wy*y*y*z*sinPhi*tanPhi2*tanPhi2
            - 2*y*cosPhi*(om0*(cspeed*cspeed*(Complex_64(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_64(0,1)*tauG*tauG*y*y) - cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_64(0,1)*wy*wy*z*z) + Complex_64(0,2)*om0*wy*wy*y*(cspeed*t - z)*tanPhi2 + Complex_64(0,1)*(Complex_64(0,-4)*cspeed*y*y*z + om0*wy*wy*(y*y - 4*(cspeed*t - z)*z))*tanPhi2*tanPhi2)
            )/(2.*cspeed*wy*wy*helpVar1*helpVar2);

            const Complex_64 helpVar5=cspeed*om0*tauG*tauG - Complex_64(0,8)*y*tan(PI/2-phi)/sinPhi/sinPhi*sinPhi2*sinPhi2*sinPhi2*sinPhi2 - Complex_64(0,2)*z*tanPhi2*tanPhi2;
            const Complex_64 result=(Complex_64::cexp(helpVar4)*tauG*Complex_64::csqrt((cspeed*om0*rho0)/helpVar3))/Complex_64::csqrt(helpVar5);			
            return result.get_real();
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
            // These objects cannot be instantiated on CUDA GPU device. Since this is done on host (see fieldBackground.param), this is no problem.
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            halfSimSize=subGrid.getGlobalDomain().size / 2;
        }
        
        template<>
        HDINLINE PMacc::math::Vector<float3_64,FieldB::numComponents>
        TWTSFieldB::getBfieldPositions_SI<DIM3>(const DataSpace<simDim>& cellIdx) const
        {
            const float_64 unit_length = picongpu::UNIT_LENGTH;
            const float3_64 cellDimensions = ( precisionCast<float_64>( picongpu::cellSize.x() ),
                                               precisionCast<float_64>( picongpu::cellSize.y() ),
                                               precisionCast<float_64>( picongpu::cellSize.z() ) ) * unit_length;
            
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center in y (usually maximum of intensity).
            float3_X laserOrigin = float3_X( halfSimSize.x(), focus_y_SI/cellDimensions.y(), halfSimSize.z() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<float3_X, FieldB::numComponents> bFieldPositions = fieldSolver::NumericalCellType::getBFieldPosition();
            PMacc::math::Vector<float3_64,FieldB::numComponents> bFieldPositions_SI;
            
            for( uint32_t i = 0; i < FieldB::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                bFieldPositions[i]   += precisionCast<float_X>(cellIdx) - laserOrigin;
                bFieldPositions_SI[i] = precisionCast<float_64>(bFieldPositions[i]) * cellDimensions;
                
                /*  Since, the laser propagation direction encloses an angle of phi with the simulation y-axis (i.e. direction of sliding window),
                 *  the positions vectors are rotated around the simulation x-axis before calling the TWTS field functions. Note: The TWTS field
                 *  functions are in non-rotated frame and only use the angle phi to determine the required amount of pulse front tilt.
                 *  RotationMatrix[PI/2+phi].(y,z) (180Deg-flip at phi=90Deg since coordinate system in paper is oriented the other way round.) */
                bFieldPositions_SI[i] = ( (bFieldPositions_SI[i]).x(),
                                          -sin(phi)*(bFieldPositions_SI[i]).y()-cos(phi)*(bFieldPositions_SI[i]).z(),
                                          +cos(phi)*(bFieldPositions_SI[i]).y()-sin(phi)*(bFieldPositions_SI[i]).z()  );
            }

            return bFieldPositions_SI;
        }
        
        template<>
        HDINLINE PMacc::math::Vector<float3_64,FieldB::numComponents>
        TWTSFieldB::getBfieldPositions_SI<DIM2>(const DataSpace<simDim>& cellIdx) const
        {
            const float_64 unit_length = picongpu::UNIT_LENGTH;
            const float2_64 cellDimensions = ( precisionCast<float_64>( picongpu::cellSize.x() ),
                                               precisionCast<float_64>( picongpu::cellSize.y() ) ) * unit_length;
                                               
            //TWTS laser coordinate origin is centered transversally and defined longitudinally by the laser center (usually maximum of intensity) in y.
            float2_X laserOrigin = float2_X( halfSimSize.x(), focus_y_SI/cellDimensions.y() );
            
            /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add that to the total cell indices. The physical field coordinate origin is transversally centered with respect to the global simulation volume. */
            PMacc::math::Vector<float3_X, FieldB::numComponents> bFieldPositions = fieldSolver::NumericalCellType::getBFieldPosition();
            PMacc::math::Vector<float3_64,FieldB::numComponents> bFieldPositions_SI;
            
            for( uint32_t i = 0; i < FieldB::numComponents; ++i ) // cellIdx Ex, Ey and Ez
            {
                bFieldPositions[i]   += ( precisionCast<float_X>(cellIdx.x() - laserOrigin.x()),
                                          precisionCast<float_X>(cellIdx.y() - laserOrigin.y()),
                                          float_X(0.0) );
                bFieldPositions_SI[i] = ( precisionCast<float_64>( (bFieldPositions[i]).x() ) * cellDimensions.x(),
                                          precisionCast<float_64>( (bFieldPositions[i]).y() ) * cellDimensions.y(),
                                          float_64(0.0) );
                
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
                 *  RotationMatrix[PI/2+phi].(y,z) (180Deg-flip at phi=90Deg since coordinate system in paper is oriented the other way round.) */
                
                /* Note: Using the x-axis as axis of rotation is fine also in 2D, because that component now contains the (non-existing) simulation z-coordinate. */
                bFieldPositions_SI[i] = ( (bFieldPositions_SI[i]).x(),                                                      // leave    2D z-component unchanged
                                          -sin(phi)*(bFieldPositions_SI[i]).y()-cos(phi)*(bFieldPositions_SI[i]).z(),       // rotates  2D y-component
                                          +cos(phi)*(bFieldPositions_SI[i]).y()-sin(phi)*(bFieldPositions_SI[i]).z()  );    // and      2D x-component
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
                 * of the TWTS-pulse. The abs()-function is for correct offset for -phi<-90Deg and +phi>+90Deg. */
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
        TWTSFieldB::getTWTSBfield_SI<DIM3>( const PMacc::math::Vector<float3_64,FieldB::numComponents>& bFieldPositions_SI, const float_64 time) const
        {
            const float_64 By_By=calcTWTSBy(bFieldPositions_SI[1], time); // Calculate By-component with the Yee-Cell offset of a By-field
            const float_64 Bz_By=calcTWTSBz(bFieldPositions_SI[1], time); // Calculate Bz-component the Yee-Cell offset of a Bz-field
            const float_64 By_Bz=calcTWTSBy(bFieldPositions_SI[2], time);
            const float_64 Bz_Bz=calcTWTSBz(bFieldPositions_SI[2], time);
            /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz, we need to back-rotate the resulting B-field vector. */
            const float_64 By_rot=-sin(+phi)*By_By+cos(+phi)*Bz_By;  // RotationMatrix[-(PI/2+phi)].(By,Bz)
            const float_64 Bz_rot=-cos(+phi)*By_Bz-sin(+phi)*Bz_Bz;  // for rotating back the Field-Vektors.
            
            // Finally, the B-field in PIConGPU units.
            return float3_X( float_X(0.0), precisionCast<float_X>(By_rot/unitField[1]), precisionCast<float_X>(Bz_rot/unitField[2]) );
        }
        
        template<>
        HDINLINE float3_X
        TWTSFieldB::getTWTSBfield_SI<DIM2>( const PMacc::math::Vector<float3_64,FieldB::numComponents>& bFieldPositions_SI, const float_64 time) const
        {
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
            // Analogous to 3D case, but replace By->By and Bz->-Bx. Hence the grid cell offset for Bx has to be used instead of Bz. Mind the -sign.
            const float_64 By_By= calcTWTSBy(bFieldPositions_SI[1], time); // Calculate By-component with the Yee-Cell offset of a By-field
            const float_64 Bx_By=-calcTWTSBz(bFieldPositions_SI[1], time); // Calculate Bx-component with the Yee-Cell offset of a By-field
            const float_64 By_Bx= calcTWTSBy(bFieldPositions_SI[0], time);
            const float_64 Bx_Bx=-calcTWTSBz(bFieldPositions_SI[0], time);
            /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz, we need to back-rotate the resulting B-field vector.
             * Now the rotation is done analogously in the (y,x)-plane. (Reverse of the position vector transformation.) */
            const float_64 By_rot=-sin(phi)*By_By+cos(phi)*Bx_By;  // RotationMatrix[-(PI/2+phi)].(By,Bx)
            const float_64 Bx_rot=-cos(phi)*By_Bx-sin(phi)*Bx_Bx;  // for rotating back the Field-Vektors.
            
            // Finally, the B-field in PIConGPU units.
            return float3_X( float_X(0.0), precisionCast<float_X>(By_rot/unitField[1]), precisionCast<float_X>(Bx_rot/unitField[2]) );
        }
        
        HDINLINE float3_X
        TWTSFieldB::operator()( const DataSpace<simDim>& cellIdx,
                                const uint32_t currentStep ) const
        {
            const float_64 time=getTime_SI(currentStep);
            const PMacc::math::Vector<float3_64,FieldB::numComponents> bFieldPositions_SI=getBfieldPositions_SI<simDim>(cellIdx);
            // Single TWTS-Pulse
            return getTWTSBfield_SI<simDim>(bFieldPositions_SI, time);
        }

        /** Calculate the By(r,t) field here
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field */
        HDINLINE float_64
        TWTSFieldB::calcTWTSBy( const float3_64& pos, const float_64 time ) const
        {
            const float_64 beta0=precisionCast<float_64>(beta_0); // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
            const float_64 phiReal=precisionCast<float_64>(this->phi);
            const float_64 alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
            const float_64 phi=2*alphaTilt; /* Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent
            to our standard definition. Question: Why is the local "phi" not equal in value to the object member "phiReal" or "this->phi"? Because the
            standard TWTS pulse is defined for beta0=1.0 and in the coordinate-system of the TWTS model phi is responsible for pulse front tilt and
            dispersion only. Hence the dispersion will (although physically correct) be slightly off the ideal TWTS pulse for beta0!=1.0. This only shows
            that this TWTS pulse is primarily designed for scenarios close to beta0=1. */
            
            /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for documentation purposes.
             * const float_64 eta = PI/2 - (phiReal - alphaTilt); */
            
            const float_64 cspeed=picongpu::SI::SPEED_OF_LIGHT_SI;
            const float_64 lambda0=precisionCast<float_64>(wavelength_SI);
            const float_64 om0=2*PI*cspeed/lambda0;
            const float_64 tauG=precisionCast<float_64>(pulselength_SI)*2.0; // factor 2 arises from definition convention in laser formula
            const float_64 w0=precisionCast<float_64>(w_x_SI); // w0 is wx here --> w0 could be replaced by wx
            const float_64 rho0=PI*w0*w0/lambda0;
            const float_64 wy=precisionCast<float_64>(w_y_SI); // Width of TWTS pulse
            const float_64 k=2*PI/lambda0;
            const float_64 x=pos.x();
            const float_64 y=pos.y();
            const float_64 z=pos.z();
            const float_64 t=time;
                            
            //Shortcuts for speeding up the field calculation.
            const float_64 sinPhi = sin(phi);
            const float_64 cosPhi = cos(phi);
            const float_64 cosPhi2 = cos(phi/2.);
            const float_64 tanPhi2 = tan(phi/2.);
            
            //The "helpVar" variables decrease the nesting level of the evaluated expressions and thus help with formal code verification through manual code inspection.
            const Complex_64 helpVar1=rho0 + Complex_64(0,1)*y*cosPhi + Complex_64(0,1)*z*sinPhi;
            const Complex_64 helpVar2=cspeed*om0*tauG*tauG + Complex_64(0,2)*(-z - y*tan(PI/2-phi))*tanPhi2*tanPhi2;
            const Complex_64 helpVar3=Complex_64(0,1)*rho0 - y*cosPhi - z*sinPhi;
            
            const Complex_64 helpVar4=-1.0*(
            cspeed*cspeed*k*om0*tauG*tauG*wy*wy*x*x
            + 2*cspeed*cspeed*om0*t*t*wy*wy*rho0
            - Complex_64(0,2)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*rho0
            + 2*cspeed*cspeed*om0*tauG*tauG*y*y*rho0
            - 4*cspeed*om0*t*wy*wy*z*rho0
            + Complex_64(0,2)*cspeed*om0*om0*tauG*tauG*wy*wy*z*rho0
            + 2*om0*wy*wy*z*z*rho0
            + 4*cspeed*om0*t*wy*wy*y*rho0*tanPhi2
            - 4*om0*wy*wy*y*z*rho0*tanPhi2
            - Complex_64(0,2)*cspeed*k*wy*wy*x*x*z*tanPhi2*tanPhi2
            + 2*om0*wy*wy*y*y*rho0*tanPhi2*tanPhi2
            - 4*cspeed*om0*t*wy*wy*z*rho0*tanPhi2*tanPhi2
            - Complex_64(0,4)*cspeed*y*y*z*rho0*tanPhi2*tanPhi2
            + 4*om0*wy*wy*z*z*rho0*tanPhi2*tanPhi2
            - Complex_64(0,2)*cspeed*k*wy*wy*x*x*y*tan(PI/2-phi)*tanPhi2*tanPhi2
            - 4*cspeed*om0*t*wy*wy*y*rho0*tan(PI/2-phi)*tanPhi2*tanPhi2
            - Complex_64(0,4)*cspeed*y*y*y*rho0*tan(PI/2-phi)*tanPhi2*tanPhi2
            + 4*om0*wy*wy*y*z*rho0*tan(PI/2-phi)*tanPhi2*tanPhi2
            + 2*z*sinPhi*(
                om0*(cspeed*cspeed*(Complex_64(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_64(0,1)*tauG*tauG*y*y) - cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_64(0,1)*wy*wy*z*z)
                + Complex_64(0,2)*om0*wy*wy*y*(cspeed*t - z)*tanPhi2 + Complex_64(0,1)*(Complex_64(0,-2)*cspeed*y*y*z + om0*wy*wy*(y*y - 2*(cspeed*t - z)*z))*tanPhi2*tanPhi2
                )
            + 2*y*cosPhi*(
                om0*(cspeed*cspeed*(Complex_64(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_64(0,1)*tauG*tauG*y*y) - cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_64(0,1)*wy*wy*z*z)
                + Complex_64(0,2)*om0*wy*wy*y*(cspeed*t - z)*tanPhi2
                + Complex_64(0,1)*(Complex_64(0,-4)*cspeed*y*y*z + om0*wy*wy*(y*y - 4*(cspeed*t - z)*z) - 2*y*(cspeed*om0*t*wy*wy + Complex_64(0,1)*cspeed*y*y - om0*wy*wy*z)*tan(PI/2-phi))*tanPhi2*tanPhi2
                )
            )/(2.*cspeed*wy*wy*helpVar1*helpVar2);

            const Complex_64 helpVar5=Complex_64(0,-1)*cspeed*om0*tauG*tauG + (-z - y*tan(PI/2-phi))*tanPhi2*tanPhi2*2;
            const Complex_64 helpVar6=(cspeed*(cspeed*om0*tauG*tauG + Complex_64(0,2)*(-z - y*tan(PI/2-phi))*tanPhi2*tanPhi2))/(om0*rho0);
            const Complex_64 result=(Complex_64::cexp(helpVar4)*tauG/cosPhi2/cosPhi2*(rho0 + Complex_64(0,1)*y*cosPhi + Complex_64(0,1)*z*sinPhi)*(Complex_64(0,2)*cspeed*t + cspeed*om0*tauG*tauG - Complex_64(0,4)*z + cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*cosPhi + Complex_64(0,2)*y*tanPhi2)*Complex_64::cpow(helpVar3,-1.5))/(2.*helpVar5*Complex_64::csqrt(helpVar6));

            return result.get_real();
        }
        
        /** Calculate the Bz(r,t) field
         *
         * \param pos Spatial position of the target field.
         * \param time Absolute time (SI, including all offsets and transformations) for calculating the field */
        HDINLINE float_64
        TWTSFieldB::calcTWTSBz( const float3_64& pos, const float_64 time ) const
        {
            const float_64 beta0=precisionCast<float_64>(beta_0); // propagation speed of overlap normalized to the speed of light. [Default: beta0=1.0]
            const float_64 phiReal=precisionCast<float_64>(this->phi);
            const float_64 alphaTilt=atan2(1-beta0*cos(phiReal),beta0*sin(phiReal));
            const float_64 phi=2*alphaTilt; /* Definition of the laser pulse front tilt angle for the laser field below. For beta0=1.0, this is equivalent
            to our standard definition. Question: Why is the local "phi" not equal in value to the object member "phiReal" or "this->phi"? Because the
            standard TWTS pulse is defined for beta0=1.0 and in the coordinate-system of the TWTS model phi is responsible for pulse front tilt and
            dispersion only. Hence the dispersion will (although physically correct) be slightly off the ideal TWTS pulse for beta0!=1.0. This only shows
            that this TWTS pulse is primarily designed for scenarios close to beta0=1. */
            
            /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for documentation purposes.
             * const float_64 eta = PI/2 - (phiReal - alphaTilt); */
            
            const float_64 cspeed=picongpu::SI::SPEED_OF_LIGHT_SI;
            const float_64 lambda0=precisionCast<float_64>(wavelength_SI);
            const float_64 om0=2*PI*cspeed/lambda0;
            const float_64 tauG=precisionCast<float_64>(pulselength_SI)*2.0; // factor 2 arises from definition convention in laser formula
            const float_64 w0=precisionCast<float_64>(w_x_SI); // w0 is wx here --> w0 could be replaced by wx
            const float_64 rho0=PI*w0*w0/lambda0;
            const float_64 wy=precisionCast<float_64>(w_y_SI); // Width of TWTS pulse
            const float_64 k=2*PI/lambda0;
            const float_64 x=pos.x();
            const float_64 y=pos.y();
            const float_64 z=pos.z();
            const float_64 t=time;
                            
            //Shortcuts for speeding up the field calculation.
            const float_64 sinPhi = sin(phi);
            const float_64 cosPhi = cos(phi);
            //const float_64 tanPhi = tan(phi);
            const float_64 sinPhi2 = sin(phi/2.);
            const float_64 cosPhi2 = cos(phi/2.);
            const float_64 tanPhi2 = tan(phi/2.);
            
            //The "helpVar" variables decrease the nesting level of the evaluated expressions and thus help with formal code verification through manual code inspection.
            const Complex_64 helpVar1=-(cspeed*z) - cspeed*y*tan(PI/2-phi) + Complex_64(0,1)*cspeed*rho0/sinPhi;
            const Complex_64 helpVar2=Complex_64(0,1)*rho0 - y*cosPhi - z*sinPhi;
            const Complex_64 helpVar3=helpVar2*cspeed;
            const Complex_64 helpVar4=cspeed*om0*tauG*tauG - Complex_64(0,1)*y*cosPhi/cosPhi2/cosPhi2*tanPhi2 - Complex_64(0,2)*z*tanPhi2*tanPhi2;
            const Complex_64 helpVar5=2*cspeed*t - Complex_64(0,1)*cspeed*om0*tauG*tauG - 2*z + 8*y/sinPhi/sinPhi/sinPhi*sinPhi2*sinPhi2*sinPhi2*sinPhi2 - 2*z*tanPhi2*tanPhi2;

            const Complex_64 helpVar6=(
            (om0*y*rho0/cosPhi2/cosPhi2/cosPhi2/cosPhi2)/helpVar1 
            - (Complex_64(0,2)*k*x*x)/helpVar2 
            - (Complex_64(0,1)*om0*om0*tauG*tauG*rho0)/helpVar2
            - (Complex_64(0,4)*y*y*rho0)/(wy*wy*helpVar2)
            + (om0*om0*tauG*tauG*y*cosPhi)/helpVar2
            + (4*y*y*y*cosPhi)/(wy*wy*helpVar2)
            + (om0*om0*tauG*tauG*z*sinPhi)/helpVar2
            + (4*y*y*z*sinPhi)/(wy*wy*helpVar2)
            + (Complex_64(0,2)*om0*y*y*cosPhi/cosPhi2/cosPhi2*tanPhi2)/helpVar3
            + (om0*y*rho0*cosPhi/cosPhi2/cosPhi2*tanPhi2)/helpVar3
            + (Complex_64(0,1)*om0*y*y*cosPhi*cosPhi/cosPhi2/cosPhi2*tanPhi2)/helpVar3
            + (Complex_64(0,4)*om0*y*z*tanPhi2*tanPhi2)/helpVar3
            - (2*om0*z*rho0*tanPhi2*tanPhi2)/helpVar3
            - (Complex_64(0,2)*om0*z*z*sinPhi*tanPhi2*tanPhi2)/helpVar3
            - (om0*helpVar5*helpVar5)/(cspeed*helpVar4)
            )/4.;
                    
            const Complex_64 helpVar7=cspeed*om0*tauG*tauG - Complex_64(0,1)*y*cosPhi/cosPhi2/cosPhi2*tanPhi2 - Complex_64(0,2)*z*tanPhi2*tanPhi2;
            const Complex_64 result=(Complex_64(0,2)*Complex_64::cexp(helpVar6)*tauG*tanPhi2*(cspeed*t - z + y*tanPhi2)*Complex_64::csqrt((om0*rho0)/helpVar3))/Complex_64::cpow(helpVar7,1.5);

            return result.get_real();
        }
        
    } /* namespace templates */
} /* namespace picongpu */
