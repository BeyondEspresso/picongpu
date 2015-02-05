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
            tdelay_user_SI(tdelay_user_SI), auto_tdelay(auto_tdelay),
            includeCollidingTWTS(includeCollidingTWTS)
        {
            const uint32_t numComponents = DIM3 ;
#if !defined(__CUDA_ARCH__)
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);
#endif
        }

        template<dimNum>
        HDINLINE PMacc::math::Vector<floatD_64,FieldE::numComponents>
        TWTSFieldE::getEfieldPositions_SI<dimNum>(const DataSpace<simDim>& cellIdx) const;
        
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
        
        HDINLINE float_64 getTime_SI(const uint32_t currentStep)
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
        
        template<dimNum>
        HDINLINE float3_X
        TWTSFieldE::getTWTSEfield_SI<dimNum>( const PMacc::math::Vector<floatD_64,FieldE::numComponents>& eFieldPositions_SI,
                                              const float_64 time) const;
        
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
            const float_64 alphaTilt=atan2(1-beta_0*cos(phiReal),beta_0*sin(phiReal));
            const float_64 phi=2*alphaTilt; // Definition of the laser pulse front tilt angle for the laser field below. For beta_0=1.0, this is equivalent to our standard definition.
            const float_64 eta = PI/2 - (phiReal - alphaTilt); // angle between the laser pulse front and the y-axis

            const float_64 cspeed=::picongpu::SI::SPEED_OF_LIGHT_SI;
            const float_64 om0=2*PI*cspeed/wavelength_SI;
            const float_64 tauG=(pulselength_SI)*2.0; // factor 2 arises from definition convention in laser formula
            const float_64 w0=w_x_SI; // w0 is wx here --> w0 could be replaced by wx
            const float_64 rho0=PI*w0*w0/wavelength_SI;
            const float_64 wy=w_y_SI; // Width of TWTS pulse
            const float_64 k=2*PI/wavelength_SI;
            const float_64 x=pos.x();
            const float_64 y=-sin(phiReal)*pos.y()-cos(phiReal)*pos.z();    // RotationMatrix[PI-phiReal].(y,z)
            const float_64 z=+cos(phiReal)*pos.y()-sin(phiReal)*pos.z();    // TO DO: For 2 counter-propagation TWTS pulses take +phiReal and -phiReal. Where do we implement this?
            const float_64 y1=(float_64)(halfSimSize[2]*::picongpu::SI::CELL_DEPTH_SI)/tan(eta); // halfSimSize[2] --> Half-depth of simulation volume (in z); By geometric projection we calculate the y-distance walkoff of the TWTS-pulse.
            const float_64 m=3.; // Fudge parameter to make sure, that TWTS pulse starts to impact simulation volume at low intensity values.
            const float_64 y2=(tauG/2*cspeed)/sin(eta)*m; // pulse length projected on y-axis, scaled with "fudge factor" m.
            const float_64 y3=focus_y_SI; // Position of maximum intensity in simulation volume along y
            const float_64 tdelay= (y1+y2+y3)/(cspeed*beta_0);

            float_64 t=time;
            if (auto_tdelay)
                t -= tdelay;
            else
                t -= tdelay_user;

            const Complex_64 helpVar1=Complex_64(0,1)*rho0 - y*cos(phi) - z*sin(phi);
            const Complex_64 helpVar2=Complex_64(0,-1)*cspeed*om0*tauG*tauG - y*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.) - 2*z*tan(phi/2.)*tan(phi/2.);
            const Complex_64 helpVar3=Complex_64(0,1)*rho0 - y*cos(phi) - z*sin(phi);

            const Complex_64 helpVar4=(
            -(cspeed*cspeed*k*om0*tauG*tauG*wy*wy*x*x)
            - 2*cspeed*cspeed*om0*t*t*wy*wy*rho0
            + Complex_64(0,2)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*rho0
            - 2*cspeed*cspeed*om0*tauG*tauG*y*y*rho0
            + 4*cspeed*om0*t*wy*wy*z*rho0
            - Complex_64(0,2)*cspeed*om0*om0*tauG*tauG*wy*wy*z*rho0
            - 2*om0*wy*wy*z*z*rho0
            - Complex_64(0,8)*om0*wy*wy*y*(cspeed*t - z)*z*sin(phi/2.)*sin(phi/2.)
            + Complex_64(0,8)/sin(phi)*(
                    +2*z*z*(cspeed*om0*t*wy*wy + Complex_64(0,1)*cspeed*y*y - om0*wy*wy*z)
                    + y*(
                        + cspeed*k*wy*wy*x*x
                        - Complex_64(0,2)*cspeed*om0*t*wy*wy*rho0
                        + 2*cspeed*y*y*rho0
                        + Complex_64(0,2)*om0*wy*wy*z*rho0
                    )*tan(PI/2-phi)/sin(phi)
                )*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)
            - Complex_64(0,2)*cspeed*cspeed*om0*t*t*wy*wy*z*sin(phi)
            - 2*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*z*sin(phi)
            - Complex_64(0,2)*cspeed*cspeed*om0*tauG*tauG*y*y*z*sin(phi)
            + Complex_64(0,4)*cspeed*om0*t*wy*wy*z*z*sin(phi)
            + 2*cspeed*om0*om0*tauG*tauG*wy*wy*z*z*sin(phi)
            - Complex_64(0,2)*om0*wy*wy*z*z*z*sin(phi)
            - 4*cspeed*om0*t*wy*wy*y*rho0*tan(phi/2.)
            + 4*om0*wy*wy*y*z*rho0*tan(phi/2.)
            + Complex_64(0,2)*y*y*(cspeed*om0*t*wy*wy + Complex_64(0,1)*cspeed*y*y - om0*wy*wy*z)*cos(phi)*cos(phi)/cos(phi/2.)/cos(phi/2.)*tan(phi/2.)
            + Complex_64(0,2)*cspeed*k*wy*wy*x*x*z*tan(phi/2.)*tan(phi/2.)
            - 2*om0*wy*wy*y*y*rho0*tan(phi/2.)*tan(phi/2.)
            + 4*cspeed*om0*t*wy*wy*z*rho0*tan(phi/2.)*tan(phi/2.)
            + Complex_64(0,4)*cspeed*y*y*z*rho0*tan(phi/2.)*tan(phi/2.)
            - 4*om0*wy*wy*z*z*rho0*tan(phi/2.)*tan(phi/2.)
            - Complex_64(0,2)*om0*wy*wy*y*y*z*sin(phi)*tan(phi/2.)*tan(phi/2.)
            - 2*y*cos(phi)*(om0*(cspeed*cspeed*(Complex_64(0,1)*t*t*wy*wy + om0*t*tauG*tauG*wy*wy + Complex_64(0,1)*tauG*tauG*y*y) - cspeed*(Complex_64(0,2)*t + om0*tauG*tauG)*wy*wy*z + Complex_64(0,1)*wy*wy*z*z) + Complex_64(0,2)*om0*wy*wy*y*(cspeed*t - z)*tan(phi/2.) + Complex_64(0,1)*(Complex_64(0,-4)*cspeed*y*y*z + om0*wy*wy*(y*y - 4*(cspeed*t - z)*z))*tan(phi/2.)*tan(phi/2.))
            )/(2.*cspeed*wy*wy*helpVar1*helpVar2);

            const Complex_64 helpVar5=cspeed*om0*tauG*tauG - Complex_64(0,8)*y*tan(PI/2-phi)/sin(phi)/sin(phi)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.)*sin(phi/2.) - Complex_64(0,2)*z*tan(phi/2.)*tan(phi/2.);
            const Complex_64 result=(Complex_64::cexp(helpVar4)*tauG*Complex_64::csqrt((cspeed*om0*rho0)/helpVar3))/Complex_64::csqrt(helpVar5);
            return result.get_real();
        }

        HDINLINE
        TWTSFieldB::TWTSFieldB()
        {
#if !defined(__CUDA_ARCH__)
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);
#endif
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
