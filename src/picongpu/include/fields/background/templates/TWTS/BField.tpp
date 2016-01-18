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
#include "math/Complex.hpp"

#include "fields/background/templates/TWTS/RotateField.tpp"
#include "fields/background/templates/TWTS/GetInitialTimeDelay_SI.tpp"
#include "fields/background/templates/TWTS/getFieldPositions_SI.tpp"
#include "fields/background/templates/TWTS/BField.hpp"

namespace picongpu
{
/** Load pre-defined background field */
namespace templates
{
/** Traveling-wave Thomson scattering laser pulse */
namespace twts
{
    namespace pmMath = PMacc::algorithms::math;
    
    HINLINE
    BField::BField( const float_64 focus_y_SI,
                    const float_64 wavelength_SI,
                    const float_64 pulselength_SI,
                    const float_64 w_x_SI,
                    const float_64 w_y_SI,
                    const float_X phi,
                    const float_X beta_0,
                    const float_64 tdelay_user_SI,
                    const bool auto_tdelay,
                    const PolarizationType pol ) :
        focus_y_SI(focus_y_SI), wavelength_SI(wavelength_SI),
        pulselength_SI(pulselength_SI), w_x_SI(w_x_SI),
        w_y_SI(w_y_SI), phi(phi), beta_0(beta_0),
        tdelay_user_SI(tdelay_user_SI), dt(SI::DELTA_T_SI),
        unit_length(UNIT_LENGTH), auto_tdelay(auto_tdelay), pol(pol), phiPositive( float_X(1.0) )
    {
        /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
         * on host (see fieldBackground.param), this is no problem. */
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        halfSimSize = subGrid.getGlobalDomain().size / 2;
        tdelay = detail::getInitialTimeDelay_SI(auto_tdelay, tdelay_user_SI, 
                                                halfSimSize, pulselength_SI,
                                                focus_y_SI, phi, beta_0);
        if ( phi < float_X(0.0) ) phiPositive = float_X(-1.0);
    }
    
    template<>
    HDINLINE float3_X
    BField::getTWTSBfield_Normalized<DIM3>(
            const PMacc::math::Vector<floatD_64,detail::numComponents>& bFieldPositions_SI,
            const float_64 time) const
    {
        PMacc::math::Vector<float3_64,detail::numComponents> pos(0.0);
        for (uint32_t k = 0; k<detail::numComponents;++k) {
            for (uint32_t i = 0; i<simDim;++i) pos[k][i] = bFieldPositions_SI[k][i];
        }
        
        /* Calculate By-component with the Yee-Cell offset of a By-field */
        const float_64 By_By = calcTWTSBy(pos[1], time);
        /* Calculate Bz-component with the Yee-Cell offset of a By-field */
        const float_64 Bz_By = calcTWTSBz_Ex(pos[1], time);
        /* Calculate By-component with the Yee-Cell offset of a Bz-field */
        const float_64 By_Bz = calcTWTSBy(pos[2], time);
        /* Calculate Bz-component with the Yee-Cell offset of a Bz-field */
        const float_64 Bz_Bz = calcTWTSBz_Ex(pos[2], time);
        /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex,
         * we need to back-rotate the resulting B-field vector. */
        /* RotationMatrix[-(PI/2+phi)].(By,Bz) for rotating back the Field-Vektors. */
        const float_64 By_rot = -pmMath::sin(+phi)*By_By+pmMath::cos(+phi)*Bz_By;
        const float_64 Bz_rot = -pmMath::cos(+phi)*By_Bz-pmMath::sin(+phi)*Bz_Bz;
        
        /* Finally, the B-field normalized to the peak amplitude. */
        return float3_X( float_X(0.0),
                         float_X(By_rot),
                         float_X(Bz_rot) );
    }
    
    template<>
    HDINLINE float3_X
    BField::getTWTSBfield_Normalized_Ey<DIM3>(
            const PMacc::math::Vector<floatD_64,detail::numComponents>& bFieldPositions_SI,
            const float_64 time) const
    {
        PMacc::math::Vector<float3_64,detail::numComponents> pos(0.0);
        for (uint32_t k = 0; k<detail::numComponents;++k) {
            for (uint32_t i = 0; i<simDim;++i) pos[k][i] = bFieldPositions_SI[k][i];
        }
        
        /* Calculate Bz-component with the Yee-Cell offset of a By-field */
        const float_64 Bz_By = calcTWTSBz_Ey(pos[1], time);
        /* Calculate Bz-component with the Yee-Cell offset of a Bz-field */
        const float_64 Bz_Bz = calcTWTSBz_Ey(pos[2], time);
        /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex,
         * we need to back-rotate the resulting B-field vector. */
        /* RotationMatrix[-(PI/2+phi)].(By,Bz) for rotating back the Field-Vektors. */
        const float_64 By_rot = +pmMath::cos(+phi)*Bz_By;
        const float_64 Bz_rot = -pmMath::sin(+phi)*Bz_Bz;
        
        /* Finally, the B-field normalized to the peak amplitude. */
        return float3_X( float_X( calcTWTSBx(pos[0], time) ),
                         float_X( By_rot ),
                         float_X( Bz_rot ) );
    }
    
    template<>
    HDINLINE float3_X
    BField::getTWTSBfield_Normalized<DIM2>(
            const PMacc::math::Vector<floatD_64,detail::numComponents>& bFieldPositions_SI,
            const float_64 time) const
    {
        PMacc::math::Vector<float3_64,detail::numComponents> pos(0.0);
        for (uint32_t k = 0; k<detail::numComponents;++k) {
            /* The 2D output of getFieldPositions_SI only returns
             * the y- and z-component of a 3D vector. */
            for (uint32_t i = 0; i<simDim;++i) pos[k][i+1] = bFieldPositions_SI[k][i];
        }
        /*  Corresponding position vector for the Field-components in 2D simulations.
         *  3D     3D vectors in 2D space (x, y)
         *  x -->  z (Meaning: In 2D-sim, insert cell-coordinate x
         *            into TWTS field function coordinate z.)
         *  y -->  y
         *  z --> -x (Since z=0 for 2D, we use the existing
         *            3D TWTS-field-function and set x = -0)
         *  Ex --> Ez (Meaning: Calculate Ex-component of existing 3D TWTS-field to obtain
         *             corresponding Ez-component in 2D.
         *             Note: the position offset due to the Yee-Cell for Ez.)
         *  By --> By
         *  Bz --> -Bx (Yes, the sign is necessary.)
         */ 
        /* Analogous to 3D case, but replace By --> By and Bz --> -Bx. Hence the grid cell offset
         * for Bx has to be used instead of Bz. Mind the -sign. */
         
        /* Calculate By-component with the Yee-Cell offset of a By-field */
        const float_64 By_By =  calcTWTSBy(pos[1], time);
        /* Calculate Bx-component with the Yee-Cell offset of a By-field */
        const float_64 Bx_By = -calcTWTSBz_Ex(pos[1], time);
        /* Calculate By-component with the Yee-Cell offset of a Bx-field */
        const float_64 By_Bx =  calcTWTSBy(pos[0], time);
        /* Calculate Bx-component with the Yee-Cell offset of a Bx-field */
        const float_64 Bx_Bx = -calcTWTSBz_Ex(pos[0], time);
        /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex, we
         * need to back-rotate the resulting B-field vector. Now the rotation is done
         * analogously in the (y,x)-plane. (Reverse of the position vector transformation.) */
        /* RotationMatrix[-(PI / 2+phi)].(By,Bx) */
        const float_64 By_rot = -pmMath::sin(phi)*By_By+pmMath::cos(phi)*Bx_By;
        /* for rotating back the Field-Vektors.*/
        const float_64 Bx_rot = -pmMath::cos(phi)*By_Bx-pmMath::sin(phi)*Bx_Bx;
        
        /* Finally, the B-field normalized to the peak amplitude. */
        return float3_X( float_X(Bx_rot),
                         float_X(By_rot),
                         float_X(0.0) );
    }
    
    template<>
    HDINLINE float3_X
    BField::getTWTSBfield_Normalized_Ey<DIM2>(
            const PMacc::math::Vector<floatD_64,detail::numComponents>& bFieldPositions_SI,
            const float_64 time) const
    {
        PMacc::math::Vector<float3_64,detail::numComponents> pos(0.0);
        for (uint32_t k = 0; k<detail::numComponents;++k) {
            /* The 2D output of getFieldPositions_SI only returns
             * the y- and z-component of a 3D vector. */
            for (uint32_t i = 0; i<simDim;++i) pos[k][i+1] = bFieldPositions_SI[k][i];
        }
        /*  Corresponding position vector for the Field-components in 2D simulations.
         *  3D     3D vectors in 2D space (x, y)
         *  x -->  z (Meaning: In 2D-sim, insert cell-coordinate x
         *            into TWTS field function coordinate z.)
         *  y -->  y
         *  z --> -x (Since z=0 for 2D, we use the existing
         *            3D TWTS-field-function and set x = -0)
         *  Ex --> Ez (Meaning: Calculate Ex-component of existing 3D TWTS-field to obtain
         *             corresponding Ez-component in 2D.
         *             Note: the position offset due to the Yee-Cell for Ez.)
         *  By --> By
         *  Bz --> -Bx (Yes, the sign is necessary.)
         */ 
        /* Analogous to 3D case, but replace By --> By and Bz --> -Bx. Hence the grid cell offset
         * for Bx has to be used instead of Bz. Mind the -sign. */
         
        /* Calculate Bx-component with the Yee-Cell offset of a By-field */
        const float_64 Bx_By = -calcTWTSBz_Ex(pos[1], time);
        /* Calculate Bx-component with the Yee-Cell offset of a Bx-field */
        const float_64 Bx_Bx = -calcTWTSBz_Ex(pos[0], time);
        /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex, we
         * need to back-rotate the resulting B-field vector. Now the rotation is done
         * analogously in the (y,x)-plane. (Reverse of the position vector transformation.) */
        /* RotationMatrix[-(PI / 2+phi)].(By,Bx) */
        const float_64 By_rot = +pmMath::cos(phi)*Bx_By;
        /* for rotating back the Field-Vektors.*/
        const float_64 Bx_rot = -pmMath::sin(phi)*Bx_Bx;
        
        /* Finally, the B-field normalized to the peak amplitude. */
        return float3_X( float_X( Bx_rot ),
                         float_X( By_rot ),
                         float_X( calcTWTSBx(pos[2], time) ) );
    }
    
    HDINLINE float3_X
    BField::operator()( const DataSpace<simDim>& cellIdx,
                            const uint32_t currentStep ) const
    {
        const float_64 time_SI = float_64(currentStep) * dt - tdelay;
        
        const PMacc::math::Vector<floatD_64,detail::numComponents> bFieldPositions_SI =
              detail::getFieldPositions_SI(cellIdx,halfSimSize,
                fieldSolver::NumericalCellType::getBFieldPosition(),unit_length,focus_y_SI,phi);
        /* Single TWTS-Pulse */
        switch (pol)
        {
            case NORMAL_TO_TILTPLANE :
            return getTWTSBfield_Normalized<simDim>(bFieldPositions_SI, time_SI);
            
            case WITHIN_TILTPLANE :
            return getTWTSBfield_Normalized_Ey<simDim>(bFieldPositions_SI, time_SI);
        }
        return getTWTSBfield_Normalized<simDim>(bFieldPositions_SI, time_SI);
    }

    /** Calculate the By(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcTWTSBy( const float3_64& pos, const float_64 time ) const
    {
        typedef PMacc::math::Complex<float_T> complex_T;
        typedef PMacc::math::Complex<float_64> complex_64;
        /** Unit of Speed */
        const double UNIT_SPEED = SI::SPEED_OF_LIGHT_SI;
        /** Unit of time */
        const double UNIT_TIME = SI::DELTA_T_SI;
        /** Unit of length */
        const double UNIT_LENGTH = UNIT_TIME*UNIT_SPEED;
        
        /* If phi < 0 the formulas below are not directly applicable.
         * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
         * z-axis of the coordinate system in this function.
         */
        const float_T phiReal = float_T( pmMath::abs(phi) );
        const float_T alphaTilt = pmMath::atan2(float_T(1.0)-float_T(beta_0)*pmMath::cos(phiReal),
                                                float_T(beta_0)*pmMath::sin(phiReal));
        const float_T phiT = float_T(2.0)*alphaTilt;
        /* Definition of the laser pulse front tilt angle for the laser field below.
         * For beta0=1.0, this is equivalent to our standard definition. Question: Why is the
         * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
         * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
         * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
         * the dispersion will (although physically correct) be slightly off the ideal TWTS
         * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
         * scenarios close to beta0 = 1. */
        
        /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
         * documentation purposes. */
        /* const float_T eta = float_T(PI/2) - (phiReal - alphaTilt); */
        
        const float_T cspeed = float_T( SI::SPEED_OF_LIGHT_SI / UNIT_SPEED );
        const float_T lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
        const float_T om0 = float_T(2.0*PI*cspeed / lambda0);
        /* factor 2  in tauG arises from definition convention in laser formula */
        const float_T tauG = float_T(pulselength_SI*2.0 / UNIT_TIME);
        /* w0 is wx here --> w0 could be replaced by wx */
        const float_T w0 = float_T(w_x_SI / UNIT_LENGTH);
        const float_T rho0 = float_T(PI*w0*w0 / lambda0);
        /* wy is width of TWTS pulse */
        const float_T wy = float_T(w_y_SI / UNIT_LENGTH);
        const float_T k = float_T(2.0*PI / lambda0);

        /* In order to calculate in single-precision and in order to account for errors in
         * the approximations far from the coordinate origin, we use the wavelength-periodicity and
         * the known propagation direction for realizing the laser pulse using relative coordinates
         * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
         * in double precision.
         */
        const float_64 tanAlpha = ( float_64(1.0) - beta_0 * pmMath::cos(phi) )
                                    / ( beta_0 * pmMath::sin(phi) );                            
        const float_64 tanFocalLine = pmMath::tan( PI / float_64(2.0) - phi );
        const float_64 deltaT = wavelength_SI / SI::SPEED_OF_LIGHT_SI
                                 * ( float_64(1.0) + tanAlpha / tanFocalLine);
        const float_64 deltaY = wavelength_SI / tanFocalLine;
        const float_64 deltaZ = -wavelength_SI;
        const float_64 numberOfPeriods = pmMath::floor( time / deltaT );
        const float_T timeMod = float_T( time - numberOfPeriods * deltaT );
        const float_T yMod = float_T( pos.y() + numberOfPeriods * deltaY );
        const float_T zMod = float_T( pos.z() + numberOfPeriods * deltaZ );

        const float_T x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
        const float_T y = float_T(phiPositive * yMod / UNIT_LENGTH);
        const float_T z = float_T(zMod / UNIT_LENGTH);
        const float_T t = float_T(timeMod / UNIT_TIME);

        /* Calculate distance from laser pulse front and set field to zero if the result
         * is 5 times the pulse length away. This sets a clear border of the laser pulse,
         * so that collisions with the front face of the simulation can be avoided
         * in a reliable, reproducible fashion. This is important for example for the Lehe solver.
         */
        const float_T offset = pmMath::abs( z - phiPositive * (float_T) tanAlpha * y ) * pmMath::cos( phiT / float_T(2.0) );
        const float_T offsetTolerance = float_T(4.0) * tauG * cspeed + lambda0;
        if ( offset > offsetTolerance ) return float_T(0.0);

        /* Calculating shortcuts for speeding up field calculation */
        const float_T sinPhi = pmMath::sin(phiT);
        const float_T cosPhi = pmMath::cos(phiT);
        const float_T cscPhi = float_T(1.0)/pmMath::sin(phiT);
        const float_T secPhi2 = float_T(1.0)/pmMath::cos(phiT / float_T(2.0));
        const float_T sinPhi2 = pmMath::sin(phiT / float_T(2.0));
        const float_T sin2Phi = pmMath::sin(phiT * float_T(2.0));
        const float_T tanPhi2 = pmMath::tan(phiT / float_T(2.0));

        const float_T sinPhi_2 = sinPhi * sinPhi;
        const float_T sinPhi_3 = sinPhi * sinPhi_2;
        const float_T sinPhi_4 = sinPhi_2 * sinPhi_2;

        const float_T sinPhi2_2 = sinPhi2 * sinPhi2;
        const float_T sinPhi2_4 = sinPhi2_2 * sinPhi2_2;
        const float_T tanPhi2_2 = tanPhi2 * tanPhi2;

        const float_T tauG2 = tauG * tauG;
        const float_T x2 = x * x;
        const float_T y2 = y * y;
        const float_T z2 = z * z;

        /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
         * thus help with formal code verification through manual code inspection.
         */
        const complex_T helpVar1 = cspeed * om0 * tauG2 * sinPhi_4
            - complex_T(0,8) * sinPhi2_4 * sinPhi * ( y * cosPhi + z * sinPhi );

        const complex_T helpVar2 = complex_T(0,1) * rho0 - y * cosPhi - z * sinPhi;

        const complex_T helpVar3 = (
            complex_T(0,float_T(-0.5)) * cscPhi * (
                complex_T(0,-8) * om0 * y * ( cspeed * t - z ) * sinPhi2_2 * sinPhi_4
                    * ( complex_T(0,1) * rho0 - z * sinPhi )
                - om0 * sinPhi_4 * sinPhi * (
                    - float_t(2.0) * z2 * rho0 - cspeed * cspeed
                       * ( k * tauG2 * x2 + float_t(2.0) * t * ( t - complex_T(0,1) * om0 * tauG2 ) * rho0 )
                    + cspeed * ( 4 * t * z * rho0 - complex_T(0,2) * om0 * tauG2 * z * rho0 )
                    - complex_T(0,2) * ( cspeed * t - z )
                        * ( cspeed * ( t - complex_T(0,1) * om0 * tauG2 ) - z ) * z * sinPhi
                    )
                + y * sinPhi * ( complex_T(0,4) * om0 * y * ( cspeed * t - z ) * sinPhi2_2 * sinPhi_2
                    + om0 * ( cspeed * t - z ) * (
                          complex_T(0,1) * cspeed * t
                        + cspeed * om0 * tauG2
                        - complex_T(0,1) * z
                        ) * sinPhi_3
                    - complex_T(0,4) * sinPhi2_4 * (
                          cspeed * k * x2
                        - om0 * (y2 - float_T(4.0) * ( cspeed * t - z ) * z ) * sinPhi
                        )
                    )*sin2Phi
                - complex_T(0,4) * sinPhi2_4 * (
                      complex_T(0,-4) * om0 * y * ( cspeed * t - z) * rho0 * cosPhi * sinPhi_2
                    + complex_T(0,2) * (
                          om0 * ( y2 + float_T(2.0) * z2 ) * rho0
                        - cspeed * z * ( complex_T(0,1) * k * x2 + float_T(2.0) * om0 * t * rho0 )
                        ) * sinPhi_3
                    - float_T(2.0) * om0 * z * ( y2 - float_T(2.0) * ( cspeed * t - z ) * z ) * sinPhi_4
                    + om0 * y2 * ( cspeed*t - z ) * sin2Phi * sin2Phi
                    )
                )
            ) / ( cspeed * helpVar2 * helpVar1 );

        const complex_T helpVar4 = cspeed * om0 * tauG * tauG
            - complex_T(0,8) * y * pmMath::tan( float_T(PI / 2.0) - phiT ) * cscPhi * cscPhi * sinPhi2_4
            - complex_T(0,2) * z * tanPhi2_2;

        const complex_T result = (
            pmMath::exp(helpVar3) * tauG * secPhi2 * secPhi2 * (
                complex_T(0,2) * cspeed * t + cspeed * om0 * tauG2 - complex_T(0,4) * z + cspeed * ( complex_T(0,2) * t + om0 * tauG2 ) * cosPhi + complex_T(0,2) * y * tanPhi2
                ) * pmMath::sqrt( cspeed * om0 * rho0 / helpVar2 )
            ) / ( float_T(2.0) * cspeed * pmMath::pow(helpVar4, float_T(1.5) ) );

        return result.get_real() / UNIT_SPEED;
    }
    
    /** Calculate the Bz(r,t) field
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcTWTSBz_Ex( const float3_64& pos, const float_64 time ) const
    {
        typedef PMacc::math::Complex<float_T> complex_T;
        /** Unit of Speed */
        const double UNIT_SPEED = SI::SPEED_OF_LIGHT_SI;
        /** Unit of time */
        const double UNIT_TIME = SI::DELTA_T_SI;
        /** Unit of length */
        const double UNIT_LENGTH = UNIT_TIME*UNIT_SPEED;
        
        /* If phi < 0 the formulas below are not directly applicable.
         * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
         * z-axis of the coordinate system in this function.
         * Note a: Another consequence is that the resulting field vectors also need to be rotated.
         * Note b: Instead of modifying field_x --> -field_x and field_y --> -field_y according
         *         to "Note a", we choose to introduce an additional 180deg shift due symmetry
         *         and modify only field_z --> -field_z .
         */
        const float_T phiReal = float_T( pmMath::abs(phi) );
        const float_T alphaTilt = pmMath::atan2(float_T(1.0)-float_T(beta_0)*pmMath::cos(phiReal),
                                                float_T(beta_0)*pmMath::sin(phiReal));
        const float_T phiT = float_T(2.0)*alphaTilt;
        /* Definition of the laser pulse front tilt angle for the laser field below.
         * For beta0=1.0, this is equivalent to our standard definition. Question: Why is the
         * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
         * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
         * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
         * the dispersion will (although physically correct) be slightly off the ideal TWTS
         * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
         * scenarios close to beta0 = 1. */
        
        /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
         * documentation purposes. */
        /* const float_T eta = float_T(float_T(PI / 2)) - (phiReal - alphaTilt); */
        
        const float_T cspeed = float_T( SI::SPEED_OF_LIGHT_SI / UNIT_SPEED );
        const float_T lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
        const float_T om0 = float_T(2.0*PI*cspeed / lambda0);
        /* factor 2  in tauG arises from definition convention in laser formula */
        const float_T tauG = float_T(pulselength_SI*2.0 / UNIT_TIME);
        /* w0 is wx here --> w0 could be replaced by wx */
        const float_T w0 = float_T(w_x_SI / UNIT_LENGTH);
        const float_T rho0 = float_T(PI*w0*w0 / lambda0);
        /* wy is width of TWTS pulse */
        const float_T wy = float_T(w_y_SI / UNIT_LENGTH);
        const float_T k = float_T(2.0*PI / lambda0);

        /* In order to calculate in single-precision and in order to account for errors in
         * the approximations far from the coordinate origin, we use the wavelength-periodicity and
         * the known propagation direction for realizing the laser pulse using relative coordinates
         * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
         * in double precision.
         */
        const float_64 tanAlpha = ( float_64(1.0) - beta_0 * pmMath::cos(phi) )
                                    / ( beta_0 * pmMath::sin(phi) );                            
        const float_64 tanFocalLine = pmMath::tan( PI / float_64(2.0) - phi );
        const float_64 deltaT = wavelength_SI / SI::SPEED_OF_LIGHT_SI
                                 * ( float_64(1.0) + tanAlpha / tanFocalLine);
        const float_64 deltaY = wavelength_SI / tanFocalLine;
        const float_64 deltaZ = -wavelength_SI;
        const float_64 numberOfPeriods = pmMath::floor( time / deltaT );
        const float_T timeMod = float_T( time - numberOfPeriods * deltaT );
        const float_T yMod = float_T( pos.y() + numberOfPeriods * deltaY );
        const float_T zMod = float_T( pos.z() + numberOfPeriods * deltaZ );

        const float_T x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
        const float_T y = float_T(phiPositive * yMod / UNIT_LENGTH);
        const float_T z = float_T(zMod / UNIT_LENGTH);
        const float_T t = float_T(timeMod / UNIT_TIME);

        /* Calculate distance from laser pulse front and set field to zero if the result
         * is 5 times the pulse length away. This sets a clear border of the laser pulse,
         * so that collisions with the front face of the simulation can be avoided
         * in a reliable, reproducible fashion. This is important for example for the Lehe solver.
         */
        const float_T offset = pmMath::abs( z - phiPositive * (float_T) tanAlpha * y ) * pmMath::cos( phiT / float_T(2.0) );
        const float_T offsetTolerance = float_T(4.0) * tauG * cspeed + lambda0;
        if ( offset > offsetTolerance ) return float_T(0.0);

        /* Calculating shortcuts for speeding up field calculation */
        const float_T sinPhi = pmMath::sin(phiT);
        const float_T cosPhi = pmMath::cos(phiT);
        const float_T cscPhi = float_T(1.0)/pmMath::sin(phiT);
        const float_T secPhi2 = float_T(1.0)/pmMath::cos(phiT / float_T(2.0));
        const float_T sinPhi2 = pmMath::sin(phiT / float_T(2.0));
        const float_T tanPhi2 = pmMath::tan(phiT / float_T(2.0));

        const float_T cscPhi_3 = cscPhi * cscPhi * cscPhi;

        const float_T sinPhi2_2 = sinPhi2 * sinPhi2;
        const float_T sinPhi2_4 = sinPhi2_2 * sinPhi2_2;
        const float_T tanPhi2_2 = tanPhi2 * tanPhi2;
        const float_T secPhi2_2 = secPhi2 * secPhi2;

        const float_T tanPI2_phi = pmMath::tan( float_T(PI / 2.0) - phiT );

        const float_T tauG2 = tauG * tauG;
        const float_T om02 = om0 * om0;
        const float_T x2 = x * x;
        const float_T y2 = y * y;
        const float_T z2 = z * z;

        /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
         * thus help with formal code verification through manual code inspection.
         */
        const complex_T helpVar1 = cspeed * om0 * tauG2 - complex_T(0,1) * y * cosPhi * secPhi2_2 * tanPhi2
            - complex_T(0,2) * z * tanPhi2_2;
        const complex_T helpVar2 = complex_T(0,1) * cspeed * rho0 - cspeed * y * cosPhi 
            - cspeed * z * sinPhi;
        const complex_T helpVar3 = rho0 + complex_T(0,1) * y * cosPhi + complex_T(0,1) * z * sinPhi;
        const complex_T helpVar4 = complex_T(0,1) * rho0 - y * cosPhi - z * sinPhi;
        const complex_T helpVar5 = -z - y * tanPI2_phi + complex_T(0,1) * rho0 * cscPhi;
        const complex_T helpVar6 = -cspeed * z - cspeed * y * tanPI2_phi
            + complex_T(0,1) * cspeed * rho0 * cscPhi;
        const complex_T helpVar7 = complex_T(0,1) * cspeed * rho0 - cspeed * y * cosPhi
            - cspeed * z * sinPhi;

        const complex_T helpVar8 = (
              om0 * y * rho0 * secPhi2_2 * secPhi2_2 / helpVar6
            + (om0 * y * tanPI2_phi * (
                    cspeed * om0 * tauG2 + float_T(8.0) * ( complex_T(0,2) * y + rho0 )
                        * cscPhi_3 * sinPhi2_4
                    )
                ) / ( cspeed * helpVar5 )
            + om02 * tauG2 * z * sinPhi / helpVar4
            - float_T(2.0) * k * x2 / helpVar3
            - om02 * tauG2 * rho0 / helpVar3
            + complex_T(0,1) * om0 * y2 * cosPhi * cosPhi * secPhi2_2 * tanPhi2 / helpVar2
            + complex_T(0,4) * om0 * y * z * tanPhi2_2 / helpVar2
            - float_T(2.0) * om0 * z * rho0 * tanPhi2_2 / helpVar2
            - complex_T(0,2) * om0 * z2 * sinPhi * tanPhi2_2 / helpVar2
            - ( om0 * pmMath::pow(
                      float_T(2.0) * cspeed * t - complex_T(0,1) * cspeed * om0 * tauG2
                    - float_T(2.0) * z + float_T(8.0) * y * cscPhi_3 * sinPhi2_4
                    - float_T(2.0) * z * tanPhi2_2
                    , float_T(2.0) )
                ) / ( cspeed * helpVar1 )
            ) / float_T(4.0);

        const complex_T helpVar9 = cspeed * om0 * tauG2
            - complex_T(0,1) * y * cosPhi * secPhi2_2 * tanPhi2 - complex_T(0,2) * z * tanPhi2_2;

        const complex_T result = float_T(phiPositive) * (
            complex_T(0,2) * pmMath::exp(helpVar8) * tauG * tanPhi2
                * (cspeed*t - z + y * tanPhi2 ) * pmMath::sqrt( om0 * rho0 / helpVar7 )
            )
            / pmMath::pow(helpVar9, float_T(1.5) );

        return result.get_real() / UNIT_SPEED;
    }
    
    /** Calculate the Bx(r,t) field
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcTWTSBx( const float3_64& pos, const float_64 time ) const
    {
        /* The Bx-field for the Ey-field is the same as
         * for the By-field for the Ex-field except for the sign. */
        return -calcTWTSBy( pos, time );
    }
    
    /** Calculate the Bz(r,t) field
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    /* CAUTION: In this branch, this fuction is not consistent with the other components.  */
    /* Thus TWTS polarization in for Ey is not available right now. */
    HDINLINE BField::float_T
    BField::calcTWTSBz_Ey( const float3_64& pos, const float_64 time ) const
    {
        typedef PMacc::math::Complex<float_T> complex_T;
        typedef PMacc::math::Complex<float_64> complex_64;
        /** Unit of Speed */
        const double UNIT_SPEED = SI::SPEED_OF_LIGHT_SI;
        /** Unit of time */
        const double UNIT_TIME = SI::DELTA_T_SI;
        /** Unit of length */
        const double UNIT_LENGTH = UNIT_TIME*UNIT_SPEED;
        
        /* propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
        const float_T beta0 = float_T(beta_0);
        /* If phi < 0 the formulas below are not directly applicable.
         * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
         * z-axis of the coordinate system in this function.
         * Note a: Another consequence is that the resulting field vectors also need to be rotated.
         * Note b: Instead of modifying field_x --> -field_x and field_y --> -field_y according
         *         to "Note a", we choose to introduce an additional 180deg shift due symmetry
         *         and modify only field_z --> -field_z .
         */
        const float_T phiReal = float_T( pmMath::abs(phi) );
        const float_T alphaTilt = pmMath::atan2(float_T(1.0)-beta0*pmMath::cos(phiReal),
                                                beta0*pmMath::sin(phiReal));
        const float_T phiT = float_T(2.0)*alphaTilt;
        /* Definition of the laser pulse front tilt angle for the laser field below.
         * For beta0=1.0, this is equivalent to our standard definition. Question: Why is the
         * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
         * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
         * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
         * the dispersion will (although physically correct) be slightly off the ideal TWTS
         * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
         * scenarios close to beta0 = 1. */
        
        /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
         * documentation purposes. */
        /* const float_T eta = float_T(float_T(PI / 2)) - (phiReal - alphaTilt); */
        
        const float_T cspeed = float_T( SI::SPEED_OF_LIGHT_SI / UNIT_SPEED );
        const float_T lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
        const float_T om0 = float_T(2.0*PI*cspeed / lambda0);
        /* factor 2  in tauG arises from definition convention in laser formula */
        const float_T tauG = float_T(pulselength_SI*2.0 / UNIT_TIME);
        /* w0 is wx here --> w0 could be replaced by wx */
        const float_T w0 = float_T(w_x_SI / UNIT_LENGTH);
        const float_T rho0 = float_T(PI*w0*w0 / lambda0);
        /* wy is width of TWTS pulse */
        const float_T wy = float_T(w_y_SI / UNIT_LENGTH);
        const float_T k = float_T(2.0*PI / lambda0);
        const float_T x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
        const float_T y = float_T(phiPositive * pos.y() / UNIT_LENGTH);
        const float_T z = float_T(pos.z() / UNIT_LENGTH);
        const float_T t = float_T(time / UNIT_TIME);

        /* Calculate distance from laser pulse front and set field to zero if the result
         * is 5 times the pulse length away. This sets a clear border of the laser pulse,
         * so that collisions with the front face of the simulation can be avoided
         * in a reliable, reproducible fashion. This is important for example for the Lehe solver.
         */
        //const float_T offset = pmMath::abs( z - phiPositive * (float_T) tanAlpha * y ) * pmMath::cos( phiT / float_T(2.0) );
        //const float_T offsetTolerance = float_T(4.0) * tauG * cspeed + lambda0;
        //if ( offset > offsetTolerance ) return float_T(0.0);
                        
        /* Shortcuts for speeding up the field calculation. */
        const float_T sinPhi = pmMath::sin(phiT);
        const float_T cosPhi = pmMath::cos(phiT);
        const float_T sinPhi2 = pmMath::sin(phiT / float_T(2.0));
        const float_T cosPhi2 = pmMath::cos(phiT / float_T(2.0));
        const float_T tanPhi2 = pmMath::tan(phiT / float_T(2.0));
        
        const float_T sinPhi2_2 = sinPhi2 * sinPhi2;
        const float_T sinPhi2_4 = sinPhi2_2 * sinPhi2_2;
        const float_T cosPhi2_2 = cosPhi2 * cosPhi2;
        const float_T tanPhi2_2 = tanPhi2 * tanPhi2;

        const float_T cspeed2 = cspeed * cspeed;
        const float_T tauG2 = tauG * tauG;
        const float_T wy2 = wy * wy;
        const float_T om02 = om0 * om0;
 
        const float_T x2 = x * x;
        const float_T y2 = y * y;
        const float_T z2 = z * z;
        const float_T t2 = t * t;

        /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
         * thus help with formal code verification through manual code inspection.
         */
        const complex_T helpVar1 =
            complex_T(0,-1)*cspeed*om0*tauG2
            - y*cosPhi / cosPhi2_2 * tanPhi2
            - float_T(2.0)*z*tanPhi2_2;
        const complex_T helpVar2 = complex_T(0,1)*rho0 - y*cosPhi - z*sinPhi;

        const complex_T helpVar3 = (
            - cspeed2*k*om0*tauG2*wy2*x2
            - float_T(2.0)*cspeed2*om0*t2*wy2*rho0
            + complex_T(0,2)*cspeed2*om02*t*tauG2*wy2*rho0
            - float_T(2.0)*cspeed2*om0*tauG2*y2*rho0
            + float_T(4.0)*cspeed*om0*t*wy2*z*rho0
            - complex_T(0,2)*cspeed*om02*tauG2*wy2*z*rho0
            - float_T(2.0)*om0*wy2*z2*rho0
            - complex_T(0,8)*om0*wy2*y*(cspeed*t - z)*z*sinPhi2_2
            + complex_T(0,8) / sinPhi *(
                float_T(2.0)*z2*(cspeed*om0*t*wy2 + complex_T(0,1)*cspeed*y2 - om0*wy2*z)
                + y*(
                    cspeed*k*wy2*x2
                    - complex_T(0,2)*cspeed*om0*t*wy2*rho0
                    + float_T(2.0)*cspeed*y2*rho0
                    + complex_T(0,2)*om0*wy2*z*rho0
                )*tan(float_T(PI) / float_T(2.0)-phiT) / sinPhi
            )*sinPhi2_4
            - complex_T(0,2)*cspeed2*om0*t2*wy2*z*sinPhi
            - float_T(2.0)*cspeed2*om02*t*tauG2*wy2*z*sinPhi
            - complex_T(0,2)*cspeed2*om0*tauG2*y2*z*sinPhi
            + complex_T(0,4)*cspeed*om0*t*wy2*z2*sinPhi
            + float_T(2.0)*cspeed*om02*tauG2*wy2*z2*sinPhi
            - complex_T(0,2)*om0*wy2*z2*z*sinPhi
            - float_T(4.0)*cspeed*om0*t*wy2*y*rho0*tanPhi2
            + float_T(4.0)*om0*wy2*y*z*rho0*tanPhi2
            + complex_T(0,2)*y2*(
                cspeed*om0*t*wy2
                + complex_T(0,1)*cspeed*y2
                - om0*wy2*z
            )*cosPhi*cosPhi / cosPhi2_2 * tanPhi2
            + complex_T(0,2)*cspeed*k*wy2*x2*z*tanPhi2_2
            - float_T(2.0)*om0*wy2*y2*rho0*tanPhi2_2
            + float_T(4.0)*cspeed*om0*t*wy2*z*rho0*tanPhi2_2
            + complex_T(0,4)*cspeed*y2*z*rho0*tanPhi2_2
            - float_T(4.0)*om0*wy2*z2*rho0*tanPhi2_2
            - complex_T(0,2)*om0*wy2*y2*z*sinPhi*tanPhi2_2
            - float_T(2.0)*y*cosPhi*(
                om0*(
                    cspeed2*(complex_T(0,1)*t2*wy2
                    + om0*t*tauG2*wy2
                    + complex_T(0,1)*tauG2*y2)
                    - cspeed*(complex_T(0,2)*t + om0*tauG2)*wy2*z
                    + complex_T(0,1)*wy2*z2
                )
                + complex_T(0,2)*om0*wy2*y*(cspeed*t - z)*tanPhi2
                + complex_T(0,1)*(
                    complex_T(0,-4)*cspeed*y2*z
                    + om0*wy2*(y2 - float_T(4.0)*(cspeed*t - z)*z)
                )*tanPhi2_2
            )
        /* The "round-trip" conversion in the line below fixes a gross accuracy bug
         * in floating-point arithmetics, when float_T is set to float_X.
         */
        ) * complex_T( float_64(1.0) / complex_64(float_T(2.0)*cspeed*wy2*helpVar2*helpVar1) );

        const complex_T helpVar4 = (
            cspeed*om0*(
                cspeed*om0*tauG2
                - complex_T(0,8)*y*pmMath::tan( float_T(PI) / float_T(2.0) - phiT )
                    / sinPhi / sinPhi * sinPhi2_4
                - complex_T(0,2)*z*tanPhi2_2
            )
        ) / rho0;

        const complex_T result = float_T(phiPositive)*float_T(-1.0)*(
            cspeed*pmMath::exp(helpVar3)*k*tauG*x*pmMath::pow( helpVar2, float_T(-1.5) )
            / pmMath::sqrt(helpVar4)
        );
        
        return result.get_real() / UNIT_SPEED;
    }

} /* namespace twts */
} /* namespace templates */
} /* namespace picongpu */

