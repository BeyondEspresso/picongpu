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

namespace picongpu
{
/** Load external TWTS field
 *
 */
namespace templates
{
namespace pmMath = PMacc::algorithms::math;

    HINLINE
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
        tdelay_user_SI(tdelay_user_SI), dt(SI::DELTA_T_SI),
        unit_length(UNIT_LENGTH), auto_tdelay(auto_tdelay)
    {
        /* Note: These objects cannot be instantiated on CUDA GPU device. Since this is done
                 on host (see fieldBackground.param), this is no problem. */
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        halfSimSize = subGrid.getGlobalDomain().size / 2;
        tdelay = detail::Get_tdelay_SI<simDim>()(auto_tdelay, tdelay_user_SI, 
                                                 halfSimSize, pulselength_SI,
                                                 focus_y_SI, phi, beta_0) ;
    }
    
    HDINLINE PMacc::math::Vector<float3_64,detail::numComponents>
    TWTSFieldE::getEfieldPositions_SI(const DataSpace<simDim>& cellIdx) const
    {
        /* Note: Neither direct precisionCast on picongpu::cellSize
           or casting on floatD_ does work. */
        const floatD_64 cellDim(picongpu::cellSize);
        const floatD_64 cellDimensions = cellDim * unit_length;
        
        /* TWTS laser coordinate origin is centered transversally and defined longitudinally by
           the laser center in y (usually maximum of intensity). */
        floatD_X laserOrigin = precisionCast<float_X>(halfSimSize);
        laserOrigin.y() = float_X( focus_y_SI/cellDimensions.y() );
        
        /* For the Yee-Cell shifted fields, obtain the fractional cell index components and add
         * that to the total cell indices. The physical field coordinate origin is transversally
         * centered with respect to the global simulation volume. */
        PMacc::math::Vector<floatD_X, detail::numComponents> eFieldPositions = 
                        fieldSolver::NumericalCellType::getEFieldPosition();
        
        PMacc::math::Vector<floatD_64,detail::numComponents> eFieldPositions_SI;
        
        for( uint32_t i = 0; i < detail::numComponents; ++i ) /* cellIdx Ex, Ey and Ez */
        {
            eFieldPositions[i]   += ( precisionCast<float_X>(cellIdx) - laserOrigin );
            eFieldPositions_SI[i] = precisionCast<float_64>(eFieldPositions[i]) * cellDimensions;
            eFieldPositions_SI[i] = detail::rotateField(eFieldPositions_SI[i],phi);
        }
        
        return eFieldPositions_SI;
    }
    
    template<>
    HDINLINE float3_X
    TWTSFieldE::getTWTSEfield_Normalized<DIM3>(
                const PMacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
                const float_64 time) const
    {
        float3_64 pos(0.0);
        for (uint32_t i = 0; i<simDim;++i) pos[i] = eFieldPositions_SI[0][i];
        return float3_X( float_X( calcTWTSEx(pos,time) ),
                         float_X(0.), float_X(0.) );
    }
    
    template<>
    HDINLINE float3_X
    TWTSFieldE::getTWTSEfield_Normalized<DIM2>(
        const PMacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
        const float_64 time) const
    {
        /* Ex->Ez, so also the grid cell offset for Ez has to be used. */
        float3_64 pos(0.0);
        for (uint32_t i = 1; i<simDim;++i) pos[i] = eFieldPositions_SI[2][i];
        return float3_X( float_X(0.), float_X(0.),
                         float_X( calcTWTSEx(pos,time) ) );
    }
    
    HDINLINE float3_X
    TWTSFieldE::operator()( const DataSpace<simDim>& cellIdx,
                            const uint32_t currentStep ) const
    {
        const float_64 time_SI = float_64(currentStep) * dt - tdelay;
        const PMacc::math::Vector<float3_64,detail::numComponents> eFieldPositions_SI =
                                                        getEfieldPositions_SI(cellIdx);
        /* Single TWTS-Pulse */
        return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI);
    }

    /** Calculate the Ex(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE TWTSFieldE::float_T
    TWTSFieldE::calcTWTSEx( const float3_64& pos, const float_64 time) const
    {
        typedef PMacc::math::Complex<float_T> complex_T;
        /** Unit of Speed */
        const double UNIT_SPEED = SI::SPEED_OF_LIGHT_SI;
        /** Unit of time */
        const double UNIT_TIME = SI::DELTA_T_SI;
        /** Unit of length */
        const double UNIT_LENGTH = UNIT_TIME*UNIT_SPEED;
    
        /* propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
        const float_T beta0 = float_T(beta_0);
        const float_T phiReal = float_T(phi);
        const float_T alphaTilt = atan2(float_T(1.0)-beta0*cos(phiReal),beta0*sin(phiReal));
        const float_T phiT = float_T(2.0)*alphaTilt;
        /* Definition of the laser pulse front tilt angle for the laser field below.
         * For beta0=1.0, this is equivalent to our standard definition. Question: Why is the
         * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
         * Because the standard TWTS pulse is defined for beta0=1.0 and in the coordinate-system
         * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
         * the dispersion will (although physically correct) be slightly off the ideal TWTS
         * pulse for beta0!=1.0. This only shows that this TWTS pulse is primarily designed for
         * scenarios close to beta0=1. */
        
        /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
         * documentation purposes. */
        /* const float_T eta = PI/2 - (phiReal - alphaTilt); */
        
        const float_T cspeed = float_T(1.0);
        const float_T lambda0 = float_T(wavelength_SI/UNIT_LENGTH);
        const float_T om0 = float_T(2.0*PI*cspeed/lambda0*UNIT_TIME);
        /* factor 2  in tauG arises from definition convention in laser formula */
        const float_T tauG = float_T(pulselength_SI*2.0/UNIT_TIME);
        /* w0 is wx here --> w0 could be replaced by wx */
        const float_T w0 = float_T(w_x_SI/UNIT_LENGTH);
        const float_T rho0 = float_T(PI*w0*w0/lambda0/UNIT_LENGTH);
        /* wy is width of TWTS pulse */
        const float_T wy = float_T(w_y_SI/UNIT_LENGTH);
        const float_T k = float_T(2.0*PI/lambda0*UNIT_LENGTH);
        const float_T x = float_T(pos.x()/UNIT_LENGTH);
        const float_T y = float_T(pos.y()/UNIT_LENGTH);
        const float_T z = float_T(pos.z()/UNIT_LENGTH);
        const float_T t = float_T(time/UNIT_TIME);
        
        /* Calculating shortcuts for speeding up field calculation */
        const float_T sinPhi = sin(phiT);
        const float_T cosPhi = cos(phiT);
        const float_T sinPhi2 = sin(phiT/float_T(2.0));
        const float_T cosPhi2 = cos(phiT/float_T(2.0));
        const float_T tanPhi2 = tan(phiT/float_T(2.0));
        
        /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
         * thus help with formal code verification through manual code inspection. */
        const complex_T helpVar1 = complex_T(0,1)*rho0 - y*cosPhi - z*sinPhi;
        const complex_T helpVar2 = complex_T(0,-1)*cspeed*om0*tauG*tauG
                                    - y*cosPhi/cosPhi2/cosPhi2*tanPhi2
                                    - float_T(2.0)*z*tanPhi2*tanPhi2;
        const complex_T helpVar3 = complex_T(0,1)*rho0 - y*cosPhi - z*sinPhi;

        const complex_T helpVar4 = (
            -(cspeed*cspeed*k*om0*tauG*tauG*wy*wy*x*x)
            - float_T(2.0)*cspeed*cspeed*om0*t*t*wy*wy*rho0 
            + complex_T(0,2)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*rho0
            - float_T(2.0)*cspeed*cspeed*om0*tauG*tauG*y*y*rho0
            + float_T(4.0)*cspeed*om0*t*wy*wy*z*rho0
            - complex_T(0,2)*cspeed*om0*om0*tauG*tauG*wy*wy*z*rho0
            - float_T(2.0)*om0*wy*wy*z*z*rho0
            - complex_T(0,8)*om0*wy*wy*y*(cspeed*t - z)*z*sinPhi2*sinPhi2
            + complex_T(0,8)/sinPhi*(
                    +float_T(2.0)*z*z*(cspeed*om0*t*wy*wy+complex_T(0,1)*cspeed*y*y-om0*wy*wy*z)
                    + y*(
                        + cspeed*k*wy*wy*x*x
                        - complex_T(0,2)*cspeed*om0*t*wy*wy*rho0
                        + float_T(2.0)*cspeed*y*y*rho0
                        + complex_T(0,2)*om0*wy*wy*z*rho0
                    )*tan(float_T(PI/2.0)-phiT)/sinPhi
                )*sinPhi2*sinPhi2*sinPhi2*sinPhi2
            - complex_T(0,2)*cspeed*cspeed*om0*t*t*wy*wy*z*sinPhi
            - float_T(2.0)*cspeed*cspeed*om0*om0*t*tauG*tauG*wy*wy*z*sinPhi
            - complex_T(0,2)*cspeed*cspeed*om0*tauG*tauG*y*y*z*sinPhi
            + complex_T(0,4)*cspeed*om0*t*wy*wy*z*z*sinPhi
            + float_T(2.0)*cspeed*om0*om0*tauG*tauG*wy*wy*z*z*sinPhi
            - complex_T(0,2)*om0*wy*wy*z*z*z*sinPhi
            - float_T(4.0)*cspeed*om0*t*wy*wy*y*rho0*tanPhi2
            + float_T(4.0)*om0*wy*wy*y*z*rho0*tanPhi2
            + complex_T(0,2)*y*y*(
                 + cspeed*om0*t*wy*wy + complex_T(0,1)*cspeed*y*y - om0*wy*wy*z
                 )*cosPhi*cosPhi/cosPhi2/cosPhi2*tanPhi2
            + complex_T(0,2)*cspeed*k*wy*wy*x*x*z*tanPhi2*tanPhi2
            - float_T(2.0)*om0*wy*wy*y*y*rho0*tanPhi2*tanPhi2
            + float_T(4.0)*cspeed*om0*t*wy*wy*z*rho0*tanPhi2*tanPhi2
            + complex_T(0,4)*cspeed*y*y*z*rho0*tanPhi2*tanPhi2
            - float_T(4.0)*om0*wy*wy*z*z*rho0*tanPhi2*tanPhi2
            - complex_T(0,2)*om0*wy*wy*y*y*z*sinPhi*tanPhi2*tanPhi2
            - float_T(2.0)*y*cosPhi*(
                + om0*(
                    + cspeed*cspeed*(
                          complex_T(0,1)*t*t*wy*wy
                        + om0*t*tauG*tauG*wy*wy
                        + complex_T(0,1)*tauG*tauG*y*y
                        )
                    - cspeed*(complex_T(0,2)*t
                    + om0*tauG*tauG)*wy*wy*z
                    + complex_T(0,1)*wy*wy*z*z
                    )
                + complex_T(0,2)*om0*wy*wy*y*(cspeed*t - z)*tanPhi2
                + complex_T(0,1)*tanPhi2*tanPhi2*(
                      complex_T(0,-4)*cspeed*y*y*z
                    + om0*wy*wy*(y*y - float_T(4.0)*(cspeed*t - z)*z)
                )
            )
        )/(float_T(2.0)*cspeed*wy*wy*helpVar1*helpVar2);

        const complex_T helpVar5 = cspeed*om0*tauG*tauG 
            - complex_T(0,8)*y*tan( float_T(PI/2)-phiT )
                                /sinPhi/sinPhi*sinPhi2*sinPhi2*sinPhi2*sinPhi2
            - complex_T(0,2)*z*tanPhi2*tanPhi2;
        const complex_T result = (pmMath::exp(helpVar4)*tauG
            *pmMath::sqrt((cspeed*om0*rho0)/helpVar3))/pmMath::sqrt(helpVar5);
        return result.get_real();
    }

} /* namespace templates */
} /* namespace picongpu */
