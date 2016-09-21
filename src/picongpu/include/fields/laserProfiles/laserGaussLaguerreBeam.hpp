/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Anton Helm, Rene Widera,
 *                     Richard Pausch, Alexander Debus
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

#include "pmacc_types.hpp"
#include "simulation_defines.hpp"

namespace picongpu
{
    namespace laserGaussLaguerreBeam
    {

        /**
         *
         * @param currentStep
         * @param subGrid
         * @param phase
         * @return
         */
        HINLINE float3_X laserLongitudinal( uint32_t currentStep, float_X& phase )
        {
            const float_64 runTime = DELTA_T*currentStep;
            const float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;

            float3_X elong(float3_X::create(0.0));

            // a symmetric pulse will be initialized at position z=0 for
            // a time of PULSE_INIT * PULSE_LENGTH = INIT_TIME.
            // we shift the complete pulse for the half of this time to start with
            // the front of the laser pulse.
            const float_64 mue = 0.5 * INIT_TIME;

            //rayleigh length (in y-direction)
            const float_64 y_R = PI * W0 * W0 / WAVE_LENGTH;
            //gaussian beam waist in the nearfield: w_y(y=0) == W0
            const float_64 w_y = W0 * sqrt( 1.0 + ( FOCUS_POS / y_R )*( FOCUS_POS / y_R ) );

            float_64 envelope = float_64( AMPLITUDE );
            if( simDim == DIM2 )
                envelope *= math::sqrt( float_64( W0 ) / w_y );
            else if( simDim == DIM3 )
                envelope *= float_64( W0 ) / w_y;
            /* no 2D representation/implementation, because Laguerre-modes are 3D only */
            /* no 1D representation/implementation */

            if( Polarisation == LINEAR_X )
            {
                elong.x() = float_X( envelope );
            }
            else if( Polarisation == LINEAR_Z )
            {
                elong.z() = float_X( envelope );
            }
            else if( Polarisation == CIRCULAR )
            {
                elong.x() = float_X( envelope / sqrt(2.0) );
                elong.z() = float_X( envelope / sqrt(2.0) );
            }

            phase = 2.0f * float_X(PI ) * float_X(f ) * ( runTime - float_X(mue ) - FOCUS_POS / SPEED_OF_LIGHT) + LASER_PHASE;

            return elong;
        }

        /**
         *  Simple iteration algorithm to implement Laguerre polynomials.
         *  @param n
         *  @param x
         *  @return
         */ 
        HDINLINE float_X simpleLaguerre( const uint32_t n, const float_X x )
        {
            uint32_t currentN = 1;
            float_X laguerreNMinus1 = 1.0f;
            float_X laguerreN = 1.0f - x;
            float_X temp;
            while (currentN < n)
            {
                temp = laguerreN;
                laguerreN = ( (2.0f * float_X(currentN) + 1.0f - x) * laguerreN - float_X(currentN) * laguerreNMinus1 ) / float_X(currentN + 1);
                laguerreNMinus1 = temp;
                currentN++;
            }
            return laguerreN;
        }

        /**
         *
         * @param elong
         * @param phase
         * @param posX
         * @param posZ
         * @return
         */
        HDINLINE float3_X laserTransversal( float3_X elong, float_X phase, const float_X posX, const float_X posZ )
        {
            const float_X r2 = posX * posX + posZ * posZ;

            //rayleigh length (in y-direction)
            const float_X y_R = float_X( PI ) * W0 * W0 / WAVE_LENGTH;
            
            // the radius of curvature of the beam's  wavefronts
            const float_X R_y = -FOCUS_POS * ( float_X(1.0) + ( y_R / FOCUS_POS )*( y_R / FOCUS_POS ) );

            uint32_t m = 0;
            float_X etrans = 0.0f;
            float_X etrans_norm = 0.0f;

            //template <class T>
            //    calculated-result-type laguerre(unsigned n, T x);

            
#if !defined(__CUDA_ARCH__) // Host code path
            //beam waist in the near field: w_y(y=0) == W0
            const float_X w_y = W0 * sqrt( float_X(1.0) + ( FOCUS_POS / y_R )*( FOCUS_POS / y_R ) );
            //! the Gouy phase shift
            const float_X xi_y = atan( -FOCUS_POS / y_R );

            if( Polarisation == LINEAR_X || Polarisation == LINEAR_Z )
            {
                for ( m=0; m<MODENUMBER ; m++ ) etrans_norm += LAGUERREMODES[m];
                for ( m=0; m<MODENUMBER ; m++ )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, 2.0f * r2 / w_y / w_y ) * math::exp( -r2 / w_y / w_y ) * cos(
                        2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS 
                      - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + ( 2*m + 1 ) * xi_y + phase 
                        )
                        * math::exp(
                            -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH )
                        );
                }
                elong *= etrans / etrans_norm;
            }
            else if( Polarisation == CIRCULAR )
            {
                for ( m=0; m<MODENUMBER ; m++ ) etrans_norm += LAGUERREMODES[m];
                for ( m=0; m<MODENUMBER ; m++ )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, 2.0f * r2 / w_y / w_y ) * math::exp( -r2 / w_y / w_y ) * cos(
                        2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS 
                      - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + ( 2*m + 1 ) * xi_y + phase 
                        )
                        * math::exp(
                            -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH )
                        );
                }
                elong.x() *= etrans / etrans_norm;
                phase += float_X( PI / 2.0 );
                for ( m=0, etrans=0.0f ; m<MODENUMBER ; m++ )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, 2.0f * r2 / w_y / w_y ) * math::exp( -r2 / w_y / w_y ) * cos(
                        2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS 
                      - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + ( 2*m + 1 ) * xi_y + phase 
                        )
                        * math::exp(
                            -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH )
                        );
                }
                elong.z() *= etrans / etrans_norm;
                phase -= float_X( PI / 2.0 );
            }

            return elong;
#else
            //beam waist in the near field: w_y(y=0) == W0
            const float_X w_y = W0 * algorithms::math::sqrt( float_X(1.0) + ( FOCUS_POS / y_R )*( FOCUS_POS / y_R ) );
            //! the Gouy phase shift
            const float_X xi_y = atanf( -FOCUS_POS / y_R );

            if( Polarisation == LINEAR_X || Polarisation == LINEAR_Z )
            {
                for ( m=0; m<MODENUMBER ; m++ ) etrans_norm += LAGUERREMODES[m];
                for ( m=0; m<MODENUMBER ; m++ )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, 2.0f * r2 / w_y / w_y ) * math::exp( -r2 / w_y / w_y ) * math::cos(
                        2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS 
                      - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + ( 2*m + 1 ) * xi_y + phase 
                        )
                        * math::exp(
                            -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH )
                        );
                }
                elong *= etrans / etrans_norm;
            }
            else if( Polarisation == CIRCULAR )
            {
                for ( m=0; m<MODENUMBER ; m++ ) etrans_norm += LAGUERREMODES[m];
                for ( m=0; m<MODENUMBER ; m++ )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, 2.0f * r2 / w_y / w_y ) * math::exp( -r2 / w_y / w_y ) * math::cos(
                        2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS 
                      - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + ( 2*m + 1 ) * xi_y + phase 
                        )
                        * math::exp(
                            -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH )
                        );
                }
                elong.x() *= etrans / etrans_norm;
                phase += float_X( PI / 2.0 );
                for ( m=0, etrans=0.0f ; m<MODENUMBER ; m++ )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, 2.0f * r2 / w_y / w_y ) * math::exp( -r2 / w_y / w_y ) * math::cos(
                        2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS 
                      - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + ( 2*m + 1 ) * xi_y + phase 
                        )
                        * math::exp(
                            -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                            / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH )
                        );
                }
                elong.z() *= etrans / etrans_norm;
                phase -= float_X( PI / 2.0 );
            }

            return elong;
#endif
        }

    }
}

