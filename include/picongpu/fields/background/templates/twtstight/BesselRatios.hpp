/* Copyright 2014-2026 Alexander Debus and LLM agent
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

#include <pmacc/attribute/unroll.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/types.hpp>

#include <array>
#include <cstdint>

namespace picongpu::templates::twtstight::detail
{
    /** Truncated asymptotic correction S_n(z) to the modified Bessel function I_n(z).
     *
     * DLMF 10.40.1 gives I_n(z) ~ exp(z) / sqrt(2*pi*z) * S_n(z).
     * With A_m(n) denoting the coefficients called a_m(n) in DLMF 10.17.1,
     *
     *   A_0(n) = 1,
     *   A_m(n) = product_{j=1}^m (4*n*n - (2*j-1)^2) / (m! * 8^m),
     *   t_m(z) = (-1)^m * A_m(n) / z^m,
     *   S_n(z) is approximated here by sum_{m=0}^{12} t_m(z).
     *
     * Dividing consecutive terms yields the recurrence used below:
     *
     *   t_0 = 1,
     *   t_m = t_{m-1} * ((2*m-1)^2 - 4*n*n) / (8*m*z).
     *
     * For checking signs: S_0(z) = 1 + 1/(8*z) + 9/(128*z*z) + ...,
     * whereas S_1(z) = 1 - 3/(8*z) - 15/(128*z*z) - ... .
     * This is an asymptotic expansion, not a convergent Taylor series.
     * We retain t_0 through t_12 (thirteen terms). At |z|=50 the first omitted
     * term has magnitude about 1.50e-18 for n=0 and 1.62e-18 for n=1; this
     * motivates twelve corrections for double precision at the chosen threshold.
     * This term estimate is not a general remainder bound; see DLMF 10.40(iii).
     * The fixed bound permits unrolling on device and also supports float.
     *
     * References (NIST Digital Library of Mathematical Functions):
     * - I_n asymptotics: https://dlmf.nist.gov/10.40.E1
     * - Coefficients: https://dlmf.nist.gov/10.17.E1
     * - Remainder bounds: https://dlmf.nist.gov/10.40.iii
     *
     * @tparam T_order Bessel order; the ratio denominator uses n=0.
     * @param z Real or complex argument; the ratio denominator supplies real a >= 50.
     */
    template<uint32_t T_order, typename T_Value>
    HDINLINE T_Value besselISeries(T_Value const& z)
    {
        T_Value term(1);
        T_Value sum(1);
        T_Value const inverseZ = T_Value(1) / z;
        PMACC_UNROLL(12)
        for(int32_t m = 1; m <= 12; ++m)
        {
            int32_t const odd = 2 * m - 1;
            term *= (T_Value(odd * odd - int32_t(4u * T_order * T_order)) / T_Value(8 * m)) * inverseZ;
            sum += term;
        }
        return sum;
    }

    /** Evaluate J_0(q)/I_0(a) and J_1(q)/I_0(a) without overflowing either Bessel function.
     *
     * TWTSTight::defineCommonHelperVariables() supplies the dimensionless arguments
     *
     *   q = k*rho*sin(phi),    a = k*k*w0*w0*sin(phi)/2,
     *   rho = sqrt(Xm*Xm + x*x),    Xm = -z_pos - i*k*w0*w0/2,
     *
     * where k is the wave number, w0 the waist, and x and z_pos spatial coordinates.
     * Lengths and k must use consistent units. The formula uses sin(phi) > 0, hence
     * a is positive: the opposite TWTS laser beam with -phi is obtained by rotating the geometry.
     * Near the beam axis |Im(q)| is close to a. Thus J_n(q) and I_0(a) can each
     * overflow even though their ratio is representable.
     *
     * The scaled PMacc functions return J_ne(q) = exp(-|Im(q)|) * J_n(q).
     * For a >= 50, the real denominator uses
     *
     *   I_0(a) ~ exp(a) / sqrt(2*pi*a) * S_0(a),
     *   R_n ~ J_ne(q) * sqrt(2*pi*a) / S_0(a) * exp(|Im(q)|-a).
     *
     * Only the real denominator needs the asymptotic series. The scaled numerator
     * handles both signs of Im(q), the real axis, and q=0 without a local rotation
     * or a separate large-complex-argument approximation.
     * Evaluate the exponential as two factors exp((|Im(q)|-a)/2), applied after
     * the algebraic normalization. This avoids prematurely underflowing exp(-a)
     * when q is small but the normalized ratio is still representable.
     * For TWTS geometry |Im(q)| <= a in exact arithmetic, so these factors do not grow.
     *
     * For a < 50, retain the direct calculation: both unscaled functions are safe
     * for TWTS arguments. The real S_0(a) expansion and its error estimate are
     * documented in besselISeries(). Input rounding in |Im(q)|-a still limits
     * the attainable accuracy near the beam axis.
     *
     * @param q Complex TWTS Bessel argument.
     * @param a Positive real normalization argument.
     * @return {J_0(q)/I_0(a), J_1(q)/I_0(a)}.
     */
    template<typename T_Float>
    HDINLINE std::array<alpaka::Complex<T_Float>, 2u> besselJOverI0(alpaka::Complex<T_Float> const& q, T_Float const a)
    {
        namespace math = pmacc::math;
        constexpr T_Float threshold = T_Float(50);
        if(a < threshold)
        {
            T_Float const denominator = math::bessel::i0(a);
            return {math::bessel::j0(q) / denominator, math::bessel::j1(q) / denominator};
        }

        T_Float const denominatorSeries = besselISeries<0u>(a);
        T_Float const normalization = math::sqrt(T_Float(2) * math::Pi<T_Float>::value * a) / denominatorSeries;
        T_Float const halfExponential = math::exp((math::abs(q.imag()) - a) * T_Float(0.5));
        return {
            ((math::bessel::j0e(q) * normalization) * halfExponential) * halfExponential,
            ((math::bessel::j1e(q) * normalization) * halfExponential) * halfExponential};
    }
} // namespace picongpu::templates::twtstight::detail
