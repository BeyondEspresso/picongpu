/* Copyright 2003-2026 Alexander Debus, C. Bond and LLM agent
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file Bessel.tpp
 *
 *  Reference: Implementation is derived from a C++ implementation of
 *             complex Bessel functions from C. Bond (2003).
 *
 *  Original source downloaded from: http://www.crbond.com
 *  Download date: 2017/07/27
 *  Files: CBESSJY.CPP, BESSEL.H
 *  File-Header:
 *      cbessjy.cpp -- complex Bessel functions.
 *      Algorithms and coefficient values from "Computation of Special
 *      Functions", Zhang and Jin, John Wiley and Sons, 1996.
 *
 *     (C) 2003, C. Bond. All rights reserved.
 *
 *  The website (http://www.crbond.com) furthermore states:
 *  "This website contains a variety of materials related to
 *  technology and engineering. Downloadable software, much of it
 *  original, is available from some of the pages. All downloadable
 *  software is offered freely and without restriction -- although
 *  in most cases the files should be considered as works in progress
 *  (alpha or beta level). Source code is also included for some
 *  applications."
 *
 *  Code history:
 *  1/03 -- Added C/C++ source files for real and complex gamma function
 *  and psi function. Also added individual C/C++ files  for Bessel and
 *  modified Bessel functions of 1st and 2nd kinds for real and complex
 *  arguments. Updated Butterworth and Bessel filter tables with files of
 *   extended parameters including polynomials, poles and component values.
 *  6/04 -- Revised bessel.zip to correct errors in complex Bessel functions.
 *
 *  Further (re-)implementation of this code has been done by FZ Juelich.
 *  URL: http://apps.jcns.fz-juelich.de/redmine/issues/569#change-2056
 *  Above URL also includes a accuracy test report of this code against
 *  SLATEC, MAPLE and MATHEMATICA.
 */

#pragma once

#include "pmacc/algorithms/math.hpp"
#include "pmacc/math/complex/Complex.hpp"

#include <limits>
#include <type_traits>

namespace pmacc::math::bessel
{
    namespace detail
    {
        /** Largest component magnitude, without squaring (also safe for tiny arguments). */
        template<typename T>
        HDINLINE T componentAbs(alpaka::Complex<T> const& z)
        {
            return math::max(math::abs(z.real()), math::abs(z.imag()));
        }

        /** Divide a Miller value by its normalization sum.
         *
         * Both operands have the same arbitrary recurrence scale. Removing the largest denominator
         * component first avoids overflow in alpaka's complex division, which squares the denominator.
         * In this use the normalized quotient is bounded: exp(-Re(u))*I_n(u), Re(u)>=0, n=0,1,
         * has magnitude at most one (the integer-order integral representation, DLMF 10.32.3).
         */
        template<typename T>
        HDINLINE alpaka::Complex<T> normalize(alpaka::Complex<T> const& value, alpaka::Complex<T> const& sum)
        {
            T const scale = componentAbs(sum);
            auto const a = value / scale;
            auto const b = sum / scale;
            return (a * alpaka::Complex<T>(b.real(), -b.imag())) / (b.real() * b.real() + b.imag() * b.imag());
        }

        /** Undo exponential scaling for one component, preserving exact zeros on the axes.
         *
         * Four factors factor=exp(y/4), y>=0, avoid overflowing exp(y) while the final component is still
         * representable. Multiplication by factors >=1 cannot overflow before the final product.
         * If exp(y/4) itself overflows, every nonzero representable scaled component overflows too.
         * This cannot recover a component already lost to underflow in the scaled result.
         */
        template<typename T>
        HDINLINE T restoreScale(T const value, T const factor)
        {
            if(value == T(0))
                return value;
            return ((value * factor) * factor) * factor * factor;
        }

        /** J_n(z) or exp(-|Im(z)|)*J_n(z), specialized to n=0,1 and float/double.
         *
         * The series is used for |z|<=4. Miller recurrence bridges to the Hankel expansion at
         * |z|=12 (float) or 32 (double). The cutoffs and fixed work bounds are accuracy/performance
         * choices validated against high-precision references, not rigorous global error bounds.
         * Near zeros, accuracy is measured against the local J0/J1 amplitude, not J_n alone.
         * The accuracy targets assume ordinary floating-point semantics, without fast-math reassociation.
         * Nonfinite arguments return NaNs. Unscaled results can overflow; use the scaled interface
         * for large imaginary arguments or combine scaling analytically when forming ratios.
         */
        template<typename T, unsigned T_order, typename T_TableA, typename T_TableB>
        struct ComplexBesselJ
        {
            using Complex = alpaka::Complex<T>;
            static_assert(T_order <= 1u);
            static constexpr bool singlePrecision = std::is_same_v<T, float>;

            /** Taylor expansion, DLMF 10.2.2, on |z|<=4.
             *
             * J_n(z)=(z/2)^n * sum_k t_k, t_0=1,
             * t_k/t_(k-1)=-z^2/(4*k*(k+n)). For n=0,1 the factorial prefactor is one.
             */
            HDINLINE static Complex series(Complex const& z)
            {
                Complex term(1);
                Complex sum(1);
                Complex const multiplier = T(-0.25) * (z * z);
                constexpr T eps = std::numeric_limits<T>::epsilon();
                for(unsigned k = 1u; k <= 24u; ++k)
                {
                    term *= multiplier / T(k * (k + T_order));
                    sum += term;
                    if(componentAbs(term) <= eps * componentAbs(sum))
                        break;
                }
                if constexpr(T_order == 1u)
                    return sum * (z * T(0.5));
                else
                    return sum;
            }

            /** Miller backward recurrence, for 4<|z|<12 (float) or 32 (double), Im(z)>=0.
             *
             * With u=-i*z, J_n(z)=i^n*I_n(u) (DLMF 10.27.6), and
             * I_(k-1)(u)=I_(k+1)(u)+(2*k/u)*I_k(u) (DLMF 10.29.1).
             * Starting with f_(N+1)=0 and arbitrary nonzero f_N approximates a multiple of I_k
             * for the small orders k of interest as N increases.
             * Normalize with exp(u)=I_0(u)+2*sum_(k>=1) I_k(u) (DLMF 10.35.1 at t=1).
             * Re(u)>=0 avoids the exponential cancellation of the alternative J0+2*sum J_(2k)=1.
             * exp(u-|Im(z)|)=exp(-i*Re(z)), so this routine directly returns the scaled value.
             *
             * N=32/80 gives small truncation in the stated float/double domains; increasing N is
             * not a remedy for rounding error. Kahan summation compensates the normalization sum,
             * but not errors in f_k. The arbitrary float seed is reduced by an exact power of two
             * to keep intermediate recurrence values finite. Storage is independent of N.
             * See also DLMF 3.6(iii): https://dlmf.nist.gov/3.6.iii
             */
            HDINLINE static Complex miller(Complex const& z)
            {
                constexpr unsigned count = singlePrecision ? 32u : 80u;
                Complex current(singlePrecision ? T(0x1p-40) : T(1));
                Complex next(0);
                Complex sum(0);
                Complex correction(0);
                Complex const u(z.imag(), -z.real());
                // Here 4<|u|<32, so the squared denominator is safe.
                Complex const twiceInverseU = T(2) / u;
                for(unsigned k = count; k > 0u; --k)
                {
                    Complex const increment = T(2) * current - correction;
                    Complex const updated = sum + increment;
                    correction = (updated - sum) - increment;
                    sum = updated;
                    Complex const previous = (T(k) * twiceInverseU) * current + next;
                    next = current;
                    current = previous;
                }
                Complex const denominator = sum + (current - correction);
                Complex const phase(math::cos(z.real()), -math::sin(z.real()));
                if constexpr(T_order == 0u)
                    return normalize(current, denominator) * phase;
                else
                {
                    Complex const value = normalize(next, denominator) * phase;
                    return Complex(-value.imag(), value.real());
                }
            }

            /** Scaled Hankel expansion in the first quadrant, DLMF 10.17.1--10.17.3.
             *
             * J_n(z) ~ sqrt(2/(pi*z))*(P_n(z)*cos(omega)-Q_n(z)*sin(omega)),
             * omega=z-n*pi/2-pi/4. P contains even inverse powers, Q odd inverse powers.
             * The tables include their alternating signs. Horner evaluation in (1/z)^2 uses
             * 4 corrections per polynomial for float, 12 for double, with no complex powers.
             * These are asymptotic expansions; adding terms indefinitely would worsen accuracy.
             * The first-quadrant mapping stays inside the sector |arg(z)|<pi of DLMF 10.17.3.
             * For complex remainder estimates see https://dlmf.nist.gov/10.17.iv
             */
            HDINLINE static Complex asymptotic(Complex const& z)
            {
                T_TableA a;
                T_TableB b;
                constexpr unsigned count = singlePrecision ? 4u : 12u;
                T const scale = componentAbs(z);
                Complex const w = z / scale;
                T const radiusSquared = w.real() * w.real() + w.imag() * w.imag();
                T const radius = math::sqrt(radiusSquared);
                Complex const inverse = (Complex(w.real(), -w.imag()) / radiusSquared) / scale;
                Complex const inverseSquared = inverse * inverse;
                Complex p(0);
                Complex q(0);
                for(unsigned k = count; k > 0u; --k)
                {
                    p = a[k - 1u] + inverseSquared * p;
                    q = b[k - 1u] + inverseSquared * q;
                }
                p = T(1) + inverseSquared * p;
                q = inverse * (T_order == 0u ? T(-0.125) + inverseSquared * q : T(0.375) + inverseSquared * q);

                // sqrt(2/(pi*z)) without forming |z| or squaring its unscaled components.
                // For w in the first quadrant, sqrt(w)=rootReal+i*w.imag()/(2*rootReal).
                T const rootReal = math::sqrt(T(0.5) * (radius + w.real()));
                // sqrt takes a const reference: bind it to a local constant, not host-only static storage.
                constexpr T doubleReciprocalPi = Pi<T>::doubleReciprocalValue;
                Complex const prefactor = Complex(rootReal, -w.imag() / (T(2) * rootReal))
                                          * ((math::sqrt(doubleReciprocalPi) / math::sqrt(scale)) / radius);

                // exp(-y)*cosh(y) and exp(-y)*sinh(y), y=Im(z)>=0, remain bounded.
                // For small y use sinh directly to preserve the small imaginary components.
                T const y = z.imag();
                T const decay = math::exp(-y);
                T const scaledCosh = T(0.5) * (T(1) + decay * decay);
                T const scaledSinh = y < T(1) ? decay * math::sinh(y) : T(0.5) * (T(1) - decay * decay);
                T const c = math::cos(z.real());
                T const s = math::sin(z.real());
                Complex const cosine(c * scaledCosh, -s * scaledSinh);
                Complex const sine(s * scaledCosh, c * scaledSinh);
                T const inverseSqrtTwo = T(0.707106781186547524400844362104849039);
                // Angle addition preserves range reduction of Re(z), even when z-pi/4 rounds to z.
                if constexpr(T_order == 0u)
                    return prefactor
                           * (p * ((cosine + sine) * inverseSqrtTwo) - q * ((sine - cosine) * inverseSqrtTwo));
                else
                    return prefactor
                           * (p * ((sine - cosine) * inverseSqrtTwo) + q * ((sine + cosine) * inverseSqrtTwo));
            }

            HDINLINE static Complex evaluate(Complex const& input, bool const scaled)
            {
                constexpr T largestFinite = std::numeric_limits<T>::max();
                if(!(math::abs(input.real()) <= largestFinite) || !(math::abs(input.imag()) <= largestFinite))
                {
                    T const nan = std::numeric_limits<T>::quiet_NaN();
                    return Complex(nan, nan);
                }
                // J_n(-z)=(-1)^n*J_n(z), J_n(conj(z))=conj(J_n(z)). The exponential scale
                // is invariant under both transformations. First map to Re(z)>=0, then Im(z)>=0.
                bool const reflect = input.real() < T(0);
                Complex const right = reflect ? -input : input;
                bool const conjugate = right.imag() < T(0);
                Complex const z(right.real(), math::abs(right.imag()));
                T const largest = componentAbs(z);
                constexpr T asymptoticThreshold = singlePrecision ? T(12) : T(32);
                // Square only small arguments: avoid overflow and do not use |z|==0 to detect zero.
                T const radiusSquared = largest <= asymptoticThreshold ? z.real() * z.real() + z.imag() * z.imag()
                                                                       : asymptoticThreshold * asymptoticThreshold;
                Complex value;
                if(radiusSquared <= T(16))
                {
                    value = series(z);
                    if(scaled)
                        value *= math::exp(-z.imag());
                }
                else
                {
                    value = radiusSquared < asymptoticThreshold * asymptoticThreshold ? miller(z) : asymptotic(z);
                    // Enforce the exact real/imaginary-axis identities before restoring a large scale.
                    if(z.imag() == T(0))
                        value.imag(T(0));
                    if(z.real() == T(0))
                    {
                        if constexpr(T_order == 0u)
                            value.imag(T(0));
                        else
                            value.real(T(0));
                    }
                    if(!scaled && z.imag() != T(0))
                    {
                        T const factor = math::exp(z.imag() * T(0.25));
                        value = Complex(restoreScale(value.real(), factor), restoreScale(value.imag(), factor));
                    }
                }
                if(conjugate)
                    value.imag(-value.imag());
                if constexpr(T_order == 1u)
                    if(reflect)
                        value = -value;
                return value;
            }
        };
    } // namespace detail

    template<typename T, unsigned T_order>
    struct ComplexBesselJ
        : detail::ComplexBesselJ<
              T,
              T_order,
              std::conditional_t<
                  std::is_same_v<T, float>,
                  std::conditional_t<T_order == 0u, aFloat_t, a1Float_t>,
                  std::conditional_t<T_order == 0u, aDouble_t, a1Double_t>>,
              std::conditional_t<
                  std::is_same_v<T, float>,
                  std::conditional_t<T_order == 0u, bFloat_t, b1Float_t>,
                  std::conditional_t<T_order == 0u, bDouble_t, b1Double_t>>>
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
        using result = alpaka::Complex<T>;

        HDINLINE result operator()(result const& z) const
        {
            return this->evaluate(z, false);
        }
    };

    template<typename T>
    struct J0<alpaka::Complex<T>> : ComplexBesselJ<T, 0u>
    {
    };

    template<typename T>
    struct J1<alpaka::Complex<T>> : ComplexBesselJ<T, 1u>
    {
    };

    template<typename T>
    struct J0e<alpaka::Complex<T>> : ComplexBesselJ<T, 0u>
    {
        using result = alpaka::Complex<T>;

        HDINLINE result operator()(result const& z) const
        {
            return this->evaluate(z, true);
        }
    };

    template<typename T>
    struct J1e<alpaka::Complex<T>> : ComplexBesselJ<T, 1u>
    {
        using result = alpaka::Complex<T>;

        HDINLINE result operator()(result const& z) const
        {
            return this->evaluate(z, true);
        }
    };
} // namespace pmacc::math::bessel
