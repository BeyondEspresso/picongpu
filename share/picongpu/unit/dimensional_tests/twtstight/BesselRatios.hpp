/* Copyright 2024-2026 Rene Widera, Alexander Debus
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

#include "Common.hpp"

template<typename T>
struct GenerateBesselRatios
{
    template<typename T_Worker, typename T_Box>
    HDINLINE void operator()(T_Worker const&, alpaka::Complex<T> const q, T const a, T_Box result) const
    {
        auto const ratios = templates::twtstight::detail::besselJOverI0(q, a);
        for(uint32_t n = 0; n < 2u; ++n)
        {
            result[2u * n] = ratios[n].real();
            result[2u * n + 1u] = ratios[n].imag();
        }
        if(a < T(50))
        {
            // Evaluate the direct formula independently for the small-a branch check.
            auto const denominator = math::bessel::i0(a);
            auto const r0 = math::bessel::j0(q) / denominator;
            auto const r1 = math::bessel::j1(q) / denominator;
            result[4] = r0.real();
            result[5] = r0.imag();
            result[6] = r1.real();
            result[7] = r1.imag();
        }
    }
};

template<typename T>
static void testBesselRatios()
{
    /* mpmath, mp.dps=100: r[n] = besselj(n, mpc(qr,qi))/besseli(0,a).
     * Exact binary inputs isolate ratio accuracy from rounding
     * when computing the ratio's inputs from beam parameters and position.
     * a, Re(q), Im(q), Re(R0), Im(R0), Re(R1), Im(R1). */
    std::array<std::array<double, 7u>, 16u> const cases{
        {{49.9990234375,
          3.0,
          49.9990234375,
          -0.984400135038261971143,
          -0.1707122907418998998654,
          0.1684080502029405851505,
          -0.9746446795707597764629},
         {50.0,
          3.0,
          50.0,
          -0.98440027075448533217,
          -0.17071171617095135334,
          0.16840753788905131908,
          -0.97464500169131651503},
         {50.0009765625,
          3.0,
          50.0009765625,
          -0.9844004064643769588637,
          -0.1707111416222631548731,
          0.1684070255943275215084,
          -0.9746453237984722486887},
         {50.0,
          3.0,
          49.0,
          -0.36578341543081707079,
          -0.063665303625172617733,
          0.062784683832877311785,
          -0.36208568879000739974},
         {50.0,
          3.0,
          51.0,
          -2.6497461592882321213,
          -0.45789495173963087234,
          0.45186087572616665663,
          -2.6239950914641511818},
         {50.0, 0.0, 0.0, 3.4099971346045609967e-21, 0.0, 0.0, 0.0},
         {50.0, 3.0, 0.0, -8.8677642106390754239e-22, 0.0, 1.1561900770354500115e-21, 0.0},
         {50.0,
          1.0,
          2.0,
          5.409140179929492353e-21,
          -4.7453603749448956942e-21,
          4.4051963389525569207e-21,
          3.4457624294722526016e-21},
         {100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.99498737300516876559},
         {1000.0, 0.0, 1000.0, 1.0, 0.0, 0.0, 0.9994998748748042802},
         {1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // Physical underflow, no NaNs at q=0.
         // Scaled numerator paths and representable subnormal inverse-I0 values.
         {50,
          3,
          4,
          -3.004938508623322361639e-20,
          -1.568066006178076536363e-20,
          1.246050558915170769864e-20,
          -2.865456143673171704378e-20},
         {50,
          31,
          2,
          7.093786936804610182313e-22,
          1.631907724937761659793e-21,
          -1.681442987671879680382e-21,
          7.133615834755543905295e-22},
         {1000,
          3,
          1000,
          -0.9897774219312946209385,
          -0.1426048835491377824143,
          0.1425320785098911801336,
          -0.9892826278440534688875},
         {104, 0, 0, 1.739626289826376347294e-44, 0.0, 0.0, 0.0},
         {746, 0, 0, 7.107273509301367849004e-323, 0.0, 0.0, 0.0}}};
    for(auto const& test : cases)
    {
        for(T const sign : {T(-1), T(1)})
        {
            T const a = T(test[0]);
            alpaka::Complex<T> const q(T(test[1]), sign * T(test[2]));
            CAPTURE(sizeof(T), a, q.real(), q.imag());
            HostBuffer<T, 1u> host(8u);
            DeviceBuffer<T, 1u> device(8u);
            device.setValue(T(0));
            PMACC_LOCKSTEP_KERNEL(GenerateBesselRatios<T>{}).template config<1u>(1u)(q, a, device.getDataBox());
            host.copyFrom(device);
            auto const result = host.getDataBox();
            auto const ratios = templates::twtstight::detail::besselJOverI0(q, a);
            for(uint32_t n = 0; n < 2u; ++n)
            {
                CAPTURE(n);
                alpaka::Complex<T> const reference(T(test[3u + 2u * n]), sign * T(test[4u + 2u * n]));
                T const referenceScale = math::max(math::abs(reference.real()), math::abs(reference.imag()));
                // Relative precision is unavailable for subnormals; allow two representable steps.
                // Keep exactly-zero references strict (including physical underflow at a=1000).
                T const tolerance = referenceScale == T(0)
                                        ? T(0)
                                        : math::max(
                                              T(128) * std::numeric_limits<T>::epsilon() * referenceScale,
                                              T(2) * std::numeric_limits<T>::denorm_min());
                CHECK(std::isfinite(result[2u * n]));
                CHECK(std::isfinite(result[2u * n + 1u]));
                CHECK(std::isfinite(ratios[n].real()));
                CHECK(std::isfinite(ratios[n].imag()));
                /* Compare components separately: squaring tiny complex errors in abs()
                 * can underflow in single precision and hide a regression. */
                CHECK(math::abs(ratios[n].real() - reference.real()) <= tolerance);
                CHECK(math::abs(ratios[n].imag() - reference.imag()) <= tolerance);
                CHECK(math::abs(result[2u * n] - reference.real()) <= tolerance);
                CHECK(math::abs(result[2u * n + 1u] - reference.imag()) <= tolerance);
                if(a < T(50))
                {
                    // Separate evaluations of the same formula need not be bitwise identical:
                    // inlining and floating-point contraction can change intermediate rounding.
                    // Allow a small componentwise relative difference; the independent reference
                    // checks above continue to enforce the numerical accuracy of the ratios.
                    constexpr T directRelativeTolerance = T(4) * std::numeric_limits<T>::epsilon();
                    CHECK(isApproxEqual(result[2u * n], result[4u + 2u * n], directRelativeTolerance));
                    CHECK(isApproxEqual(result[2u * n + 1u], result[5u + 2u * n], directRelativeTolerance));
                    T const denominator = math::bessel::i0(a);
                    auto const direct
                        = n == 0u ? math::bessel::j0(q) / denominator : math::bessel::j1(q) / denominator;
                    CHECK(isApproxEqual(ratios[n].real(), direct.real(), directRelativeTolerance));
                    CHECK(isApproxEqual(ratios[n].imag(), direct.imag(), directRelativeTolerance));
                }
            }
        }
    }
}

// Host entry point implemented in the double-precision translation unit.
void testTWTSTightBesselRatiosDouble();
