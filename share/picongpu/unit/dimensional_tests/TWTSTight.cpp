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

#include <pmacc/boost_workaround.hpp>

#include <pmacc/test/PMaccFixture.hpp>

// STL
#include <pmacc/Environment.hpp>
#include <pmacc/algorithms/math.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/math/ConstVector.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <picongpu/fields/background/templates/twtstight/TWTSTight.hpp>
#include <picongpu/param/precision.param>

using namespace picongpu;
using namespace pmacc;

//! Helper to setup the PMacc environment
static pmacc::test::PMaccFixture<simDim> pmaccFixture;

/** check if floating point result is equal
 *
 * Allows an error of one epsilon.
 * @return true if equal, else false
 */
template<typename T>
static bool isApproxEqual(T const& a, T const& b, T const& epsilon)
{
    T const epsilonScaled = epsilon * math::max(math::abs(a), math::abs(b));
    return a == Catch::Approx(b).epsilon(0).margin(epsilonScaled);
}

template<typename T>
static bool isApproxEqual(T const& a, T const& b)
{
    T const epsilon = std::numeric_limits<T>::epsilon() * math::max(math::abs(a), math::abs(b));
    return a == Catch::Approx(b).epsilon(0).margin(epsilon);
}

template<uint32_t T_numThreadsPerBlock>
struct GenerateEvals
{
    templates::twtstight::EField const testEfield;
    templates::twtstight::BField const testBfield;
    using float_T = templates::twtstight::float_T;

    HINLINE GenerateEvals(float_64 const waist)
        : testEfield(0.0, 800.0e-9, 30.0e-15, waist, 5. * (PI / 180.), 1.0, 0.0, false, 0.0, 30. * (PI / 180.))
        , testBfield(0.0, 800.0e-9, 30.0e-15, waist, 5. * (PI / 180.), 1.0, 0.0, false, 0.0, 30. * (PI / 180.))
    {
    }

    template<class T_Box, typename T_Worker>
    HDINLINE void operator()(
        T_Worker const& worker,
        uint32_t const numValues,
        uint32_t const numValuesPerThread,
        float3_64 const pos,
        float_64 const time,
        T_Box result) const
    {
        using namespace ::pmacc;
        uint32_t const blockIdx = worker.blockDomIdxND().x();
        auto forEach = lockstep::makeForEach<T_numThreadsPerBlock>(worker);

        forEach(
            [&](uint32_t const idx)
            {
                auto valueIdx = blockIdx * T_numThreadsPerBlock + idx;
                if(valueIdx < numValues)
                {
                    result(0u + valueIdx) = testEfield.calcTWTSFieldX(pos, time);
                    result(1u + valueIdx) = testEfield.calcTWTSFieldY(pos, time);
                    result(2u + valueIdx) = testEfield.calcTWTSFieldZ(pos, time);
                    result(3u + valueIdx) = testBfield.calcTWTSFieldX(pos, time);
                    result(4u + valueIdx) = testBfield.calcTWTSFieldY(pos, time);
                    result(5u + valueIdx) = testBfield.calcTWTSFieldZ(pos, time);
                }
            });
    }
};

/** Test TWTSTight laser functions
 *
 * Compares the on host and on device computed result to analytical results.
 *
 */
struct twtsTightNumberTest
{
    void operator()(float_64 const waist, float_64 const zSign, std::array<float_64, 6u> const& reference)
    {
        using namespace ::pmacc;
        templates::twtstight::EField const testEfield = templates::twtstight::EField(
            0.0,
            800.0e-9,
            30.0e-15,
            waist,
            5. * (PI / 180.),
            1.0,
            0.0,
            false,
            0.0,
            30. * (PI / 180.));
        templates::twtstight::BField const testBfield = templates::twtstight::BField(
            0.0,
            800.0e-9,
            30.0e-15,
            waist,
            5. * (PI / 180.),
            1.0,
            0.0,
            false,
            0.0,
            30. * (PI / 180.));
        using float_T = templates::twtstight::float_T;
        using float3_T = ::pmacc::math::Vector<float_T, 3u>;

        constexpr uint32_t numBlocks = 1;
        constexpr uint32_t numThreadsPerBlock = 1;
        constexpr uint32_t numThreads = numBlocks * numThreadsPerBlock;
        constexpr uint32_t numValuesPerThread = 6;
        constexpr uint32_t numValues = numThreads * numValuesPerThread;
        float3_64 const pos = float3_64{1.0e-6, 1.0e-6, zSign * 150.0e-6};
        float_64 const time = float_64(1.0e-15);

        HostBuffer<float_T, 1u> resultHost(numValues);
        DeviceBuffer<float_T, 1u> resultDevice(numValues);
        resultDevice.setValue(float_T(0.0));

        PMACC_LOCKSTEP_KERNEL(GenerateEvals<numThreadsPerBlock>{waist})
            .template config<numThreadsPerBlock>(
                numBlocks)(numValues, numValuesPerThread, pos, time, resultDevice.getDataBox());

        resultHost.copyFrom(resultDevice);

        auto res = resultHost.getDataBox();
        auto hostEfield = float3_T(
            testEfield.calcTWTSFieldX(pos, time),
            testEfield.calcTWTSFieldY(pos, time),
            testEfield.calcTWTSFieldZ(pos, time));
        auto hostBfield = float3_T(
            testBfield.calcTWTSFieldX(pos, time),
            testBfield.calcTWTSFieldY(pos, time),
            testBfield.calcTWTSFieldZ(pos, time));

// This combination of compilers has a bug that is triggered by Catch2 internally suppressing warnings.
// See https://github.com/ComputationalRadiationPhysics/picongpu/pull/5174#issuecomment-2467890326
#if (__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
        float3_64 const refEfield(reference[0], reference[1], reference[2]);
        float3_64 const refBfield(reference[3], reference[4], reference[5]);
        float3_T const refEfieldT = precisionCast<float_T>(refEfield);
        float3_T const refBfieldT = precisionCast<float_T>(refBfield);
        /* epsilon to compare to Mathematica implementation.
         * Note: Reduction of epsilon would require replacing complex-valued bessel function support in
         * PMacc with boost library calls that also work on device. */
        float_T const epsilonSmallWaist = std::is_same<float_T, float_64>::value ? float_T(5.0e-13) : float_T(5.0e-5);
        /* The scaled ratio in besselJOverI0() (BesselRatios.hpp) uses exp(u-a).
         * See that implementation for the definitions of u and a, whose values are
         * computed separately from position and beam parameters.
         * Since Re(u) is close to a near the beam axis, rounding in those inputs can
         * perturb the exponent u-a by an amount proportional to a * machine epsilon. */
        float_64 const k = 2.0 * PI / 800.0e-9;
        float_T const a = float_T(k * k * waist * waist * std::sin(PI / 36.0) / 2.0);
        float_T const epsilonAlgebra
            = math::max(epsilonSmallWaist, float_T(8) * a * std::numeric_limits<float_T>::epsilon());
        // Allow for accumulated host/device rounding differences in the field expressions.
        float_T const epsilonSmallWaistHostDevice = std::is_same<float_T, float_64>::value
                                                        ? float_T(5.0e-15)
                                                        : float_T(8) * std::numeric_limits<float_T>::epsilon();
        float_T const epsilonHostDevice = waist == 2.5e-6 ? epsilonSmallWaistHostDevice : epsilonAlgebra;
        INFO(
            "waist = " << waist << ", zSign = " << zSign
                       << ", field precision bits = " << std::numeric_limits<float_T>::digits);
        for(uint32_t i = 0; i < 3; i++)
        {
            float_T const difference = math::abs(hostEfield[i] - res[i]);
            float_T const scale = math::max(math::abs(hostEfield[i]), math::abs(res[i]));
            INFO(
                std::setprecision(std::numeric_limits<float_T>::max_digits10)
                << "E[" << i << "]: reference = " << refEfieldT[i] << ", host = " << hostEfield[i]
                << ", device = " << res[i] << ", absolute difference = " << difference
                << ", relative difference = " << (scale > float_T(0) ? difference / scale : float_T(0))
                << ", host/device relative tolerance = " << epsilonHostDevice
                << ", reference relative tolerance = " << epsilonAlgebra);
            CHECK(std::isfinite(res[i]));
            CHECK(std::isfinite(hostEfield[i]));
            CHECK(isApproxEqual(refEfieldT[i], res[i], epsilonAlgebra));
            CHECK(isApproxEqual(refEfieldT[i], hostEfield[i], epsilonAlgebra));
            CHECK(isApproxEqual(hostEfield[i], res[i], epsilonHostDevice));
        }
        for(uint32_t i = 0; i < 3; i++)
        {
            float_T const difference = math::abs(hostBfield[i] - res[i + 3]);
            float_T const scale = math::max(math::abs(hostBfield[i]), math::abs(res[i + 3]));
            INFO(
                std::setprecision(std::numeric_limits<float_T>::max_digits10)
                << "B[" << i << "]: reference = " << refBfieldT[i] << ", host = " << hostBfield[i]
                << ", device = " << res[i + 3] << ", absolute difference = " << difference
                << ", relative difference = " << (scale > float_T(0) ? difference / scale : float_T(0))
                << ", host/device relative tolerance = " << epsilonHostDevice
                << ", reference relative tolerance = " << epsilonAlgebra);
            CHECK(std::isfinite(res[i + 3]));
            CHECK(std::isfinite(hostBfield[i]));
            CHECK(isApproxEqual(refBfieldT[i], res[i + 3], epsilonAlgebra));
            CHECK(isApproxEqual(refBfieldT[i], hostBfield[i], epsilonAlgebra));
            CHECK(isApproxEqual(hostBfield[i], res[i + 3], epsilonHostDevice));
        }
#endif
    }
};

TEST_CASE("unit::TWTSTight", "[TWTSTight laser math test]")
{
    // Keep the original small-waist reference and tolerances as a regression test.
    twtsTightNumberTest()(
        2.5e-6,
        1.0,
        {0.18329124052693974,
         -0.009402050968104002,
         0.1054028749666347,
         3.5299706879027803e-10,
         5.334111474127282e-11,
         -6.090365721598194e-10});
}

TEST_CASE("unit::TWTSTight scaled fields", "[TWTSTight laser math test]")
{
    /* References: original unscaled EField.tpp/BField.tpp expressions evaluated
     * with mpmath 1.3.0 at 100 decimal digits, using SI coordinates throughout.
     * lambda=800 nm, phi=5 deg, polarization=30 deg, tauG=60 fs, beta0=1,
     * c=299792458 m/s, (x,y,z)=(1,1,+/-150) um, t=1 fs.
     * Reversing z conjugates q and exercises both signs of Im(q). */
    struct Case
    {
        float_64 waist;
        float_64 zSign;
        std::array<float_64, 6u> field;
    };

    std::array<Case, 7u> const cases{
        {{2.5e-6,
          -1.0,
          {-0.22683717993203585622,
           0.011251485613919733954,
           -0.13048222260606555559,
           -4.3684157558245886412e-10,
           -6.5854329518267636854e-11,
           7.5380375043473663924e-10}},
         {16e-6,
          1.0,
          {0.73754875564986787601,
           -0.037117431689055956813,
           0.42420412958321451586,
           1.4203916732076328423e-9,
           2.1422114929815096752e-10,
           -2.4508661202040180123e-9}},
         {16e-6,
          -1.0,
          {-0.18399639266221673632,
           0.0093058325492185407241,
           -0.10582238571614977021,
           -3.5434462563779760335e-10,
           -5.332183512425163856e-11,
           6.114304564921202476e-10}},
         {20e-6,
          1.0,
          {0.74604912664393651582,
           -0.03753992944570809842,
           0.42909329387168937052,
           1.4367633589838776775e-9,
           2.1676546010587792285e-10,
           -2.4791018459636873501e-9}},
         {20e-6,
          -1.0,
          {-0.16575947333791697116,
           0.008373850198588232851,
           -0.09533442729189659863,
           -3.1922418671979663029e-10,
           -4.8085000320549771577e-11,
           5.508219663379615372e-10}},
         {32e-6,
          1.0,
          {0.75359015484100959997,
           -0.037918303007427995671,
           0.4334302649004970305,
           1.4512877978759074381e-9,
           2.1903585065762717283e-10,
           -2.5041484172978545767e-9}},
         {32e-6,
          -1.0,
          {-0.14527966308675931874,
           0.0073242642037703979932,
           -0.083556918719714990594,
           -2.7978413485473628064e-10,
           -4.2197076287447526837e-11,
           4.8276091663120122494e-10}}}};
    for(auto const& test : cases)
    {
        CAPTURE(test.waist, test.zSign);
        twtsTightNumberTest()(test.waist, test.zSign, test.field);
    }
}

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
            // Check that the small-a branch keeps the original direct calculation.
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
     * when computing the ratio’s inputs from beam parameters and position.
     * a, Re(q), Im(q), Re(R0), Im(R0), Re(R1), Im(R1). */
    std::array<std::array<double, 7u>, 11u> const cases{{
        {49.9990234375,
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
        {1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} // Physical underflow, no NaNs at q=0.
    }};
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
                alpaka::Complex<T> const reference(T(test[3u + 2u * n]), sign * T(test[4u + 2u * n]));
                T const tolerance = T(128) * std::numeric_limits<T>::epsilon()
                                    * math::max(math::abs(reference.real()), math::abs(reference.imag()));
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
                    CHECK(result[2u * n] == result[4u + 2u * n]);
                    CHECK(result[2u * n + 1u] == result[5u + 2u * n]);
                    T const denominator = math::bessel::i0(a);
                    auto const direct
                        = n == 0u ? math::bessel::j0(q) / denominator : math::bessel::j1(q) / denominator;
                    CHECK(ratios[n].real() == direct.real());
                    CHECK(ratios[n].imag() == direct.imag());
                }
            }
        }
    }
}

TEST_CASE("unit::TWTSTight Bessel ratios", "[TWTSTight laser math test]")
{
    testBesselRatios<float>();
    testBesselRatios<double>();
}
