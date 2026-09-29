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
        /* The scaled ratio in besselJOverI0() (BesselRatios.hpp) uses exp(abs(Im(q))-a).
         * The Bessel arguments q and a are computed separately from position and beam parameters.
         * Since abs(Im(q)) is close to a near the beam axis, rounding in those inputs can
         * perturb the exponent by an amount proportional to a * machine epsilon. */
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
    // Mathematica reference for the corrected envelope, evaluated with unscaled Bessel functions.
    twtsTightNumberTest()(
        2.5e-6,
        1.0,
        {0.1727409809769384,
         -0.008861178965595797,
         0.09933584040072169,
         3.3267856482525846e-10,
         5.027189941003478e-11,
         -5.739801666328101e-10});
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
          {-0.2404407138786834031309,
           0.01192634975335576813892,
           -0.1383072882006389890513,
           -4.630392034744425089699e-10,
           -6.980481442322865747846e-11,
           7.990095267043634620785e-10}},
         {16e-6,
          1.0,
          {0.696474751878454896751,
           -0.03505049656288141308007,
           0.4005802396535182897393,
           1.341290228665441072431e-9,
           2.022910065080756614205e-10,
           -2.314377686041806740596e-9}},
         {16e-6,
          -1.0,
          {-0.196145313751402320313,
           0.009919817576899993953916,
           -0.1128096694784181925942,
           -3.777413118051959150407e-10,
           -5.684369471818129081026e-11,
           6.518018971095310405847e-10}},
         {20e-6,
          1.0,
          {0.7045402691238972829893,
           -0.03545135629888131804292,
           0.4052192925273568417814,
           1.356824380395596856796e-9,
           2.047048780170902131612e-10,
           -2.341169002022269115955e-9}},
         {20e-6,
          -1.0,
          {-0.1768586973877699815439,
           0.008934219690544825810004,
           -0.1017180324755257647238,
           -3.405993823624053431777e-10,
           -5.130553114257804753234e-11,
           5.877048301994784520533e-10}},
         {32e-6,
          1.0,
          {0.7117036614287385983734,
           -0.03581073694115641523918,
           0.4093390849589198851009,
           1.370621461760648545124e-9,
           2.068612136618168502997e-10,
           -2.364961364562449495166e-9}},
         {32e-6,
          -1.0,
          {-0.1551962973055136450206,
           0.007824052446382259627384,
           -0.08926043824735833669929,
           -2.988819006999841071541e-10,
           -4.507772744142058639344e-11,
           5.157136301943491165455e-10}}}};
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

TEST_CASE("unit::TWTSTight Bessel ratios", "[TWTSTight laser math test]")
{
    testBesselRatios<float>();
    testBesselRatios<double>();
}

// This combination of compilers has a bug that is triggered by Catch2 internally suppressing warnings.
// See https://github.com/ComputationalRadiationPhysics/picongpu/pull/5174#issuecomment-2467890326
#if (__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
/** Check the cutoff against the envelope geometry in SI coordinates.
 *
 * The center satisfies y = beta*c*t - z*(beta-cos(phi))/sin(phi).
 * The Gaussian width follows independently from q = tan(alpha)^2*z/(c*sin(phi)),
 * avoiding the guard's nu/xi expression when constructing the expected boundary.
 */
struct twtsTightEnvelopeGuardTest
{
    static constexpr float_64 wavelength = 800.0e-9;
    static constexpr float_64 pulseLength = 30.0e-15;
    // Keep individual Bessel values representable in float in the end-to-end check by using a small waist.
    static constexpr float_64 waist = 0.5e-6;

    static void checkEnvelope(float_64 const beta, float_64 const phi)
    {
        templates::twtstight::EField const
            eField{0.0, wavelength, pulseLength, waist, phi, beta, 0.0, false, 0.0, PI / 6.0};
        templates::twtstight::BField const
            bField{0.0, wavelength, pulseLength, waist, phi, beta, 0.0, false, 0.0, PI / 6.0};
        float_64 const c = sim.si.getSpeedOfLight();
        float_64 const sinPhi = std::sin(std::abs(phi));
        float_64 const cosPhi = std::cos(phi);
        float_64 const tanAlpha = (1.0 - beta * cosPhi) / (beta * sinPhi);
        float_64 const tau = 2.0 * pulseLength;
        float_64 const omega = 2.0 * PI * c / wavelength;
        float_64 const period = wavelength / (c * (1.0 - beta * cosPhi));
        float_64 const phiSign = phi < 0.0 ? -1.0 : 1.0;

        // Include no broadening and both signs of the imaginary part of sigma.
        for(float_64 const broadening : {-2.0, 0.0, 2.0})
        {
            float_64 const z = broadening * omega * tau * tau * c * sinPhi / (2.0 * tanAlpha * tanAlpha);
            float_64 const width = tau * std::sqrt(1.0 + broadening * broadening);
            // The second time exercises the reduced-coordinate translation by many periods.
            for(float_64 const periods : {0.25, 17.25})
            {
                float_64 const time = periods * period;
                float_64 const centerY = beta * c * time - z * (beta - cosPhi) / sinPhi;
                // Stay clear of equality: coordinates are rounded to float_T by the functor.
                for(float_64 const offset : {-1.05, -0.95, 0.0, 0.95, 1.05})
                {
                    float_64 const u = offset * templates::twtstight::numSigmas * width;
                    float3_64 const pos{0.0, centerY - beta * c * u, phiSign * z};
                    bool const outside = std::abs(offset) > 1.0;
                    INFO("beta = " << beta << ", phi = " << phi << ", broadening = " << broadening);
                    INFO("periods = " << periods << ", boundary fraction = " << offset);
                    CHECK(eField.isOutsideTWTSEnvelope(eField.defineMinimalCoordinates(pos, time)) == outside);
                    CHECK(bField.isOutsideTWTSEnvelope(bField.defineMinimalCoordinates(pos, time)) == outside);
                }
            }
        }
    }

    /** Exercise the actual field early-outs, including the negative-angle coordinate rotation. */
    template<typename T_Field>
    static void checkNegativePhi(float_64 const beta)
    {
        using float_T = templates::twtstight::float_T;
        float_64 const phi = -PI / 6.0;
        T_Field const field{0.0, wavelength, pulseLength, waist, phi, beta, 0.0, false, 0.0, PI / 6.0};
        float_64 const c = sim.si.getSpeedOfLight();
        float_64 const sinPhi = std::sin(-phi);
        float_64 const cosPhi = std::cos(phi);
        float_64 const tanAlpha = (1.0 - beta * cosPhi) / (beta * sinPhi);
        float_64 const tau = 2.0 * pulseLength;
        float_64 const omega = 2.0 * PI * c / wavelength;
        float_64 const z = 300.0e-6;
        float_64 const broadening = 2.0 * tanAlpha * tanAlpha * z / (c * sinPhi * omega * tau * tau);
        float_64 const width = tau * std::sqrt(1.0 + broadening * broadening);
        float_64 const time = 17.25 * wavelength / (c * (1.0 - beta * cosPhi));
        float_64 const centerY = beta * c * time - z * (beta - cosPhi) / sinPhi;

        INFO("negative-angle field check, beta = " << beta);
        for(float_64 const offset : {-1.05, 0.0, 1.05})
        {
            float_64 const u = offset * templates::twtstight::numSigmas * width;
            float3_64 const pos{-0.1e-6, centerY - beta * c * u, -z};
            std::array<float_T, 3u> const values{
                field.calcTWTSFieldX(pos, time),
                field.calcTWTSFieldY(pos, time),
                field.calcTWTSFieldZ(pos, time)};
            INFO("boundary fraction = " << offset);
            bool anyNonzero = false;
            for(auto const value : values)
            {
                CHECK(std::isfinite(value));
                anyNonzero = anyNonzero || value != float_T(0.0);
                if(offset != 0.0)
                    CHECK(value == float_T(0.0));
            }
            // Test the vector rather than requiring each oscillatory component to be nonzero.
            if(offset == 0.0)
                CHECK(anyNonzero);
        }
    }
};
#endif

TEST_CASE("unit::TWTSTightEnvelopeGuard", "[TWTSTight laser envelope guard test]")
{
#if (__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
    for(float_64 const beta : {0.9, 1.0, 1.1})
    {
        for(float_64 const phi : {-PI / 2.0, -PI / 6.0, PI / 6.0, PI / 2.0})
            twtsTightEnvelopeGuardTest::checkEnvelope(beta, phi);
        twtsTightEnvelopeGuardTest::checkNegativePhi<templates::twtstight::EField>(beta);
        twtsTightEnvelopeGuardTest::checkNegativePhi<templates::twtstight::BField>(beta);
    }
#endif
}
