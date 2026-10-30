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

#include "twtstight/FieldEvaluation.hpp"

#include <pmacc/test/PMaccFixture.hpp>

// Initialize and finalize PMacc exactly once for the combined test executable.
static pmacc::test::PMaccFixture<simDim> pmaccFixture;

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
                    result(0u + valueIdx) = evaluateTWTSComponent<0u>(testEfield, pos, time);
                    result(1u + valueIdx) = evaluateTWTSComponent<1u>(testEfield, pos, time);
                    result(2u + valueIdx) = evaluateTWTSComponent<2u>(testEfield, pos, time);
                    result(3u + valueIdx) = evaluateTWTSComponent<0u>(testBfield, pos, time);
                    result(4u + valueIdx) = evaluateTWTSComponent<1u>(testBfield, pos, time);
                    result(5u + valueIdx) = evaluateTWTSComponent<2u>(testBfield, pos, time);
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
            evaluateTWTSComponent<0u>(testEfield, pos, time),
            evaluateTWTSComponent<1u>(testEfield, pos, time),
            evaluateTWTSComponent<2u>(testEfield, pos, time));
        auto hostBfield = float3_T(
            evaluateTWTSComponent<0u>(testBfield, pos, time),
            evaluateTWTSComponent<1u>(testBfield, pos, time),
            evaluateTWTSComponent<2u>(testBfield, pos, time));

// This combination of compilers has a bug that is triggered by Catch2 internally suppressing warnings.
// See https://github.com/ComputationalRadiationPhysics/picongpu/pull/5174#issuecomment-2467890326
#if(__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
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
