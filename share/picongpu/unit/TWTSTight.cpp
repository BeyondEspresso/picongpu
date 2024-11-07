/* Copyright 2024 Rene Widera, Alexander Debus
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

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <typeinfo>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <picongpu/fields/background/templates/twtstight/TWTSTight.hpp>
#include <picongpu/param/precision.param>

//! Helper to setup the PMacc environment
using TestFixture = pmacc::test::PMaccFixture<TEST_DIM>;
static TestFixture fixture;

using namespace picongpu;
using namespace pmacc;

/** check if floating point result is equal
 *
 * Allows an error of one epsilon.
 * @return true if equal, else false
 */
template<typename T>
static bool isApproxEqual(T const& a, T const& b)
{
    return a == Catch::Approx(b).margin(std::numeric_limits<T>::epsilon());
}

template<typename T>
static bool isApproxEqual(T const& a, T const& b, T const& epsilon)
{
    return a == Catch::Approx(b).margin(epsilon);
}

template<uint32_t T_numThreadsPerBlock>
struct GenerateEvals
{
    const templates::twtstight::EField testEfield;
    const templates::twtstight::BField testBfield;
    using float_T = templates::twtstight::float_T;

    HINLINE GenerateEvals()
        : testEfield(0.0, 800.0e-9, 30.0e-15, 2.5e-6, 5. * (PI / 180.), 1.0, 0.0, false, 30. * (PI / 180.))
        , testBfield(0.0, 800.0e-9, 30.0e-15, 2.5e-6, 5. * (PI / 180.), 1.0, 0.0, false, 30. * (PI / 180.))
    {
    }

    template<class T_Box, typename T_Worker>
    HDINLINE void operator()(
        const T_Worker& worker,
        const uint32_t numValues,
        const uint32_t numValuesPerThread,
        const float3_64 pos,
        const float_64 time,
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
    void operator()()
    {
        using namespace ::pmacc;
        const templates::twtstight::EField testEfield = templates::twtstight::EField(
            0.0,
            800.0e-9,
            30.0e-15,
            2.5e-6,
            5. * (PI / 180.),
            1.0,
            0.0,
            false,
            30. * (PI / 180.));
        const templates::twtstight::BField testBfield = templates::twtstight::BField(
            0.0,
            800.0e-9,
            30.0e-15,
            2.5e-6,
            5. * (PI / 180.),
            1.0,
            0.0,
            false,
            30. * (PI / 180.));
        using float_T = templates::twtstight::float_T;

        constexpr uint32_t numBlocks = 1;
        constexpr uint32_t numThreadsPerBlock = 1;
        constexpr uint32_t numThreads = numBlocks * numThreadsPerBlock;
        constexpr uint32_t numValuesPerThread = 6;
        constexpr uint32_t numValues = numThreads * numValuesPerThread;
        const float3_64 pos = float3_64{1.0e-6, 1.0e-6, 100.0 * 1.5e-6};
        const float_64 time = float_64(1.0e-15);

        HostBuffer<float_T, 1u> resultHost(numValues);
        DeviceBuffer<float_T, 1u> resultDevice(numValues);
        resultDevice.setValue(float_T(0.0));

        PMACC_LOCKSTEP_KERNEL(GenerateEvals<numThreadsPerBlock>{})
            .template config<numThreadsPerBlock>(
                numBlocks)(numValues, numValuesPerThread, pos, time, resultDevice.getDataBox());

        resultHost.copyFrom(resultDevice);

        auto res = resultHost.getDataBox();
        const float_64 Ex = float_64(0.18329124052693974);
        const float_64 Ey = float_64(-0.009402050968104002);
        const float_64 Ez = float_64(0.1054028749666347);
        const float_64 Bx = float_64(3.5299706879027803e-10);
        const float_64 By = float_64(5.334111474127282e-11);
        const float_64 Bz = float_64(-6.090365721598194e-10);
        const float_T epsilon = float_T(5.0e-6);
        const float_T epsilon2 = float_T(5.0e-15);
        bool isCorrect = isApproxEqual(precisionCast<float_T>(Ex), res[0], epsilon)
            && isApproxEqual(testEfield.calcTWTSFieldX(pos, time), res[0], epsilon2);
        std::cerr << std::setprecision(std::numeric_limits<float_T>::digits10);
        if(!isCorrect)
            std::cerr << "Ex (analytical reference):   " << Ex << "  , EField.calcTWTSFieldX (Device):   " << res[0]
                      << "  , EField.calcTWTSFieldX (Host):   " << testEfield.calcTWTSFieldX(pos, time) << std::endl;
        REQUIRE(isCorrect);

        isCorrect = isApproxEqual(precisionCast<float_T>(Ey), res[1], epsilon)
            && isApproxEqual(testEfield.calcTWTSFieldY(pos, time), res[1], epsilon2);
        std::cerr << std::setprecision(std::numeric_limits<float_T>::digits10);
        if(!isCorrect)
            std::cerr << "Ey (analytical reference):   " << Ey << "  , EField.calcTWTSFieldY (Device):   " << res[1]
                      << "  , EField.calcTWTSFieldY (Host):   " << testEfield.calcTWTSFieldY(pos, time) << std::endl;
        REQUIRE(isCorrect);

        isCorrect = isApproxEqual(precisionCast<float_T>(Ez), res[2], epsilon)
            && isApproxEqual(testEfield.calcTWTSFieldZ(pos, time), res[2], epsilon2);
        std::cerr << std::setprecision(std::numeric_limits<float_T>::digits10);
        if(!isCorrect)
            std::cerr << "Ez (analytical reference):   " << Ez << "  , EField.calcTWTSFieldZ (Device):   " << res[2]
                      << "  , EField.calcTWTSFieldZ (Host):   " << testEfield.calcTWTSFieldZ(pos, time) << std::endl;
        REQUIRE(isCorrect);

        isCorrect = isApproxEqual(precisionCast<float_T>(Bx), res[3], epsilon)
            && isApproxEqual(testBfield.calcTWTSFieldX(pos, time), res[3], epsilon2);
        std::cerr << std::setprecision(std::numeric_limits<float_T>::digits10);
        if(!isCorrect)
            std::cerr << "Bx (analytical reference):   " << Bx << "  , BField.calcTWTSFieldX (Device):   " << res[3]
                      << "  , BField.calcTWTSFieldX (Host):   " << testBfield.calcTWTSFieldX(pos, time) << std::endl;
        REQUIRE(isCorrect);

        isCorrect = isApproxEqual(precisionCast<float_T>(By), res[4], epsilon)
            && isApproxEqual(testBfield.calcTWTSFieldY(pos, time), res[4], epsilon2);
        std::cerr << std::setprecision(std::numeric_limits<float_T>::digits10);
        if(!isCorrect)
            std::cerr << "By (analytical reference):   " << By << "  , BField.calcTWTSFieldY (Device):   " << res[4]
                      << "  , BField.calcTWTSFieldY (Host):   " << testBfield.calcTWTSFieldY(pos, time) << std::endl;
        REQUIRE(isCorrect);

        isCorrect = isApproxEqual(precisionCast<float_T>(Bz), res[5], epsilon)
            && isApproxEqual(testBfield.calcTWTSFieldZ(pos, time), res[5], epsilon2);
        std::cerr << std::setprecision(std::numeric_limits<float_T>::digits10);
        if(!isCorrect)
            std::cerr << "Bz (analytical reference):   " << Bz << "  , BField.calcTWTSFieldZ (Device):   " << res[5]
                      << "  , BField.calcTWTSFieldZ (Host):   " << testBfield.calcTWTSFieldZ(pos, time) << std::endl;
        REQUIRE(isCorrect);
    }
};

TEST_CASE("unit::TWTSTight", "[TWTSTight laser math test]")
{
    twtsTightNumberTest()();
}
