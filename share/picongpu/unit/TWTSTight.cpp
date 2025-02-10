/* Copyright 2024-2024 Rene Widera, Alexander Debus
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
static bool isApproxEqual(T const& a, T const& b, T const& epsilon = std::numeric_limits<T>::epsilon())
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
        using float3_T = ::pmacc::math::Vector<float_T, 3u>;

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
        auto hostEfield = float3_X(
            testEfield.calcTWTSFieldX(pos, time),
            testEfield.calcTWTSFieldY(pos, time),
            testEfield.calcTWTSFieldZ(pos, time));
        auto hostBfield = float3_X(
            testBfield.calcTWTSFieldX(pos, time),
            testBfield.calcTWTSFieldY(pos, time),
            testBfield.calcTWTSFieldZ(pos, time));

// This combination of compilers has a bug that is triggered by Catch2 internally suppressing warnings.
// See https://github.com/ComputationalRadiationPhysics/picongpu/pull/5174#issuecomment-2467890326
#if(__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
        const float3_64 refEfield = float3_64(0.18329124052693974, -0.009402050968104002, 0.1054028749666347);
        const float3_64 refBfield = float3_64(3.5299706879027803e-10, 5.334111474127282e-11, -6.090365721598194e-10);
        const float3_T refEfieldT = precisionCast<float_T>(refEfield);
        const float3_T refBfieldT = precisionCast<float_T>(refBfield);
        /* epsilon to compare to Mathematica implementation.
         * Note: Reduction of epsilon would require replacing complex-valued bessel function support in
         * PMacc with boost library calls that also work on device. */
        const float_T epsilonAlgebra = float_T(5.0e-6);
        /* Epsilon to compare host implementation to device implementation */
        const float_T epsilonHostDevice = float_T(5.0e-15);
        for(uint32_t i = 0; i < 3; i++)
        {
            CHECK(isApproxEqual(refEfieldT[i], res[i], epsilonAlgebra));
            CHECK(isApproxEqual(hostEfield[i], res[i], epsilonHostDevice));
        }
        for(uint32_t i = 0; i < 3; i++)
        {
            CHECK(isApproxEqual(refBfieldT[i], res[i + 3], epsilonAlgebra));
            CHECK(isApproxEqual(hostBfield[i], res[i + 3], epsilonHostDevice));
        }
#endif
        float3_64 const cellDim(picongpu::sim.pic.getCellSize());
        float3_64 const cellDimensions = cellDim * sim.unit.length();
        // float3_X const amplitude = precisionCast<float_X>(float_64( 2.0 ) * float3_64( 1.0, 2.0, 3.0 ));
        std::cout << "halfSimSize    : { " << testEfield.halfSimSize[0] << ",  " << testEfield.halfSimSize[1] << ",  "
                  << testEfield.halfSimSize[2] << " }" << std::endl;
        std::cout << "cellDimensions : { " << cellDimensions[0] << ",  " << cellDimensions[1] << ",  "
                  << cellDimensions[2] << " }" << std::endl;
        float3_X const eFieldProbe(testEfield(DataSpace<3>{0, 1, 2}, 0u));
        std::cout << "eFieldProbe    : { " << eFieldProbe[0] << ",  " << eFieldProbe[1] << ",  " << eFieldProbe[2]
                  << " }" << std::endl;
        picongpu::traits::FieldPosition<fields::YeeCell, FieldE> const fieldPos;
        pmacc::math::Vector<floatD_64, templates::twtstight::detail::numComponents> const fieldPositions_SI
            = templates::twtstight::detail::getFieldPositions_SI(
                float3_X(0., 1., 2.),
                testEfield.halfSimSize,
                fieldPos(),
                sim.unit.length(),
                0.0,
                30. * (PI * 180.));
        float3_X const eFieldProbe2(testEfield.getTWTSField_Normalized(fieldPositions_SI, float_64(0.0)));
        std::cout << "eFieldProbe2   : { " << eFieldProbe2[0] << ",  " << eFieldProbe2[1] << ",  " << eFieldProbe2[2]
                  << " }" << std::endl;
        std::cout << "eFieldProbe3   : { " << testEfield.calcTWTSFieldX(fieldPositions_SI[0], float_64(0.0)) << ",  "
                  << testEfield.calcTWTSFieldY(fieldPositions_SI[1], float_64(0.0)) << ",  "
                  << testEfield.calcTWTSFieldZ(fieldPositions_SI[2], float_64(0.0)) << " }" << std::endl;
    }
};

TEST_CASE("unit::TWTSTight", "[TWTSTight laser math test]")
{
    twtsTightNumberTest()();
}
