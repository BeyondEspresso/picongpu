/* Copyright 2026 Alexander Debus and LLM agent
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

#include <pmacc/boost_workaround.hpp>

#include "BesselReference.hpp"

#include <pmacc/lockstep.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{
    static pmacc::test::PMaccFixture<TEST_DIM> fixture;

    template<typename T>
    HDINLINE std::array<alpaka::Complex<T>, 4u> evaluate(alpaka::Complex<T> const& z)
    {
        namespace bessel = pmacc::math::bessel;
        return {bessel::j0(z), bessel::j1(z), bessel::j0e(z), bessel::j1e(z)};
    }

    struct EvaluateBessel
    {
        template<typename T_Worker, typename T_Input, typename T_Output>
        DINLINE void operator()(T_Worker const& worker, T_Input input, T_Output output, uint32_t const count) const
        {
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const local)
                {
                    uint32_t const index = worker.blockDomIdx() * T_Worker::blockDomSize() + local;
                    if(index < count)
                    {
                        auto const values = evaluate(input[index]);
                        for(uint32_t k = 0u; k < 4u; ++k)
                            output[4u * index + k] = values[k];
                    }
                });
        }
    };

    template<typename T>
    auto evaluateOnDevice(std::vector<alpaka::Complex<T>> const& arguments)
    {
        using Complex = alpaka::Complex<T>;
        uint32_t const count = static_cast<uint32_t>(arguments.size());
        pmacc::HostDeviceBuffer<Complex, 1u> input(pmacc::DataSpace<1u>{count});
        pmacc::HostDeviceBuffer<Complex, 1u> output(pmacc::DataSpace<1u>{4u * count});
        for(uint32_t i = 0u; i < count; ++i)
            input.getHostBuffer().data()[i] = arguments[i];
        input.hostToDevice();
        uint32_t const blocks = (count + 31u) / 32u;
        PMACC_LOCKSTEP_KERNEL(EvaluateBessel{})
            .config<32u>(blocks)(input.getDeviceBuffer().getDataBox(), output.getDeviceBuffer().getDataBox(), count);
        output.deviceToHost();
        std::vector<std::array<Complex, 4u>> result(count);
        for(uint32_t i = 0u; i < count; ++i)
            for(uint32_t k = 0u; k < 4u; ++k)
                result[i][k] = output.getHostBuffer().data()[4u * i + k];
        return result;
    }

    template<typename T>
    void checkReferenceValues()
    {
        using Complex = alpaka::Complex<T>;
        std::vector<besselTest::Reference> references(besselTest::common.begin(), besselTest::common.end());
        if constexpr(std::is_same_v<T, double>)
            references.insert(references.end(), besselTest::doubleOnly.begin(), besselTest::doubleOnly.end());

        std::vector<Complex> arguments;
        std::vector<std::array<T, 2u>> signs;
        for(auto const& reference : references)
            for(T const realSign : {T(1), T(-1)})
                for(T const imagSign : {T(1), T(-1)})
                {
                    arguments.emplace_back(realSign * T(reference.real), imagSign * T(reference.imag));
                    signs.push_back({realSign, imagSign});
                }
        auto const device = evaluateOnDevice(arguments);
        for(uint32_t i = 0u; i < arguments.size(); ++i)
        {
            auto const& ref = references[i / 4u];
            auto const z = arguments[i];
            auto const host = evaluate(z);
            for(uint32_t k = ref.unscaled ? 0u : 2u; k < 4u; ++k)
            {
                bool const orderOne = k % 2u == 1u;
                T const realSign = signs[i][0];
                T const imagSign = signs[i][1];
                // Parity and conjugation transform first-quadrant high-precision references.
                Complex const expected(
                    T(ref.values[2u * k]) * (orderOne ? realSign : T(1)),
                    T(ref.values[2u * k + 1u]) * (orderOne ? imagSign : realSign * imagSign));
                T scale = std::max(std::abs(expected.real()), std::abs(expected.imag()));
                if(std::max(ref.real, ref.imag) >= 1.)
                {
                    // Near zeros use the J0/J1 pair's amplitude. No fixed absolute floor:
                    // small arguments must retain relative accuracy in the small J1 value.
                    uint32_t const pairStart = k < 2u ? 0u : 4u;
                    for(uint32_t c = 0u; c < 4u; ++c)
                        scale = std::max(scale, T(std::abs(ref.values[pairStart + c])));
                }
                constexpr T epsilonFactor = std::is_same_v<T, float> ? T(32) : T(64);
                T const tolerance = (epsilonFactor * std::numeric_limits<T>::epsilon()) * scale;
                CAPTURE(sizeof(T), z.real(), z.imag(), k, expected.real(), expected.imag(), tolerance);
                for(auto const& actual : {host[k], device[i][k]})
                {
                    CHECK(std::isfinite(actual.real()));
                    CHECK(std::isfinite(actual.imag()));
                    // Do not square tiny errors or use Catch's default relative tolerance.
                    CHECK(std::abs(actual.real() - expected.real()) <= tolerance);
                    CHECK(std::abs(actual.imag() - expected.imag()) <= tolerance);
                    if(z.imag() == T(0))
                        CHECK(actual.imag() == T(0));
                    if(z.real() == T(0))
                    {
                        if(orderOne)
                            CHECK(actual.real() == T(0));
                        else
                            CHECK(actual.imag() == T(0));
                    }
                }
            }
        }
    }

    template<typename T>
    void checkRangeAndSpecialValues()
    {
        using Complex = alpaka::Complex<T>;
        T const infinity = std::numeric_limits<T>::infinity();
        T const nan = std::numeric_limits<T>::quiet_NaN();
        std::vector<Complex> const arguments{
            Complex(0, 1000),
            Complex(0, -1000),
            Complex(infinity, 0),
            Complex(0, infinity),
            Complex(-infinity, 1),
            Complex(nan, 0),
            Complex(0, nan)};
        auto const device = evaluateOnDevice(arguments);
        for(uint32_t i = 0u; i < arguments.size(); ++i)
        {
            CAPTURE(sizeof(T), i);
            auto const host = evaluate(arguments[i]);
            for(auto const& values : {host, device[i]})
            {
                if(i < 2u)
                {
                    // The unscaled functions overflow legitimately, but axes must not acquire NaNs.
                    CHECK(values[0].real() == infinity);
                    CHECK(values[0].imag() == T(0));
                    CHECK(values[1].real() == T(0));
                    CHECK(values[1].imag() == (i == 0u ? infinity : -infinity));
                    CHECK(std::isfinite(values[2].real()));
                    CHECK(values[2].real() > T(0));
                    CHECK(values[2].imag() == T(0));
                    CHECK(values[3].real() == T(0));
                    CHECK(std::isfinite(values[3].imag()));
                }
                else
                    for(auto const& value : values)
                    {
                        CHECK(std::isnan(value.real()));
                        CHECK(std::isnan(value.imag()));
                    }
            }
        }
    }
} // namespace

TEST_CASE("complex Bessel J0/J1 and exponential scaling: float", "[math][bessel]")
{
    checkReferenceValues<float>();
    checkRangeAndSpecialValues<float>();
}

TEST_CASE("complex Bessel J0/J1 and exponential scaling: double", "[math][bessel]")
{
    checkReferenceValues<double>();
    checkRangeAndSpecialValues<double>();
}
