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

#include "FieldEvaluation.hpp"

// This combination of compilers has a bug that is triggered by Catch2 internally suppressing warnings.
// See https://github.com/ComputationalRadiationPhysics/picongpu/pull/5174#issuecomment-2467890326
#if(__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
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
                evaluateTWTSComponent<0u>(field, pos, time),
                evaluateTWTSComponent<1u>(field, pos, time),
                evaluateTWTSComponent<2u>(field, pos, time)};
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
#if(__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
    for(float_64 const beta : {0.9, 1.0, 1.1})
    {
        for(float_64 const phi : {-PI / 2.0, -PI / 6.0, PI / 6.0, PI / 2.0})
            twtsTightEnvelopeGuardTest::checkEnvelope(beta, phi);
        twtsTightEnvelopeGuardTest::checkNegativePhi<templates::twtstight::EField>(beta);
        twtsTightEnvelopeGuardTest::checkNegativePhi<templates::twtstight::BField>(beta);
    }
#endif
}
