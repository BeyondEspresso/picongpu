#!/usr/bin/env python3
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Alexander Debus's LLM agent
License: GPLv3+

Reference values for TWTSTight.cpp; Python 3 + mpmath 1.3.0.

Run: python3 twtstight_reference.py [--dps 100] [--check]
Uses 100 decimal digits and direct Bessel functions, not the scaled asymptotic
implementation. Decimal input strings avoid initial Python float rounding.
The field formulas are the EField.tpp/BField.tpp expressions in SI units,
specialized to the test geometry (positive phi); beta0 defaults to 1.
--check validates the C++ field table, the retained Mathematica values,
precision convergence, and general-beta envelope identities.
"""

import argparse
from pathlib import Path
import re

from mpmath import mp


def row(values):
    return "{" + ", ".join(mp.nstr(v, 22) for v in values) + "}"


def ratio_references():
    # a, Re(q), Im(q): all exactly representable in float and double.
    cases = [
        ("49.9990234375", "3", "49.9990234375"),
        ("50", "3", "50"),
        ("50.0009765625", "3", "50.0009765625"),
        ("50", "3", "49"),
        ("50", "3", "51"),
        ("50", "0", "0"),
        ("50", "3", "0"),
        ("50", "1", "2"),
        ("100", "0", "100"),
        ("1000", "0", "1000"),
        ("1000", "0", "0"),
    ]
    print("// a, Re(q), Im(q), Re(R0), Im(R0), Re(R1), Im(R1)")
    for inputs in cases:
        a, qr, qi = map(mp.mpf, inputs)
        q = mp.mpc(qr, qi)
        r0, r1 = [mp.besselj(n, q) / mp.besseli(0, a) for n in (0, 1)]
        if qr == 0:
            # J0(i*y) is real; J1(i*y) is imaginary. Remove tiny numerical
            # remnants in components that vanish identically by symmetry.
            r0, r1 = mp.mpc(r0.real, 0), mp.mpc(0, r1.imag)
        print(row([a, qr, qi, r0.real, r0.imag, r1.real, r1.imag]))
    # For negative Im(q), conjugate both ratios: J_n(conj(q))=conj(J_n(q)).
    # At a=1000, q=0, R0 is nonzero here but underflows to zero in float/double;
    # the C++ table therefore stores zero for this entry.


def fields(waist_um, z_sign, beta0="1"):
    beta0 = mp.mpf(beta0)
    c = mp.mpf(299792458)
    wavelength = mp.mpf("800e-9")
    phi, polarization = mp.pi / 36, mp.pi / 6
    tauG = mp.mpf("60e-15")  # twice the 30 fs constructor parameter
    w0 = mp.mpf(waist_um) * mp.mpf("1e-6")
    x = y = mp.mpf("1e-6")
    z = z_sign * mp.mpf("150e-6")
    t = mp.mpf("1e-15")
    # The full-precision envelope and carrier are invariant under the
    # wavelength-period reduction, so physical coordinates can be used directly.
    k = 2 * mp.pi / wavelength
    omega = c * k
    s, C = mp.sin(phi), mp.cos(phi)
    s2, C2 = s * s, C * C
    P, Q = mp.sin(polarization), mp.cos(polarization)
    cot = C / s
    tanAlpha = (1 - beta0 * C) / (beta0 * s)
    nu = (y * C + z * s) / c
    xi = (-z * C + y * s) * tanAlpha / c
    # Corrected envelope; q_envelope is distinct from the Bessel argument q.
    q_envelope = tanAlpha * (nu * tanAlpha - xi * cot)
    sigma = omega * tauG**2 - 2j * q_envelope
    envelope = tauG * mp.sqrt(omega / sigma) * mp.exp(-omega * (t - nu - xi) ** 2 / sigma)

    Xm = -z - mp.j * k * w0**2 / 2
    Xm2, x2 = Xm**2, x**2
    rho2 = Xm2 + x2
    rho = mp.sqrt(rho2)  # principal complex square root
    q, a = k * rho * s, k**2 * w0**2 * s / 2
    J0, J1 = mp.besselj(0, q), mp.besselj(1, q)
    # Direct, unscaled numerator and denominator are safe with mpmath.
    F = mp.exp(mp.j * (omega * t - k * y * C)) * envelope * (2 / k) / mp.besseli(0, a)

    # Shorthand above: s=sin(phi), C=cos(phi), P=sin(pol), Q=cos(pol).
    Ex = (
        mp.j
        * F
        / (4 * rho * rho2)
        * (
            k * rho * J0 * ((rho2 - x2 + x * Xm * C) * P * s2 + Q * (rho2 + rho2 * C2 - x2 * s2 - x * C * s2 * Xm))
            + J1
            * s
            * (
                P * (-rho2 + 2 * x2 - mp.j * rho2 * Xm * k * s + x * C * (-2 * Xm - mp.j * rho2 * k * s))
                + Q * (-rho2 + 2 * x2 + mp.j * rho2 * Xm * k * s + x * C * (2 * Xm + mp.j * rho2 * k * s))
            )
        )
    )
    Ey = (
        F
        * k
        * s
        / (4 * rho)
        * (J1 * (Q * (Xm - 2 * x * C - Xm * C2) + (1 + C2) * P * Xm) + mp.j * rho * J0 * (Q - P) * s2)
    )
    Ez = (
        mp.j
        * F
        / (8 * rho * rho2)
        * (
            2 * k * rho * J0 * (x * (Q + P) * s2 * Xm + C * (Q * s2 * Xm2 + P * (2 * rho2 - Xm2 * s2)))
            + J1
            * s
            * (
                Q * (-4 * x * Xm + 2 * C * (rho2 - 2 * Xm2) + 2j * rho2 * (x - Xm * C) * k * s)
                + P
                * (
                    -4 * x * Xm
                    - 2 * C * (rho2 - 2 * Xm2)
                    - 2j * rho2 * k * x * s
                    + mp.j * rho2 * Xm * k * mp.sin(2 * phi)
                )
            )
        )
    )
    Bx = (
        -mp.j
        * F
        / (4 * c * rho * rho2)
        * (
            k * rho * J0 * (Q * (-rho2 + x2 + x * C * Xm) * s2 - P * (rho2 + rho2 * C2 - x2 * s2 + x * C * s2 * Xm))
            + J1
            * (
                Q * s * (rho2 - 2 * x2 + mp.j * Xm * rho2 * k * s + x * C * (-2 * Xm - mp.j * rho2 * k * s))
                + P * ((rho2 - 2 * x2) * s + mp.j * rho2 * (-Xm + x * C) * k * s2 + x * mp.sin(2 * phi) * Xm)
            )
        )
    )
    By = (
        F
        * k
        * s
        / (4 * c * rho)
        * (-J1 * (Q * (1 + C2) * Xm + (Xm + 2 * x * C - Xm * C2) * P) + mp.j * rho * J0 * (Q - P) * s2)
    )
    Bz = (
        -mp.j
        * F
        / (4 * c * rho * rho2)
        * (
            J1
            * s
            * (
                P * (x * (2 * Xm - mp.j * k * rho2 * s) + C * (rho2 - 2 * Xm2 - mp.j * Xm * k * rho2 * s))
                + Q * (x * (2 * Xm + mp.j * k * rho2 * s) + C * (-rho2 + 2 * Xm2 + mp.j * Xm * k * rho2 * s))
            )
            + k * rho * J0 * (Xm * (-x + Xm * C) * P * s2 - Q * (x * s2 * Xm + C * (-2 * rho2 + Xm2 * s2)))
        )
    )
    return [v.real for v in (Ex, Ey, Ez, Bx, By, Bz)]


def check_references():
    """Check the committed decimal constants without replacing the Mathematica row."""
    source = Path(__file__).with_name("TWTSTight.cpp").read_text()
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    original = source.split('TEST_CASE("unit::TWTSTight",', 1)[1].split("TEST_CASE(", 1)[0]
    match = re.search(r"twtsTightNumberTest\(\)\(\s*2\.5e-6,\s*1\.0,\s*\{([^}]+)\}", original)
    if match is None:
        raise AssertionError("Cannot find the Mathematica field reference")
    mathematica = list(map(mp.mpf, re.findall(number, match[1])))
    reference = fields("2.5", 1)
    print("// Mathematica comparison: component, absolute deviation, relative deviation")
    for name, expected, actual in zip(("Ex", "Ey", "Ez", "Bx", "By", "Bz"), mathematica, reference):
        absolute = abs(expected - actual)
        relative = absolute / abs(actual)
        print(name, mp.nstr(absolute, 8), mp.nstr(relative, 8))
        assert relative < mp.mpf("5e-13"), name
    assert len(mathematica) == 6

    table = source.split("std::array<Case, 7u> const cases", 1)[1].split("for(auto const& test", 1)[0]
    rows = re.findall(r"\{\s*(" + number + r"),\s*(" + number + r"),\s*\{([^}]+)\}", table)
    assert len(rows) == 7
    for waist, sign, values in rows:
        reference = fields(mp.mpf(waist) / mp.mpf("1e-6"), int(mp.mpf(sign)))
        values = list(map(mp.mpf, re.findall(number, values)))
        assert len(values) == 6
        for expected, actual in zip(values, reference):
            assert abs(expected / actual - 1) < mp.mpf("1e-21")

    # Increasing arithmetic precision must leave the field references stable.
    for beta in ("0.9", "1", "1.1"):
        for waist in ("2.5", "16", "20", "32"):
            for sign in (1, -1):
                values = fields(waist, sign, beta)
                with mp.workdps(mp.dps + 60):
                    finer = fields(waist, sign, beta)
                    assert max(abs(v / f - 1) for v, f in zip(values, finer)) < mp.mpf("1e-80")

    # Independently expand sigma and apply the reduced-coordinate translation.
    for beta in map(mp.mpf, ("0.9", "1", "1.1")):
        for degrees in (5, 30, 90):
            phi = degrees * mp.pi / 180
            S, C = mp.sin(phi), mp.cos(phi)
            a = (1 - beta * C) / (beta * S)
            c, tau = mp.mpf(299792458), mp.mpf("60e-15")
            omega = 2 * mp.pi * c / mp.mpf("800e-9")
            dt = 2 * mp.pi / (omega * (1 - beta * C))
            for sign in (1, -1):
                y, z, t = mp.mpf("1e-6"), sign * mp.mpf("150e-6"), mp.mpf("1e-15")
                nu, xi = (y * C + z * S) / c, (y * S - z * C) * a / c
                sigma = omega * tau**2 - 2j * a * (a * nu - C / S * xi)
                expanded = (
                    omega * tau**2
                    - 2j * (nu + xi) * (C / S) ** 2
                    + 2j / beta * (2 * nu + xi) * C / S**2
                    - 2j / beta**2 * nu / S**2
                )
                spatial = omega * tau**2 - 2j * a**2 * z / (c * S)
                shifted_nu = nu - 17 * beta * C * dt
                shifted_xi = xi - 17 * (1 - beta * C) * dt
                shifted = omega * tau**2 - 2j * a * (a * shifted_nu - C / S * shifted_xi)
                for other in (expanded, spatial, shifted):
                    assert abs(other / sigma - 1) < mp.mpf("1e-80")
                assert abs(((t - 17 * dt) - shifted_nu - shifted_xi) - (t - nu - xi)) / tau < mp.mpf("1e-80")
    print("// PASS: Mathematica, seven C++ rows, 24 precision comparisons, 18 envelope/translation checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, default=100, help="decimal precision (at least 100)")
    parser.add_argument("--check", action="store_true", help="validate references and envelope identities")
    args = parser.parse_args()
    if args.dps < 100:
        parser.error("use at least 100 decimal digits for reference generation")
    mp.dps = args.dps
    if args.check:
        check_references()
    else:
        ratio_references()
        print("\n// waist [m], zSign, {Ex, Ey, Ez, Bx, By, Bz}")
        for waist in ("2.5", "16", "20", "32"):
            for sign in (1, -1):
                print(
                    "{"
                    + mp.nstr(mp.mpf(waist) * mp.mpf("1e-6"))
                    + ", "
                    + str(sign)
                    + ", "
                    + row(fields(waist, sign))
                    + "}"
                )
        # The +z, 2.5 um test retains its Mathematica reference;
        # the corresponding mpmath row is printed for comparison only.
