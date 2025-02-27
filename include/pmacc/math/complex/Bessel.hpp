/* Copyright 2003-2025 Alexander Debus
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

/** @file Bessel.hpp
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

#include "pmacc/math/ConstVector.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    namespace math
    {
        namespace bessel
        {
        } // namespace bessel
    } // namespace math
} // namespace pmacc

#include "pmacc/math/complex/Bessel.tpp"
