/* Copyright 2013-2017 Heiko Burau, Rene Widera, Richard Pausch
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/lambda/Expression.hpp"
#include "pmacc/algorithms/math/defines/trigo.hpp"

namespace pmacc
{
namespace math
{
namespace math_functor
{

struct Cosf
{
    typedef float result_type;

    DINLINE result_type operator()(const result_type& value) const
    {
        return algorithms::math::cos(value);
    }
};

lambda::Expression<lambda::exprTypes::terminal, mpl::vector<Cosf> > _cosf;

} // math_functor
} // math
} // pmacc

