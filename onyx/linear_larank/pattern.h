/* Onyx: Linear LaRank: Pattern
 * Copyright (C) 2008- Antoine Bordes
 * Copyright (C) 2015 Rok Mandeljc
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ONYX__LINEAR_ARANK__PATTERN_H
#define ONYX__LINEAR_LARANK__PATTERN_H

#include <Eigen/Core>


namespace Onyx {
namespace LinearLaRank {


class Pattern
{
public:
    Pattern (int id, const Eigen::VectorXf &features, int label, float weight = 1.0);
    Pattern ();

    virtual ~Pattern ();

    bool isValid () const;

    void invalidate ();

public:
    int64_t id; // ID (effectively sample number)
    Eigen::VectorXf features; // Feature vector
    int label; // Label
    float weight; // Weight
};


} // LinearLaRank
} // Onyx


#endif
