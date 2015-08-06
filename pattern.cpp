/* Linear LaRank: Pattern
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

#include "pattern.h"

namespace LinearLaRank {


Pattern::Pattern (int id, const Eigen::VectorXd &features, int label, double weight)
    : id(id), features(features), label(label), weight(weight)
{
}

Pattern::Pattern ()
    : id(-1)
{
}

Pattern::~Pattern ()
{
}


bool Pattern::isValid () const
{
    return id >= 0;
}

void Pattern::clear ()
{
    id = -1;
}


}; // LinearLaRank
