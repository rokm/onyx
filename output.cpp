/* Linear LaRank: Decision function
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

#include "output.h"

namespace LinearLaRank {


Output::Output (int numFeatures)
    : wy(Eigen::VectorXd::Zero(numFeatures))
{
}

Output::~Output ()
{
}


double Output::computeGradient (const Eigen::VectorXd &features, int label, int this_label) const
{
    return (label == this_label ? 1.0 : 0.0) - computeScore(features);
}

double Output::computeScore (const Eigen::VectorXd &features) const
{
    return wy.dot(features);
}

void Output::update (const Eigen::VectorXd &features, double lambda, int pattern_id)
{
    wy += lambda * features;

    // Update indicator value
    double beta_value = getBeta(pattern_id) + lambda;
    if (beta_value != 0.0)  {
        beta[pattern_id] = beta_value;
    } else {
        beta.erase(pattern_id);
    }
}

double Output::getBeta (int pattern_id) const
{
    std::unordered_map<int, double>::const_iterator it = beta.find(pattern_id);
    return (it == beta.end()) ? 0.0 : it->second;
}

bool Output::isSupportVector (int pattern_id) const
{
    return getBeta(pattern_id) != 0;
}

int Output::getNSV () const
{
    return beta.size();
}

double Output::getW2 () const
{
    return wy.squaredNorm();
}


}; // LinearLaRank
