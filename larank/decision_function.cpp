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

#include "decision_function.h"

namespace LinearLaRank {


DecisionFunction::DecisionFunction (int numFeatures)
    : w(Eigen::VectorXd::Zero(numFeatures))
{
}

DecisionFunction::~DecisionFunction ()
{
}


double DecisionFunction::computeGradient (const Eigen::VectorXd &features, int true_label, int predicted_label) const
{
    return (true_label == predicted_label ? 1.0 : 0.0) - computeScore(features);
}

double DecisionFunction::computeScore (const Eigen::VectorXd &features) const
{
    return w.dot(features);
}

void DecisionFunction::update (const Eigen::VectorXd &features, double lambda, int64_t pattern_id)
{
    // Update hyperplane weights
    w += lambda * features;

    // Update indicator value
    double beta_value = getBeta(pattern_id) + lambda;
    if (std::fabs(beta_value) < 1e-15)  {
        beta.erase(pattern_id); // Clear the value (close enough to 0.0)
    } else {
        beta[pattern_id] = beta_value; // Update
    }
}

double DecisionFunction::getBeta (int64_t pattern_id) const
{
    auto it = beta.find(pattern_id);
    return (it == beta.end()) ? 0.0 : it->second;
}

bool DecisionFunction::isSupportVector (int64_t pattern_id) const
{
    return getBeta(pattern_id) != 0;
}

int DecisionFunction::getNSV () const
{
    return beta.size();
}

double DecisionFunction::getW2 () const
{
    return w.squaredNorm();
}


}; // LinearLaRank
