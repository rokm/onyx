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

#ifndef LINEAR_LARANK__LARANK__DECISION_FUNCTION_H
#define LINEAR_LARANK__LARANK__DECISION_FUNCTION_H

#include <Eigen/Core>

#include <unordered_map>


namespace LinearLaRank {


class DecisionFunction
{
public:
    DecisionFunction (int numFeatures = 0);

    virtual ~DecisionFunction ();

    double computeGradient (const Eigen::VectorXf &features, int label, int this_label) const;
    double computeScore (const Eigen::VectorXf &features) const;

    void update (const Eigen::VectorXf &features, double lambda, int64_t pattern_id);

    double getBeta (int64_t pattern_id) const;
    bool isSupportVector (int64_t pattern_id) const;

    int getNSV () const;
    double getW2 () const;

private:
    // Beta (indicator) values of each support vector
    std::unordered_map<int64_t, double> beta;

    // Hyperplane weights
    Eigen::VectorXf w;
};


}; // LinearLaRank


#endif
