/* Demo application: Dataset
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

#ifndef LINEAR_LARANK__APPS__DATASET_H
#define LINEAR_LARANK__APPS__DATASET_H

#include <vector>
#include <set>

#include <Eigen/Core>


namespace Example {


class Dataset
{
public:
    void load (const std::string &featuresFilename, const std::string &labelsFilename);

protected:
    void findFeatureRange ();

public:
    std::vector<Eigen::VectorXf> features;
    std::vector<int> labels;
    std::set<int> classes;

    unsigned int numSamples;
    unsigned int numFeatures;
    unsigned int numClasses;

    Eigen::VectorXf minFeatureRange;
    Eigen::VectorXf maxFeatureRange;
};


}; // Example


#endif
