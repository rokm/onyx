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

#include "dataset.h"

#include <iostream>
#include <fstream>


namespace Example {


void Dataset::findFeatureRange ()
{
    minFeatureRange = Eigen::VectorXf(numFeatures);
    maxFeatureRange = Eigen::VectorXf(numFeatures);

    float minVal, maxVal;
    for (unsigned int f = 0; f < numFeatures; f++) {
        minVal = features[0](f);
        maxVal = features[0](f);

        for (unsigned int s = 1; s < features.size(); s++) {
            if (features[s](f) < minVal) {
                minVal = features[s](f);
            }
            if (features[s](f) > maxVal) {
                maxVal = features[s](f);
            }
        }

        minFeatureRange(f) = minVal;
        maxFeatureRange(f) = maxVal;
    }
}

void Dataset::load (const std::string &featuresFilename, const std::string &labelsFilename)
{
    std::ifstream featuresStream(featuresFilename.c_str(), std::ios::binary);
    if (!featuresStream) {
        throw std::runtime_error("Could not open input file " + featuresFilename + " !");
    }

    std::ifstream labelsStream(labelsFilename.c_str(), std::ios::binary);
    if (!labelsStream) {
        throw std::runtime_error("Could not open input file " + labelsFilename + " !");
    }

    // Read the header
    unsigned int tmp;
    featuresStream >> numSamples;
    featuresStream >> numFeatures;

    labelsStream >> tmp;
    if (tmp != numSamples) {
        throw std::runtime_error("Number of samples in data and labels file is different!");
    }
    labelsStream >> tmp;

    features.clear();
    labels.clear();

    classes.clear();

    // Read data
    for (unsigned int s = 0; s < numSamples; s++) {
        Eigen::VectorXf sampleFeatures(numFeatures);
        int sampleLabel;

        labelsStream >> sampleLabel;
        for (unsigned int f = 0; f < numFeatures; f++) {
            featuresStream >> sampleFeatures(f);
        }

        labels.push_back(sampleLabel);
        features.push_back(sampleFeatures);

        classes.insert(sampleLabel);
    }

    featuresStream.close();
    labelsStream.close();

    numClasses = classes.size();

    // Find the data range
    findFeatureRange();
}


}; // Example
