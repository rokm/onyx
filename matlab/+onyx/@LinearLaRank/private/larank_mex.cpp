/* Onyx: Linear LaRank: MEX interface for Matlab wrapper class
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

#include <onyx/larank/larank.h>

#include <cstring>
#include <map>
#include <memory>

#include <iostream>

// Matlab MEX and matrix API
#include "mex.h"
#include "matrix.h"


// MEX interface commands
enum {
    CommandCreate,
    CommandDelete,
    CommandPredict,
    CommandUpdate,
    CommandSerialize,
    CommandDeserialize,
    CommandGetC,
    CommandSetC,
    CommandGetTau,
    CommandSetTau,
};


// Monotonic counter to ensure we always get unique handle ID
int counter = 0;

// Stored objects
std::map< int, std::unique_ptr<Onyx::LinearLaRank::Classifier> > objects;


// *********************************************************************
// *                              Create                               *
// *********************************************************************
// id = classifier_create ()
static void classifier_create (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 0) {
        mexErrMsgTxt("'Ceate' command requires no input arguments!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("'Create' command requires one output argument!");
    }

    // *** Create LinearLaRank object ***
    objects[counter] = std::unique_ptr<Onyx::LinearLaRank::Classifier>( Onyx::LinearLaRank::create_classifier() );

    // Return the ID
    plhs[0] = mxCreateDoubleScalar(counter);

    // Increment counter
    counter++;
}

// *********************************************************************
// *                              Delete                               *
// *********************************************************************
// classifier_delete (id)
static void classifier_delete (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("'Delete' command requires one argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First argument needs to be a numeric ID!");
    }

    // Get handle ID
    int id = mxGetScalar(prhs[0]);

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }

    objects.erase(iterator);
}

// *********************************************************************
// *                              Predict                              *
// *********************************************************************
template <typename FeatureType>
void __classifier_predict (std::unique_ptr<Onyx::LinearLaRank::Classifier> &classifier, const FeatureType *featuresPtr, unsigned int numClasses, unsigned int numSamples, unsigned int numFeatures, float *labelsPtr, float *scoresPtr)
{
    int predictedLabel;

    // Process all samples
    for (unsigned int i = 0; i < numSamples; i++) {
        Eigen::Map< const Eigen::Matrix<FeatureType, Eigen::Dynamic, 1> > features(featuresPtr + i*numFeatures, numFeatures);

        // Predict
        if (scoresPtr) {
            predictedLabel = classifier->predict(features, Eigen::Ref<Eigen::VectorXf>(Eigen::Map<Eigen::VectorXf>(scoresPtr + i*numClasses, numClasses)));
        } else {
            predictedLabel = classifier->predict(features);
        }

        // Copy label
        if (labelsPtr) {
            labelsPtr[i] = predictedLabel;
        }
    }
}

// [ label, confidence ] = classifier_predict (id, features)
static void classifier_predict (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 2) {
        mexErrMsgTxt("'Predict' command requires two arguments!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First argument needs to be a numeric ID!");
    }

    if (!mxIsSingle(prhs[1]) && !mxIsDouble(prhs[1])) {
        mexErrMsgTxt("Second argument needs to be a single or double vector/matrix!");
    }

    // Get handle ID
    int id = mxGetScalar(prhs[0]);

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Get feature matrix/vector: DxN matrix for N samples
    unsigned int numFeatures = mxGetM(prhs[1]); // rows = feature dimension
    unsigned int numSamples = mxGetN(prhs[1]); // columns = number of samples
    unsigned int numClasses = classifier->getNumClasses();

    // Initialize output
    if (nlhs >= 1) {
        plhs[0] = mxCreateNumericMatrix(1, numSamples, mxSINGLE_CLASS, mxREAL);
    }
    if (nlhs >= 2) {
        plhs[1] = mxCreateNumericMatrix(numClasses, numSamples, mxSINGLE_CLASS, mxREAL);
    }

    // Process all samples
    float *labelsPtr = (nlhs >= 1) ? static_cast<float *>(mxGetData(plhs[0])) : 0;
    float *scoresPtr = (nlhs >= 2) ? static_cast<float *>(mxGetData(plhs[1])) : 0;

    if (mxIsDouble(prhs[1])) {
        // Double-precision features
        /*__classifier_predict<double>(classifier,
            static_cast<const double *>(mxGetData(prhs[1])),
            numClasses, numSamples, numFeatures,
            labelsPtr,
            scoresPtr);*/
    } else {
        // Single-precision features
        __classifier_predict<float>(classifier,
            static_cast<const float *>(mxGetData(prhs[1])),
            numClasses, numSamples, numFeatures,
            labelsPtr,
            scoresPtr);
    }
}


// *********************************************************************
// *                              Update                               *
// *********************************************************************
template <typename FeatureType>
void __classifier_update (std::unique_ptr<Onyx::LinearLaRank::Classifier> &classifier, const FeatureType *featuresPtr, const double *labelsPtr, const double *weightsPtr, unsigned int numSamples, unsigned int numFeatures)
{
    for (unsigned int i = 0; i < numSamples; i++) {
        Eigen::Map< const Eigen::Matrix<FeatureType, Eigen::Dynamic, 1> > features(featuresPtr + i*numFeatures, numFeatures);
        classifier->update(features, labelsPtr[i], weightsPtr ? weightsPtr[i] : 1.0);
    }
}

// classifier_update (id, features, labels, weights)
void classifier_update (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 3 && nrhs != 4) {
        mexErrMsgTxt("'Update' command requires three or four arguments!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First argument needs to be a numeric ID!");
    }

    if (!mxIsSingle(prhs[1]) && !mxIsDouble(prhs[1])) {
        mexErrMsgTxt("Second argument needs to be a single or double vector/matrix!");
    }

    if (!mxIsDouble(prhs[2])) {
        mexErrMsgTxt("Third argument needs to be a dobule type!");
    }

    if (nrhs >= 4 && !mxIsDouble(prhs[3])) {
        mexErrMsgTxt("Fourth argument needs to be a double type!");
    }

    // Get handle ID
    int id = mxGetScalar(prhs[0]);

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Get feature matrix/vector: DxN matrix for N samples
    unsigned int numFeatures = mxGetM(prhs[1]); // rows = feature dimension
    unsigned int numSamples = mxGetN(prhs[1]); // columns = number of samples

    // Validate number of labels
    if (numSamples != mxGetNumberOfElements(prhs[2])) {
        mexErrMsgTxt("Number of labels must match number of samples!");
    }

    // Validate number of weights
    if (nrhs >= 4 && (numSamples != mxGetNumberOfElements(prhs[3]))) {
        mexErrMsgTxt("Number of weights must match number of samples!");
    }

    double *labelsPtr = static_cast<double *>(mxGetData(prhs[2]));
    double *weightsPtr = (nrhs >= 4) ? static_cast<double *>(mxGetData(prhs[3])) : 0;

    // Process all samples
    if (mxIsDouble(prhs[1])) {
        // Double-precision features
        /*__classifier_update<double>(classifier,
            static_cast<const double *>(mxGetData(prhs[1])),
            labelsPtr,
            weightsPtr,
            numSamples,
            numFeatures);*/
    } else {
        // Single-precision features
        __classifier_update<float>(classifier,
            static_cast<const float *>(mxGetData(prhs[1])),
            labelsPtr,
            weightsPtr,
            numSamples,
            numFeatures);
    }
}


// *********************************************************************
// *                             Serialize                             *
// *********************************************************************
// data = classifier_serialize (id)
void classifier_serialize (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("'Serialize' command requires one argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First argument needs to be a numeric ID!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("'Serialize' command requires one output argument!");
    }

    // Get handle ID
    int id = mxGetScalar(prhs[0]);

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Create stream buffer
    std::stringstream stream;

    // Serialize
    classifier->saveToStream(stream);

    // Determine length of the data in the stream; after saveToStream(),
    // the output stream pointer should already be at the end of the
    // stream
    int streamLength = stream.tellp();

    // Allocate Matlab buffer
    plhs[0] = mxCreateNumericMatrix(1, streamLength, mxUINT8_CLASS, mxREAL);

    // Copy the buffer to output
    stream.read(static_cast<char *>(mxGetData(plhs[0])), streamLength);
}


// *********************************************************************
// *                            Deerialize                             *
// *********************************************************************
// data = classifier_deserialize (id, stream)
void classifier_deserialize (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 2) {
        mexErrMsgTxt("'Serialize' command requires two argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First argument needs to be a numeric ID!");
    }

    if (!mxIsUint8(prhs[1])) {
        mexErrMsgTxt("Second argument needs to be uint8 array!");
    }

    if (nlhs != 0) {
        mexErrMsgTxt("'Deserialize' command requires no output arguments!");
    }

    // Get handle ID
    int id = mxGetScalar(prhs[0]);

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Copy input data into stream
    std::stringstream stream;
    stream.write(static_cast<char *>(mxGetData(prhs[0])), mxGetNumberOfElements(prhs[0]));

    // Deserialize stream
    classifier->loadFromStream(stream);
}


// *********************************************************************
// *                        Matlab entry point                         *
// *********************************************************************
void mexFunction (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    int command;

    // Cleanup function - register only once
    static bool cleanupRegistered = false;
    if (!cleanupRegistered) {
        // Yay for C++11 lambdas
        mexAtExit([] () {
            std::cout << "Cleaning up the onyx Linear LaRank MEX wrapper!" << std::endl;
            // Clear all the objects to free their memory
            objects.clear();
        });
    }

    // We need at least one argument - command
    if (nrhs < 1) {
        mexErrMsgTxt("Wrapper requires at least one argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First argument needs to be a numeric command!");
    }

    command = mxGetScalar(prhs[0]);

    // Skip the command
    nrhs--;
    prhs++;

    switch (command) {
        case CommandCreate: {
            return classifier_create(nlhs, plhs, nrhs, prhs);
        }
        case CommandDelete: {
            return classifier_delete(nlhs, plhs, nrhs, prhs);
        }
        case CommandPredict: {
            return classifier_predict(nlhs, plhs, nrhs, prhs);
        }
        case CommandUpdate: {
            return classifier_update(nlhs, plhs, nrhs, prhs);
        }
        case CommandSerialize: {
            return classifier_serialize(nlhs, plhs, nrhs, prhs);
        }
        case CommandDeserialize: {
            return classifier_deserialize(nlhs, plhs, nrhs, prhs);
        }
        default: {
            mexErrMsgTxt("Unrecognized command value!");
        }
    }
}
