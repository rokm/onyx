/* Onyx: Linear LaRank: MEX interface for Matlab wrapper class
 * Copyright (C) 2015-2016 Rok Mandeljc
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

#include <onyx/linear_larank/larank.h>

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
    CommandGetNumFeatures,
    CommandGetNumClasses,
    CommandGetClassLabels,
    CommandGetNumSeenSamples,
    CommandGetDecisionFunctionWeights,
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
    std::ignore = prhs;

    // Validate arguments
    if (nrhs != 0) {
        mexErrMsgTxt("Command requires no input arguments!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
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
    std::ignore = nlhs;
    std::ignore = plhs;

    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("Command requires one input argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

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
static void __classifier_predict (std::unique_ptr<Onyx::LinearLaRank::Classifier> &classifier, const FeatureType *featuresPtr, unsigned int numClasses, unsigned int numSamples, unsigned int numFeatures, float *labelsPtr, float *scoresPtr)
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
            labelsPtr[i] = static_cast<float>(predictedLabel);
        }
    }
}

// [ label, confidence ] = classifier_predict (id, features)
static void classifier_predict (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 2) {
        mexErrMsgTxt("Command requires two input arguments!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (!mxIsSingle(prhs[1])) {
        mexErrMsgTxt("Second input argument needs to be a single-precision vector/matrix!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Get feature matrix/vector: DxN matrix for N samples
    unsigned int numFeatures = static_cast<unsigned int>(mxGetM(prhs[1])); // rows = feature dimension
    unsigned int numSamples = static_cast<unsigned int>(mxGetN(prhs[1])); // columns = number of samples
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


    // Single-precision features
    __classifier_predict<float>(classifier,
        static_cast<const float *>(mxGetData(prhs[1])),
        numClasses, numSamples, numFeatures,
        labelsPtr,
        scoresPtr);
}


// *********************************************************************
// *                              Update                               *
// *********************************************************************
template <typename FeatureType>
static void __classifier_update (std::unique_ptr<Onyx::LinearLaRank::Classifier> &classifier, const FeatureType *featuresPtr, const double *labelsPtr, const double *weightsPtr, unsigned int numSamples, unsigned int numFeatures)
{
    for (unsigned int i = 0; i < numSamples; i++) {
        Eigen::Map< const Eigen::Matrix<FeatureType, Eigen::Dynamic, 1> > features(featuresPtr + i*numFeatures, numFeatures);
        classifier->update(features, static_cast<int>(labelsPtr[i]), weightsPtr ? static_cast<float>(weightsPtr[i]) : 1.0f);
    }
}

// classifier_update (id, features, labels, weights)
static void classifier_update (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    std::ignore = nlhs;
    std::ignore = plhs;

    // Validate arguments
    if (nrhs != 3 && nrhs != 4) {
        mexErrMsgTxt("Command requires three or four input arguments!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (!mxIsSingle(prhs[1])) {
        mexErrMsgTxt("Second input argument needs to be a single-precision vector/matrix!");
    }

    if (!mxIsDouble(prhs[2])) {
        mexErrMsgTxt("Third input argument needs to be a dobule type!");
    }

    if (nrhs >= 4 && !mxIsDouble(prhs[3])) {
        mexErrMsgTxt("Fourth input argument needs to be a double type!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Get feature matrix/vector: DxN matrix for N samples
    unsigned int numFeatures = static_cast<unsigned int>(mxGetM(prhs[1])); // rows = feature dimension
    unsigned int numSamples = static_cast<unsigned int>(mxGetN(prhs[1])); // columns = number of samples

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
    // Single-precision features
    __classifier_update<float>(classifier,
        static_cast<const float *>(mxGetData(prhs[1])),
        labelsPtr,
        weightsPtr,
        numSamples,
        numFeatures);
}


// *********************************************************************
// *                             Serialize                             *
// *********************************************************************
// data = classifier_serialize (id)
static void classifier_serialize (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("Command requires one input argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

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
    size_t streamLength = static_cast<size_t>(stream.tellp());

    // Allocate Matlab buffer
    plhs[0] = mxCreateNumericMatrix(1, streamLength, mxUINT8_CLASS, mxREAL);

    // Copy the buffer to output
    stream.read(static_cast<char *>(mxGetData(plhs[0])), streamLength);
}


// *********************************************************************
// *                            Deserialize                            *
// *********************************************************************
// data = classifier_deserialize (id, stream)
static void classifier_deserialize (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    std::ignore = plhs;

    // Validate arguments
    if (nrhs != 2) {
        mexErrMsgTxt("Command requires two input arguments!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (!mxIsUint8(prhs[1])) {
        mexErrMsgTxt("Second input argument needs to be uint8 array!");
    }

    if (nlhs != 0) {
        mexErrMsgTxt("Command requires no output arguments!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Copy input data into stream
    std::stringstream stream;
    stream.write(static_cast<char *>(mxGetData(prhs[1])), mxGetNumberOfElements(prhs[1]));

    // Deserialize stream
    classifier->loadFromStream(stream);
}


// *********************************************************************
// *                               GetC                                *
// *********************************************************************
// value = classifier_get_c (id)
static void classifier_get_c (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("Command requires one input argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Return value
    plhs[0] = mxCreateDoubleScalar(classifier->getC());
}


// *********************************************************************
// *                               SetC                                *
// *********************************************************************
// classifier_set_c (id, value)
static void classifier_set_c (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    std::ignore = plhs;

    // Validate arguments
    if (nrhs != 2) {
        mexErrMsgTxt("Command requires two input arguments!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 0) {
        mexErrMsgTxt("Command requires no output arguments!");
    }

    if (mxGetNumberOfElements(prhs[1]) != 1 || !mxIsNumeric(prhs[1])) {
        mexErrMsgTxt("Second argument needs to be a numeric scalar!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Set value
    float value = static_cast<float>(mxGetScalar(prhs[1]));
    classifier->setC(value);
}


// *********************************************************************
// *                              GetTau                               *
// *********************************************************************
// value = classifier_get_tau (id)
static void classifier_get_tau (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("Command requires one input argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Return value
    plhs[0] = mxCreateDoubleScalar(classifier->getTau());
}

// *********************************************************************
// *                              SetTau                               *
// *********************************************************************
// classifier_set_tau (id, value)
static void classifier_set_tau (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    std::ignore = plhs;

    // Validate arguments
    if (nrhs != 2) {
        mexErrMsgTxt("Command requires two input arguments!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 0) {
        mexErrMsgTxt("Command requires no output arguments!");
    }

    if (mxGetNumberOfElements(prhs[1]) != 1 || !mxIsNumeric(prhs[1])) {
        mexErrMsgTxt("Second argument needs to be a numeric scalar!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Set value
    float value = static_cast<float>(mxGetScalar(prhs[1]));
    classifier->setTau(value);
}


// *********************************************************************
// *                          GetNumFeatures                           *
// *********************************************************************
// value = classifier_get_num_features (id)
static void classifier_get_num_features (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("Command requires one input argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Return value
    plhs[0] = mxCreateDoubleScalar(classifier->getNumFeatures());
}


// *********************************************************************
// *                          GetNumClasses                            *
// *********************************************************************
// value = classifier_get_num_classes (id)
static void classifier_get_num_classes (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("Command requires one input argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Return value
    plhs[0] = mxCreateDoubleScalar(classifier->getNumClasses());
}


// *********************************************************************
// *                          GetClassLabels                           *
// *********************************************************************
// value = classifier_get_class_labels (id)
static void classifier_get_class_labels (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("Command requires one input argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Get vector of class labels and convert it to double array
    std::vector<int> labels = classifier->getClassLabels();

    plhs[0] = mxCreateDoubleMatrix(1, labels.size(), mxREAL);
    double *outptr = static_cast<double *>(mxGetData(plhs[0]));
    for (unsigned int i = 0; i < labels.size(); i++) {
        outptr[i] = labels[i];
    }
}


// *********************************************************************
// *                        GetNumSeenSamples                          *
// *********************************************************************
// value = classifier_get_num_seen_samples (id)
static void classifier_get_num_seen_samples (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 1) {
        mexErrMsgTxt("Command requires one input argument!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Return value
    double num = static_cast<double>(classifier->getNumSeenSamples());
    plhs[0] = mxCreateDoubleScalar(num);
}


// *********************************************************************
// *                    GetDecisionFunctionWeights                     *
// *********************************************************************
// value = classifier_get_decision_function_weights (id, label)
static void classifier_get_decision_function_weights (int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    // Validate arguments
    if (nrhs != 2) {
        mexErrMsgTxt("Command requires two input arguments!");
    }

    if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First input argument needs to be a numeric ID!");
    }

    if (mxGetNumberOfElements(prhs[1]) != 1 || !mxIsNumeric(prhs[1])) {
        mexErrMsgTxt("Second input argument needs to be a numeric label!");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Command requires one output argument!");
    }

    // Get handle ID
    int id = static_cast<int>(mxGetScalar(prhs[0]));

    // Try to find the classifier
    auto iterator = objects.find(id);
    if (iterator == objects.end()) {
        mexErrMsgTxt("Invalid handle ID!");
    }
    auto &classifier = iterator->second;

    // Get the weights and copy them to output vector
    int label = static_cast<int>(mxGetScalar(prhs[1]));
    const Eigen::VectorXf &weights = classifier->getDecisionFunctionWeights(label);

    plhs[0] = mxCreateNumericMatrix(weights.size(), 1, mxDOUBLE_CLASS, mxREAL);
    double *W = static_cast<double *>(mxGetData(plhs[0]));
    for (int i = 0; i < weights.size(); i++) {
        *W++ = weights[i];
    }
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

    command = static_cast<int>(mxGetScalar(prhs[0]));

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
        case CommandGetC: {
            return classifier_get_c(nlhs, plhs, nrhs, prhs);
        }
        case CommandSetC: {
            return classifier_set_c(nlhs, plhs, nrhs, prhs);
        }
        case CommandGetTau: {
            return classifier_get_tau(nlhs, plhs, nrhs, prhs);
        }
        case CommandSetTau: {
            return classifier_set_tau(nlhs, plhs, nrhs, prhs);
        }
        case CommandGetNumFeatures: {
            return classifier_get_num_features(nlhs, plhs, nrhs, prhs);
        }
        case CommandGetNumClasses: {
            return classifier_get_num_classes(nlhs, plhs, nrhs, prhs);
        }
        case CommandGetClassLabels: {
            return classifier_get_class_labels(nlhs, plhs, nrhs, prhs);
        }
        case CommandGetNumSeenSamples: {
            return classifier_get_num_seen_samples(nlhs, plhs, nrhs, prhs);
        }
        case CommandGetDecisionFunctionWeights: {
            return classifier_get_decision_function_weights(nlhs, plhs, nrhs, prhs);
        }
        default: {
            mexErrMsgTxt("Unrecognized command value!");
        }
    }
}
