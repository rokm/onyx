/* Onyx: Demo application
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

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>

#include <boost/program_options.hpp>

#include <onyx/linear_larank/larank.h>
#include "dataset.h"


int main (int argc, char **argv)
{
    Onyx::LinearLaRank::Classifier *classifier;

    Onyx::Example::Dataset datasetTrain;
    Onyx::Example::Dataset datasetTest;

    std::chrono::time_point<std::chrono::system_clock> start, end; // Timings

    std::string filenameTrainingData;
    std::string filenameTrainingLabels;
    std::string filenameTestData;
    std::string filenameTestLabels;

    std::string saveClassifier;
    std::string loadClassifier;

    bool doTraining = false;
    bool doTesting = false;

    unsigned int numEpochs = 10;
    double C = 1.0;
    double tau = 0.0001;

    // *** Print banner ***
    std::cout << "Onyx v.1.0, (C) 2015 Rok Mandeljc <rok.mandeljc@gmail.com>" << std::endl;
    std::cout << std::endl;

    // *** Command-line parser ***
    boost::program_options::options_description commandLineArguments("Onyx - Online Classifier Application");
    boost::program_options::variables_map optionsMap;
    boost::program_options::positional_options_description positionalArguments;

    boost::program_options::options_description argArguments("Arguments");
    argArguments.add_options()
        ("help", "produce help message")
        ("training-data", boost::program_options::value<std::string>(&filenameTrainingData), "name of training .data file")
        ("training-labels", boost::program_options::value<std::string>(&filenameTrainingLabels), "name of training .labels file")
        ("test-data", boost::program_options::value<std::string>(&filenameTestData), "name of test .data file")
        ("test-labels", boost::program_options::value<std::string>(&filenameTestLabels), "name of test .labels file")
        ("save-classifier", boost::program_options::value<std::string>(&saveClassifier), "optional filename to store classifier")
        ("load-classifier", boost::program_options::value<std::string>(&loadClassifier), "optional filename to load classifier")
        ("epochs", boost::program_options::value<unsigned int>(&numEpochs)->default_value(numEpochs), "number of epochs for (re)training")
    ;
    commandLineArguments.add(argArguments);

    boost::program_options::options_description argParameters("Algorithm parameters");
    argParameters.add_options()
        ("C", boost::program_options::value<double>(&C)->default_value(C), "SVM regularization parameter")
        ("tau", boost::program_options::value<double>(&tau)->default_value(tau), "tolerance for choosing new support vectors")
    ;
    commandLineArguments.add(argParameters);

    positionalArguments.add("training-data", 1);
    positionalArguments.add("training-labels", 1);
    positionalArguments.add("test-data", 1);
    positionalArguments.add("test-labels", 1);

    // Parse command-line
    try {
        boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(commandLineArguments).positional(positionalArguments).run(), optionsMap);
    } catch (std::exception &error) {
        std::cout << commandLineArguments << std::endl << std::endl;
        std::cout << "Command-line error: " << error.what() << std::endl;
        return -1;
    }

    // Display help?
    if (optionsMap.count("help")) {
        std::cout << commandLineArguments << std::endl;
        return 1;
    }

    // Validate
    try {
        boost::program_options::notify(optionsMap);
    } catch (std::exception &error) {
        std::cout << commandLineArguments << std::endl << std::endl;
        std::cout << "Argument error: " << error.what() << std::endl;
        return -1;
    }

    doTraining = !filenameTrainingData.empty() && !filenameTrainingLabels.empty();
    doTesting = !filenameTestData.empty() && !filenameTestLabels.empty();

    if (!doTraining && !doTesting) {
        std::cout << "Neither training nor testing dataset provided; nothing to do!" << std::endl;
        return 1;
    }

    if (!doTraining && loadClassifier.empty()) {
        std::cout << "Neither training dataset nor pre-trained classifier provided!" << std::endl;
        return -1;
    }

    // *** Load datasets ***
    if (doTraining) {
        try {
            datasetTrain.load(filenameTrainingData, filenameTrainingLabels);
        } catch (std::exception &error) {
            std::cout << "Failed to load training dataset: " << error.what() << std::endl;
            return -2;
        }
        std::cout << "Loaded training set:" << std::endl;
        std::cout << " data file: " << filenameTrainingData << std::endl;
        std::cout << " labels file: " << filenameTrainingLabels << std::endl;
        std::cout << " samples: " << datasetTrain.numSamples << std::endl;
        std::cout << " features: " << datasetTrain.numFeatures << std::endl;
        std::cout << " classes: " << datasetTrain.numClasses << std::endl;
        std::cout << std::endl;
    }
    if (doTesting) {
        try {
            datasetTest.load(filenameTestData, filenameTestLabels);
        } catch (std::exception &error) {
            std::cout << "Failed to load testing dataset: " << error.what() << std::endl;
            return -2;
        }
        std::cout << "Loaded testing set:" << std::endl;
        std::cout << " data file: " << filenameTestData << std::endl;
        std::cout << " labels file: " << filenameTestLabels << std::endl;
        std::cout << " samples: " << datasetTest.numSamples << std::endl;
        std::cout << " features: " << datasetTest.numFeatures << std::endl;
        std::cout << " classes: " << datasetTest.numClasses << std::endl;
        std::cout << std::endl;
    }

    // *** Classifier ***
    classifier = Onyx::LinearLaRank::create_classifier();
    if (!loadClassifier.empty()) {
        // Load from file
        std::cout << "Loading classifier from file: " << loadClassifier << std::endl;
        std::cout << std::endl;

        std::ifstream stream(loadClassifier, std::ios::binary);
        classifier->loadFromStream(stream);
    } else {
        // Create new classifier
        std::cout << "Creating new classifier..." << std::endl;
        std::cout << std::endl;

        classifier->setC(C);
        classifier->setTau(tau);
    }

    // *** Training (with optional testing) ***
    if (doTraining) {
        start = std::chrono::system_clock::now();

        int sampleRatio = datasetTrain.numSamples / 10;
        std::vector<float> trainError(numEpochs, 0.0);
        std::vector<float> testError(numEpochs, 0.0);

        std::vector<int> sampleIndices(datasetTrain.numSamples);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);

        for (unsigned int epoch = 0; epoch < numEpochs; epoch++) {
            std::cout << "Epoch " << epoch << std::endl;

            // *** Train ***
            // Randomly permute the sample indices
            std::vector<std::vector<int>::iterator> shuffledSampleIndices(sampleIndices.size());
            std::iota(shuffledSampleIndices.begin(), shuffledSampleIndices.end(), sampleIndices.begin());

            std::shuffle(shuffledSampleIndices.begin(), shuffledSampleIndices.end(), std::mt19937{std::random_device{}()});

            for (unsigned int s = 0; s < shuffledSampleIndices.size(); s++) {
                int idx = *shuffledSampleIndices[s];
                const Eigen::VectorXf &sampleFeature = datasetTrain.features[idx];
                int sampleLabel = datasetTrain.labels[idx];

                // Estimate training error
                int label = classifier->predict(sampleFeature);

                if (label != sampleLabel) {
                    trainError[epoch]++;
                }

                // Update
                classifier->update(sampleFeature, sampleLabel, 1.0f);

                // Print progress
                if (s && s % sampleRatio == 0) {
                    std::cout << "Epoch: " << epoch << ": ";
                    std::cout << (10 * s) / sampleRatio << "%";
                    std::cout << " -> training error: " << trainError[epoch];
                    std::cout << "/" << s;
                    std::cout << " = "  << trainError[epoch]/s*100 << "%";
                    std::cout << std::endl;
                }
            }

            if (doTesting) {
                // *** Test ***
                for (unsigned int s = 0; s < datasetTest.numSamples; s++) {
                    const Eigen::VectorXf &sampleFeature = datasetTest.features[s];
                    int sampleLabel = datasetTest.labels[s];

                    if (classifier->predict(sampleFeature) != sampleLabel) {
                        testError[epoch]++;
                    }
                }

                std::cout << "Test error: " << testError[epoch] << "/" << datasetTest.numSamples << " = " << testError[epoch]/datasetTest.numSamples*100 << "%" << std::endl;
                std::cout << std::endl;
            }
        }

        end = std::chrono::system_clock::now();
        std::cout << "Elapsed time: " << std::chrono::duration<float>(end - start).count() << " seconds." << std::endl;
    }

    // *** Save classifier ***
    if (!saveClassifier.empty()) {
        std::cout << "Saving classifier to file: " << saveClassifier << std::endl;
        std::ofstream stream(saveClassifier, std::ios::binary);
        classifier->saveToStream(stream);
        std::cout << "Done!" << std::endl;
    }

    // *** Test - only if we didn't do the training ***
    if (!doTraining && doTesting) {
        float testError = 0.0f;

        start = std::chrono::system_clock::now();

        for (unsigned int s = 0; s < datasetTest.numSamples; s++) {
            const Eigen::VectorXf &sampleFeature = datasetTest.features[s];
            int sampleLabel = datasetTest.labels[s];

            if (classifier->predict(sampleFeature) != sampleLabel) {
                testError++;
            }
        }

        end = std::chrono::system_clock::now();

        std::cout << "Test error: " << testError << "/" << datasetTest.numSamples << " = " << testError/datasetTest.numSamples*100 << "%" << std::endl;
        std::cout << "Elapsed time: " << std::chrono::duration<float>(end - start).count() << " seconds." << std::endl;
        std::cout << std::endl;
    }

    // Cleanup
    delete classifier;

    return 0;
}
