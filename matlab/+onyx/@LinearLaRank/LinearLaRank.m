classdef LinearLaRank < handle
    % LINEARLARANK Linear LaRank classifier
    %
    % This class wraps a MEX imlementation of Linear LaRank
    %
    % (C) 2015 Rok Mandeljc <rok.mandeljc@gmail.com>

    % Commands enum - keep it in sync with MEX file!
    properties (Access = private, Constant)
        CommandCreate = 0
        CommandDelete = 1
        CommandPredict = 2
        CommandUpdate = 3
        CommandSerialize = 4
        CommandDeserialize = 5
        CommandGetC = 6
        CommandSetC = 7
        CommandGetTau = 8
        CommandSetTau = 9
        CommandGetNumFeatures = 10
        CommandGetNumClasses = 11
        CommandGetClassLabels = 12
        CommandGetNumSeenSamples = 13
    end

    properties (Access = private)
        handle = -1
    end

    properties (SetAccess = private)
        C
        tau
        num_features
        num_classes
        num_seen_samples
        class_labels
    end

    %% Core API
    methods (Access = public)
        function self = LinearLaRank (varargin)
            % self = LINEARLARANK (varargin)
            %
            % Constructs a LinearLaRank classifier.
            %
            % Input:
            %  - key/value pairs:
            %     - C: regularization parameter
            %     - tau: threshold
            %
            % Output:
            %  - self: @LinearLaRank instance

            % Input parser
            parser = inputParser();
            parser.addParameter('C', [], @isnumeric);
            parser.addParameter('tau', [], @isnumeric);

            parser.parse(varargin{:});

            self.handle = linear_larank_mex(self.CommandCreate);

            % Apply C and tau if they are given; otherwise leave at the
            % library-provided defaults
            if ~isempty(parser.Results.C),
                linear_larank_mex(self.CommandSetC, self.handle, parser.Results.C);
            end
            if ~isempty(parser.Results.tau),
                linear_larank_mex(self.CommandSetTau, self.handle, parser.Results.tau);
            end
        end

        function [ labels, scores ] = predict (self, features)
            % [ labels, scores ] = PREDICT (self, features)
            %
            % Performs classification of the provided feature vectors.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %  - features: DxN feature matrix in single- or
            %    double-precision format, where each column corresponds to
            %    a sample
            %
            % Output:
            %  - labels: 1xN vector of predicted labels
            %  - scores: CxN matrix of prediction scores, where each column
            %    corresponds to a scores vector for a sample and C is
            %    number of classes

            if nargout > 1,
                [ labels, scores ] = linear_larank_mex(self.CommandPredict, self.handle, features);
            else
                labels = linear_larank_mex(self.CommandPredict, self.handle, features);
            end
        end

        function update (self, features, labels, weights)
            % UPDATE (self, features, labels, weights)
            %
            % Perform on-line update of the classifier.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %  - features: DxN feature matrix in single- or
            %    double-precision format, where each column corresponds to
            %    a sample
            %  - labels: 1xN vector of sample labels
            %  - weights: optional 1xN vector of sample weights

            if ~exist('weights', 'var') || isempty(weights),
                linear_larank_mex(self.CommandUpdate, self.handle, features, labels);
            else
                linear_larank_mex(self.CommandUpdate, self.handle, features, labels, weights);
            end
        end

    end

    %% Batch training
    methods (Access = public)
        function [ error ] = train (self, features, labels, varargin)
            % [ error ] = TRAIN (self, features, labels, ...)
            %
            % Perform batch training of the classifier using provided
            % features and labels.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %  - features: DxN feature matrix in single- or
            %    double-precision format, where each column corresponds to
            %    a sample
            %  - labels: 1xN vector of sample labels
            %  - additional parameters are specified via key/value pairs:
            %     - num_epochs: number of epochs (default: 10)
            %     - test_features: DxM feature matrix for test samples used
            %       to evaluate classification error at the end of each
            %       epoch (default: [])
            %     - test_labels: 1xM vector of test sample labels (default:
            %       [])
            %     - verbose: verbosity flag (default: false)
            %
            % Output:
            %  - training_error: training error at the end of each epoch
            %  - error: testing error at the end of each epoch (if testing
            %    samples are provided)

            % Parameter parsing
            parser = inputParser();
            parser.addParameter('num_epochs', 10, @isnumeric);
            parser.addParameter('test_features', [], @isnumeric);
            parser.addParameter('test_labels', [], @isvector);
            parser.addParameter('verbose', false, @isvector);

            parser.parse(varargin{:});

            num_epochs = parser.Results.num_epochs;
            test_features = parser.Results.test_features;
            test_labels = parser.Results.test_labels;
            verbose = parser.Results.verbose;

            % Validate input data
            num_train_samples = size(features, 2);
            num_test_samples = 0;

            assert(num_train_samples == numel(labels), 'Number of labels in training dataset does not match the number of samples!');

            if ~isempty(test_features),
                num_test_samples = size(test_features, 2);
                assert(num_test_samples == numel(test_labels), 'Number of labels in test dataset does not match the number of samples!');
            end

            % Train for several epochs
            error = nan(num_epochs, 1);
            for i = 1:num_epochs,
                % Randomly permute the training dataset
                idx = randperm(num_train_samples);

                % Batch update; this precludes us from estimating training
                % error, but should be significantly faster than calling
                % update on each sample individually...
                self.update(features(:, idx), labels(idx));

                % If we have test samples, estimate the error at the end of
                % epoch
                if num_test_samples,
                    predicted_labels = self.predict(test_features);
                    incorrect = sum(predicted_labels ~= test_labels);
                    error(i) = incorrect / numel(test_labels);

                    if verbose,
                        fprintf('Epoch #%d: test error: %d / %d (%.05f%%)\n', i, incorrect, numel(test_labels), error(i)*100);
                    end
                end
            end
        end
    end

    %% Destructor and property getters
    methods
        function delete (self)
            linear_larank_mex(self.CommandDelete, self.handle);
        end

        function value = get.C (self)
            value = linear_larank_mex(self.CommandGetC, self.handle);
        end

        function value = get.tau (self)
            value = linear_larank_mex(self.CommandGetTau, self.handle);
        end

        function value = get.num_features (self)
            value = linear_larank_mex(self.CommandGetNumFeatures, self.handle);
        end

        function value = get.num_classes (self)
            value = linear_larank_mex(self.CommandGetNumClasses, self.handle);
        end

        function value = get.class_labels (self)
            value = linear_larank_mex(self.CommandGetClassLabels, self.handle);
        end

        function value = get.num_seen_samples (self)
            value = linear_larank_mex(self.CommandGetNumSeenSamples, self.handle);
        end
    end

    %% Load/save functions
    methods
        function export_to_file (self, filename)
            % EXPORT_TO_FILE (self, filename)
            %
            % Exports the classifier to a file. Intended for
            % interoperability with other applications using the onyx
            % C++ library.
            %
            % Input:
            %  - self: @LinearLaRank instance
            %  - filename: name of file to export classifier to

            % Open file
            fid = fopen(filename, 'w+');

            % Serialize
            data = linear_larank_mex(self.CommandSerialize, self.handle);

            % Write and close
            fwrite(fid, data, 'uint8');
            fclose(fid);
        end

        function s = saveobj (self)
            % Serialize our classifier's data
            s.SerializedData = linear_larank_mex(self.CommandSerialize, self.handle);
        end
    end

    methods (Static)
        function self = import_from_file (filename)
            % self = IMPORT_FROM_FILE (filename)
            %
            % Restores the classifier from a file. Intended for
            % interoperability with other applications using the onyx
            % library.
            %
            % Input:
            %  - filename: name of file to restore classifier from
            %
            % Output:
            %  - self: restored @LinearLaRank instance

            % Read data from file
            fid = fopen(filename, 'r');
            data = fread(fid, inf, 'uint8=>uint8');
            fclose(fid);

            % Restore classifier from serialized data
            self = onyx.LinearLaRank();
            linear_larank_mex(self.CommandDeserialize, self.handle, data);
        end

        function self = loadobj (s)
            if isstruct(s),
                % Construct LinearLaRank and deserial
                self = onyx.LinearLaRank();
                linear_larank_mex(self.CommandDeserialize, self.handle, self.SerializedData);
            else
                self = s;
            end
        end
    end
end
