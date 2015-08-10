function onyx_app (dataset_prefix, varargin)
    % ONYX_APP (dataset_prefix, ...)
    %
    % Matlab port of the onyx demo application.
    %
    % The training/testing dataset is assumed to be provided in libsvm
    % format, in files named:
    % ${dataset_prefix}-train.data, ${dataset_prefix}-train.labels,
    % ${dataset_prefix}-test.data, and ${dataset_prefix}-test.labels
    %
    % Input:
    %  - dataset_prefix: dataset name prefix
    %  - additional parameters are specified via key/value pairs:
    %     - load_classifier: file to restore classifier from before
    %       training and testing (default: '')
    %     - save_classifier: file to export classifier to after training is
    %       complete (default: '')
    %     - training: enable training (default: true)
    %     - test: enable testing (default: true)
    %     - online_test: enable online testing (default: false)
    %     - num_epochs: number of epochs in training (default: 10)
    %     - classifier_parameters: a cell array of parameters to pass to
    %       the classifier's constructor

    %% Parse arguments
    parser = inputParser();
    parser.addParameter('load_classifier', '', @ischar);
    parser.addParameter('save_classifier', '', @ischar);
    parser.addParameter('training', true, @islogical);
    parser.addParameter('test', true, @islogical);
    parser.addParameter('online_test', true, @islogical);
    parser.addParameter('num_epochs', 10, @isnumeric);
    parser.addParameter('classifier_parameters', {}, @iscell);
    parser.parse(varargin{:});

    %% Initialize variables
    load_classifier = parser.Results.load_classifier;
    save_classifier = parser.Results.save_classifier;

    enable_training = parser.Results.training;
    enable_testing = parser.Results.test;
    enable_online_testing = parser.Results.online_test;
    num_epochs = parser.Results.num_epochs;

    classifier_parameters = parser.Results.classifier_parameters;

    %% Load datasets
    if enable_training,
        try
            fprintf('Loading training dataset...\n');
            training_features = load_data_file( sprintf('%s-train.data', dataset_prefix) );
            training_labels = load_data_file( sprintf('%s-train.labels', dataset_prefix) );
            fprintf('Loaded training dataset!\n\n');
        catch
            fprintf('Failed to load train set; disabling training!\n');
            training_features = [];
            training_labels = [];
            enable_training = false;
        end
    end

    if enable_testing || enable_online_testing,
        try
            fprintf('Loading test dataset...\n');
            testing_features = load_data_file( sprintf('%s-test.data', dataset_prefix) );
            testing_labels = load_data_file( sprintf('%s-test.labels', dataset_prefix) );
            fprintf('Loaded test dataset!\n\n');
        catch
            fprintf('Failed to load test set; disabling testing!\n');
            testing_features = [];
            testing_labels = [];
            enable_testing = false;
            enable_online_testing = false;
        end
    end

    assert(enable_training || ~isempty(load_classifier), 'Neither training dataset nor pre-trained classifier provided!');

    %% Create classifier
    if ~isempty(load_classifier),
        % Load from file
        classifier = onyx.LinearLaRank.import_from_file(load_classifier);
    else
        % Create new
        classifier = onyx.LinearLaRank(classifier_parameters{:});
    end

    %% Training
    if enable_training,
        t = tic();
        classifier.train(training_features, training_labels, ...
            'num_epochs', num_epochs, ...
            'verbose', true, ...
            'test_features', testing_features, ...
            'test_labels', testing_labels);
        t = toc(t);

        fprintf('Elapsed time: %f seconds\n', t);
    end

    %% Save classifier
    if ~isempty(save_classifier),
        classifier.export_to_file(save_classifier);
    end

    %% Test
    % Only if we didn't do the training...
    if ~enable_training && enable_testing,
        t = tic();
        predicted_labels = classifier.predict(testing_features);
        t = toc(t);

        incorrect = sum(predicted_labels ~= testing_labels);

        fprintf('Test error: %d/%d (%.05f%%)\n', incorrect, numel(testing_labels), incorrect/numel(testing_labels)*100);
        fprintf('Elapsed time: %f seconds\n', t);
    end

    %% Online test
    if enable_online_testing,
        t = tic();

        % Permute test samples
        num_test_samples = numel(testing_labels);
        permuted_indices = randperm(num_test_samples);

        incorrect = 0;
        for i = 1:numel(permuted_indices),
            idx = permuted_indices(i);

            % Predict
            predicted_label = classifier.predict(testing_features(:, idx));
            if predicted_label ~= testing_labels(idx),
                incorrect = incorrect + 1;
            end

            % Update
            classifier.update(testing_features(:, idx), testing_labels(idx));
        end

        t = toc(t);

        fprintf('\n');
        fprintf('Online test error: %d/%d (%.05f%%)\n', incorrect, numel(testing_labels), incorrect/numel(testing_labels)*100);
        fprintf('Elapsed time: %f seconds\n', t);
    end
end

function data = load_data_file (filename)
    fid = fopen(filename, 'r');

    header = fgetl(fid);
    header_values = sscanf(header, '%d %d');

    num_samples = header_values(1);
    num_features = header_values(2);

    data = textscan(fid, '%f', inf);
    data = data{1};

    data = reshape(data, [ num_features, num_samples ]);
end
